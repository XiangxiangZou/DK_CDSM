# Plan 02：Hao 等人的累积式在线更新方法

> 状态：待实施
>
> 前置依赖：Plan 01 已验收
>
> 方法标识：`dktv_accumulative`
>
> 对照方法：`fixed_dko`

## 1. 目标

在 Plan 01 冻结的 encoder、normalizer、初始模型和数据流上，实现 Hao 等人的
累积式在线更新方法。该方法持续加入新数据，但不删除旧数据，是后续评价 Zhang
滑动窗口遗忘机制的必要在线基线。

## 2. 算法边界

固定 encoder 后使用：

```math
z_k=g(x_k),
```

```math
z_{k+1}=A_\tau z_k+B_\tau u_k.
```

这里的公式严格使用 Plan 01 artifact schema v2 冻结的归一化坐标：

```text
x_norm = (x_phys - x_mean) / x_std
u_norm = (applied_torque_phys - u_mean) / u_std
z = [x_norm, phi(x_norm), 1]
z_next = A_tau z + B_tau u_norm
x_phys_next = (C0 z_next) * x_std + x_mean
```

因此公式中的 `u_k` 是 `normalized_applied_torque`，`C0` 输出的是归一化状态。
online update、`direct_refit` oracle 和 fixed DKO 必须加载同一份冻结 normalizer，
禁止把物理单位力矩直接送入 `B_tau`，也不得在在线阶段重新拟合 normalizer。

每收到大小为 `b` 的新批次，就把新批次加入全部历史辨识数据，更新
`A_tau/B_tau`；主实验保持 `C=[I,0]`，论文一致性对照可支持一般 `C_tau`。

本计划不删除旧样本、不维护固定窗口、不实现误差阈值或负更新拒绝。数值异常时
拒绝覆盖当前模型属于安全保护，不等同于 Zhang 的选择性更新。

## 3. 实现内容

1. 开始编码前核对 Zhang 论文中对 Hao 方法的定义及所引用原文，建立公式、符号
   和项目数组之间的对应表；
2. 全历史数据上的 `direct_refit`，作为数值参考；
3. 仅添加新批次的累积递推或等价充分统计量更新；
4. ridge、秩、最小奇异值、条件数和有限性检查；
5. 当前模型、候选模型、更新编号和累计样本数记录；
6. 可完整重放的更新历史；
7. `fixed_dko` 与 `dktv_accumulative` 的统一预测对比；
8. 预测验收通过后，再接入现有同一 MPC 做可选闭环对比。

核心实现应集中在最小模块中，例如：

```text
prediction/dktv/
  least_squares.py
  accumulative_update.py
  online_model.py

tests/dktv/
  test_accumulative_update.py
  test_accumulative_replay.py
```

实验入口只负责加载 Plan 01 artifact、按顺序喂入数据、调用算法和保存结果，不在
入口中重复实现矩阵更新。

## 4. 实施顺序

### Step 01：直接累计拟合

- 第 `tau` 次更新时在全部历史样本上直接拟合；
- 明确 linear/affine 与 exact/learned readout 的矩阵形状；
- 保存每次结果，作为累积递推的数值真值。

### Step 02：累积递推

- 初始化累积统计量；
- 每批只加入新样本，累计样本数必须单调增加；
- 病态时显式使用 ridge 或安全重拟合；
- 保存递推与 direct refit 的矩阵和预测差异。

### Step 03：算法测试

- 良态数据上验证递推与全历史重拟合一致；
- 秩亏和相关数据上保证无未处理的 NaN/Inf；
- 不同批次大小下验证样本计数和模型版本；
- encoder 版本改变时显式判定旧统计量失效。

### Step 04：预测实验

- 在完全相同的数据流上同步运行 fixed 与 accumulative；
- 保存 one-step、多步 rollout 和时变阶段分段误差；
- 记录累计样本数、更新耗时、估计内存和矩阵诊断量；
- commanded/applied input 只作为单独标记的消融。

### Step 05：可选控制验证

只有预测指标通过后，才把已接受模型接入当前同一个 lifted MPC。模型更新后刷新
预测矩阵；控制不可行时继续使用最近可行模型。此步骤不修改控制权重或增加新的
控制算法。

## 5. 第一轮实验设置

| 维度 | 设置 |
| --- | --- |
| 方法 | fixed、accumulative |
| 场景 | 无时变、正弦扰动 |
| 批大小 `b` | 5、10、20 |
| 物理输入字段 | `applied_torque` |
| 模型输入坐标 | `normalized_applied_torque`（使用 Plan 01 固定 `u_normalizer`） |
| 噪声 | 0、低噪声 |
| 变化速率 | slow、medium、fast |
| seed | 开发 1–3 个，最终不少于 10 个 |

指标除 Plan 01 公共指标外，还包括更新前后新批次误差、更新次数、平均/最大更新
时间、累计样本数、内存、秩、条件数和 Koopman 矩阵谱性质。

## 6. 输出

```text
outputs/results/dktv/plan_02/<run>/
  manifest.json
  metrics/{one_step,rollout,update_summary}.json
  arrays/{predictions,update_history}.npz
  figures/
  logs/
```

更新历史至少记录批次边界、累计样本数、模型版本、更新耗时、矩阵诊断量、
direct/recursive 差异、接受状态和失败原因。

## 7. 验收标准

- 累积递推在良态数据上与全历史 direct refit 达到预设容差；
- 历史样本数单调增加，没有隐式删除或遗忘；
- 病态数据不会产生未处理的 NaN/Inf；
- 方法可通过 `dktv_accumulative` 独立运行；
- 与 fixed 使用完全相同的数据、encoder、normalizer 和初始模型；
- one-step、rollout、分时段指标和更新轨迹均已保存；
- 多个 seed 下形成可解释的比较结果，无论结果是否优于 fixed；
- 相关测试通过，`git diff --check` 无错误。

若递推无法稳定复现全历史重拟合，不得进入 Plan 03。
