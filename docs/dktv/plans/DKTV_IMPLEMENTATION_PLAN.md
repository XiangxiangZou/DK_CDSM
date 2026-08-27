# DKTV / OTVDKL 实施计划总索引

> 文档版本：v0.3
>
> 当前分支：`OTVDKL`
>
> 更新日期：2026-08-25
>
> 当前状态：计划已拆分，尚未开始算法代码实现

## 1. 总体目标

本分支研究绳驱空间机械臂（CDSM）的在线时变 Deep Koopman 建模。当前工作
拆成三个按顺序验收、可独立运行的小计划：

```text
Plan 01：基础设置与实验契约
    ↓
Plan 02：Hao 等人的累积式在线更新
    ↓
Plan 03：Zhang 等人的滑动窗口在线更新
```

这次拆分以当前仓库已有的 `traj_data / prediction / control / common /
visualization` 工作流为基础，不做大规模目录重构。新增内容遵循根目录
`AGENTS.md` 的职责边界，但不批量移动或复制现有程序。

## 2. 三个子计划

| 计划 | 核心目标 | 方法标识 | 前置依赖 |
| --- | --- | --- | --- |
| [Plan 01](DKTV_PLAN_01_FOUNDATION.md) | 复用当前架构，冻结最小配置、时变数据、初始模型、评价与输出契约 | `fixed_dko` | 无 |
| [Plan 02](DKTV_PLAN_02_HAO_ACCUMULATIVE.md) | 实现只加入新数据、不删除旧数据的累积式在线更新 | `dktv_accumulative` | Plan 01 |
| [Plan 03](DKTV_PLAN_03_ZHANG_SLIDING_WINDOW.md) | 实现加入新批次并删除旧批次的滑动窗口，以及选择性更新 | `otvdkl_window`、`otvdkl_selective` | Plan 01、Plan 02 |

## 3. 方法命名

| 名称 | 含义 | 所属计划 |
| --- | --- | --- |
| DKO | 离线训练后保持固定的 Deep Koopman | Plan 01 基线 |
| DKTV | Hao 等人的累积式在线更新 | Plan 02 |
| OTVDKL | Zhang 等人的基础滑动窗口更新 | Plan 03 |
| OTVDKL* | 带阈值触发和负更新拒绝的滑动窗口更新 | Plan 03 |

当前分支名是 `OTVDKL`，但代码、配置和结果中仍必须区分以上四个方法标识，不能
把 DKTV 和 OTVDKL 混称为同一种算法。

主要依据为 Zhang 等人在 Automatica 2026 发表的论文
“Deep Koopman iterative learning and stability-guaranteed control for unknown
nonlinear time-varying systems”，本地 Markdown 文献位于：

```text
/home/zouxx/Documents/pdf2md_storge/
Zhang 等 - 2026 - Deep Koopman iterative learning and stability-guaranteed control for unknown nonlinear time-varying/
```

## 4. 计划边界

### 4.1 Plan 01：只建立最小公共基础

Plan 01 复用当前仓库已有数据采集、DKUC、控制和绘图能力，只增加一份公共基础
配置、一类正弦时变扰动、必要数据字段、唯一初始模型以及统一指标和 manifest。
不实现在线更新，不引入复杂模型管理，不同时开展多种物理时变场景。

### 4.2 Plan 02：只实现累积

Plan 02 每次加入新批次，并让全部历史数据继续参与模型估计。此阶段不删除旧
数据，不实现滑动窗口，也不实现 Zhang 的阈值触发和候选拒绝。

### 4.3 Plan 03：再实现遗忘与选择

Plan 03 才实现固定长度窗口的“加入新批次、移除最旧批次”。基础窗口通过数值
验收后，再加入阈值触发与负更新拒绝，形成 OTVDKL*。

## 5. 共用约束

### 5.1 公平比较

四种方法必须使用相同的：

- 数据流、样本顺序和数据划分；
- 固定 encoder、normalizer 与初始 `A0/B0/C0`；
- 随机种子和时变扰动；
- one-step、rollout 和分时段指标；
- 若进入控制实验，则使用同一参考、MPC 权重、约束和求解器设置。

在线辨识默认使用实际施加的等效关节力矩 `applied_torque`。第一版模型采用
DKUC 风格的 `g(x)=[x,phi(x)]`，而不是当前 DKAC 的状态相关输入编码。

### 5.2 时变判据

状态、参考或控制输入随时间变化不等于动力学时变。Plan 01 必须通过“相同状态、
相同输入、不同绝对时刻”的下一状态对比，证明启用扰动后演化规律确实随时间
改变。

### 5.3 输出

新结果只保存在根目录：

```text
outputs/data/{raw,processed,rejected}/
outputs/models/dktv/
outputs/results/dktv/{plan_01,plan_02,plan_03}/
```

每次结果至少包含 `manifest.json`、数值指标、可重绘原始数组、图表和日志。旧
目录中的历史输出不在本轮迁移或删除。

## 6. 执行顺序与停止条件

1. Plan 01 冻结公共契约，并证明 fixed DKO 在时变阶段存在可识别退化；
2. Plan 02 证明累积递推能复现全历史 direct refit；
3. Plan 03 证明滑动窗口递推能复现候选窗口 direct refit；
4. 最后才在同一数据流上汇总四种方法的预测对比；
5. 预测结论稳定后，才接入同一 MPC 做闭环比较。

任何阶段的递推公式若无法通过 direct refit 数值对照，就停止进入下一计划。算法
没有优于基线也应保留为有效的否定结果，不通过挑选单条有利轨迹掩盖。

## 7. 本轮暂不包含

Zhang 论文中的 SDP 终端控制设计和稳定性条件验证暂不放入这三个计划。它应在
Plan 03 完成后另立控制计划，以分别回答：

1. 在线模型更新是否有效；
2. 预测改善是否转化为闭环改善；
3. 稳定性假设、状态相关绳张力约束和实时性是否成立。

这样可以避免把算法复现、CDSM 适配和稳定性控制一次性堆在同一个计划中。
