# Plan 03：Zhang 等人的滑动窗口在线更新方法

> 状态：待实施
>
> 前置依赖：Plan 01、Plan 02 已验收
>
> 方法标识：`otvdkl_window`、`otvdkl_selective`
>
> 对照方法：`fixed_dko`、`dktv_accumulative`

## 1. 目标

实现 Zhang 等人的固定长度滑动窗口更新方法，并在基础窗口方法可靠后加入阈值
触发和负更新拒绝。重点比较：遗忘旧数据能否比 Hao 累积方法更快适应时变动力学，
以及这种收益是否足以抵消小窗口带来的噪声和病态风险。

## 2. 算法定义

```text
S_cur    长度为 w 的当前拟合窗口
S_new    长度为 b 的新批次
S_out    每次移出的最旧 b 个样本
```

基础 OTVDKL 每次执行：

```text
S_candidate = S_cur - S_out + S_new
```

然后递推更新 `A_tau/B_tau` 及相关逆 Gram 矩阵。窗口长度保持为 `w`，理论上
至少满足 `w >= r+m`，工程上还必须检查实际秩、最小奇异值和条件数。

## 3. 两层实现

### 3.1 基础滑动窗口：`otvdkl_window`

- 每累计 `b` 个样本触发更新；
- 同时加入 `S_new`、删除 `S_out`；
- 数值检查通过后接受候选；
- 数据内存不随总运行时间持续增加。

### 3.2 选择性窗口：`otvdkl_selective`

在基础方法上增加：

1. 当前模型在 `S_new` 上误差不超过 `epsilon` 时跳过更新；
2. 候选模型在同一 `S_new` 上不优于当前模型时拒绝候选。

论文对拒绝后的缓冲处理不够明确，因此显式支持：

```text
discard_on_reject     # 第一版默认
retain_on_reject      # 消融实验
```

manifest 必须保存实际策略，两个策略的结果不能混用。

## 4. 实现内容

1. 当前窗口、新批次和移出批次的数据结构；
2. 候选窗口上的 `direct_refit` 数值参考；
3. 加入新批次与删除旧批次的递推公式；
4. 可逆性、ridge、秩、条件数和有限性检查；
5. accepted、rejected、skipped、failed 状态机；
6. 两种 reject buffer 策略的参数化测试；
7. 四种方法在同一数据流上的预测比较；
8. 预测验收后接入同一 MPC 的可选闭环比较。

增量代码保持最小：

```text
prediction/dktv/
  window_update.py
  selective_update.py

tests/dktv/
  test_window_update.py
  test_selective_update.py
  test_window_replay.py
```

本计划不在线修改 encoder，不重训对 OTVDKL 有利的初始模型，也不包含 SDP
终端控制设计。

## 5. 实施顺序

### Step 01：窗口和直接拟合

- 明确样本对、窗口边界和批次时间戳；
- 实现候选窗口 direct refit；
- 验证每次恰好移入和移出 `b` 个样本；
- 验证窗口长度、样本顺序和矩阵形状。

### Step 02：加入/删除递推

- 分别实现新批次加入和最旧批次删除；
- 检查论文低维逆矩阵的可逆性和条件数；
- 失败时安全回退到候选窗口 direct refit 或保留当前模型；
- 比较递推与 direct refit 的矩阵、预测和耗时。

### Step 03：基础 OTVDKL 实验

- 在良态、秩亏、噪声和快速变化数据上测试；
- 验证固定窗口内存；
- 完成 `otvdkl_window` 独立运行；
- 与 fixed、accumulative 做同步数据流比较。

### Step 04：选择性更新

- 加入 `epsilon` 触发判断；
- 在同一新批次上比较 current 与 candidate；
- 区分 accepted、rejected、skipped_threshold 和 failed_numerical；
- 验证两种拒绝缓冲策略不会导致窗口错位或重复计数。

### Step 05：完整预测比较

- 同步重放四种方法；
- 比较整体与分时段 rollout、适应延迟、耗时和内存；
- 对 `w/b/epsilon`、噪声与变化速率进行消融；
- 最终结论使用不少于 10 个 seed。

### Step 06：可选控制验证

预测 Gate 通过后，使用同一个现有 MPC 比较四种模型。只在接受模型后刷新预测
矩阵，并记录更新时控制跳变、求解失败、张力和关节限制。本步骤不修改 MPC
代价，也不宣称已经获得论文中的稳定性保证。

## 6. 初始实验设置

```text
window_size_w:         100
batch_size_b:          5 or 10
epsilon:               用 Plan 01 验证数据标定
reject_buffer_policy:  discard_on_reject
ridge_lambda:          1.0e-3
```

后续比较 `w={50,100,200}`、`b={5,10,20}`、多档 `epsilon`、不同噪声和
slow/medium/fast 变化速率。

除公共预测指标外，还需保存：

- 相对 fixed 和 accumulative 的分时段改善；
- 动力学变化后的适应延迟；
- candidate/accepted/rejected/skipped 次数；
- 窗口秩、条件数和最小奇异值；
- direct/recursive 差异；
- 平均、P95、P99 和最大更新时间；
- 实际窗口内存。

## 7. 输出

```text
outputs/results/dktv/plan_03/<run>/
  manifest.json
  metrics/{one_step,rollout,segmented,update_summary}.json
  arrays/{predictions,update_history,window_diagnostics}.npz
  figures/
  logs/
```

更新历史必须能重放窗口边界、移入/移出样本、current/candidate 误差、决策、
模型版本、缓冲策略和失败原因。

## 8. 验收标准

- 滑动窗口递推在良态数据上与候选窗口 direct refit 达到预设容差；
- 每次正确移入/移出 `b` 个样本，窗口长度恒定；
- 窗口算法的数据内存不随运行时间增长；
- 选择性更新的四类状态均有测试覆盖并可重放；
- 两种拒绝缓冲策略没有窗口错位或重复计数；
- 四种方法使用相同数据、初始模型和评价器；
- 保存 one-step、rollout、分时段误差、适应延迟、耗时和内存；
- 多个 seed 下形成滑动窗口相对累积方法的可信结论；
- 相关测试通过，`git diff --check` 无错误。

若递推不能复现候选窗口 direct refit，应停止实验并修复公式或批次定义。若窗口
方法没有优于累积方法，应优先检查窗口长度、激励、实际输入和变化速率，并如实
保留否定结果。

Zhang 论文中的 SDP-MPC 和稳定性条件验证在本计划完成后另行规划，不与窗口
更新算法的有效性混为一个验收目标。
