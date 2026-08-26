# DKTV Plan 02 执行情况 Review

> 审查对象：`DKTV_PLAN_02_HAO_ACCUMULATIVE.md`、
> `DKTV_PLAN_02_EXECUTION_FEEDBACK.md`
>
> 审查日期：2026-08-25
>
> 审查结论：**核心算法 Gate 通过；Plan 02 整体有条件通过**

## 1. 总体结论

Plan 02 已经实现了固定 encoder/normalizer 下的累积式 `A/B` 在线更新，并形成
direct-refit oracle、因果预测、三个 batch size、10-seed 汇总和可重放产物。
本次独立复跑没有发现递推公式或主要数值结果错误。

以下核心结论成立：

- updater 只对 Gram/Cross 充分统计量做加法，没有删除历史统计量；
- `b=5/10/20` 的递推结果均在 `1e-8` 容差内复现全历史 direct refit；
- 10 个 seed 共 4200 次更新全部接受，最大 `A` 差为
  `2.007867483388992e-11`；
- fixed 与 accumulative 使用相同 Plan 01 数据、模型、normalizer、输入序列和评价
  窗口；
- 报告保留了 50-step 略差于 fixed 的结果，没有只汇报正结果；
- 固定 encoder 是对 Hao 累积更新机制的受控复现，而不是 Hao 全算法的逐项复现，
  报告已经明确披露这一边界。

因此，可以进入 Plan 03 的滑动窗口算法开发。但当前结果仍是 noncanonical，且
汇总器、异常 replay 和更新历史契约存在需要修正的问题。在这些问题解决前，不应
把现有 10-seed aggregate 定义为最终论文基线，也不建议立即开展 Plan 03 的正式
canonical 数值对比。

## 2. 本次独立核对

### 2.1 代码与产物

本次检查了：

- Plan 02 计划、执行报告和公式—数组映射；
- `configs/dktv/plan_02.json`；
- `least_squares.py`、`accumulative_update.py`、`online_model.py`；
- Plan 02 单次运行入口和多 seed 汇总入口；
- Plan 01 seed override 改动；
- Plan 02 自动化测试；
- primary full、10-seed aggregate、更新器状态、metrics、arrays、logs 和代表性图。

独立核对结果：

- 10 个 source run 的 Plan 02 config SHA-256 完全一致；
- 10 个 run 的源码文件 hash 集合完全一致；
- batch size 均为 `[5, 10, 20]`；
- 坐标契约、`time_major_then_trajectory` 排序和
  `predict_then_observe_then_update` 因果顺序一致；
- seed 为 `20260825` 至 `20260834`，无重复；
- 累计接受更新数为 4200，失败数为 0；
- aggregate 中 one-step 均值和 sample standard deviation 与报告一致；
- `b=5/10/20` 在 10 个 seed 的 one-step paired comparison 中均为 10/10 优于
  对应 fixed DKO；
- aggregate PNG 可以正常打开，标题中的 `n=10` 与数据一致；
- primary full 的 5 个 metrics、5 个 arrays/updater、3 张图和审计日志均存在。

### 2.2 独立运行检查

使用解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

独立执行结果：

```text
Python verification: configured executable, Linux
py_compile:          passed
Plan 02 CLI --help:  passed
pytest:              22 passed in 10.10s
git diff --check:    passed
new-file checks:     passed
```

其中 Plan 02 测试为 9 项；其余 13 项为 Plan 01 foundation 回归测试。

## 3. 验收条目复核

| Plan 02 验收条目 | Review 结论 | 说明 |
| --- | --- | --- |
| 递推复现 direct refit | 通过 | 10-seed 最大 `A` 差约 `2.01e-11`，远小于 `1e-8` |
| 历史样本只增不减 | 通过 | 每个 seed 从 1440 增至 2640；更新次数与 `b=5/10/20` 一致 |
| 病态数据有限 | updater 层通过 | rank-deficient 和 non-finite 单元测试通过；完整 replay 的拒绝路径仍有缺口 |
| 方法可独立运行 | 通过 | module CLI、smoke E2E 和 full runs 均存在 |
| 与 fixed 共用模型和数据 | 通过 | Plan 01 artifact、normalizer、stream 和窗口一致 |
| one-step、rollout、分阶段与更新轨迹 | 基本通过 | 预测指标齐全；更新历史缺少逐批 `accepted/reason` 等字段 |
| 多 seed 比较 | 当前场景通过 | 已有 10 seed；low-noise 与 slow/fast 网格尚未执行 |
| 自动化检查 | 通过 | 本次独立复跑 `22 passed`，编译和 CLI 检查通过 |
| canonical 论文基线 | 未通过 | source Plan 01 和 Plan 02 工作树均为 noncanonical |

## 4. 优先问题

### P0-01：当前 Plan 02 结果仍不能作为 canonical 基线

执行报告对此判断正确。primary full manifest 同时记录了：

```text
git_worktree_dirty
source_plan01_noncanonical
```

所有 10 个 source run 和 aggregate 都是 `accepted_noncanonical`。现有结果可以保留
为开发与审查证据，但最终论文数值应在明确 Git 版本和 canonical Plan 01 artifact
上重新生成。

建议在完成本 review 的必要修正后：

1. 固化 Plan 01/Plan 02 源码版本；
2. 从 canonical Plan 01 多 seed runs 重新生成 Plan 02；
3. 保留当前 run，不覆盖、不删除；
4. canonical aggregate 只接收 canonical source runs。

### P0-02：aggregate 的 canonical 与可比性 Gate 不完整

`aggregate_plan_02.py` 当前只强制检查：source run 通过、seed 不重复、方法名一致。
它没有在代码中强制比较：

- Plan 02 config hash；
- source-file hash；
- batch sizes；
- 坐标契约；
- stream 排序、步数和 trajectory 数；
- horizons 与阶段定义；
- source Plan 01 schema/场景；
- 指标是否 finite。

本次 10 个 run 经独立检查后确实满足这些一致性条件，所以当前 aggregate 数字本身
仍然有效。但是汇总器理论上可以把不兼容实验合并后仍写成 `acceptance.passed=true`。

此外，aggregate manifest 只记录 `git dirty`，没有记录 aggregate 自身的 Git
branch/commit 和 `aggregate_plan_02.py` hash。未来即使工作树 clean，它仍缺少精确
追溯汇总代码的证据。

建议为 aggregate 增加 `comparison_contract`，逐项校验并写入 manifest；同时记录
Git provenance、汇总入口 hash，并要求 final profile 至少 10 个 seed。

## 5. 需要完善的问题

### P1-01：数值失败保护只在 updater 层闭合，完整 replay 仍会中断

`AccumulativeKoopmanUpdater.update()` 能在 non-finite batch 上保持原模型和统计量
不变，这一点测试通过。但是 `run_accumulative_replay()` 在调用 updater 前已经把
该批数据追加到 direct-oracle history，随后无条件执行 `direct_refit()`。

本次向真实 replay 注入一个 NaN 后，完整流程直接抛出：

```text
ValueError: z_next must be finite and match z_current
```

同时，`update_summary()` 默认每条记录都有非空 diagnostics，也无法汇总 rejected
update。因此当前实现还不能在完整实验中“拒绝坏批次、继续使用最近可行模型并保存
失败原因”。

建议：

- 只有 recursive update 接受后才把 batch 加入 oracle history；或给 oracle 使用
  独立的 accepted-history；
- rejected update 不执行包含坏批次的 direct refit；
- `update_summary()` 支持 diagnostics 为 `None`；
- 增加 replay/CLI 级 non-finite batch 测试，而不只测试 updater；
- 明确失败后 pending batch 是丢弃、隔离还是重试。

### P1-02：保存的更新历史没有满足计划中的审计字段

内存中的 `UpdateResult` 包含 `accepted`、`reason` 和完整 diagnostics，但
`arrays/update_history.npz` 没有保存：

- `accepted`；
- `reason`；
- recursive rank、minimum singular value、condition number；
- recursive `spectral_radius_A` 和 `finite`。

当前 NPZ 主要保存时间、样本计数、batch RMSE 和 oracle diagnostics。由于本次所有
更新都接受，summary 能说明结果；但一旦出现拒绝，就无法仅依赖 update history
完整重建逐批决策。这与计划要求的“接受状态和失败原因”不完全一致。

建议定义 update-history schema version，并完整保存决策字段。字符串原因可单独保存
为 JSONL，数值字段继续保存为 NPZ，避免使用 pickle。

### P1-03：最终 10-seed manifest 仍写着 “Three seeds”

`20260825_160810_plan02_aggregate_final_10seed/manifest.json` 的
`seed_count=10` 且 source run 数也是 10，但 `interpretation_limit` 仍为：

```text
Three seeds satisfy the development-stage multi-seed check...
```

这是 `aggregate_plan_02.py` 中的硬编码文本错误。建议根据 seed 数量和明确的
`development/final` profile 生成说明；修复后重新生成 aggregate，不要手工修改旧
manifest，因为旧 manifest 的 hash 已被记录。

### P1-04：batch size 的物理含义需要明确

在线 stream 含 5 条验证轨迹，排序为 `time_major_then_trajectory`。每个时间步先对
5 条轨迹全部预测，再把 5 个 snapshot 放入 pending batch。因此：

```text
b=5  -> 每个仿真时间步更新一次
b=10 -> 每两个仿真时间步更新一次
b=20 -> 每四个仿真时间步更新一次
```

这是一种合法的“并行轨迹在线学习”设置，也没有当前目标泄漏；但它不等同于单台
真实机械臂连续运行时每个时间步只产生一个 snapshot。若论文目标包括真实单机在线
适应，需要增加 single-trajectory stream，或明确说明当前结果来自 5 个同步仿真
trajectory 的批量更新。

Plan 03 必须沿用完全相同的 ordering 才能公平比较遗忘机制，不能在不标记的情况
下把 `b` 改解释为时间步数。

### P1-05：4896 bytes 只是 updater 统计量，不是完整实验内存

`statistics_memory_bytes=4896` 正确等于 Gram 和 Cross 的固定内存，但实验 replay
还保留并不断拼接 `history_z/history_next/history_u`，用于逐批 direct refit。因而：

- deployable accumulative updater 的统计量内存是常数；
- 带 oracle 的验收实验总内存随历史样本增长。

执行报告目前写的是“每个 updater 的充分统计量内存”，措辞基本准确。后续论文
比较应分别报告 updater-only 与 oracle-enabled benchmark，更新时间也不要把
recursive solve 和 direct-oracle 验证混为同一指标。

### P1-06：计划中的实验矩阵尚未全部执行

本轮已完成 zero-noise、公共正弦扰动、三阶段和 10 seed，但尚未完成：

- low-noise；
- slow/medium/fast 变化速率网格中的 slow/fast；
- 可选 MPC。

MPC 本来就是可选项，不阻断预测 Gate。noise/rate 消融则属于计划第一轮实验设置，
因此 Plan 02 更准确的状态应是“核心实现及主预测实验完成，消融实验待补”，而不是
全部工作无条件完成。

## 6. 论文结果层面的建议

### P2-01：多 seed 应增加 paired statistics

当前汇总保存各方法的 mean ± sample standard deviation，这是有效的描述统计。由于
每个 accumulative run 与 fixed DKO 使用同一个 seed、数据和 artifact，更合适的
论文统计单位是逐 seed paired difference 或 paired ratio。

本次独立复算得到 one-step paired difference 全部为正，10 个 seed 中：

```text
b=5:  10/10 wins, mean(fixed - method) = 0.00151408
b=10: 10/10 wins, mean(fixed - method) = 0.00147746
b=20: 10/10 wins, mean(fixed - method) = 0.00143129
```

建议 aggregate 额外保存 paired improvement 的均值、标准差、置信区间和逐 seed
数组。不要仅根据两个独立 error bar 是否重叠判断差异。

### P2-02：方法命名应持续保留复现边界

当前实现冻结 encoder，只更新 `A/B`；Hao 原方法还涉及 DNN 参数继续优化。建议
论文和图例使用类似：

```text
Hao-style accumulative DKTV (fixed encoder)
```

`dktv_accumulative` 可继续作为代码标识，但正文不要简称为“完整 Hao 方法复现”。

## 7. 建议的最小修正顺序

1. 修复完整 replay 的 rejected-update 路径；
2. 补齐 update-history 的 accepted/reason/diagnostics 契约及测试；
3. 给 aggregate 增加同配置可比性检查和自身 provenance；
4. 修正 10-seed manifest 的 “Three seeds” 文本；
5. 在公式映射和报告中明确五轨迹并行 stream 与 batch size 含义；
6. 给 aggregate 增加 paired statistics；
7. 决定 noise/rate 消融是在 Plan 02 补齐，还是统一放入最终对比计划；
8. 固化源码后重新生成 canonical Plan 01、Plan 02 和 aggregate。

## 8. Plan 03 Gate

可以立即开展：

- Zhang 滑动窗口 add/remove 充分统计量；
- sliding-window direct-refit oracle；
- window/selective updater 的单元测试；
- 与 Plan 02 相同坐标、stream ordering 和评价窗口的接口复用。

建议在正式运行 Plan 03 对比实验前完成：

- P1-01 的 rejected-update replay 修复；
- P1-02 的统一 update-history schema；
- P0-02 的 aggregate comparison contract。

最终 canonical 对比仍需等待 P0-01 完成。现有
`20260825_155915_plan02_full_accumulative_reviewed` 和
`20260825_160810_plan02_aggregate_final_10seed` 应作为 noncanonical 审查证据保留。
