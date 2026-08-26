# DKTV Plan 02 第二轮 Review

> 审查对象：`DKTV_PLAN_02_REVIEW_FEEDBACK.md` 及其对应代码、测试和新产物
>
> 审查日期：2026-08-25
>
> 审查结论：**上一轮代码与审计问题已关闭；Plan 02 核心算法 Gate 通过，整体仍为有条件通过。**

## 1. 总体结论

本轮修订与反馈报告基本一致。上一轮指出的 aggregate 可比性 Gate、异常批次回放、
更新历史契约、seed 解释文本、并行轨迹语义、内存口径、paired statistics 和方法命名
问题均已落实到代码、测试及新产物中。本次独立复核没有发现新的 P0/P1 级算法错误。

当前可以继续开展 Plan 03 的算法实现和开发性对比，但仍不能把现有 Plan 02 结果称为
canonical 论文基线，原因有两项：

1. 10 个 Plan 02 source run 及 aggregate 都明确标记为 `accepted_noncanonical`；
2. 计划中的 low-noise 与 slow/fast 变化速率消融尚未执行。

因此，准确状态仍应保持为：**核心实现及主预测实验完成，消融实验待补，canonical
重跑待源码版本固化。**

## 2. 上一轮问题关闭情况

| Review 项 | 二次审核结论 | 核对证据 |
| --- | --- | --- |
| P0-01 canonical 基线 | 仍未关闭，状态正确 | source run 阻断项为 `git_worktree_dirty`、`source_plan01_noncanonical`；aggregate 阻断项为 `git_worktree_dirty`、`source_plan02_runs_noncanonical` |
| P0-02 aggregate Gate | 已关闭 | final 至少 10 个不同 seed；11 项比较契约全部通过；记录 aggregate Git、入口源码 hash 和 source manifest hash |
| P1-01 rejected replay | 已关闭 | 坏批次不进入 updater/oracle history；第 6、7 次尝试拒绝后，第 8 次恢复接受，之后继续完成 5 次更新 |
| P1-02 update history | 已关闭 | schema v2 NPZ 保存数值轨迹，JSONL 保存接受状态、原因、策略及完整 diagnostics；`allow_pickle=False` 可读 |
| P1-03 seed 解释文本 | 已关闭 | final aggregate 明确写入 `n=10`，解释文本根据 profile 和实际 seed 数生成 |
| P1-04 batch 物理语义 | 已关闭 | manifest 与公式映射均声明 5 条同步仿真轨迹，`b=5/10/20` 对应 1/2/4 个仿真步 |
| P1-05 内存口径 | 已关闭 | updater statistics 与 oracle raw history 分开统计，recursive/oracle 耗时分开报告 |
| P1-06 实验矩阵 | 部分完成 | zero-noise、medium rate 和三阶段主实验已完成；low-noise、slow/fast 尚未完成；MPC 为可选项 |
| P2-01 paired statistics | 已关闭 | 保存逐 seed difference/ratio、Student-t 95% CI、win/tie count 和 NPZ 数组 |
| P2-02 方法命名 | 已关闭 | 图例和 manifest 均使用 `Hao-style accumulative DKTV (fixed encoder, b=...)` |

## 3. 独立执行与产物核对

本次使用仓库规定解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

独立执行结果：

```text
interpreter verification: passed, Linux
py_compile:               passed
Plan 02 CLI --help:       passed
aggregate CLI --help:     passed
focused pytest:           12 passed in 5.19s
full pytest:              25 passed in 10.31s
```

此外，使用仅含 2 个 seed 的 `final` aggregate 调用按预期在创建结果目录前失败：

```text
ValueError: final profile requires at least 10 distinct seeds
```

### 3.1 异常批次恢复路径

对真实 Plan 01 smoke 模型和 validation stream 注入一个 NaN 后，独立检查得到：

```text
rejected attempts:                 [6, 7]
rejected time steps:               [5, 6]
first accepted attempt afterwards: 8
accepted updates after rejection:  5
final updater sample count:         212
rejected sample count:              4
```

这证明 replay 会丢弃受污染批次、保持最近已接受模型，并在输入恢复有效后继续更新。
rejected record 未执行 oracle check，坏样本也未进入 oracle history。

现有自动化测试只分别断言了 `rejected` 和 `accepted` 均存在，没有直接断言
“最后一次拒绝之后仍有接受更新”。实现行为已经独立确认正确，但建议后续把上述时序
关系补成显式回归断言，避免未来修改造成静默退化。该项属于测试加固，不阻断当前 Gate。

### 3.2 更新历史与数值等价性

主 full run：

```text
outputs/results/dktv/plan_02/20260825_plan02_full_review2b_seed20260825/
```

其更新历史核对结果为：

```text
JSONL decision records: 420
rejected records:       0
NPZ schema version:     2
NPZ arrays:             88
object arrays:          0
accepted records:       420
```

10 个 full run 合计结果为：

```text
accepted updates:               4200
failed updates:                 0
maximum oracle A difference:    2.007867483388992e-11
maximum oracle B difference:    4.5824455341403336e-14
all source acceptance passed:   true
all accepted updates finite:    true
```

该数值远小于 `1e-8` oracle 容差，足以继续作为 Plan 03 滑动窗口更新器的数值参照。

### 3.3 聚合可比性与来源证明

final aggregate：

```text
outputs/results/dktv/plan_02/20260825_plan02_aggregate_final_review_round2b/
```

独立检查确认：

- 10 个 seed 为 `20260825` 至 `20260834`，没有重复；
- config、Plan 02 source files、batch sizes、坐标、stream、方法、rollout horizon、
  stage、Plan 01 scenario 和 update-history schema 的 11 项 Gate 均为 `true`；
- 当前 Plan 02 源码 hash 与 aggregate reference 全部一致；
- 当前 aggregate 入口 hash 与 manifest 一致；
- 10 个 source manifest 文件 hash 与 aggregate 记录全部一致；
- aggregate metrics 和 arrays 均可读取，代表性 PNG 内容、标题 `n=10` 和方法命名正确。

## 4. 数值结论复核

one-step paired difference 定义为 `fixed_dko_rmse - method_rmse`。本次从逐 seed 数组
独立重算得到：

| 方法 | paired mean difference | 95% Student-t CI | wins |
| --- | ---: | ---: | ---: |
| `b=5` | `0.0015140838` | `[0.0007962580, 0.0022319096]` | `10/10` |
| `b=10` | `0.0014774584` | `[0.0007639668, 0.0021909499]` | `10/10` |
| `b=20` | `0.0014312918` | `[0.0007245935, 0.0021379901]` | `10/10` |

三组置信区间均位于零以上，说明在当前数据生成、固定 encoder、五轨迹同步 stream 和
zero-noise/medium-rate 设置下，累积更新的 one-step RMSE 稳定优于 fixed DKO。

但该结论不能扩展成“所有预测时域均更好”。50-step rollout 的均值对三种 batch size
均略差于 fixed，且更长时域结果不存在一致优势。论文中应分别呈现 one-step 与 rollout，
并保留这一负面结果；不能用 one-step 改善替代长时域模型能力结论。

## 5. 剩余工作与风险

### 5.1 正式论文基线前必须完成

1. 由用户确认并固化 Git 版本；
2. 基于 canonical Plan 01 多 seed artifact 重跑 Plan 02；
3. 仅用 canonical source runs 生成 canonical final aggregate；
4. 补齐 low-noise 与 slow/fast rate 消融，或正式修改计划并说明这些实验被移至统一
   最终对比计划。

### 5.2 Plan 03 必须保持的公平比较契约

- 使用相同 Plan 01 artifact、normalizer、encoder 和初始 `A/B`；
- 使用相同 5 条同步轨迹、`time_major_then_trajectory` ordering 和因果顺序；
- 对相同 batch/window 语义、评价窗口、horizon 和 stage 进行比较；
- 继续分开报告 one-step、rollout 和分阶段指标；
- 明确 Hao-style 与 Zhang-style 都是在当前仓库的 fixed-encoder 受控边界下比较。

若要支持真实单机器人在线更新，应另建 single-trajectory 场景，不应与本次五轨迹
aggregate 混合。

### 5.3 非阻断改进

- 在 non-finite replay 测试中显式断言“拒绝后恢复接受”；
- 后续论文图可增加逐 seed paired 点图或 difference CI 图；当前 mean ± sample std
  柱状图可作为概览，但不应作为 paired inference 的唯一展示；
- MPC 闭环对比仍是 Plan 02 可选项，可在模型预测边界稳定后统一放入控制实验。

## 6. Plan 03 Gate

**允许进入 Plan 03 的算法实现、单元测试和 noncanonical 开发实验。**

进入正式论文级 Plan 02/Plan 03 数值对比前，必须先完成 canonical 重跑，并决定如何
处理尚未完成的 noise/rate 消融。本轮没有发现需要阻止 Plan 03 编码工作的核心算法
缺陷。
