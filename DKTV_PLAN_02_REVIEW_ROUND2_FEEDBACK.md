# DKTV Plan 02 第二轮 Review 修正反馈

> 对应审核：`DKTV_PLAN_02_REVIEW_ROUND2.md`
>
> 修正日期：2026-08-26
>
> 结论：**Round 2 建议中的回归测试加固和 paired 可视化已完成；Plan 02 核心 Gate
> 继续通过，结果仍为 noncanonical，noise/rate 消融仍待补。**

## 1. 本轮修改范围

Round 2 审核确认上一轮 P0-02、P1 和 P2 代码/审计问题均已关闭，没有发现新的算法
错误。本轮只修改审核提出的非阻断改进，不改动累积递推公式、数据、模型或评价窗口：

1. 在真实 Plan 01 smoke artifact 的 NaN replay 测试中，显式固定“拒绝后恢复接受”的
   时序关系；
2. final aggregate 增加逐 seed paired difference 散点及 95% Student-t CI 图；
3. 因测试文件属于 source provenance，重新生成 10 个 full runs 和 final aggregate，
   保证产物记录的 source hash 与当前代码一致。

未执行 commit、push、旧产物覆盖或删除。

## 2. 显式恢复回归断言

`tests/test_dktv_accumulative.py` 现在直接断言：

```text
rejected attempts:                 [6, 7]
rejected time steps:               [5, 6]
first accepted attempt afterwards: 8
accepted updates after rejection:  5
final updater sample count:         212
rejected sample count:              4
```

同时继续验证 rejected record 不执行 oracle check、batch disposition 为
`discarded_invalid_batch`、坏批次不进入 updater/oracle history。这样后续若出现“可以
拒绝，但拒绝后不再恢复更新”的静默退化，自动化测试会直接失败。

## 3. Paired statistics 图

`experiments/dktv/aggregate_plan_02.py` 新增：

```text
figures/paired_one_step_difference_ci.png
```

图中每种 batch size 同时显示：

- 10 个 seed 的 `fixed_dko_rmse - method_rmse` paired difference；
- paired mean；
- 95% Student-t confidence interval；
- `difference=0` 参考线。

原 `mean ± sample std` 概览图继续保留，两张图分别承担描述统计和 paired inference
展示，避免用独立 error bar 代替配对结论。

## 4. 新结果产物

10 个 full runs：

```text
outputs/results/dktv/plan_02/20260826_plan02_full_round2_reviewfix_seed20260825/
...
outputs/results/dktv/plan_02/20260826_plan02_full_round2_reviewfix_seed20260834/
```

最终聚合：

```text
outputs/results/dktv/plan_02/20260826_plan02_aggregate_final_round2_reviewfix/
```

重点核查文件：

```text
manifest.json
metrics/multiseed_summary.json
arrays/multiseed_metrics.npz
figures/multiseed_one_step_rmse.png
figures/paired_one_step_difference_ci.png
logs/command.json
```

paired 图为 `1800 x 990` PNG，已实际打开检查；点、均值、置信区间、零参考线和标签均
可辨认。manifest 已记录该图的路径、SHA-256 和文件大小。

## 5. 数值与契约复核

算法及输入未改变，因此数值与上一轮一致：10 个 seed 共 4200 次更新全部接受，最大
recursive/direct-refit `A` 差为 `2.007867483388992e-11`，小于 `1e-8` 容差。

| batch | paired mean difference | 95% Student-t CI | wins |
| --- | ---: | ---: | ---: |
| `b=5` | `0.0015140838` | `[0.0007962580, 0.0022319096]` | `10/10` |
| `b=10` | `0.0014774584` | `[0.0007639668, 0.0021909499]` | `10/10` |
| `b=20` | `0.0014312918` | `[0.0007245935, 0.0021379901]` | `10/10` |

新 aggregate 核查结果：

```text
profile:                         final
seed count / minimum:            10 / 10
comparison-contract checks:      11/11 true
all source acceptance passed:    true
all numeric metrics finite:      true
source hashes match current:     true
aggregate entry hash match:      true
aggregate NPZ arrays:            81
paired arrays:                   48
object arrays:                   0
```

50-step rollout 略差于 fixed 的负面结果没有改变，也没有从报告或产物中移除。

## 6. 测试和检查

解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

完成的验证：

- interpreter verification：通过；
- `py_compile`：通过；
- aggregate CLI `--help`：通过；
- Plan 02 focused pytest：`12 passed in 5.23s`；
- repository full pytest：`25 passed in 8.83s`；
- 10 个重新生成的 full run：全部 engineering acceptance passed；
- final aggregate：acceptance passed，11 项 comparison contract 全部通过；
- aggregate NPZ：`allow_pickle=False` 可读；
- 两张 aggregate PNG：尺寸和内容已检查；
- `git diff --check`：通过（最终交付检查）。

## 7. 未关闭条件

1. 当前结果仍为 `accepted_noncanonical`，阻断项为 `git_worktree_dirty` 和
   `source_plan02_runs_noncanonical`。canonical 重跑需要用户先确认并固化 Git 版本，
   再提供 canonical Plan 01 多 seed artifacts；本轮未自行 commit。
2. low-noise 与 slow/fast rate 消融尚未执行。当前状态仍应表述为“核心实现及主预测
   实验完成，消融实验待补”。
3. 当前证据适用于 fixed encoder、五轨迹同步 stream、zero-noise/medium-rate 场景，
   不应外推为完整 Hao 在线 DNN 方法或单机器人在线流结果。
4. Plan 03 可以继续进行算法实现和 noncanonical 开发实验；正式论文级 Plan 02/03
   对比仍需 canonical 重跑，并保持相同 artifact、ordering、batch/window 和评价契约。

综上，Round 2 审核中唯一明确的测试加固项已闭合，并补充了更适合 paired inference
的可视化证据；没有新的核心算法阻断项。
