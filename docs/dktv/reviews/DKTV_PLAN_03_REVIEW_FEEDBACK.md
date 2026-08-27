# DKTV Plan 03 审查整改反馈报告

> 对照审查：`DKTV_PLAN_03_REVIEW.md`
>
> 整改日期：2026-08-26
>
> 最终结论：P1、P2 意见均已实现并经新 smoke、10-seed full 和 final 聚合验证；
> engineering acceptance 通过，但结果仍为 noncanonical，预测 Gate 仍未通过。

## 1. 整改结论

本轮没有修改旧 gatefinal manifest 或旧结果。所有修复均通过代码实现后重新运行，旧产物
保留为历史证据，新结果使用 `reviewfix` 标识。

| 审查项 | 整改状态 | 核查证据 |
| --- | --- | --- |
| P1-01 `run.log` 哈希失配 | 已完成 | 最终日志先写入，再快照 `result_files`；10 个 full run 共 190/190 个结果文件验证通过 |
| P1-02 aggregate canonical/provenance | 已完成 | 聚合时 Git dirty、source canonical、acceptance 共同决定 canonical；保存聚合源码 provenance 和 blockers；聚合前验证全部 source 结果文件 |
| P1-03 计时混用 | 已完成 | 分离 recursive、direct oracle、fallback、total 四类耗时，并纳入 10-seed 聚合 |
| P2-01 fallback 统计重锚定 | 已完成 | fallback 从 raw candidate window 重算 `G/H/P`；新增精确统计同步测试 |
| P2-02 epsilon 同流标定 | 已完成 | 使用固定 seed `20260824` 的独立 MuJoCo nominal calibration stream，明确无评价流重叠 |
| P2-03 噪声 clean truth | 已完成 | 更新使用 observed state；同时保存 clean/observed truth 的 one-step、rollout 和聚合结果 |
| P2-04 finite 检查 | 已完成 | 显式检查 Woodbury 系统、统计量、inverse、recursive `A/B/theta` 和差值；新增溢出输入测试 |

## 2. 主要实现变化

### 2.1 窗口递推与数值稳健性

`SlidingWindowKoopmanUpdater` 现在分别记录：

```text
recursive_candidate_time_s
direct_refit_oracle_time_s
fallback_time_s
total_update_time_s
```

其中 recursive 时间不包含 direct-refit oracle；total 时间覆盖完整更新、oracle、判定和
可能的 fallback。`update_time_s` 仅作为兼容字段保留，并等于 total。

发生 `direct_refit_fallback` 时，不再沿用累计加减得到的 `G/H/P`，而是从
`candidate_z/candidate_u/candidate_next` 重建完整窗口统计量并重新求逆。极端尺度导致的
非有限 Woodbury 系统会返回 `failed_numerical`，且不会改变当前模型、窗口或统计量。

### 2.2 独立 epsilon 标定与双真值噪声评估

epsilon 标定合同冻结为：

```text
source              independent_mujoco_nominal_calibration_stream
seed                20260824
trajectory_count    5
configured_steps    120
evaluation overlap  false
```

每个 run 保存 `arrays/epsilon_calibration_stream.npz`、质量检查、标定批次 RMSE 和数据文件
哈希。标定 seed 与 10 个评价 seed `20260825..20260834` 均不同。

噪声场景现在保存：

- `states_observed`：作为 lift、在线更新和预测初始状态；
- `states_clean`：作为主要物理真值；
- clean-truth 和 observed-truth 两套 one-step、全部 rollout horizon 与分段指标。

聚合器同时汇总所有 rate/noise 场景的两套 one-step 和 rollout 指标，而不再只汇总场景
one-step。

### 2.3 产物完整性与 canonical 合同

Plan 03 在写入最终状态日志后才生成 `result_files` 快照。CLI 集成测试会逐个重算
manifest 所列文件的 SHA-256 和字节数。

聚合器现在：

- 在读取 source run 后逐个验证其 `result_files`；
- 保存配置、聚合器、统计依赖、在线模型和 `AGENTS.md` 的源码哈希；
- 使用 `acceptance passed AND all sources canonical AND aggregate git clean` 判定 canonical；
- 显式保存 `canonical_blockers`。

## 3. 新运行与输出

### 3.1 Smoke

```text
outputs/results/dktv/plan_03/20260826_plan03_smoke_reviewfix
```

结果：`accepted_noncanonical`，engineering acceptance 全部通过，19/19 个结果文件哈希
与字节数一致。

### 3.2 10 个 full runs

| Seed | Plan 01 source | 新 Plan 03 run |
| ---: | --- | --- |
| 20260825 | `20260825_145620_plan01_full_baseline_reviewed` | `20260826_plan03_full_reviewfix_seed20260825` |
| 20260826 | `20260825_155934_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260826` |
| 20260827 | `20260825_155956_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260827` |
| 20260828 | `20260825_160539_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260828` |
| 20260829 | `20260825_160555_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260829` |
| 20260830 | `20260825_160610_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260830` |
| 20260831 | `20260825_160625_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260831` |
| 20260832 | `20260825_160641_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260832` |
| 20260833 | `20260825_160656_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260833` |
| 20260834 | `20260825_160711_plan01_full_plan02_seed` | `20260826_plan03_full_reviewfix_seed20260834` |

10/10 engineering acceptance 为 true，10/10 prediction Gate 为 false，10/10 状态为
`accepted_noncanonical`。150 个 Plan 03 source provenance 记录与当前文件一致；10 个 run
共 190 个结果文件全部通过聚合器的 SHA-256 与字节数复核。

### 3.3 最终聚合

最终替代聚合：

```text
outputs/results/dktv/plan_03/20260826_plan03_aggregate_final_reviewfix2_10seed
```

状态为 `accepted_noncanonical`，10 个 seed、比较合同、finite metrics 和 source 结果文件
验证均通过。聚合 manifest 的 6 个源码 provenance 记录及 4 个聚合 artifact 哈希均已
再次核对。较早的 `20260826_plan03_aggregate_final_reviewfix_10seed` 已保留，但不包含最终
增加的 10-seed timing 摘要，应由 `reviewfix2` 结果替代。

## 4. 关键数值结果

### 4.1 主预测结果

| 方法 | one-step RMSE mean |
| --- | ---: |
| fixed DKO | `0.0047948954` |
| accumulative | `0.0033174370` |
| window | `0.0023809613` |
| selective（独立标定） | `0.0032741586` |

window 相对 accumulative 的配对改善：

```text
mean difference = +0.000936476
95% Student-t CI = [+0.000707550, +0.001165401]
wins             = 10/10
```

独立标定后的 selective 相对 accumulative：

```text
mean difference = +0.000043278
95% Student-t CI = [-0.000542213, +0.000628770]
wins/losses      = 8/2
```

因此本轮不再沿用旧报告中 selective 明确优于 accumulative 的表述；独立标定后的配对区间
跨越 0，只能视为当前配置下无明确优势。

window rollout 均值：

| Horizon | accumulative | window | 结论 |
| ---: | ---: | ---: | --- |
| 10 | `0.0173204` | `0.0126233` | window 较好 |
| 20 | `0.0318646` | `0.0295022` | window 较好 |
| 50 | `0.0693609` | `0.1188000` | window 明显较差 |
| 100 | `0.0829999` | `0.3215372` | window 明显较差 |

长 rollout 结论未改变，所以 10 个 run 的控制 Gate 均为 false，未执行可选 MPC。

### 4.2 拆分计时

主 window 的 10-seed 平均口径：

| 指标 | 均值 |
| --- | ---: |
| total update | `0.697825 ms` |
| recursive candidate（不含 oracle） | `0.315601 ms` |
| direct-refit oracle | `0.254777 ms` |
| fallback（按全部 candidate 平均） | `0.002518 ms` |
| recursive/direct ratio | `1.23785` |
| fallback count | `3.8/run`，合计 `38` |

当前小维度 `w=100,b=10` 配置下，纯 recursive candidate 反而约为 direct oracle 的
`1.24x`，不能宣称递推已有在线速度优势。direct oracle 仍只用于验证，不属于目标部署
路径；该结论现在由分离计时而非混合总耗时支持。

### 4.3 clean/observed truth 示例

在 `rate1_noise0.001` 的主 window 聚合中：

| Truth | one-step mean | rollout-100 mean |
| --- | ---: | ---: |
| clean | `0.00263546` | `0.30535977` |
| observed | `0.00281541` | `0.30536214` |

这两套结果和对应原始数组均已保存，噪声鲁棒性结论不再把 noisy observation 当作唯一
物理真值。

## 5. 测试与检查

使用解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

已执行：

- Plan 03 修改文件 `py_compile`：通过；
- window/selective/replay 聚焦测试：`18 passed in 6.16s`；
- 全仓测试：`43 passed in 15.59s`；
- aggregate canonical/hash 定向回归：`2 passed, 4 deselected`；
- smoke 端到端：通过；
- 10 个 full 端到端：全部通过；
- final 10-seed aggregate：通过；
- 代表性聚合图 `paired_one_step_difference_ci.png`：已打开检查，散点、均值、CI、零线和标签正常；
- `git diff --check`：通过。

新增或强化的自动化覆盖包括：fallback raw-window 统计重锚定、极端溢出不修改状态、
四类计时字段、独立 calibration seed、clean/observed 状态保存、全结果树哈希验证、source
文件被篡改时拒绝聚合，以及 aggregate dirty 状态阻止 canonical。

## 6. 修改文件

```text
prediction/dktv_window_config.json
prediction/dktv/window_update.py
prediction/dktv/online_model.py
prediction/dktv_window_prediction.py
prediction/dktv_window_aggregate.py
tests/dktv/test_window_update.py
tests/dktv/test_window_replay.py
DKTV_PLAN_03_FORMULA_MAPPING.md
DKTV_PLAN_03_REVIEW_FEEDBACK.md
```

未修改审查报告 `DKTV_PLAN_03_REVIEW.md`，也未删除或覆盖旧数据、模型和结果。

## 7. 剩余限制与最终 Gate

- 当前工作树为 dirty，且上游 Plan 01 source runs 为 noncanonical，因此 source 和 aggregate
  均正确标记为 `accepted_noncanonical`；blockers 为
  `source_runs_noncanonical` 与 `aggregate_worktree_dirty`。
- 本轮完成的是可追溯的 noncanonical 工程整改，不等同于论文最终 canonical 复现实验。
- 50/100-step rollout 仍显著恶化，prediction Gate 未通过；不得进入 MPC，也不得宣称
  长时域稳定或控制性能改善。
- independent epsilon 修复改变了 selective 结论，后续若优化阈值或校准集，必须保持
  calibration 与 test seeds 隔离并重新执行完整 10-seed 比较。

本轮 review 意见已经完成代码、测试和重跑闭环；后续若需要 canonical 结果，应先固定并
提交完整源码，再从 canonical Plan 01/02 来源重新生成全链路产物。
