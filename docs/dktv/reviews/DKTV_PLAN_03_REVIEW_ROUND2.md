# DKTV Plan 03 第二轮执行审查报告

> 审查对象：`DKTV_PLAN_03_REVIEW_FEEDBACK.md`
>
> 第一轮审查：`DKTV_PLAN_03_REVIEW.md`
>
> 审查日期：2026-08-26
>
> 分支：`OTVDKL`
>
> 审查结论：**第一轮提出的 P1-01～P2-04 均已在工程范围内关闭；新 smoke、10-seed full 和 final aggregate 可以作为可信的 noncanonical 开发证据。canonical 与控制 Gate 仍未通过，另外保留两个非阻塞的部署/产物合同强化项。**

## 1. 总体结论

本轮整改形成了完整的“修改代码—补测试—重新运行—重新聚合”闭环，没有直接改写旧
gatefinal manifest。反馈中报告的主要实现和数值结论均能独立复核：

- 10 个新 full run 的 manifest 所列结果文件 190/190 哈希与字节数匹配；
- aggregate 会验证 source result files，并将自身 dirty 状态纳入 canonical 判定；
- fallback 会从 raw candidate window 重建 `G/H/P`；
- recursive、direct oracle、fallback 和 total 计时已经分离；
- epsilon 使用 seed `20260824` 的独立 nominal calibration stream；
- 噪声场景同时保存 clean/observed truth，并聚合全部 rollout horizon；
- 极端非有限递推路径能够失败且不修改 updater 状态；
- 独立重新聚合与 `reviewfix2` 的 metrics、comparison contract 完全一致。

因此，第一轮审查的七项整改可以关闭。当前不能标记 canonical 的原因是工作树和上游
Plan 01 来源状态，而不是本轮整改失败。

## 2. 第一轮问题逐项复核

| 第一轮问题 | 第二轮结论 | 独立证据 |
| --- | --- | --- |
| P1-01 `run.log` 哈希失配 | 已关闭 | 最终日志在快照前写入；10 runs 共 190/190 manifest-listed files 通过 SHA-256/bytes 检查 |
| P1-02 aggregate canonical/provenance | 已关闭 | dirty/source/acceptance 共同决定状态；6/6 aggregate source hashes 匹配；blockers 正确 |
| P1-03 recursive/direct 计时混用 | 已关闭 | 四类字段进入 JSONL、NPZ、summary 和 10-seed aggregate |
| P2-01 fallback 未重锚统计 | 已关闭 | raw window 重建 `G/H/P`；单元测试逐项比较 Gram、cross 和 inverse |
| P2-02 epsilon 同流标定 | 已关闭 | 10/10 使用 seed 20260824，均不同于评价 seed；独立 calibration artifact 已保存并校验 |
| P2-03 噪声缺少 clean truth | 已关闭 | observed 用于 lift/update，clean 用于主评价；两套 one-step/rollout 均保存和聚合 |
| P2-04 recursive finite 检查 | 已关闭 | 系统矩阵、统计量、inverse、`A/B/theta` 和差值均检查；溢出测试不改变状态 |

## 3. 独立自动化验证

指定解释器验证通过：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

本轮独立执行：

| 检查 | 结果 |
| --- | --- |
| Plan 03 修改文件 `py_compile` | 通过 |
| Plan 03 CLI `--help` | 通过 |
| aggregate CLI `--help` | 通过 |
| `tests/dktv` + Plan 02 回归 | `30 passed in 10.56s` |
| 全仓测试 | `43 passed in 16.63s` |
| final aggregate 少于 10 seed | 按预期拒绝，未创建输出目录 |
| final 10-seed 独立重聚合 | 通过 |
| `git diff --check` | 通过 |

测试数量与反馈的全仓 `43 passed` 一致。聚焦测试数量不同，是因为本轮同时纳入了完整
`tests/dktv` 和 Plan 02 回归，覆盖面更大，不构成矛盾。

## 4. 产物与 provenance 复核

核查的新 source runs：

```text
20260826_plan03_full_reviewfix_seed20260825
...
20260826_plan03_full_reviewfix_seed20260834
```

结果如下：

- 10 个 source seed 均不同；
- 10/10 engineering acceptance 为 true；
- 10/10 prediction Gate 为 false；
- 10/10 canonical 为 false；
- 150/150 个 Plan 03 source provenance 哈希与字节数匹配；
- 190/190 个 manifest-listed result files 哈希与字节数匹配；
- 10/10 calibration dataset 哈希、独立 seed 和 no-overlap 合同匹配；
- 10/10 `scenario_predictions.npz` 全部 finite、无 object 数组。

核查的最终 aggregate：

```text
outputs/results/dktv/plan_03/20260826_plan03_aggregate_final_reviewfix2_10seed
```

结果如下：

- 6/6 aggregate source provenance 记录匹配；
- 4/4 aggregate artifact 哈希与字节数匹配；
- 10/10 source manifest 和 result-file validation 匹配；
- aggregate NPZ 含 266 个数组，全部 finite，无 object 数组；
- 状态为 `accepted_noncanonical`；
- blockers 正确为 `source_runs_noncanonical`、`aggregate_worktree_dirty`。

独立重聚合输出：

```text
outputs/results/dktv/plan_03/20260826_plan03_aggregate_final_review_round2_audit
```

该输出的 metrics 和 comparison contract 与 `reviewfix2` 完全一致。代表性配对 CI 图已
打开检查，散点、均值、区间、零线和标签正常。

## 5. 关键数值复核

### 5.1 One-step 与配对统计

| 方法 | one-step RMSE mean |
| --- | ---: |
| fixed DKO | `0.0047948954` |
| accumulative | `0.0033174370` |
| window | `0.0023809613` |
| selective（独立标定） | `0.0032741586` |

独立重算确认：

```text
window vs accumulative
mean difference = +0.000936476
wins            = 10/10

selective vs accumulative
mean difference = +0.000043278
wins/losses     = 8/2
95% CI          = [-0.000542213, +0.000628770]
```

因此反馈中撤回“selective 明确优于 accumulative”的旧表述是正确的。当前证据只支持
“selective 在该独立阈值下无统计明确优势”。

### 5.2 拆分计时

主 window 的独立复核值：

| 指标 | 10-seed mean |
| --- | ---: |
| total update | `0.697825 ms` |
| recursive candidate | `0.315601 ms` |
| direct-refit oracle | `0.254777 ms` |
| fallback（按全部 candidate 平均） | `0.002518 ms` |
| recursive/direct ratio | `1.237846` |
| fallback | 38 次，`3.8/run` |

计时口径与反馈一致。当前维度下 recursive candidate 没有速度优势，报告未对此作过度
解释。

### 5.3 Clean/observed truth

`rate1_noise0.001` 主 window 的聚合值复核为：

| Truth | one-step mean | rollout-100 mean |
| --- | ---: | ---: |
| clean | `0.00263546` | `0.30535977` |
| observed | `0.00281541` | `0.30536214` |

该结果证明 clean/observed 双真值合同已经进入实际 full artifacts，而不只是测试或配置
字段。

## 6. 新发现的非阻塞强化项

### R2-P2-01：尚无可直接执行的 oracle-free deployment 模式

目前 `SlidingWindowKoopmanUpdater.propose()` 仍会在每个候选上无条件执行
`direct_refit()`。新的计时字段正确说明 recursive 时间“不含 oracle”，但实际 updater
调用路径仍包含 oracle，而且 `direct_refit_fallback` 的触发也依赖 recursive 与 oracle
的差值。

这不影响当前工程验证，因为 Plan 03 本身要求逐批 oracle 核对；但在真正部署到绳驱
机械臂前，建议增加明确模式，例如：

```text
oracle_mode = always       # 研究验证
oracle_mode = periodic     # 周期审计/重锚
oracle_mode = disabled     # 部署路径
```

`periodic/disabled` 模式还需定义不依赖 oracle 的数值触发条件。当前不得把
`recursive_candidate_time_s` 描述成一个已经可独立运行的部署 CLI 实测耗时。

### R2-P2-02：`config_snapshot.json` 尚未纳入结果完整性记录

每个 full/smoke 目录实际有 21 个文件：其中 manifest 自身通常不自哈希，另外 19 个
位于 `metrics/arrays/figures/logs` 并已验证；根目录的 `config_snapshot.json` 目前没有
进入 `result_files` 或独立 artifact record。

稳定配置原文件及其 SHA-256 已由 manifest 记录，所以该遗漏不影响当前数值重现和
190/190 的既有结论；但若使用“全结果树完整性”这一表述，建议再增加
`config_snapshot` 的哈希记录，或删除这份未参与合同的重复快照。

## 7. 最终 Gate

当前可以：

- 关闭第一轮 P1-01～P2-04；
- 将 `reviewfix` 结果作为可信的 noncanonical 工程基线；
- 继续诊断 window 的 50/100-step rollout 恶化；
- 在后续部署计划中实现并测试 oracle-free/periodic 模式。

当前仍不可以：

- 将结果标记为 canonical 或论文最终复现证据；
- 宣称 selective 明确优于 accumulative；
- 宣称 Woodbury 递推在当前维度具有速度优势；
- 将拆分后的 recursive timing 当作已实际执行的 oracle-free 部署耗时；
- 在 50/100-step Gate 失败时进入 MPC 或宣称稳定性保证。

canonical 重跑前仍应先固定并提交完整源码，再从 canonical Plan 01/02 来源生成完整
Plan 03 链路。R2-P2-01 建议在机械臂实时部署前完成；R2-P2-02 可与下一次 artifact
schema 调整一起处理。
