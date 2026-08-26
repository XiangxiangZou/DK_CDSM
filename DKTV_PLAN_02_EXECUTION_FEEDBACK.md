# DKTV Plan 02 执行情况 Feedback 报告

> 执行计划：`DKTV_PLAN_02_HAO_ACCUMULATIVE.md`
>
> 执行日期：2026-08-25
>
> 结论：**核心算法和 10-seed 预测实验通过工程验收；结果因工作树及源 Plan 01
> 非 canonical 而标记为 noncanonical**

## 1. 完成结论

Plan 02 的 direct-refit oracle、累积充分统计量更新、数值诊断、模型版本、状态保存
与重放、因果在线预测、统一 rollout/分阶段评价、smoke/full CLI 和 10-seed 汇总均已
实现并实际运行。

10 个 seed、每个 seed 的 `b=5/10/20` 合计执行：

```text
b=5:   240 updates/seed × 10 = 2400
b=10:  120 updates/seed × 10 = 1200
b=20:   60 updates/seed × 10 =  600
total:                            4200
```

4200 次更新全部接受，没有 NaN/Inf 或数值失败。递推结果相对全历史 ridge direct
refit 的最大 `A` 绝对差为 `2.007867483388992e-11`，小于验收容差 `1e-8`。
因此满足进入 Plan 03 前最关键的“递推复现 direct refit”Gate。

所有新结果均为 `accepted_noncanonical`。原因是运行时 Git worktree 包含本轮未提交
实现，而且所复用的 Plan 01 artifact 也明确标记为 noncanonical。本轮遵守仓库
规则，没有擅自 commit，也没有把结果描述为 canonical 论文基线。

## 2. 文献与实现边界

核对来源：

- Hao et al., Automatica 159 (2024) 111372，DOI
  `10.1016/j.automatica.2023.111372`，公开稿
  `https://arxiv.org/abs/2210.06272`；
- Zhang et al., Automatica 190 (2026) 113054，DOI
  `10.1016/j.automatica.2026.113054`。

Hao 原方法持续加入新数据而不移除旧数据；Zhang 将其作为 accumulative DKTV
基线。Hao 原文还会继续优化 DNN 参数。本仓库根据 Plan 02 的公平比较约束，采用
受控复现：固定 Plan 01 encoder、normalizer 和 exact `C0`，只在线更新 `A/B`。
该差异已在 `DKTV_PLAN_02_FORMULA_MAPPING.md` 和 manifest 中明确记录，不宣称逐项
复现 Hao 的在线 DNN 优化。

本仓库实现的坐标为：

```text
x_norm = (x_phys - x_mean) / x_std
u_norm = (applied_torque_phys - u_mean) / u_std
z = [x_norm, phi(x_norm), 1]
z_next = A_tau @ z + B_tau @ u_norm
x_phys_next = (C0 @ z_next) * x_std + x_mean
```

每个 snapshot 使用 `r_k=[z_k,u_norm,k]`。updater 只累积：

```text
Gram  += R_new.T @ R_new
Cross += R_new.T @ Z_next,new
count += batch_size
```

随后求解同一个 ridge 系统。direct oracle 则每批重新拼接全部历史 snapshot 并直接
拟合，二者逐批保存矩阵差。

## 3. 实现内容

### 算法层

- ridge direct refit 和充分统计量求解；
- `A/B` affine constant row 约束；
- rank、最小奇异值、raw/regularized condition number、谱半径和 finite 检查；
- 只加不减的 `AccumulativeKoopmanUpdater`；
- encoder/normalizer/architecture 指纹检查；
- 数值失败保留当前模型和统计量；
- updater `.npz` 保存、加载和后续重放；
- 每批 current/pre-update、candidate/post-update 和 direct oracle 证据。

### 在线评价层

使用因果顺序：

```text
当前模型预测 x_{k+1}
→ 观测真实 x_{k+1}
→ 加入 pending batch
→ 满 b 个 snapshot 后更新
→ 新模型从后续预测开始生效
```

fixed 和 accumulative 使用相同 Plan 01 stream、样本顺序、encoder、normalizer、
`A0/B0/C0`、future input windows 和评价函数。保存 one-step、10/20/50/100-step
rollout、nominal/transition/time-varying 分阶段结果及逐时刻模型版本。

### 输出与汇总

- 每个 run 保存 manifest、5 类 metrics JSON、预测与更新历史 NPZ、最终 updater、
  3 张图及审计日志；
- 10 个独立 seed 分别运行 Plan 01 full 和 Plan 02 full；
- aggregate run 保存逐 seed 原始值、均值、sample standard deviation、NPZ 和汇总图。

## 4. 文件变更

- `DKTV_PLAN_02_FORMULA_MAPPING.md`
- `configs/dktv/plan_02.json`
- `src/koopman_control/dktv/least_squares.py`
- `src/koopman_control/dktv/accumulative_update.py`
- `src/koopman_control/dktv/online_model.py`
- `src/koopman_control/dktv/__init__.py`
- `experiments/dktv/plan_02.py`
- `experiments/dktv/aggregate_plan_02.py`
- `experiments/dktv/plan_01.py`：增加显式 `--seed` override，用于独立多 seed
  Plan 01 数据和初始 artifact；
- `tests/test_dktv_accumulative.py`
- `DKTV_PLAN_02_EXECUTION_FEEDBACK.md`（本报告）

本轮开始时工作树为 clean。`AGENTS.md` 和既有 Plan 01/Plan 03 实现或文档未被
改写。

## 5. 测试和核查

使用解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

主要检查：

| 检查 | 结果 |
| --- | --- |
| 7 个相关 Python 文件 compile | 通过 |
| 全仓测试 | `22 passed in 8.59s` |
| `b=5/10/20` synthetic recursive/direct | 通过 |
| rank-deficient ridge | finite，通过 |
| encoder fingerprint 改变 | 显式拒绝，通过 |
| non-finite batch | 不覆盖模型/统计量，通过 |
| updater save/load/replay | 通过 |
| Plan 02 smoke CLI 端到端 | 通过 |
| 实际 final updater 重载 | 3/3 通过，`A=(16,16)`、`B=(16,2)` |
| 同源 full 重放 predictions | 20/20 字段 `array_equal=True` |
| 同源 full 重放非计时 history | 45/45 字段 `array_equal=True` |
| 同源 full 重放 final updater | 3/3 完全一致 |
| primary full manifest/source/result hash | 32/32 匹配 |
| 10-seed aggregate/source hash | 14/14 匹配 |
| 代表性单 seed 与 10-seed PNG | 均已打开检查 |

自动化测试覆盖了：充分统计量、direct oracle、三个批大小、秩亏、encoder 失效、
失败安全、状态重放和真实 artifact CLI 端到端。

## 6. 运行记录

### Smoke

```text
Plan 01 source: 20260825_145551_plan01_smoke_baseline_reviewed
Plan 02 run:    20260825_155856_plan02_smoke_accumulative_reviewed
```

### 10 个 Full seed

| Seed | Plan 01 source run | Plan 02 full run |
| --- | --- | --- |
| 20260825 | `20260825_145620_plan01_full_baseline_reviewed` | `20260825_155915_plan02_full_accumulative_reviewed` |
| 20260826 | `20260825_155934_plan01_full_plan02_seed` | `20260825_160017_plan02_full_accumulative_multiseed` |
| 20260827 | `20260825_155956_plan01_full_plan02_seed` | `20260825_160021_plan02_full_accumulative_multiseed` |
| 20260828 | `20260825_160539_plan01_full_plan02_seed` | `20260825_160735_plan02_full_accumulative_10seed` |
| 20260829 | `20260825_160555_plan01_full_plan02_seed` | `20260825_160739_plan02_full_accumulative_10seed` |
| 20260830 | `20260825_160610_plan01_full_plan02_seed` | `20260825_160743_plan02_full_accumulative_10seed` |
| 20260831 | `20260825_160625_plan01_full_plan02_seed` | `20260825_160747_plan02_full_accumulative_10seed` |
| 20260832 | `20260825_160641_plan01_full_plan02_seed` | `20260825_160751_plan02_full_accumulative_10seed` |
| 20260833 | `20260825_160656_plan01_full_plan02_seed` | `20260825_160755_plan02_full_accumulative_10seed` |
| 20260834 | `20260825_160711_plan01_full_plan02_seed` | `20260825_160759_plan02_full_accumulative_10seed` |

10-seed aggregate：

```text
20260825_160810_plan02_aggregate_final_10seed
```

另保留重复性检查 run `20260825_160301_plan02_full_replay_check`，未覆盖任何原结果。

## 7. 10-seed 关键指标

下表为 mean ± sample standard deviation：

### One-step RMSE

| 方法 | RMSE |
| --- | --- |
| fixed DKO | `0.0047948954 ± 0.0013165907` |
| accumulative `b=5` | `0.0032808116 ± 0.0003569213` |
| accumulative `b=10` | `0.0033174370 ± 0.0003616385` |
| accumulative `b=20` | `0.0033636036 ± 0.0003703873` |

`b=5` 相对 fixed 的均值降低约 31.6%。

### Rollout RMSE

| Horizon | fixed | `b=5` | `b=10` | `b=20` |
| --- | --- | --- | --- | --- |
| 10 | `0.02175634` | `0.01719973` | `0.01732043` | `0.01755860` |
| 20 | `0.03644973` | `0.03165326` | `0.03186464` | `0.03225545` |
| 50 | `0.06609152` | `0.06911809` | `0.06936093` | `0.06973346` |
| 100 | `0.08352748` | `0.08285624` | `0.08299985` | `0.08312134` |

结果不是所有 horizon 都更好：`b=5` 在 horizon 10/20 分别降低约 20.9%/13.2%，
但 horizon 50 比 fixed 高约 4.6%；horizon 100 仅小幅降低约 0.8%。该负结果已保留，
没有挑选性删除。

### 分阶段 20-step rollout RMSE

| 阶段 | fixed | `b=5` | `b=10` | `b=20` |
| --- | --- | --- | --- | --- |
| nominal | `0.01823341` | `0.01588575` | `0.01603169` | `0.01631459` |
| transition | `0.03586017` | `0.03084668` | `0.03106275` | `0.03147393` |
| time-varying | `0.05267761` | `0.04821468` | `0.04849110` | `0.04904704` |

`b=5` 在 time-varying 阶段均值降低约 8.5%。

### 更新诊断

- 10-seed 最大 recursive/direct `A` 差：`2.007867483388992e-11`；
- primary full 最大 recursive/direct `B` 差：`4.5824455341403336e-14`；
- `b=5/10/20` 平均递推更新时间约
  `0.266/0.261/0.269 ms`；
- 每个 updater 的充分统计量内存：`4896 bytes`，不随 stream 长度增长；
- 每个 seed 最终历史样本数：从 initial `1440` 增至 `2640`；
- 样本数严格单调增加，没有删除或遗忘。

## 8. 输出位置

Primary full：

```text
outputs/results/dktv/plan_02/20260825_155915_plan02_full_accumulative_reviewed/
```

10-seed aggregate：

```text
outputs/results/dktv/plan_02/20260825_160810_plan02_aggregate_final_10seed/
```

Primary full 的关键文件：

```text
manifest.json
metrics/{one_step,rollout,segmented,update_summary,comparison}.json
arrays/predictions.npz
arrays/update_history.npz
arrays/dktv_accumulative_b{5,10,20}_updater_final.npz
figures/{one_step_rmse_by_step,rollout_rmse_by_horizon,update_diagnostics}.png
logs/{command,environment,run,pytest,py_compile,reproducibility_check,
      hash_verification,updater_reload,acceptance_summary}.*
```

Aggregate 保存：

```text
manifest.json
metrics/multiseed_summary.json
arrays/multiseed_metrics.npz
figures/multiseed_one_step_rmse.png
logs/{command,hash_verification}.json|log
```

## 9. 未执行项和剩余风险

1. **Canonical Gate**：当前工作树和所有源 Plan 01 run 非 canonical，因此 Plan 02
   结果不能作为最终冻结论文基线。需要用户审查、授权形成 Git 版本后，在 clean
   worktree 上重新生成 canonical Plan 01/Plan 02。
2. **噪声/变化速率消融**：本轮完成 zero-noise、公共 Plan 01 正弦扰动及其
   nominal/transition/time-varying 阶段，并满足 10-seed 最终数量；尚未跑 low-noise
   和 slow/fast 频率网格。这些不阻断递推公式 Gate，但在论文比较前仍应补齐。
3. **MPC**：Plan 02 将闭环 MPC 标为可选步骤。本轮只完成预测 Gate，没有执行
   MPC，以免把模型更新有效性和控制器效果混为一个结论。
4. **Hao 原方法差异**：本实现按计划冻结 encoder；不能声称包含 Hao 原文中的
   在线 DNN 参数优化。
5. **长 horizon**：50-step rollout 的 10-seed 均值略差于 fixed，应在 Plan 03
   比较时保留并解释，不应仅依据 one-step 改善宣称全时域占优。

## 10. Plan 03 Gate

允许进入 Plan 03 的算法开发，因为：

- 三个 batch size 的累积统计量均稳定复现 direct refit；
- rank-deficient、non-finite、encoder mismatch 和 replay 均有自动化测试；
- 更新历史严格累积且可重放；
- 10 个 seed 的 fixed/accumulative 公平比较已经保存；
- 正结果和 50-step 负结果均完整保留。

进入 Plan 03 的正式 canonical 数值对比前，仍需解决第 9 节的 canonical Gate。
