# DKTV Plan 03 执行情况 Feedback 报告

> 执行计划：`DKTV_PLAN_03_ZHANG_SLIDING_WINDOW.md`
>
> 执行日期：2026-08-26
>
> 最终结论：**滑动窗口与选择性更新实现、消融和 10-seed 预测实验通过工程验收；
> 短期预测改善但长 rollout 明显恶化，因此控制 Gate 未通过，未执行可选 MPC；
> 所有结果因 dirty worktree 与源 Plan 01 非 canonical 而标记为 noncanonical。**

## 1. 完成范围

已完成：

- 固定长度窗口、精确 `b` 样本移入/移出、样本 ID 和窗口边界；
- ridge direct-refit oracle；
- Woodbury 新批加入和旧批删除递推、低维系统/秩/奇异值/条件数/finite 检查；
- 递推超容差时可审计的 direct-refit fallback；
- `accepted/rejected/skipped_threshold/failed_numerical` 四状态；
- `discard_on_reject` 和 `retain_on_reject` 两种窗口语义；
- fixed、accumulative、window、selective 在同一因果数据流和评价器上的比较；
- `w/b/epsilon`、拒绝策略、观测噪声和 slow/medium/fast 扰动频率消融；
- one-step、10/20/50/100-step rollout、分阶段误差、适应延迟、更新耗时和内存；
- 10 个独立 seed 的配对统计及 95% Student-t CI。

没有在线更新 encoder，没有重训偏向窗口方法的初始模型，也没有实现 SDP-MPC 或
稳定性证明。

## 2. 算法与状态语义

公式和工程边界记录于 `DKTV_PLAN_03_FORMULA_MAPPING.md`。窗口维护
`G=R.T@R`、`H=R.T@Y` 和 `P=(G+lambda I)^-1`，每批先加入 `S_new`，再删除最旧
`S_out`。候选逐批与相同候选窗口上的 ridge direct refit 比较。

选择性状态语义为：

- current batch RMSE `<= epsilon`：`skipped_threshold`，窗口推进，模型不更新；
- candidate 不优于 current：`rejected`；
- `discard_on_reject`：窗口和模型均保持；
- `retain_on_reject`：数据进入窗口，但模型保持；
- 数值失败：`failed_numerical`，窗口和模型均保持。

`epsilon` 使用每个源 Plan 01 的 nominal validation stream，在固定初始模型上的
batch latent RMSE 中位数标定；10 个 seed 的实际 epsilon 均写入各 run manifest。

## 3. 文件变更

新增：

- `DKTV_PLAN_03_FORMULA_MAPPING.md`
- `configs/dktv/plan_03.json`
- `src/koopman_control/dktv/window_update.py`
- `src/koopman_control/dktv/selective_update.py`
- `experiments/dktv/plan_03.py`
- `experiments/dktv/aggregate_plan_03.py`
- `tests/dktv/__init__.py`
- `tests/dktv/test_window_update.py`
- `tests/dktv/test_selective_update.py`
- `tests/dktv/test_window_replay.py`
- `DKTV_PLAN_03_EXECUTION_FEEDBACK.md`（本报告）

修改：

- `src/koopman_control/dktv/online_model.py`：窗口 replay、统一评价接入和汇总；
- `src/koopman_control/dktv/__init__.py`：导出窗口 updater。

Plan 01/02 既有修改和用户审查文档均保留，未回滚、覆盖、提交或推送。

## 4. 自动化验证

解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

主要结果：

| 检查 | 结果 |
| --- | --- |
| Plan 03 Python `py_compile` | 通过 |
| Plan 03 与 aggregate CLI `--help` | 通过 |
| Plan 03 聚焦测试及 Plan 02 回归 | `25 passed` |
| 全仓测试 | `38 passed in 12.60s` |
| synthetic `w={50,100}`, `b={5,10,20}` direct/recursive | 通过 |
| 长重放窗口长度、顺序、移入/移出和固定内存 | 通过 |
| rank-deficient/non-finite 安全失败 | 通过 |
| 四状态和两 reject policy | 通过 |
| 真实 Plan 01 artifact 因果 replay | 通过 |
| Plan 03 smoke CLI 端到端 | 通过 |
| final 聚合少于 10 seed | 按预期拒绝，未创建结果目录 |
| 10 个 final source manifest hash | `10/10` 匹配 |
| 代表性 run 的 source provenance hash | `15/15` 匹配 |
| aggregate NPZ | 38 arrays，全部 finite，无 object/pickle |
| `git diff --check` | 通过 |
| 代表性 PNG | 已打开检查，尺寸正常、文字可读 |

四状态由参数化单元测试覆盖；真实 full stream 中基础 window 为 1200 次 accepted、
0 failed。主 selective 的 10-seed 总计为 250 accepted、945 skipped、5 rejected、
0 failed；retain 策略另出现 2 次 rejected，因此拒绝策略不仅停留在配置字段。

## 5. 实验合同与运行量

每个 full seed 的 base scenario 执行：

- 主 accumulative：`b=10`；
- 主 window：`w=100,b=10`；
- 主 selective：`w=100,b=10,epsilon=calibrated,discard_on_reject`；
- window 消融：`(w,b)=(50,10),(100,5),(100,10),(100,20),(200,10)`；
- epsilon 消融：`0.5x,1x,2x`；
- reject policy：discard 与 retain；
- rate/noise：`rate={0.5,1,2} × state measurement noise std={0,0.001}`。

rate=0.5/2 的 MuJoCo streams 重新采集并通过 finite、状态范围、饱和、关节限制、
张力异常和分配残差检查；rate=1 复用已验收 Plan 01 validation stream。每个场景内四
种主方法共享数据、固定 encoder/normalizer、初始 `A0/B0`、样本顺序和评价器。

10 个 final run：

```text
20260826_plan03_full_gatefinal_seed20260825
...
20260826_plan03_full_gatefinal_seed20260834
```

最终 aggregate：

```text
outputs/results/dktv/plan_03/20260826_plan03_aggregate_final_gatefinal_10seed
```

全部 10 个 source run 为 `accepted_noncanonical`，工程验收均通过，控制预测 Gate
均为 false。早期 smoke/full 诊断结果未删除，但已被上述 `gatefinal` runs 取代。

## 6. 10-seed 主要预测结果

### 6.1 One-step RMSE（mean ± sample std）

| 方法 | RMSE |
| --- | --- |
| fixed DKO | `0.00479490 ± 0.00131659` |
| accumulative | `0.00331744 ± 0.00036164` |
| window `w100,b10` | `0.00238096 ± 0.00026431` |
| selective | `0.00300257 ± 0.00039567` |

window 相对 accumulative 的配对差定义为 `accumulative_rmse-window_rmse`：均值
`+0.000936476`，95% Student-t CI
`[+0.000707550,+0.001165401]`，10/10 seed 改善。selective 的配对均值为
`+0.000314863`，CI `[+0.000147971,+0.000481756]`，9/10 seed 改善。

### 6.2 Rollout RMSE

| Horizon | fixed | accumulative | window | selective |
| --- | ---: | ---: | ---: | ---: |
| 10 | `0.0217563` | `0.0173204` | `0.0126233` | `0.0175791` |
| 20 | `0.0364497` | `0.0318646` | `0.0295022` | `0.0394205` |
| 50 | `0.0660915` | `0.0693609` | `0.1188000` | `0.1348528` |
| 100 | `0.0835275` | `0.0829999` | `0.3215372` | `0.5849929` |

window 在 horizon 10 为 10/10、horizon 20 为 9/10 seed 优于 accumulative；但在
horizon 50 和 100 均为 0/10。该长时域负结果是控制 Gate 失败的直接原因。

### 6.3 分阶段 20-step rollout

| 阶段 | fixed | accumulative | window | selective |
| --- | ---: | ---: | ---: | ---: |
| nominal | `0.0182334` | `0.0160317` | `0.0254196` | `0.0282975` |
| transition | `0.0358602` | `0.0310627` | `0.0259227` | `0.0391970` |
| time_varying | `0.0526776` | `0.0484911` | `0.0400261` | `0.0582487` |

window 对 accumulative 在 transition 为 9/10、time-varying 为 10/10 改善，但 nominal
仅 1/10。按已保存的“连续 3 步进入阶段尾部 RMSE 的 1.1 倍”定义，window 在
time-varying 阶段平均适应延迟 `0.5 step`，accumulative 为 `6.4 steps`；该指标是
自适应收敛描述，不替代长 rollout 稳定性证据。

## 7. 消融结果

### 7.1 `w/b/epsilon` one-step 均值

| 配置 | RMSE |
| --- | ---: |
| window `w50,b10` | `0.00218574` |
| window `w100,b5` | `0.00220917` |
| window `w100,b10` | `0.00238096` |
| window `w100,b20` | `0.00251926` |
| window `w200,b10` | `0.00273473` |
| selective `epsilon×0.5` | `0.00256166` |
| selective `epsilon×1` | `0.00300257` |
| selective `epsilon×2` | `0.00403312` |
| selective retain policy | `0.00299914` |

one-step 上更短窗口和更小 batch 较好；过大的 epsilon 跳过过多更新并明显恶化。
retain/discard 的总体差很小，因为实际 negative-update rejection 很少。

### 7.2 速率/噪声 one-step

| rate / noise std | accumulative | window | 配对胜 seed |
| --- | ---: | ---: | ---: |
| 0.5 / 0 | `0.00321622` | `0.00225075` | 10/10 |
| 0.5 / 0.001 | `0.00351331` | `0.00270857` | 10/10 |
| 1 / 0 | `0.00331744` | `0.00238096` | 10/10 |
| 1 / 0.001 | `0.00360293` | `0.00281541` | 10/10 |
| 2 / 0 | `0.00282680` | `0.00245642` | 10/10 |
| 2 / 0.001 | `0.00316351` | `0.00289426` | 10/10 |

噪声会同时抬高两种方法误差；window 在所有场景仍保持 one-step 配对优势，但随
rate 加快优势缩小。此结论仅针对当前物理状态观测噪声模型和扰动频率倍数。

## 8. 数值、耗时与内存

主 window 的 10-seed 最大 raw recursive `A` 差为 `2.02154e-8`；42/1200 个候选
超过 `1e-8` 后执行已记录的 direct fallback，fallback 后最大候选差为
`9.98380e-9`。最大窗口条件数约 `1.497e6`，最小奇异值约 `9.93e-6`，所有候选
保持满秩 18 且 finite。全部 base window/selective 消融合计记录 378 次 fallback；
因此结果不能解释为“所有实际更新均由纯 Woodbury 递推达到容差”。

主方法 10-seed 平均耗时与内存：

| 方法 | mean update | mean P95 | mean P99 | observed max | updater memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| accumulative | `0.259 ms` | `0.296 ms` | 未单列 | `0.471 ms` | `4,896 B` statistics |
| window | `0.540 ms` | `0.616 ms` | `0.682 ms` | `0.885 ms` | `37,792 B` fixed |
| selective | `0.517 ms` | `0.613 ms` | `0.670 ms` | `0.846 ms` | `37,792 B` fixed |

window updater 的 `37,792 B` 包含固定 100-sample 原始窗口、统计量、逆 Gram 和
当前模型，所有时刻不增长。direct-refit oracle 的临时诊断开销不计入部署内存。

## 9. 输出位置

最终汇总：

- `outputs/results/dktv/plan_03/20260826_plan03_aggregate_final_gatefinal_10seed/manifest.json`
- `.../metrics/multiseed_summary.json`
- `.../arrays/multiseed_metrics.npz`
- `.../figures/multiseed_one_step_rmse.png`
- `.../figures/paired_one_step_difference_ci.png`

每个 source run 均包含：

```text
manifest.json
metrics/{one_step,rollout,segmented,update_summary,adaptation_delay,scenario_ablation}.json
arrays/{predictions,scenario_predictions,update_history,window_diagnostics}.npz
arrays/update_history_schema.json
figures/
logs/{command,environment,run,update_history}.json/jsonl/log
```

## 10. 剩余风险与后续条件

- 结果为 noncanonical：工作树包含未提交实现，源 Plan 01 也是 noncanonical；
- 矩阵递推在部分实际候选上需要 direct fallback，纯递推数值策略仍可继续改进；
- selective 的负更新拒绝样本较少，retain/discard 结论证据弱于 threshold 消融；
- one-step/短 rollout 的优势没有延伸到 50/100-step，不能据此进入 MPC 或声明稳定；
- 因严格预测 Gate 未通过，Plan 03 Step 06 未执行；需要先解决长 rollout 恶化，再规划
  同一 MPC 的闭环比较；
- Zhang 论文中的 SDP-MPC 和稳定性条件仍明确属于后续计划。
