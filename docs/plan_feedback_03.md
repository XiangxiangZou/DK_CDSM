# Hao-DKTV 审查整改执行反馈 03

> 执行日期：2026-08-28  
> 对照文档：`docs/plan_review_02.md`  
> 当前结论：探索性完整运行启动门槛已满足；fixed DKUC one-step 和 fixed-horizon 均已严格冻结；full 坐标重建增加了精确诊断、事务回滚和分项耗时。已完成一次归档 smoke 和一次单 trial、不中断的 exploratory full run。

## 1. 代码整改

### 1.1 fixed-horizon DKUC 串扰修复

主程序现在单独从原 artifact 加载 `fixed_model`。fixed one-step 使用 replay 内不可变网络，fixed-horizon 使用主程序中的独立 `fixed_model`，两者均不接触在线 DKTV encoder。

归档 smoke 中，horizon 2 和 horizon 4 保存结果与重新加载原 DKUC artifact 后独立复算的最大绝对偏差均为 `0.0`。

### 1.2 full 坐标重建事务

每个完整更新事务现包含：

1. 旧坐标矩阵候选更新；
2. encoder 在线训练；
3. 全部已消费物理转移重新 lifting；
4. ridge 统计量及 `A/B/C/P` 重建；
5. post-refit 有限值检查；
6. 同时提交 encoder、统计量、矩阵和物理消费历史。

若训练、relift、refit 或有限值检查失败，会恢复 batch 前网络和 `HaoDKTVState`，并记录 `rolled_back_*` 原因。

### 1.3 诊断与耗时

更新记录现在明确区分：

- `pre_theta_update_diagnostics`；
- `post_coordinate_refit_diagnostics`；
- `coordinate_relift_time_s`；
- `coordinate_refit_time_s`；
- `total_transaction_time_s`。

summary 另外保存 relift、refit、训练和完整事务的均值，以及完整事务最大耗时。矩阵类 docstring 已改为“固定 encoder 坐标下的递推状态”，并说明 full 模式全历史重建的线性样本复杂度。

### 1.4 resume 风险隔离

在 checkpoint 尚未保存已消费物理转移前，`full` 模式若传入 resume 参数会明确拒绝，不再允许静默产生 sample count 回退。`frozen_encoder` 的固定坐标 state 续算能力保留。

### 1.5 测试增强

新增或增强：

- 病态但有限批次的 ridge 稳定性；
- full 最终 state 对全历史直接 ridge refit oracle；
- 篡改未来观测不影响过去预测的 causality；
- fixed one-step 高学习率串扰检查；
- trial isolation、batch 划分和固定坐标 state 续算。

## 2. 验收 smoke

命令使用默认输出根目录，未使用 `/tmp` 保存最终证据：

```bash
env -u PYTHONPATH MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dktv_review03_mpl \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  prediction/dktv_prediction.py \
  --artifact_dir prediction/outputs/smoke_test/dkuc/20260703_105217_dkuc_prediction_split \
  --stream_dataset traj_data/outputs/smoke_test/20260828_102529_time_varying_sine_dktv_plan_smoke/dataset.npz \
  --mode full --batch_size 4 --online_epochs 4 --online_lr 0.00001 \
  --device cpu --run_type smoke_test --rollout_horizons 2 4 \
  --rollout_stride 2 --tag full_sine_review03_seed50
```

归档目录：

```text
prediction/outputs/smoke_test/dktv/20260828_123019_dktv_full_sine_review03_seed50/
prediction/outputs/smoke_test/figures/20260828_123019_dktv_full_sine_review03_seed50/
```

验收结果：

- 所有数组有限；
- 两个更新均 `coordinate_consistent=true`；
- 两个更新均保存 post-refit 精确诊断；
- fixed-horizon h2/h4 独立复算最大偏差为 0；
- fixed DKUC one-step RMSE：`0.1203049681`；
- full DKTV one-step RMSE：`0.0503477608`；
- 平均完整事务耗时：`0.4293 s`。

## 3. 探索性完整运行

### 3.1 数据

数据目录：

```text
traj_data/outputs/full_run/20260828_123055_time_varying_sine_exploratory_sine_seed50/
```

配置为 1 trial、200 steps、50 warm-up steps、seed 50。最低检查：

- 必需数组全部有限；
- 时间严格递增；
- 无 joint limit violation；
- 峰值等效关节力矩：`185.9714291`；
- 峰值缆索张力：`3694.2647618`；
- 峰值未观测扰动力矩：`1.0`。

执行器和缆索物理允许阈值尚未确定，因此该数据只用于探索性仿真预测。

### 3.2 DKTV 运行

运行约束：单 trial、单次不中断、未传 resume、结果标记 `exploratory`。

产物目录：

```text
prediction/outputs/full_run/dktv/20260828_123109_dktv_full_sine_exploratory_seed50/
prediction/outputs/full_run/figures/20260828_123109_dktv_full_sine_exploratory_seed50/
```

参数：batch size 20、online epochs 20、online learning rate `1e-5`、ridge `1e-3`、seed 50、CPU。

验收结果：

- 10/10 更新接受，10 次 encoder 更新；
- arrays、loss、`A/B/C/P` 和指标全部有限；
- 10 次更新均 `coordinate_consistent=true`；
- 最终 state 对最终 encoder 全历史直接 refit 的最大绝对偏差为 `0.0`；
- 平均事务耗时 `0.2248 s`，最大 `0.7516 s`；
- 平均 encoder 训练耗时 `0.0809 s`；
- 平均全历史 relift 耗时 `0.0376 s`；
- 平均 ridge refit 耗时 `0.0173 s`。

探索性指标：

| 指标 | fixed DKUC | full DKTV |
|---|---:|---:|
| one-step RMSE | 0.0582715 | 0.0266593 |
| horizon-10 RMSE | 0.3432380 | 0.1375131 |
| horizon-25 RMSE | 0.7604495 | 0.2452570 |
| horizon-50 RMSE | 1.2964461 | 0.3709575 |

DKTV batch/horizon-20 RMSE 为 `0.1988324`。上述结果只有一个 seed、一个场景，不构成方法优越性或论文最终结论。

## 4. 验证命令与结果

指定解释器验证为：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

结果：

```text
py_compile：通过
全仓库 pytest：11 passed in 1.80s
归档 smoke：通过
探索性 full run：通过
指标数组重算：通过（按 evaluate_predictions 排除窗口初始帧）
最终坐标 oracle：最大绝对偏差 0.0
```

## 5. 尚未完成

- full resume 的完整 checkpoint 尚未实现；当前是明确禁用，而不是宣称支持。
- checkpoint fingerprint、normalizer/config/data 消费进度绑定和错误配对拒绝尚未实现。
- 多 trial 逐 trial checkpoint 尚未实现。
- 执行器和缆索物理安全阈值、无扰动/有扰动成对数据及确定性测试尚未完成。
- full、frozen_encoder、fixed DKUC 的多 seed 独立正式实验尚未执行。
- 当前只有探索性墙钟证据，不宣称实时性。

## 6. 下一步建议

先建立包含物理转移或可验证重放位置的统一 checkpoint，再做连续/恢复完全等价测试。随后推导 XML 执行器和缆索约束并建立数据 accepted/rejected 契约，生成成对确定性数据。最后独立运行 frozen_encoder 与 full 的多 seed 参数实验，形成论文级统计结果。
