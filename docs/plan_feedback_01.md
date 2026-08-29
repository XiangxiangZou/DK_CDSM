# Hao-DKTV 计划执行反馈 01

> 执行日期：2026-08-28  
> 对照计划：`docs/Plan.md`  
> 执行范围：Hao-DKTV 核心算法、时变数据入口、轻量 CDSM 验证  
> 结论：核心链路已实现并通过公式与端到端 smoke 验证；正式参数扫描、完整三类 rollout 和论文级结论尚未完成。

## 1. 本轮完成内容

### 阶段 0：模式与因果契约

- CLI 新增 `--mode {full,frozen_encoder}`，默认 `full`。
- `full` 更新 `A/B/C/theta`；`frozen_encoder` 更新 `A/B/C`，不训练 encoder。
- 保留原 `AccumulativeKoopmanUpdater`，以兼容既有固定编码器 A/B oracle 测试。
- replay 顺序固定为“使用当前版本预测 -> 观察真实转移 -> 满批后更新”。
- 多轨迹不再按同一时刻拼批，每条 trial 从相同初始网络和递推状态独立运行。

迁移关系：

| 旧对象/字段 | 新契约 | 说明 |
|---|---|---|
| `AccumulativeKoopmanUpdater` | 保留 | 仅作为旧 A/B 累积基线与兼容 oracle |
| 无模式参数 | `--mode full` | 完整 Hao-DKTV 默认模式 |
| 隐含冻结 encoder | `--mode frozen_encoder` | 明确标记的消融模式 |
| `gram/cross` | `gram_chi/cross_chi` | A/B 累积统计 |
| 无 C 状态 | `gram_g/cross_g/P_g/C` | 重构矩阵递推状态 |
| 单一 model version | `model_version/encoder_version` | 区分线性表示和 encoder 版本 |

### 阶段 1：CDSM 时变数据

新增 `traj_data/collect_data_time_varying.py`：

- 复用 `MujocoCDSM`、缆索分配和参考轨迹工具；
- warm-up 阶段扰动力矩为零；
- 在线阶段施加 `amplitude*sin(2*pi*frequency*t+phase)` 外部关节力矩；
- `inputs` 只保存缆索目标等效关节力矩，`disturbance_torque` 单独保存；
- 保存严格递增时间、状态、输入、扰动力矩、缆索张力、随机种子、XML、Git 信息和安全摘要；
- 支持 `smoke_test/full_run` 两种独立实验规模。

本轮生成的代表性 smoke 数据：

`traj_data/outputs/smoke_test/20260828_102529_time_varying_sine_dktv_plan_smoke/dataset.npz`

参数为 1 条轨迹、8 步、2 步无扰动 warm-up、seed 123。该目录属于 stage-local 生成产物，不应提交 Git。

### 阶段 2：A/B/C 递推

新增 `HaoDKTVState`，保存：

- `A/B/C`；
- `gram_chi/cross_chi/P_chi`；
- `gram_g/cross_g/P_g`；
- `sample_count/update_index/model_version/encoder_version`。

实现内容：

- ridge 初始化使用线性求解，不裸调用矩阵求逆；
- `P_chi` 和 `P_g` 使用 Woodbury 批量递推；
- 候选矩阵全部通过有限值检查后才原子提交；
- 失败时保持旧状态；
- 记录批次秩、最小奇异值、条件数、A/B/C 谱范数及 A 谱半径；
- checkpoint 可保存和恢复完整递推状态。

公式测试表明，固定 encoder 时，分批递推结果与全部累计样本直接 ridge refit 在 `1e-9` 容差内一致。

### 阶段 3：在线 theta 与 Hao 损失

- 网络直接复用 `dkuc_prediction.make_network_class()` 生成的 DKUC encoder，没有复制 MLP。
- 实现 `L = w*L1 + (1-w)*L2`。
- 在线优化器只接收 encoder 参数，A/B/C 使用当前批次更新后的冻结 tensor。
- 每批从上一 encoder warm start，保存最低总损失对应权重。
- 保存训练前后 L1/L2/L、逐 epoch history、训练耗时和 encoder checksum。
- 非有限 loss/梯度时恢复原 encoder；线性状态与 encoder 作为一个批次整体回滚。
- manifest 明确记录 full 模式保留历史特征坐标的 Hao 累积近似。

### 阶段 4：轨迹隔离与恢复

- 每条 trial 单独运行和重置，trial 之间不共享在线更新。
- 数组保存 `trial_id/time_index/model_version/encoder_version`。
- 最后不足一个 batch 的样本不更新，并记录 `pending_incomplete_batch`。
- CLI 新增 `--resume_state` 和 `--resume_model`，分别恢复递推状态和 encoder/network 权重。
- checkpoint 输出为 `final_dktv_state.npz` 和 `final_dktv_model.pt`。

### 阶段 5：产物与评估

已保存：

- `manifest.json`
- `metrics.json`
- `prediction_arrays.npz`
- `update_history.json`
- `training_history.json`
- `final_dktv_state.npz`
- `final_dktv_model.pt`
- one-step 状态图和误差图

已有 one-step 指标和 fixed DKUC 的 fixed-horizon rollout。完整 DKTV 的 batch rollout 与 fixed-horizon rollout 尚未补齐，因此阶段 5 只算部分完成。

## 2. 验证记录

解释器验证：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

执行的关键命令：

```bash
env -u PYTHONPATH .../bin/python -m py_compile \
  prediction/dktv_prediction.py traj_data/collect_data_time_varying.py

env -u PYTHONPATH .../bin/python prediction/dktv_prediction.py --help

env -u PYTHONPATH .../bin/python -m pytest \
  tests/time_varying/test_core_updates.py \
  tests/time_varying/test_dktv_algorithm.py -q

env -u PYTHONPATH MUJOCO_GL=egl .../bin/python \
  traj_data/collect_data_time_varying.py --run_type smoke_test \
  --traj 1 --steps 8 --warmup_steps 2 --seed 123 --tag dktv_plan_smoke

env -u PYTHONPATH MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dktv_mpl2 .../bin/python \
  prediction/dktv_prediction.py \
  --artifact_dir prediction/outputs/smoke_test/dkuc/20260703_105217_dkuc_prediction_split \
  --stream_dataset traj_data/outputs/smoke_test/20260828_102529_time_varying_sine_dktv_plan_smoke/dataset.npz \
  --mode full --batch_size 4 --online_epochs 20 --online_lr 1e-6 \
  --device cpu --run_type smoke_test --out_dir /tmp/dktv_plan_smoke2 \
  --rollout_horizons 4 --rollout_stride 2 --tag plan
```

聚焦测试结果：`6 passed`。

端到端 smoke 结果：

- 接受矩阵更新：2 次；
- encoder checksum 改变：2 次；
- 平均 encoder 训练耗时：以最终 manifest 为准（CPU、短批次）；
- fixed DKUC one-step total RMSE：约 `0.1203`；
- full DKTV one-step total RMSE：约 `0.0503`（仅 1 条 8 步 smoke，不构成论文结论）；
- 完整临时产物：`/tmp/dktv_plan_smoke2/dktv/20260828_102610_dktv_plan/`。

## 3. 与论文一致项和工程扩展

一致项：A/B 与 C 的累积最小二乘表示、Woodbury 逆状态、A/B/C 更新后固定矩阵训练 theta、L1/L2 加权损失、先预测后更新的在线顺序。

工程扩展：ridge 默认为 `1e-3`；支持梯度裁剪、AdamW、checkpoint 和独立 trial replay。ridge 与优化器设置均写入 manifest，不能描述为论文原始无正则设置。

已知方法近似：encoder 更新后，历史统计仍位于采集当时的旧特征坐标；实现保留该累积统计并记录 encoder version，没有错误宣称等价于用最新 encoder 对全部历史重编码 refit。

## 4. 未完成项与风险

以下内容没有在本轮伪装为完成：

1. 尚未实现 full DKTV 的 `batch_rollout` 和 fixed-horizon rollout；目前 fixed-horizon 只报告 fixed DKUC。
2. 尚未新增专门的 causality、trial isolation 和中断恢复自动化集成测试；代码路径已按该契约实现，但仍需测试锁定。
3. resume 尚未校验 normalizer fingerprint、优化配置和 state/model 是否属于同一 checkpoint。
4. 尚未执行无扰动对照、中等长度多 trial、弱/中/强幅值和频率扫描。
5. 尚未完成 fixed DKUC、frozen-encoder DKTV、full DKTV 的正式同数据统计比较。
6. 尚未测量峰值内存和 GPU/采样周期实时性。
7. 时变采集目前只实现外部正弦扰动，未实现负载、质量或阻尼时变。
8. 本轮 smoke 的改善仅用于链路检查，样本过少，不能支持“DKTV 优于 DKUC”的研究结论。

## 5. 下一轮建议

优先顺序应为：补齐 full DKTV 的两类 rollout与数组重算测试；增加 causality/isolation/resume pipeline 测试；生成无扰动和三档扰动数据；在相同 trial 上独立运行三种方法；最后再开展 batch size、online epochs 和 loss weight 的单因素实验。阶段 6 的论文级 review 应在这些正式实验完成后另行撰写。
