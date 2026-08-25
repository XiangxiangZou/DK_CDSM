# DKTV Plan 01 执行情况反馈报告

> 对应计划：`DKTV_PLAN_01_FOUNDATION.md`
>
> 执行日期：2026-08-25
>
> 最终结论：**Plan 01 验收通过，可以进入 Plan 02**

## 1. 执行结论

Plan 01 规定的公共配置、时变数据、唯一初始模型、统一评价和输出契约均已实现。
smoke 与 full-run 均由同一个 `configs/dktv/base.json` 驱动并得到
`status=accepted`。full-run 中 fixed DKO 的 20 步阶段 rollout RMSE 从 nominal
阶段的 `0.0159488` 增至 time-varying 阶段的 `0.0566936`，退化比为
`3.55473`，超过配置要求的 `1.1`。

本阶段没有实现累积更新、滑动窗口、选择性更新、在线 encoder、SDP-MPC 或其他
物理时变场景，也没有迁移或删除旧五目录及历史结果。

## 2. 实施内容与计划步骤对照

| 计划步骤 | 实施结果 | 核查证据 |
| --- | --- | --- |
| 复用采集与 DKUC 入口 | 复用 `traj_data` 的 MuJoCo plant、张力分配和参考生成；复用 `prediction` 的 DKUC 训练、加载和绘图 | `src/cdsm/dktv_data.py`、`src/koopman_control/dktv/foundation.py` |
| 公共配置与校验 | 冻结四个方法名、状态/输入、16 维 affine lift、固定 encoder、ridge、seed、扰动、质量与评价参数 | `configs/dktv/base.json`、`src/koopman_control/dktv/config.py` |
| 正弦扰动与数据字段 | MuJoCo 主动关节通过 `qfrc_applied` 接收正弦外扰；保存计划要求的 11 个字段和 4 个旧接口兼容别名 | full raw `dataset.npz` |
| 时变性与质量检查 | 实现相同 x/u 不同绝对时刻测试；检查有限值、限位、饱和、张力异常、残差和范围 | `metrics/time_variation.json`、`metrics/data_quality.json` |
| smoke 后 full-run | smoke 和 full 均通过同一入口、配置和 seed 运行 | 下文产物路径 |
| 冻结唯一初始模型 | 固定 `encoder.pt`、`normalizers.json`、`A0/B0/C0` 和标准 DKUC 加载文件；四个方法共享同一 artifact | `artifact_manifest.json` |
| fixed DKO 统一评价 | 保存 one-step、完整 rollout、10/20/50/100 步 rollout 和分阶段 20 步 rollout | `metrics/`、`arrays/fixed_dko_predictions.npz` |
| manifest/数组/图形/日志 | 每次结果均包含 `manifest.json`、`metrics/`、`arrays/`、`figures/`、`logs/` | full result 目录 |

## 3. 代码与配置变更

- `configs/dktv/base.json`
  - 唯一公共基础配置；seed 为 `20260825`。
  - lift 为 `[x, phi(x), 1]`，总维数 16，其中 encoder 输出 11 维。
- `traj_data/mujoco_cdsm.py`
  - 新增外部关节扰动和绳张力等效关节力矩接口。
  - `set_state` 同时复位绝对仿真时间和外力，保证时变性对照公平。
- `prediction/dkuc_prediction.py`
  - 增加默认关闭的常数 observable；旧 artifact 行为保持不变。
- `src/cdsm/dktv_data.py`
  - 时变受控采集、字段契约、质量验收、确定性拆分和时变性证明。
- `src/koopman_control/dktv/config.py`
  - 公共配置和阶段边界校验。
- `src/koopman_control/dktv/foundation.py`
  - 固定 encoder 后的 ridge `A0/B0` 重估、artifact 冻结和统一评价。
- `experiments/dktv/plan_01.py`
  - smoke/full 端到端模块入口与可复现 manifest。
- `tests/test_dktv_foundation.py`
  - 7 个配置、数据、affine lift 和真实 MuJoCo 时变性测试。

## 4. 执行命令

以下命令均在仓库根目录执行，并按 `AGENTS.md` 清除了继承的 ROS
`PYTHONPATH`：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dktv_plan01_matplotlib \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m experiments.dktv.plan_01 --run-type smoke --device cpu --tag acceptance

env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dktv_plan01_matplotlib \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m experiments.dktv.plan_01 --run-type full --device cpu --tag baseline

env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dktv_plan01_matplotlib \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python -m pytest
```

另执行了 `py_compile`、CLI `--help`、旧 DKUC artifact 加载、逐数组数据重放、
逐项评价复算、代表性 PNG 打开检查和 `git diff --check`。

## 5. 产物位置

### 5.1 Smoke

- run id：`20260825_112618_plan01_smoke_acceptance`
- raw：`outputs/data/raw/20260825_112618_plan01_smoke_acceptance/`
- processed：`outputs/data/processed/20260825_112618_plan01_smoke_acceptance/`
- model：`outputs/models/dktv/20260825_112618_plan01_smoke_acceptance/`
- result：`outputs/results/dktv/plan_01/20260825_112618_plan01_smoke_acceptance/`
- 状态：`accepted`

### 5.2 Full-run（后续计划应使用的基线）

- run id：`20260825_112709_plan01_full_baseline`
- raw：`outputs/data/raw/20260825_112709_plan01_full_baseline/dataset.npz`
- processed：`outputs/data/processed/20260825_112709_plan01_full_baseline/`
- model：`outputs/models/dktv/20260825_112709_plan01_full_baseline/`
- result：`outputs/results/dktv/plan_01/20260825_112709_plan01_full_baseline/`
- manifest：`outputs/results/dktv/plan_01/20260825_112709_plan01_full_baseline/manifest.json`
- 状态：`accepted`

`outputs/` 已由根 `.gitignore` 排除。没有产生 rejected 数据，也没有删除或迁移旧
结果。

## 6. Full-run 数据质量

| 指标 | 结果 |
| --- | ---: |
| 轨迹数 × 步数 | 20 × 240 |
| 必需字段有限值 | 全部通过 |
| 饱和标志 | 0 |
| 关节限位标志 | 0 |
| 张力异常 | 0 |
| 最大命令力矩 | 20.2488 Nm |
| 最大实际力矩 | 20.2488 Nm |
| 最大有效绳张力 | 43.9885 N |
| 最大分配残差 | 1.33839e-7 Nm |

原始数据包含：`t`、`states`、`commanded_torque`、`applied_torque`、
`commanded_tensions`、`effective_tensions`、`allocation_residual`、
`disturbance_torque`、`reference_state`、`saturation_flags`、
`joint_limit_flags`。在线辨识兼容字段 `inputs` 与 `applied_torque` 是同一数组。

## 7. 时变性证明

对相同状态 `[0.1, -0.1, 0.05, -0.03]` 和相同命令力矩
`[2.0, -1.5]`，分别在绝对时刻 `0.11 s` 和 `0.37 s` 前进一步：

- 固定 plant 下一状态差异范数：`3.46945e-18`；
- 正弦时变 plant 下一状态差异范数：`0.0196820`；
- 判定：通过。

该检查证明变化来自绝对时间相关外扰，而不是仅由参考轨迹随时间变化造成。

## 8. 初始模型 artifact

- `A0`：`16 × 16`
- `B0`：`16 × 2`
- `C0`：`4 × 16`，前四列为单位阵，其余严格为零
- affine 常数行：`A0[-1,-1]=1`，其余常数行和 `B0[-1]` 为零
- encoder：固定，在线更新关闭
- normalizer：固定并独立保存
- ridge lambda：`1.0e-3`
- DKUC 最佳验证损失：`0.00419649`
- ridge latent fit RMSE：`0.00982091`
- 控制接口：DKUC adapter 独立加载成功，`A/B/C` 形状正确
- 共享方法：`fixed_dko`、`dktv_accumulative`、`otvdkl_window`、
  `otvdkl_selective`

Plan 02 和 Plan 03 应直接加载 full-run model 目录，不重新训练初始 encoder。

## 9. Fixed DKO 基线指标

### 9.1 One-step 与长序列

| 评价 | 总 RMSE | 最大绝对误差 | 非有限预测 |
| --- | ---: | ---: | ---: |
| one-step | 0.00378845 | 0.0538216 | 0 |
| 从每条验证轨迹起点完整 rollout | 0.0915095 | 0.414228 | 0 |

one-step 各状态 RMSE：

```text
qa   2.46948e-5
qb   3.49481e-5
dqa  0.00388707
dqb  0.00650371
```

### 9.2 统一 rollout horizon

| Horizon | 总 RMSE | 最大绝对误差 | 非有限预测 |
| ---: | ---: | ---: | ---: |
| 10 | 0.0198429 | 0.164820 | 0 |
| 20 | 0.0352470 | 0.297351 | 0 |
| 50 | 0.0632844 | 0.479342 | 0 |
| 100 | 0.0648991 | 0.404273 | 0 |

### 9.3 分阶段 20 步 rollout

| 阶段 | 扰动比例 | 总 RMSE | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| nominal | 0.0 | 0.0159488 | 0.0782575 |
| transition | 0.5 | 0.0321644 | 0.165515 |
| time_varying | 1.0 | 0.0566936 | 0.295080 |

time-varying/nominal RMSE 比为 `3.55473`，fixed DKO 的预测退化清晰可识别。

## 10. 测试与复现结果

- 自动化测试：`7 passed in 1.17s`。
- 旧 68 维 DKUC artifact：加载成功，`A=(68,68)`、`B=(68,2)`，预测有限。
- full raw 数据重放：17 个保存字段逐数组 `array_equal=True`。
- 确定性拆分：训练索引与验证索引完全一致。
- one-step、10/20/50/100 步和全部阶段指标复算：与保存 JSON 精确一致。
- 代表性图像：`one_step_states.png` 和 `rollout_prediction_error.png` 可正常打开。
- `git diff --check`：通过。

## 11. 验收标准逐条核查

- [x] 未批量搬迁或重复实现当前五部分工作流。
- [x] 一份基础配置驱动数据、初始模型和固定基线评价。
- [x] 相同 x/u、不同绝对时刻能区分固定 plant 和时变 plant。
- [x] 数据没有未记录的非有限值、限位、饱和或张力异常。
- [x] 初始模型可独立加载，包含固定 encoder、normalizer 和 A0/B0/C0。
- [x] fixed DKO 同时输出 one-step、rollout 和分阶段结果。
- [x] 相同 seed 复现数据、拆分和评价。
- [x] 自动化检查和 `git diff --check` 通过。
- [x] fixed DKO 在时变阶段表现出可识别退化。

## 12. 剩余边界与后续建议

- 本轮只确认模型接口可被控制适配器加载，没有修改或运行闭环 MPC；这符合
  Plan 01 边界。
- 当前执行会话没有可用 CUDA 设备，因此 smoke/full 均使用 CPU；manifest 已记录
  `device_used=cpu`。
- Plan 02 应以 full-run 的 `validation_stream.npz` 作为同一时序数据流，并直接加载
  full-run model artifact。首次累积递推实现必须与全历史 direct refit 做数值对照，
  未通过前不进入 Plan 03。
