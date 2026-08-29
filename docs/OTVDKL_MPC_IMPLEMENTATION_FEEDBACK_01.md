# OTVDKL-MPC 实施反馈 01

> 执行日期：2026-08-28  
> 对应计划：`docs/OTVDKL_MPC_IMPLEMENTATION_PLAN.md`  
> 结论：第一轮实现完成了 Algorithm 1 核心、控制数值原型和预测 smoke；未达到计划第 11 节定义的整项完成条件。

## 1. 本轮结论

本轮把仓库原有 OTVDKL 从“仅 A/B、latent 误差、跳过仍推进窗口”整改为：

- 固定 DKUC encoder，并维护 `A_tau/B_tau/C_tau`；
- A/B 与 C 分别保存回归统计量、正则化逆矩阵和直接 refit oracle；
- 式 (17)、(18) 使用 `C[A B][g(x);u]` 的 normalized physical-state RMSE；
- OTVDKL* 先判断阈值，跳过时不构造候选；跳过、拒绝、数值失败均不移动窗口，只有接受才提交模型和窗口；
- 提供不可变 `OTVDKLModelSnapshot` 和包含原始窗口、统计量、版本、样本 ID、pending 数据的 checkpoint；
- 新增独立 `control/otvdkl_control.py`，实现保守对称输入边界、终端 SDP 数值原型、residual 重算、lifted error-coordinate MPC 和确定性安全 fallback；
- 完成预测时间有序 replay smoke，并保存 A/B/C by-step、更新历史、one-step/rollout 数组和指标。

目前不得称为 “stability-guaranteed OTVDKL-MPC”。尚未完成 MuJoCo 闭环、30 s full run、多 seed 对比、模型跳变 residual、Assumption 5 与 ISS Lyapunov 数值验收。

## 2. 公式—代码映射

| 论文对象 | 当前代码对象 | 本轮状态 |
|---|---|---|
| 式 (6)、(7)：lifted dynamics/readout | `SlidingWindowKoopmanUpdater.A/B/C`、`C_struct` | 已实现并区分 learned/structured readout |
| 式 (10)、(11)：A/B 滑窗加减 | `propose()` 的 Gram/cross 与 Woodbury add-delete | 已有并补强 oracle |
| 式 (12)、(13)：C 滑窗更新 | `c_gram/c_cross/inverse_regularized_c_gram` | regularized-safe 已实现；未正则 paper-exact 模式未实现 |
| 式 (17) | `physical_state_rmse()` 和 threshold-first 分支 | 已实现 |
| 式 (18) | `SelectiveWindowKoopmanUpdater.update()` | 已实现，负更新不提交 |
| Algorithm 1 checkpoint | `save_checkpoint()/load_checkpoint()` | 已实现核心状态和 pending buffer |
| 控制模型事务 | `OTVDKLModelSnapshot` | 已实现只读快照；完整 plant 事务尚未实现 |
| 式 (21)：lifted MPC | `LiftedMPC.solve()` | 低维数值原型完成 |
| 式 (22)、(23)：终端设计 | `solve_terminal_sdp()`、`terminal_sdp_margins()` | 可重算数值原型；仍需逐式论文审查 |

## 3. 文件变更

- `prediction/otvdkl_prediction.py`
  - 新增物理状态误差、C 递推状态、C oracle、不可变快照、checkpoint；
  - 修正 OTVDKL* 窗口提交语义；
  - replay 保存 `C_by_step` 和最终 C 统计量。
- `control/otvdkl_control.py`
  - 新增独立 CLI；
  - 新增 SDP/MPC 数值组件、输入边界转换和 fallback。
- `tests/time_varying/test_core_updates.py`
  - 将旧的“阈值跳过推进窗口”断言修正为论文默认语义。
- `tests/time_varying/test_otvdkl_algorithm.py`
  - 覆盖 A/B/C oracle、阈值路径、checkpoint 和物理误差坐标。
- `tests/time_varying/test_otvdkl_control.py`
  - 覆盖对称输入边界、SDP residual 重算和 constrained MPC。

未修改或回滚工作区中已有的 DKTV、数据采集、`.gitignore` 等用户改动。

## 4. 执行与验证

解释器检查：

```text
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -c "import platform, sys; print(sys.executable); print(platform.system())"
```

结果：`/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python`，`Linux`。

静态与单元测试：

```text
env -u PYTHONPATH .../env_dk_cdsm/bin/python -m py_compile \
  prediction/otvdkl_prediction.py control/otvdkl_control.py \
  tests/time_varying/test_otvdkl_algorithm.py \
  tests/time_varying/test_otvdkl_control.py

env -u PYTHONPATH .../env_dk_cdsm/bin/python -m pytest -q \
  tests/time_varying/test_core_updates.py \
  tests/time_varying/test_otvdkl_algorithm.py \
  tests/time_varying/test_otvdkl_control.py
```

结果：`10 passed`。另有两条 OSQP API deprecation warning，不影响数值结果。

两个独立入口均通过 `--help`；控制入口的 `--dry_run` 仅验证配置。没有合适且已审查的物理闭环配置时，普通运行会明确停止，不会伪造控制成功。

## 5. Prediction smoke

命令：

```text
env -u PYTHONPATH MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dk_cdsm_otvdkl_mpl \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  prediction/otvdkl_prediction.py \
  --artifact_dir prediction/outputs/smoke_test/dkuc/20260703_105217_dkuc_prediction_split \
  --stream_dataset traj_data/outputs/full_run/20260828_123055_time_varying_sine_exploratory_sine_seed50/dataset.npz \
  --variant both --window_size 100 --batch_size 10 --epsilon 0.01 \
  --run_type smoke_test --tag plan01
```

输出：

```text
prediction/outputs/smoke_test/otvdkl/20260828_172840_otvdkl_plan01/
```

关键结果：

| 指标 | OTVDKL | OTVDKL* |
|---|---:|---:|
| accepted | 20 | 16 |
| threshold skipped | 0 | 4 |
| rejected / numerical failed | 0 / 0 | 0 / 0 |
| model/window version | 20 / 20 | 16 / 16 |
| max A oracle diff | 9.062e-9 | 8.733e-9 |
| max B oracle diff | 1.034e-9 | 1.034e-9 |
| max C oracle diff | 1.869e-10 | 9.460e-11 |
| mean update time | 1.879 ms | 1.464 ms |

Fixed DKUC one-step total RMSE 为 `0.05827`。本次 OTVDKL one-step total RMSE 为 `0.09407`；因此该 smoke 证明执行链路和保存契约可用，但不证明在线更新提升预测性能。需要多 seed、窗口/阈值消融和更匹配的初始 artifact。

## 6. 分阶段完成度

| 阶段 | 状态 | 说明 |
|---|---|---|
| 0 公式映射与基线 | 基本完成 | 映射见第 2 节，保存了 replay 基线 |
| 1 OTVDKL 预测核心 | 大部分完成 | A/B/C、选择语义、oracle、checkpoint 已完成；paper-exact 未正则模式和恢复后完整 stream 等价测试待补 |
| 2 在线控制事务 | 部分完成 | 只读快照和 fallback 已有；尚未组合 measurement-update-SDP-MPC-actuation 日志事务 |
| 3 Zhang SDP | 原型完成 | 低维可行案例和 residual 重算通过；需依据论文原式再次逐块核查，不作稳定性声明 |
| 4 lifted MPC | 原型完成 | error-coordinate、非对称输入约束和版本诊断通过；独立 QP oracle/终端约束待补 |
| 5 MuJoCo 闭环 | 未完成 | 没有生成闭环或 30 s 产物 |
| 6 稳定性与论文对比 | 未完成 | 多 seed、ISS、Assumption 5、固定 DKUC 对比均未完成 |

## 7. 残余风险和下一轮审查重点

1. 计划引用 2026 Automatica 论文；本轮仅依据仓库计划中的公式描述实施，终端 SDP 必须对论文式 (22)、(23) 原文逐块复核后才能提升声明等级。
2. 当前 SDP 对高维 DKUC latent 每控制步同步求解的耗时和可行率未知，尚无实时性证据。
3. 第一轮控制尚未接入 cable torque bounds、tension allocation、MuJoCo plant 和完整 artifact 输出契约。
4. C_tau 用 regularized direct/recursive statistics；`C_struct` 保留供控制稳定性结构使用，两者不可无记录切换。
5. 当前 replay 的 OTVDKL 误差劣于 fixed DKUC，需审查历史窗口覆盖、窗口宽度、ridge、epsilon 与 stream 分布差异。
6. 尚未检查 joint limit、tension、allocation residual、deadline miss、模型跳变 residual 和 ISS Lyapunov 差分。

因此建议后续 review 首先审查 SDP 的论文逐式一致性和控制事务 API，再批准 MuJoCo smoke；不建议直接运行 30 s 并据此宣称实现完成。
