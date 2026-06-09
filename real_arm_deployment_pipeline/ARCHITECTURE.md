# Real-Arm Deployment Pipeline Architecture

目标：把当前 MuJoCo 绳驱空间机械臂先当作实验室真实机械臂，用同一套数据、预测评估、闭环控制和日志接口，对 DKUC、DKAC、EDMD、DKN 做模型预测和跟踪控制对比。

后续替换真实机械臂时，只替换设备接口和必要的绳张力映射，不改训练、预测评估和控制对比主流程。

## 统一实验流程

1. 离线采集数据
   - 从被控对象读取 `q, dq`。
   - 用激励控制器生成 `tau_cmd`。
   - 通过绳驱映射得到 8 根绳张力 `cable_tensions`。
   - 执行一步并记录下一时刻状态。

2. 离线训练模型
   - 所有模型读取同一份训练/验证数据。
   - 统一使用状态 `x = [qa, qb, dqa, dqb]`。
   - 统一使用控制输入 `u = [tau_a, tau_b]`。
   - 保存每个模型自己的权重、字典、标准化参数和训练历史。

3. 离线 rollout 验证
   - 使用验证集记录的 `x0` 和 `u_seq`。
   - 统一比较 `one_step` 和 `rollout`，以后优先看 `rollout`。
   - 输出每个模型的 `total_rmse`、`rmse_by_state`、`step_rmse` 和预测轨迹。

4. 在线闭环控制
   - 每个控制周期读取真实 `q, dq`。
   - 用当前真实状态计算模型内部表示 `z`。
   - 控制器根据当前反馈和参考轨迹求当前 `tau_cmd`。
   - `tau_cmd` 映射为 8 根绳张力并执行一步。
   - 下一周期重新读取真实反馈。

## 模型接口分层

所有预测模型至少实现：

```text
name
fit(train_data, val_data, output_dir)
load(artifact_dir)
lift(x_phys) -> z
step_latent(z, u_phys, x_phys optional) -> z_next
recover_state(z) -> x_phys
rollout(x0, u_seq) -> x_pred_seq
```

可直接进入同一套 Koopman LQR 跟踪控制的模型还需要暴露：

```text
A
B
C
control_mode
recover_control(x_phys, v_or_u_internal) -> u_phys
```

## 四类模型的定位

### EDMD

定位：
- 固定 RBF 字典 Koopman baseline。
- 形式：`z = [x_n, rbf(x_n)]`, `z_next = A z + B u_n`。
- 可以做模型预测。
- 可以直接用 Koopman-space LQR/MPC 做跟踪控制。

现有来源：
- `cdsm_koopman_vs_edmd_model_compare.py`
- `cdsm_dkac_vs_edmd_tracking_control.py`

### DKUC

定位：
- Deep Koopman with unchanged control。
- 形式：`z = [x_n, phi_x(x_n)]`, `z_next = A z + B u_n`。
- 可以做模型预测。
- 可以直接用 Koopman-space LQR/MPC 做跟踪控制。

现有来源：
- `cdsm_dkuc_vs_dkac_tracking_control.py`

### DKAC

定位：
- Deep Koopman with autoencoded/control-transformed control。
- 形式：`z = [x_n, phi_x(x_n)]`, `v = G(x_n) u_n`, `z_next = A z + B v`。
- 可以做模型预测。
- 可以用 Koopman-space LQR/MPC 求内部控制 `v`，但执行前必须通过 DKAC runtime 恢复或求解实际 `u_phys`。

现有来源：
- `cdsm_dkuc_vs_dkac_tracking_control.py`
- `cdsm_dkac_vs_edmd_tracking_control.py`

### DKN

定位：
- Deep Koopman Nonlinear prediction model。
- 当前脚本只用于模型预测对比，不直接做 LQR/MPC。
- 原因：DKN 的控制项是状态相关非线性控制编码，不能直接套 `z_next = A z + B u` 的同一套线性 LQR。

第一阶段处理：
- 纳入统一预测评估：`one_step`、`rollout`、RMSE。
- 暂不纳入同一套 Koopman LQR 跟踪控制排名。

第二阶段可选扩展：
- 为 DKN 单独实现 nonlinear MPC。
- 或构造局部线性化/控制反演层，再进入跟踪控制对比。
- 只有完成上述控制接口后，DKN 才能与 DKUC、DKAC、EDMD 做公平闭环跟踪控制对比。

现有来源：
- `cdsm_dkn_vs_edmd_prediction_compare.py`

## 新目录建议文件清单

```text
real_arm_deployment_pipeline/
  ARCHITECTURE.md
  configs/
    experiment_common.json
    models.json
    tracking_reference.json
  data/
    raw/
    processed/
  artifacts/
    edmd/
    dkuc/
    dkac/
    dkn/
  results/
    prediction/
    tracking/
  plant_interface.py
  mujoco_plant.py
  real_arm_plant.py
  cable_mapping.py
  data_collection.py
  datasets.py
  normalizers.py
  model_base.py
  model_edmd.py
  model_dkuc.py
  model_dkac.py
  model_dkn.py
  prediction_eval.py
  tracking_controller.py
  tracking_runtime.py
  tracking_eval.py
  cartesian_reference.py
  mujoco_ik.py
  run_01_collect_data.py
  run_02_train_all_models.py
  run_03_validate_prediction.py
  run_04_tracking_compare.py
  run_05_cartesian_ik_tracking_compare.py
  run_06_render_mujoco_animation.py
  run_08_render_combined_mujoco_trajectory_gif.py
```

## 调用关系

```text
run_01_collect_data.py
  -> plant_interface.MujocoPlant or RealArmPlant
  -> cable_mapping.tau_to_cable_tensions
  -> data_collection.collect_trajectories
  -> data/raw/*.npz

run_02_train_all_models.py
  -> datasets.load_dataset
  -> normalizers.fit/load/save
  -> model_edmd.EDMDModel
  -> model_dkuc.DKUCModel
  -> model_dkac.DKACModel
  -> model_dkn.DKNModel
  -> artifacts/<model_name>/*

run_03_validate_prediction.py
  -> model_base.PredictiveModel.rollout
  -> prediction_eval.evaluate_one_step
  -> prediction_eval.evaluate_rollout
  -> results/prediction/*

run_04_tracking_compare.py
  -> plant_interface.MujocoPlant or RealArmPlant
  -> tracking_controller.KoopmanLqrTracker
  -> tracking_runtime.run_joint_space_closed_loop_model
  -> model_edmd / model_dkuc / model_dkac control runtimes
  -> cable_mapping.tau_to_cable_tensions
  -> tracking_eval.metrics
  -> results/tracking/*

run_05_cartesian_ik_tracking_compare.py
  -> cartesian_reference.generate_cartesian_reference
  -> mujoco_ik.MujocoSiteIK
  -> tracking_runtime.run_joint_space_closed_loop_model
  -> model_edmd / model_dkuc / model_dkac control runtimes
  -> plotting.plot_tracking_figures / plot_cartesian_tracking_figures
  -> results/cartesian_tracking/*

笛卡尔参考轨迹默认使用 `time_scaling=quintic`：
- 圆/8 字轨迹对整条路径相位做五次多项式时间缩放，使起点和终点速度、加速度为 0。
- 正方形轨迹对每条边分别做五次多项式插值，使四个拐角处速度、加速度为 0。
- 如需复现实验中的原始匀速相位/线性分段轨迹，可显式传入 `--time_scaling linear`。

run_06_render_mujoco_animation.py
  -> results/cartesian_tracking/<run>/<trajectory>/closed_loop_<model>.npz
  -> mujoco_plant.MujocoCablePlant
  -> mujoco.Renderer
  -> results/cartesian_tracking/<run>/<trajectory>/animations/*.gif
  - 动画中白色虚线表示期望末端轨迹，红色实线表示该方法的真实末端运动轨迹。

run_08_render_combined_mujoco_trajectory_gif.py
  -> results/cartesian_tracking/<run>/<trajectory>/cartesian_ik_reference.npz
  -> results/cartesian_tracking/<run>/<trajectory>/closed_loop_edmd.npz
  -> results/cartesian_tracking/<run>/<trajectory>/closed_loop_dkuc.npz
  -> results/cartesian_tracking/<run>/<trajectory>/closed_loop_dkac.npz
  -> mujoco_plant.MujocoCablePlant
  -> mujoco.Renderer
  -> results/cartesian_tracking/<run>/<trajectory>/animations/*combined*.gif
  - 合并 GIF 中白色虚线表示期望末端轨迹，蓝/橙/绿实线分别表示 EDMD/DKUC/DKAC 的真实末端运动轨迹。
```

## 必须保存的数据

离线数据：
- `states`: `(traj, steps+1, 4)`，`[qa, qb, dqa, dqb]`
- `inputs`: `(traj, steps, 2)`，`[tau_a, tau_b]`
- `q_ref`: `(traj, steps, 2)`
- `dq_ref`: 如果采集器能提供，必须保存
- `cable_ctrl`: `(traj, steps, 8)`
- `meta.json`: `dt`、采样策略、PD 参数、绳张力限制、随机种子、XML/真实机械臂版本

模型产物：
- `model_config.json`
- `normalizers.json`
- `training_history.csv`
- `checkpoint.pt` 或 `model.npz`
- `runtime_matrices.npz`: 能用于控制的模型保存 `A, B, C`

预测结果：
- `prediction_rollouts.npz`
- `prediction_metrics.json`
- `one_step_metrics.json`
- `rollout_metrics.json`

跟踪结果：
- `closed_loop_<model>.npz`
- `tracking_metrics.json`
- `solve_time_ms`
- `tau_cmd`
- `cable_tensions`
- `x_meas`
- `x_ref`

末端笛卡尔轨迹跟踪结果：
- `cartesian_ik_reference.npz`: `ee_ref`、`q_ref`、`dq_ref`、`ee_ik`、`ik_error`
- `closed_loop_<model>.npz`: 每个模型的 `x_meas`、`ee_meas`、`tau_cmd`、`cable_tensions`
- `cartesian_tracking_metrics.json`: 关节 RMSE、末端 RMSE、IK 几何误差
- `cartesian_tracking_summary.json`: 多轨迹汇总
- `animations/*.gif`: 读取 `closed_loop_<model>.npz` 回放出的 MuJoCo 运动动画，含期望/实际末端轨迹叠加
- `animations/*combined*.gif`: 同一个 MuJoCo 动画中叠加三种方法的末端实际轨迹

## 近期拆分顺序

1. 先抽出 `plant_interface.py`、`mujoco_plant.py`、`cable_mapping.py`，保证 MuJoCo 被当作真实机械臂调用。
2. 再抽出 `datasets.py`、`normalizers.py`，保证四类模型共用同一份数据。
3. 再抽出 `model_base.py` 和四个模型适配器。
4. 先完成统一预测对比：EDMD、DKUC、DKAC、DKN。
5. 再完成统一跟踪控制对比：EDMD、DKUC、DKAC。
6. DKN 的跟踪控制单独立项，不和前三者混用同一个线性 LQR 接口。
