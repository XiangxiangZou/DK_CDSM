# others 工具程序整理说明

本文件夹用于保存五个主目录之外仍有复用价值的小工具程序。这里不放数据采集、Koopman 预测训练、LQR/MPC/KILC 控制、正式可视化主流程代码；这些功能已经由 `traj_data/`、`prediction/`、`control/`、`common/`、`visualization/` 承担。

## 目录结构

```text
others/
  assets/
    multi_joint_cable_driven_space_robot.xml
  diagnostics/
    test_mujoco_CDSM.py
    mujoco_cdsm_jacobian.py
    cdsm_cable_tau_mapping_audit.py
    mujoco_cdsm_kinematic_planning.py
    mujoco_cdsm_antagonistic_pd_tracking.py
    utils_plot.py
    utils_mujoco_log.py
  reporting/
    report_dkac_circle_tracking.py
```

## diagnostics

| 文件 | 原始来源 | 功能 |
| --- | --- | --- |
| `test_mujoco_CDSM.py` | `archive/diagnostics/test_mujoco_CDSM.py` | MuJoCo 模型结构检查：加载 XML，打印关节、绳索、执行器、传感器、约束和 ctrlrange 信息；可用 `--mode sine/preload/step` 打开 viewer 做开环绳驱检查。 |
| `mujoco_cdsm_jacobian.py` | `archive/diagnostics/mujoco_cdsm_jacobian.py` | 静态校验 tendon Jacobian、绳索张力到关节广义力的映射、同侧预紧是否抵消净力矩，以及手算力矩与 MuJoCo `qfrc_actuator` 是否一致。 |
| `cdsm_cable_tau_mapping_audit.py` | `archive/diagnostics/cdsm_cable_tau_mapping_audit.py` | 审计 `tau_cmd -> F_cable -> tau_act` 的实际误差；支持 PD 轨迹采样和构型网格随机力矩两种模式，并输出 JSON、NPZ 和图。 |
| `mujoco_cdsm_kinematic_planning.py` | `archive/diagnostics/mujoco_cdsm_kinematic_planning.py` | 运动学规划与解析绳长校验：比较解析几何绳长与 MuJoCo tendon length，记录轨迹、张力估计、图像和可回放 MuJoCo 日志/GIF。 |
| `mujoco_cdsm_antagonistic_pd_tracking.py` | `archive/diagnostics/mujoco_cdsm_antagonistic_pd_tracking.py` | 真实 8 绳拮抗驱动下的关节空间 PD 跟踪诊断，用于观察绳索几何限制、张力需求和跟踪误差。 |
| `utils_plot.py` | `archive/support/utils_plot.py` | 旧式科研图保存工具，默认把 PNG/SVG/PDF 保存到 `outputs/figures/<program>/<timestamp>/`。 |
| `utils_mujoco_log.py` | `archive/support/utils_mujoco_log.py` | MuJoCo 原生可回放日志和 GIF 工具，可保存 XML、MJB、NPZ、GIF、metadata，并支持 viewer 回放。 |

这些脚本已经改为从 `others/assets/multi_joint_cable_driven_space_robot.xml` 读取模型，避免继续依赖旧的根目录 `assets/models/`。

## reporting

| 文件 | 原始来源 | 功能 |
| --- | --- | --- |
| `report_dkac_circle_tracking.py` | `experiments/control/report_dkac_circle_tracking.py` | 从已有 DKAC circle `closed_loop_dkac.npz` 结果中重新生成末端轨迹、误差、RMSE、关节、力矩、绳索张力等汇报图片和摘要 JSON。 |

## 已检查但未放入 others 的内容

| 原始位置 | 处理结论 |
| --- | --- |
| `experiments/deployment_pipeline/` | 已由 `traj_data/`、`prediction/`、`control/`、`visualization/` 分别承接，不再重复复制。 |
| `src/koopman_control/`、`src/cdsm/`、`src/cable_robotics/` | 可复用核心已经拆到 `common/` 及五个主目录中；不放进 `others`，避免形成第二套核心库。 |
| `archive/deployment_pipeline/` | 旧版完整流水线，当前功能已经被五目录新结构覆盖；只保留其中与诊断相关的轻量 support。 |
| `experiments/model_comparison/` | 依赖旧 `src/koopman_control` 模型注册与 artifact 结构，属于历史对比流程，不是独立小工具。 |
| `archive/hybrid_models/` | 名义模型 + 残差 Koopman 的历史完整实验，依赖旧 archive 模块链，当前不纳入轻量工具目录。 |
| `archive/diagnostics/test_deep_koopman_cdsm.py` | 依赖旧 `deep_koopman_cdsm` 历史实现，预测验证功能已由 `prediction/` 接管，因此未复制。 |

## 运行方式

从仓库根目录使用项目 Python 环境运行，例如：

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\others\diagnostics\mujoco_cdsm_jacobian.py
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\others\diagnostics\cdsm_cable_tau_mapping_audit.py --traj 10 --steps 100
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\others\reporting\report_dkac_circle_tracking.py --result_dir <dkac_circle_result_dir>
```

注意：`test_mujoco_CDSM.py`、`mujoco_cdsm_kinematic_planning.py`、`mujoco_cdsm_antagonistic_pd_tracking.py` 会打开 MuJoCo viewer 或生成较重的离屏渲染结果，适合作为模型/绳索几何诊断，不作为正式训练或控制入口。
