# 旧系统到新系统对应关系

本文件用于快速从旧实现定位到当前新主线。

| 旧系统位置 | 当前主线位置 | 说明 |
| --- | --- | --- |
| `experiments/deployment_pipeline/collect_data.py`、`src/cdsm/data_collection.py` | `traj_data/` | 数据采集已拆成受控/非受控两个入口，并保留独立 MuJoCo 适配器。 |
| `experiments/deployment_pipeline/train_models.py`、`src/koopman_control/training/*` | `prediction/` | EDMD、DKUC、DKAC、DKN 各自拥有独立训练与评估脚本。 |
| `src/koopman_control/data/*` | `prediction/common.py`、`common/prediction_utils.py` | 数据加载、归一化、窗口、预测评估等公共逻辑已迁移。 |
| `src/koopman_control/control/finite_horizon_lqr.py`、`src/cdsm/runtime/tracking.py` | `control/lqr_control.py` | LQR 控制核心和运行入口合并为一个可直接运行脚本。 |
| `archive/control/cdsm_mpc_tracking_compare.py`、旧 constrained MPC 相关逻辑 | `control/mpc_control.py` | DKAC-MPC 控制主线，支持 `circle/star`。 |
| `src/koopman_control/control/yu_tan_kilc.py`、`src/cdsm/runtime/kilc_tracking.py` | `control/kilc_control.py`、`common/kilc_model.py` | KILC 主线保留为 continuous-DKUC 专用控制流程。 |
| `src/cable_robotics/*` | `common/packages/cable_robotics/`、`common/cable_interface.py`、`control/cable_interface.py` | 绳索张力分配、安全和控制接口已成为公共模块。 |
| `src/cdsm/*` | `common/packages/cdsm/` | CDSM plant、IK、reference、常量等运行期依赖已精简迁移。 |
| `src/cdsm/visualization/*`、`archive/deployment_pipeline/run_06_render_mujoco_animation.py` | `visualization/` | MuJoCo 动画、合并动画、控制结果绘图已归入可视化目录。 |
| `archive/diagnostics/*` | `others/diagnostics/` | 模型检查、Jacobian 校验、力矩到张力映射审计等小工具已单独整理。 |
| `experiments/control/report_dkac_circle_tracking.py` | `others/reporting/`、`visualization/reports/` | DKAC circle tracking 报告图生成工具已保留。 |
| `assets/models/multi_joint_cable_driven_space_robot.xml` | `traj_data/assets/`、`common/assets/`、`others/assets/` | 当前主线使用各自目录内复制的 XML，避免依赖旧根 `assets/`。 |
| `configs/deployment/*.json` | `prediction/dataset_selections.json`、`control/model_selections.json` | 当前选择层拆成数据集选择和模型 artifact 选择。 |

## 新系统运行入口

首选阅读：

```text
FIVE_FOLDER_RUN_GUIDE.md
```

半自动运行：

```text
run_interactive_fullflow.bat
```

手工全流程：

```text
traj_data/collect_data_controlled.py
prediction/dkac_prediction.py
control/mpc_control.py
visualization/entrypoints/render_animation.py
```
