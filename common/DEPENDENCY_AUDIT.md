# 公共依赖梳理记录

本文件按“后续只保留 `traj_data/`、`prediction/`、`control/`、`common/`”的目标整理。检查范围是这四个目录中的 Python 入口脚本、公共模块、JSON 配置和默认 XML 路径。

## 结论

当前主流程不应再依赖根目录下的 `src/` 和 `assets/`：

- 数据采集：`traj_data/` 内部自洽，使用 `traj_data/assets/multi_joint_cable_driven_space_robot.xml`。
- 模型预测：`prediction/` 主要依赖自身脚本和 `prediction/common.py`；根 `common/__init__.py` 也导出了 `prediction_utils.py`，用于兼容从仓库根目录以模块方式运行预测脚本的情况。
- 控制：`control/` 仍需要 `cdsm`、`cable_robotics` 两组运行期公共包，现已迁移到 `common/packages/`；旧 `koopman_control` 依赖已拆分为 `common/control_metrics.py`、`common/artifacts.py`、`common/kilc_model.py`。
- 控制默认 XML：从原 `assets/models/multi_joint_cable_driven_space_robot.xml` 复制到 `common/assets/multi_joint_cable_driven_space_robot.xml`。

## 已迁移到 common 的运行期包

### `common/packages/cable_robotics/`

来源：`src/cable_robotics/`

用途：

- 关节力矩到绳索张力的 antagonistic allocation。
- 绳索张力上下限到等效关节力矩边界的推导。
- 被 `control/cable_interface.py` 和 `common/cable_interface.py` 使用。

保留文件：

- `__init__.py`
- `interfaces.py`
- `metrics.py`
- `safety.py`
- `tension_allocator.py`

### `common/packages/cdsm/`

来源：`src/cdsm/`

用途：

- CDSM 命名常量、主动关节、mimic 关节、绳索名、执行器名。
- MuJoCo plant 封装：读状态、施加绳索张力、计算 tendon Jacobian。
- MuJoCo site IK：末端笛卡尔轨迹到关节参考轨迹。
- 笛卡尔参考轨迹生成。

关键被调用路径：

- `control/cable_interface.py`
- `control/references.py`
- `control/lqr_control.py`
- `control/mpc_control.py`
- `control/kilc_control.py`

注意：

- `common/packages/cdsm/constants.py` 的 `DEFAULT_XML_PATH` 已改为指向 `common/assets/multi_joint_cable_driven_space_robot.xml`。
- `cdsm.kinematics.__init__`、`cdsm.plants.__init__`、`cdsm.references.__init__` 已收窄导入，避免无关模块引入额外依赖。

### Koopman 相关公共功能的归属

不再保留 `common/packages/koopman_control/`。原先临时放在该包中的功能已按用途拆分：

- `common/control_metrics.py`：控制阶段 tracking metrics。
- `common/artifacts.py`：JSON 和 runtime matrix 保存工具。
- `common/kilc_model.py`：KILC 专用 continuous-DKUC artifact 运行时加载。

EDMD、DKUC、DKAC、DKN 的训练与预测仍归 `prediction/`，控制脚本通过 `common/model_artifacts.py` 读取 `prediction/outputs/...` 中的固定模型。

## 保留目录内部依赖

### `traj_data/`

入口：

- `collect_data_controlled.py`
- `collect_data_uncontrolled.py`

内部公共模块：

- `mujoco_cdsm.py`
- `references.py`
- `data_io.py`

默认资源：

- `traj_data/assets/multi_joint_cable_driven_space_robot.xml`

当前不依赖根 `src/`。

### `prediction/`

入口：

- `edmd_prediction.py`
- `dkuc_prediction.py`
- `dkac_prediction.py`
- `dkn_prediction.py`

内部公共模块：

- `common.py`

选择文件：

- `dataset_selections.json`

输出规则：

- 非图片模型结果：`prediction/outputs/<run_type>/<method>/<run_id>/`
- 图片/PDF：`prediction/outputs/<run_type>/figures/<run_id>/`

### `control/`

入口：

- `lqr_control.py`
- `mpc_control.py`
- `kilc_control.py`

内部公共模块：

- `io_utils.py`
- `cable_interface.py`
- `model_artifacts.py`
- `references.py`
- `plotting.py`

选择文件：

- `model_selections.json`

输出规则：

- 数值结果：`control/outputs/<run_type>/<method>/<run_id>/`
- 图片/PDF：`control/outputs/<run_type>/figures/<run_id>/`

默认资源：

- `common/assets/multi_joint_cable_driven_space_robot.xml`

## 删除旧目录前的核对项

删除 `src/`、根 `assets/`、`archive/`、`experiments/`、`configs/`、`docs/`、`tests/` 前，至少重新运行：

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\traj_data\collect_data_controlled.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\traj_data\collect_data_uncontrolled.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\prediction\edmd_prediction.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\prediction\dkuc_prediction.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\prediction\dkac_prediction.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\prediction\dkn_prediction.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\control\lqr_control.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\control\mpc_control.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\control\kilc_control.py --help
```

更严格的检查是临时把 `src/` 改名后再运行上述命令和一个最小 smoke test；正式删除前不要直接删除已有实验结果。
