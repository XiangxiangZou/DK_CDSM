# common 公共模块说明

`common/` 用来临时集中整理当前脚本中已经出现复用需求的公共模块。现阶段先保留源文件夹中的原模块，不直接改动现有入口脚本的 import，避免影响已经能运行的数据采集、预测和控制流程。

## 模块分类

- `io_utils.py`
  - 项目根目录、`common/packages/` 路径注入、默认 MuJoCo XML 路径。
  - 不再注入根目录 `src/`，用于验证五目录独立运行。
  - 控制阶段输出目录、JSON 保存、Git 元信息和 manifest 生成。

- `cable_interface.py`
  - 关节力矩到绳索张力的分配接口。
  - 绳索张力下的关节力矩边界计算。
  - 控制器向 MuJoCo plant 施加绳索张力的公共接口。

- `model_artifacts.py`
  - 从 `prediction/outputs/...` 读取 EDMD、DKUC、DKAC 模型。
  - 将预测模型封装成控制器可直接使用的 `ControlModelAdapter`。
  - 根据 `control/model_selections.json` 或命令行路径解析控制器使用的固定预测模型。

- `control_metrics.py`
  - 控制阶段共享的关节空间和笛卡尔空间跟踪指标。
  - 从旧 `koopman_control.evaluation.tracking` 拆出，避免公共目录中继续保留 Koopman 包。

- `artifacts.py`
  - JSON、NumPy 数组等结果文件保存辅助函数。
  - 当前主要供可视化脚本保存 metadata 使用。

- `kilc_model.py`
  - KILC 专用的 continuous-DKUC artifact 加载和运行时封装。
  - 只服务 `control/kilc_control.py`，不承担 EDMD/DKUC/DKAC/DKN 的预测训练。

- `references.py`
  - 关节空间 ramp 参考轨迹。
  - 笛卡尔圆轨迹到关节参考轨迹的 IK 转换。
  - 控制脚本共享的 IK 参数构造。

- `control_plotting.py`
  - 控制结果的关节位置、关节速度、力矩、绳索张力和末端轨迹绘图。
  - 按 `control/outputs/<run_type>/figures/<run_id>/` 规则保存 PNG/PDF。
  - 可从已有 `closed_loop_<model>.npz` 结果目录补画图片。

- `prediction_utils.py`
  - 从 `prediction/common.py` 复制而来。
  - 预测阶段共享的数据加载、归一化、训练/验证集划分、输出目录、评估指标和预测结果绘图。
  - 放入 `common/` 后仍然指向原 `prediction/dataset_selections.json` 和 `prediction/outputs/...`。

- `packages/`
  - 从原 `src/` 中迁移出的运行期公共包。
  - 当前只保留 `cable_robotics`、`cdsm` 两个精简包，用于支撑控制脚本和公共接口。
  - 原先临时放入的 `koopman_control` 包已拆分到 `control_metrics.py`、`artifacts.py`、`kilc_model.py`。

- `assets/`
  - 控制阶段默认使用的 MuJoCo XML。
  - 当前文件来自原 `assets/models/multi_joint_cable_driven_space_robot.xml`，用于保持控制阶段默认模型不变。

## 后续整理建议

后续可以逐步把 `prediction/*.py` 和 `control/*.py` 的 import 切换到 `common.*`，每切换一类入口脚本就运行对应的 `--help` 和一个最小 smoke test。这样可以避免一次性大范围改 import 导致预测、控制流程同时失效。
