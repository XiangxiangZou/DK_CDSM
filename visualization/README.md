# visualization 可视化模块说明

`visualization/` 用于集中整理仓库中的绘图、MuJoCo 动画渲染、GIF 生成和历史可视化工具。当前整理策略是复制并归类源码，不移动已有实验输出，避免破坏已经生成的 `outputs/`、`prediction/outputs/`、`control/outputs/` 结果。

## 推荐入口

- `entrypoints/render_animation.py`
  - 单个模型的 MuJoCo 运动 GIF 渲染入口。
  - 调用 `mujoco/mujoco_animation.py`。

- `entrypoints/render_combined_animation.py`
  - EDMD/DKUC/DKAC 多方法末端轨迹合并 GIF 渲染入口。
  - 调用 `mujoco/combined_mujoco_animation.py`。

- `plots/control_result_plots.py`
  - 控制结果绘图工具。
  - 读取 `closed_loop_<model>.npz`，绘制关节位置、关节速度、关节力矩、绳索张力和末端轨迹。

- `plots/plotting.py`
  - 从原 `src/koopman_control/visualization/plotting.py` 复制而来。
  - 用于预测和 tracking 结果的通用绘图。

- `reports/dkac_circle_tracking_figures.py`
  - 从已有 DKAC circle tracking 结果生成报告式图片。

## MuJoCo 动画

- `mujoco/mujoco_animation.py`
  - 从闭环日志回放每个模型的 MuJoCo 运动，分别生成 GIF。

- `mujoco/combined_mujoco_animation.py`
  - 在一个 MuJoCo GIF 中叠加多种方法的末端轨迹。
  - 视觉约定：白色虚线为期望轨迹，EDMD 蓝色，DKUC 橙色，DKAC 绿色。

这两个脚本已经适配当前精简仓库结构：

- Python 包依赖从 `common/packages/` 加载。
- 默认 MuJoCo XML 使用 `common/assets/multi_joint_cable_driven_space_robot.xml`。
- 不再依赖 `experiments._paths` 或根目录 `src/`。

## 历史可视化工具

- `legacy/deployment_pipeline/`
  - 原 `archive/deployment_pipeline/` 中的渲染和 plotting 历史脚本。
  - 保留用于追溯旧 pipeline 的动画/绘图逻辑。

- `legacy/support/`
  - 原 `archive/support/` 中的 `utils_plot.py` 和 `utils_mujoco_log.py`。
  - 主要用于旧脚本的 Matplotlib 保存和 MuJoCo GIF 记录。

- `legacy/examples/`
  - 历史 figure-8 MuJoCo tracking 示例。
  - 已补入旧的 `multi_joint_cdsm_model.py` 和 `legacy/support/` 路径，但仍建议只作为历史参考，不作为新流程推荐入口。

## 未搬运的内容

以下内容没有移动到 `visualization/`：

- 已生成的 PNG/PDF/GIF 结果文件。
  - 这些文件仍保留在 `prediction/outputs/.../figures/`、`control/outputs/.../figures/` 或原结果目录中。
  - 这样可以保持实验结果路径和 metrics/manifest 中记录的路径一致。

- 训练、模型对比、诊断大脚本中内嵌的绘图代码。
  - 例如一些 `archive/model_comparison/`、`archive/diagnostics/` 脚本包含 `matplotlib` 调用，但它们本质是实验或诊断程序，不是纯可视化模块。
  - 如果后续需要保留某个具体旧实验的绘图逻辑，应单独抽取绘图函数，而不是整体复制实验脚本。

## 基本检查命令

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\visualization\entrypoints\render_animation.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\visualization\entrypoints\render_combined_animation.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\visualization\plots\control_result_plots.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\visualization\reports\dkac_circle_tracking_figures.py --help
```
