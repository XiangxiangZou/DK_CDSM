# 五目录独立运行与全流程手工实验指南

本文档说明当前仓库中 `traj_data/`、`prediction/`、`control/`、`common/`、`visualization/` 五个主目录在单独剥离后的运行方式，并区分哪些文件是可以直接运行的入口脚本，哪些文件只是被入口脚本调用的函数/模块文件。

这里的“五个程序”按五个主目录理解。推荐剥离后的目录结构保持为：

```text
new_project_root/
  traj_data/
  prediction/
  control/
  common/
  visualization/
```

后续所有命令都应在 `new_project_root/` 下运行。不要进入某个子目录后再运行脚本，否则相对路径、输出目录和 `common/packages/` 路径注入容易不一致。

## 0. 基本运行约定

本项目固定使用以下 Python：

```powershell
$PY = 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe'
& $PY -c "import sys; print(sys.executable)"
```

如果做论文级训练或控制，CUDA 阶段建议显式使用：

```powershell
--device cuda
```

剥离五目录时，至少要保留：

- `common/assets/multi_joint_cable_driven_space_robot.xml`
- `common/packages/`
- `traj_data/assets/multi_joint_cable_driven_space_robot.xml`
- `prediction/dataset_selections.json`
- `control/model_selections.json`

其中 `common/` 是公共运行依赖，不是一个独立实验阶段。

## 1. 总体流程关系

手工全流程是：

```text
traj_data 采集数据
  -> prediction 训练/评估 Koopman 预测模型
  -> control 读取固定模型做 LQR/MPC/KILC 控制
  -> visualization 读取控制结果生成 MuJoCo GIF 或报告图
```

每个阶段向下一阶段传递的关键内容如下：

| 阶段 | 需要交给下一阶段的东西 | 典型路径 |
| --- | --- | --- |
| 数据采集 | `dataset.npz` | `traj_data/outputs/full_run/<run_id>/dataset.npz` |
| 模型预测 | 整个模型输出目录，不只是单个 `.pt` | `prediction/outputs/full_run/dkac/<run_id>/` |
| 控制实验 | 整个控制结果目录 | `control/outputs/full_run/mpc/<run_id>/` |
| 可视化 | GIF/metadata/报告图片 | `visualization/outputs/full_run/media/<run_id>/` |

注意：控制阶段需要的是“模型目录”，例如 DKAC 需要 `best_dkac.pt`、`normalizers.json`、`model_config.json` 等一整套文件。不要只拷贝一个 `best_dkac.pt`。

## 1.1 半自动交互式脚本

根目录提供了两个半自动运行脚本：

| 文件 | 作用 |
| --- | --- |
| `run_interactive_fullflow.bat` | Windows 双击/右键运行入口，会自动调用 PowerShell 脚本。 |
| `run_interactive_fullflow.ps1` | 实际菜单逻辑，负责询问采集方法、预测模型、控制器、末端轨迹，并按顺序运行各阶段。 |

推荐普通使用方式：

```text
右键/双击 run_interactive_fullflow.bat
```

脚本会依次询问：

1. 运行规模：`full_run` 或 `smoke_test`。
2. 数据采集方法：`controlled`、`uncontrolled random`、`uncontrolled passive`。
3. 预测方法：`dkac`、`dkuc`、`edmd`、`dkn`。
4. 控制器：`mpc`、`lqr`、`kilc`、`none`。
5. 末端轨迹或参考类型：
   - `mpc`: `star` 或 `circle`。
   - `lqr`: `circle` 或 `joint`。
   - `kilc`: 当前固定为 `circle`。
6. 是否渲染 MuJoCo GIF。

脚本内置以下兼容限制：

- `MPC` 当前只接受 `DKAC` artifact；如果选择了其他预测模型再选 `MPC`，脚本会自动改为训练 `DKAC`。
- `LQR` 支持 `EDMD`、`DKUC`、`DKAC`。
- `DKN` 当前只作为 prediction-only；如果选择 `DKN` 后又选择控制器，脚本会自动跳过控制阶段。
- `KILC` 需要已有 continuous-DKUC artifact，不直接使用普通 `dkuc_prediction.py` 的训练输出；脚本会要求手工输入该 artifact 目录。

脚本仍然会把每一步实际执行的命令打印到终端，并在最后汇总：

```text
Dataset
Prediction artifact
Control result
Media output
```

如果需要完全可复现的论文实验，建议运行结束后把终端中打印的这些路径和对应 `manifest.json/metrics.json` 一起记录下来。

## 2. `traj_data/`：数据采集

### 可直接运行的入口脚本

| 文件 | 是否独立运行 | 作用 |
| --- | --- | --- |
| `traj_data/collect_data_controlled.py` | 是 | 受控 PD 数据采集，输出带 `q_ref/dq_ref` 的训练数据。 |
| `traj_data/collect_data_uncontrolled.py` | 是 | 非受控数据采集，支持 `random` 随机开环力矩和 `passive` 自由响应。 |

### 函数/模块文件

| 文件 | 作用 |
| --- | --- |
| `traj_data/mujoco_cdsm.py` | MuJoCo CDSM 适配器，负责状态读写、tendon Jacobian、力矩到 8 根绳索张力分配。 |
| `traj_data/references.py` | 随机参考轨迹、初始状态采样、软限位保护。 |
| `traj_data/data_io.py` | `dataset.npz`、`metadata.json`、`summary.json` 保存与数据检查。 |
| `traj_data/assets/*.xml` | MuJoCo 模型文件。 |
| `traj_data/PROGRAMS.md` | 该目录内部的详细程序说明。 |

### 受控数据采集命令

```powershell
$PY = 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe'
& $PY .\traj_data\collect_data_controlled.py `
  --out_dir .\traj_data\outputs\full_run `
  --tag manual_full
```

默认完整规模是 `--traj 40 --steps 500 --dt 0.01 --seed 42`。

输出目录形如：

```text
traj_data/outputs/full_run/<timestamp>_controlled_pd_manual_full/
  dataset.npz
  metadata.json
  summary.json
```

### 非受控数据采集命令

```powershell
& $PY .\traj_data\collect_data_uncontrolled.py `
  --mode random `
  --out_dir .\traj_data\outputs\full_run `
  --tag manual_full
```

输出目录形如：

```text
traj_data/outputs/full_run/<timestamp>_uncontrolled_random_manual_full/
  dataset.npz
  metadata.json
  summary.json
```

## 3. `prediction/`：Koopman 模型预测

### 可直接运行的入口脚本

| 文件 | 是否独立运行 | 作用 | 控制阶段是否可直接用 |
| --- | --- | --- | --- |
| `prediction/edmd_prediction.py` | 是 | EDMD 训练、one-step/rollout 评估、图像输出。 | 可用于 LQR。 |
| `prediction/dkuc_prediction.py` | 是 | DKUC 训练、one-step/rollout 评估、图像输出。 | 可用于 LQR，KILC 另有 continuous-DKUC artifact 要求。 |
| `prediction/dkac_prediction.py` | 是 | DKAC 训练、one-step/rollout 评估、图像输出。 | 可用于 LQR/MPC，当前 DKAC-MPC 主流程推荐用它。 |
| `prediction/dkn_prediction.py` | 是 | DKN 训练、one-step/rollout 评估、图像输出。 | 当前主要是 prediction-only，不作为现有 LQR/MPC 线性控制模型直接入口。 |

### 函数/模块文件

| 文件 | 作用 |
| --- | --- |
| `prediction/common.py` | 预测阶段公共工具：数据加载、归一化、窗口采样、训练/验证划分、指标、绘图、输出目录。 |
| `prediction/dataset_selections.json` | 数据集选择表，可用 `--dataset_key` 让不同方法选择不同数据集。 |
| `prediction/outputs/` | 训练模型、metrics、预测数组、图片输出目录。 |

### 推荐的数据指定方式

最稳妥方式是直接把上一步生成的 `dataset.npz` 传给 `--train_dataset`：

```powershell
$DATASET = 'traj_data/outputs/full_run/<data_run_id>/dataset.npz'
```

也可以使用 `prediction/dataset_selections.json` 中的 `--dataset_key`，但前提是 JSON 里的路径已经指向实际存在的数据集。多次采集时，直接传 `--train_dataset` 更不容易混淆。

### DKAC 完整训练与预测评估

```powershell
& $PY .\prediction\dkac_prediction.py `
  --train_dataset $DATASET `
  --run_type full_run `
  --pred_mode both `
  --device cuda `
  --tag manual_full
```

输出目录形如：

```text
prediction/outputs/full_run/dkac/<timestamp>_dkac_manual_full/
  best_dkac.pt
  normalizers.json
  model_config.json
  dataset_train.npz
  dataset_val.npz
  one_step_metrics.json
  rollout_metrics.json
  one_step_prediction_rollouts.npz
  rollout_prediction_rollouts.npz
  dkac_training_history.csv
  run_summary.json
```

图像会保存到：

```text
prediction/outputs/full_run/figures/<timestamp>_dkac_manual_full/
```

其他预测方法把脚本名换成对应入口即可：

```powershell
& $PY .\prediction\edmd_prediction.py --train_dataset $DATASET --run_type full_run --pred_mode both --tag manual_full
& $PY .\prediction\dkuc_prediction.py --train_dataset $DATASET --run_type full_run --pred_mode both --device cuda --tag manual_full
& $PY .\prediction\dkn_prediction.py  --train_dataset $DATASET --run_type full_run --pred_mode both --device cuda --tag manual_full
```

## 4. `control/`：控制实验

### 可直接运行的入口脚本

| 文件 | 是否独立运行 | 作用 |
| --- | --- | --- |
| `control/lqr_control.py` | 是 | 有限时域 Koopman LQR，可接 EDMD/DKUC/DKAC artifact。 |
| `control/mpc_control.py` | 是 | DKAC Koopman MPC，支持 `circle/star` 末端轨迹。 |
| `control/kilc_control.py` | 是 | continuous-DKUC KILC 控制，需要对应 artifact。 |
| `control/plotting.py` | 是 | 从已有闭环结果目录补画关节、末端、力矩、绳索张力图。 |

### 函数/模块文件

| 文件 | 作用 |
| --- | --- |
| `control/model_artifacts.py` | 从 `prediction/outputs/...` 读取 EDMD/DKUC/DKAC，并封装成控制可用模型。 |
| `control/model_selections.json` | 控制阶段模型选择表，可用 `--model_key` 固定某一次训练模型。 |
| `control/cable_interface.py` | 力矩到绳索张力、张力约束、实际 plant 施加接口。 |
| `control/references.py` | 关节 ramp、笛卡尔 circle/star 到关节参考轨迹的 IK。 |
| `control/io_utils.py` | 输出目录、manifest、JSON 保存、项目路径。 |
| `control/plotting.py` | 同时也是可运行绘图入口。 |

### 控制阶段如何指定模型

推荐方式一：直接传模型目录。

```powershell
$MODEL_DIR = 'prediction/outputs/full_run/dkac/<prediction_run_id>'
```

推荐方式二：先更新 `control/model_selections.json`，然后用：

```powershell
--model_key dkac_full
```

如果你每次训练都会产生新模型，并且后续控制只想固定某一次模型，最清晰的做法是把那一次模型目录写进 `control/model_selections.json`，然后控制脚本只用 `--model_key`。

### DKAC-MPC 跟踪 20s 五角星轨迹

```powershell
& $PY .\control\mpc_control.py `
  --run_type full_run `
  --device cuda `
  --artifact_dir $MODEL_DIR `
  --trajectory star `
  --period 20 `
  --num_cycles 1 `
  --start_hold 0 `
  --radius 0.45 `
  --inner_radius_ratio 0.382 `
  --tag manual_full
```

输出目录形如：

```text
control/outputs/full_run/mpc/<timestamp>_mpc_manual_full/
  manifest.json
  arrays/
    closed_loop_dkac.npz
    reference.npz
  metrics/
    tracking_metrics.json
```

控制图像会保存到：

```text
control/outputs/full_run/figures/<timestamp>_mpc_manual_full/
```

如需从已有结果补画图：

```powershell
$CONTROL_RESULT = 'control/outputs/full_run/mpc/<control_run_id>'
& $PY .\control\plotting.py --result_dir $CONTROL_RESULT --model dkac
```

### LQR 示例

```powershell
& $PY .\control\lqr_control.py `
  --run_type full_run `
  --device cuda `
  --artifact_dir $MODEL_DIR `
  --model dkac `
  --task circle `
  --tag manual_lqr
```

### KILC 示例

```powershell
& $PY .\control\kilc_control.py `
  --run_type full_run `
  --device cuda `
  --artifact_dir <continuous_dkuc_artifact_dir> `
  --tag manual_kilc
```

KILC 不是直接读取普通 `dkuc_prediction.py` 的所有 artifact 都能保证可用，它依赖 continuous-DKUC/KILC 运行时需要的 artifact 结构。

## 5. `visualization/`：绘图与 MuJoCo 动画

### 推荐可运行入口

| 文件 | 是否独立运行 | 作用 |
| --- | --- | --- |
| `visualization/entrypoints/render_animation.py` | 是 | 单模型 MuJoCo GIF 渲染。 |
| `visualization/entrypoints/render_combined_animation.py` | 是 | 多模型轨迹叠加的 MuJoCo GIF 渲染。 |
| `visualization/plots/control_result_plots.py` | 是 | 控制结果静态图绘制。 |
| `visualization/reports/dkac_circle_tracking_figures.py` | 是 | DKAC circle tracking 报告式图片生成。 |

### 函数/模块文件

| 文件/目录 | 作用 |
| --- | --- |
| `visualization/path_setup.py` | 设置 `common/packages/` 等运行路径。 |
| `visualization/mujoco/*.py` | MuJoCo GIF 实际渲染实现，推荐通过 `entrypoints/` 调用。 |
| `visualization/plots/plotting.py` | 通用绘图函数。 |
| `visualization/legacy/` | 历史可视化脚本，仅用于追溯，不推荐作为新流程入口。 |

### 对 MPC 结果生成单模型 GIF

```powershell
$CONTROL_RESULT = 'control/outputs/full_run/mpc/<control_run_id>'
$MEDIA_DIR = 'visualization/outputs/full_run/media/<timestamp>_mpc_dkac_star'

& $PY .\visualization\entrypoints\render_animation.py `
  --result_dir $CONTROL_RESULT `
  --models dkac `
  --trajectory star `
  --actual_trail_color red `
  --out_dir $MEDIA_DIR `
  --tag manual_full
```

输出文件形如：

```text
visualization/outputs/full_run/media/<timestamp>_mpc_dkac_star/
  <timestamp>_star_dkac_mujoco_motion_manual_full.gif
  <timestamp>_star_mujoco_motion_metadata_manual_full.json
```

## 6. `common/`：公共模块

`common/` 主要是函数库和公共依赖，不是一个实验阶段。

### 主要函数/模块文件

| 文件/目录 | 作用 |
| --- | --- |
| `common/io_utils.py` | 项目路径、`common/packages/` 注入、JSON/manifest 保存。 |
| `common/cable_interface.py` | 关节力矩到绳索张力公共接口。 |
| `common/model_artifacts.py` | 模型 artifact 加载与控制模型适配。 |
| `common/control_metrics.py` | 控制指标计算。 |
| `common/control_plotting.py` | 控制结果绘图公共实现，也可作为补画图工具。 |
| `common/prediction_utils.py` | 预测阶段共享数据、训练、评估和绘图工具。 |
| `common/kilc_model.py` | KILC 专用 continuous-DKUC artifact 加载。 |
| `common/references.py` | 控制参考轨迹和 IK 参数工具。 |
| `common/packages/cdsm/` | CDSM plant、kinematics、reference 等精简运行包。 |
| `common/packages/cable_robotics/` | 绳索分配、安全、metrics、接口。 |
| `common/assets/` | 控制/可视化默认 MuJoCo XML。 |

### 可运行但不作为主流程入口的文件

以下文件含有 `main()` 或调试入口，但推荐只作为工具或开发检查，不作为论文实验主流程：

| 文件 | 建议 |
| --- | --- |
| `common/control_plotting.py` | 可从结果目录补画图；优先用 `control/plotting.py` 或 `visualization/plots/control_result_plots.py`。 |
| `common/packages/cdsm/references/cartesian.py` | 可单独生成/检查笛卡尔参考轨迹；主流程由控制脚本调用。 |
| `common/packages/cdsm/kinematics/nominal_model.py` | 名义模型自检/演示；主流程不直接运行。 |

## 7. 一次完整手工实验命令顺序

下面给出一套“受控数据采集 -> DKAC 预测训练 -> DKAC-MPC 五角星控制 -> MuJoCo GIF”的手工运行流程。路径中的 `<...>` 需要替换成上一步实际生成的目录名。

### 7.1 检查环境

```powershell
$PY = 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe'
& $PY -c "import sys; print(sys.executable)"
& $PY -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

### 7.2 采集受控数据

```powershell
& $PY .\traj_data\collect_data_controlled.py `
  --out_dir .\traj_data\outputs\full_run `
  --tag manual_full
```

记录输出目录，例如：

```powershell
$DATASET = 'traj_data/outputs/full_run/<data_run_id>/dataset.npz'
```

### 7.3 训练 DKAC 并绘制 one-step/rollout 预测误差

```powershell
& $PY .\prediction\dkac_prediction.py `
  --train_dataset $DATASET `
  --run_type full_run `
  --pred_mode both `
  --device cuda `
  --tag manual_full
```

记录模型目录，例如：

```powershell
$MODEL_DIR = 'prediction/outputs/full_run/dkac/<prediction_run_id>'
```

检查至少存在：

```text
best_dkac.pt
normalizers.json
model_config.json
one_step_metrics.json
rollout_metrics.json
```

图像目录应位于：

```text
prediction/outputs/full_run/figures/<prediction_run_id>/
```

### 7.4 用 DKAC-MPC 跟踪 20s 五角星

```powershell
& $PY .\control\mpc_control.py `
  --run_type full_run `
  --device cuda `
  --artifact_dir $MODEL_DIR `
  --trajectory star `
  --period 20 `
  --num_cycles 1 `
  --start_hold 0 `
  --radius 0.45 `
  --inner_radius_ratio 0.382 `
  --tag manual_full
```

记录控制结果目录，例如：

```powershell
$CONTROL_RESULT = 'control/outputs/full_run/mpc/<control_run_id>'
```

检查至少存在：

```text
arrays/closed_loop_dkac.npz
arrays/reference.npz
metrics/tracking_metrics.json
manifest.json
```

控制图像目录应位于：

```text
control/outputs/full_run/figures/<control_run_id>/
```

### 7.5 渲染 MuJoCo 动画

```powershell
$MEDIA_DIR = 'visualization/outputs/full_run/media/<timestamp>_mpc_dkac_star'

& $PY .\visualization\entrypoints\render_animation.py `
  --result_dir $CONTROL_RESULT `
  --models dkac `
  --trajectory star `
  --actual_trail_color red `
  --out_dir $MEDIA_DIR `
  --tag manual_full
```

检查至少存在：

```text
*.gif
*_metadata_*.json
```

## 8. 最容易混淆的点

1. `prediction/` 的输出不是只拿一个模型权重文件，而是拿整个 artifact 目录。
2. `control/` 可以通过 `--artifact_dir` 临时指定模型，也可以通过 `control/model_selections.json` 固定选择模型。
3. `prediction/dataset_selections.json` 只负责“预测阶段选哪个数据集”，`control/model_selections.json` 只负责“控制阶段选哪个预测模型”。
4. `common/` 不是实验入口目录，是公共函数和运行依赖目录。
5. `visualization/mujoco/*.py` 虽然也有 `main()`，但推荐从 `visualization/entrypoints/*.py` 运行。
6. `visualization/legacy/` 是历史可视化代码，除非追溯旧实验，不建议作为新全流程入口。
7. 多次采集、多次训练时，最可靠的做法是在命令中显式写 `--train_dataset` 和 `--artifact_dir`，不要只依赖“最新目录”。
