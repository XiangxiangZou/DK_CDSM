# DK_CDSM

DK_CDSM 是一个面向绳驱空间机械臂（Cable-Driven Space Manipulator，CDSM）的 Koopman 动力学建模、在线更新、闭环控制与 MuJoCo 可视化研究仓库。项目采用“数据采集、模型预测、控制实验、可视化展示相互独立”的组织方式，各阶段通过明确的文件产物衔接。

当前仓库不提供自动全流程编排脚本。正式实验应逐阶段运行，并显式记录上一阶段的产物路径。`smoke_test` 和 `full_run` 仍然是各独立阶段的两种实验规模：前者用于快速连通性检查，后者用于正式规模实验。

## 1. 工作流概览

```text
traj_data/ 数据采集
    │
    │ dataset.npz
    ▼
prediction/ 模型训练、离线预测或在线更新
    │
    │ 完整 prediction artifact 目录
    ▼
control/ LQR、MPC 或 KILC 闭环实验
    │
    │ 完整 control result 目录
    ▼
visualization/ 静态图、报告图和 MuJoCo GIF
```

最成熟的主流程是：

```text
受控数据采集
  → DKAC 训练及 one-step/rollout 评估
  → DKAC-MPC 闭环跟踪
  → 控制结果绘图和 MuJoCo 动画
```

阶段之间应传递明确路径，不应通过“查找最新目录”的方式隐式选择产物：

```text
DATASET       = traj_data/outputs/full_run/<data_run_id>/dataset.npz
MODEL_DIR     = prediction/outputs/full_run/<method>/<prediction_run_id>/
CONTROL_RESULT= control/outputs/full_run/<controller>/<control_run_id>/
MEDIA_DIR     = visualization/outputs/full_run/media/<media_run_id>/
```

完整命令和参数示例见 [FIVE_FOLDER_RUN_GUIDE.md](FIVE_FOLDER_RUN_GUIDE.md)。

## 2. 仓库结构

```text
DK_CDSM/
├── AGENTS.md                    Agent 工作规则、环境和实验规范
├── README.md                    仓库总览（本文档）
├── FIVE_FOLDER_RUN_GUIDE.md     各阶段独立运行及手工衔接指南
├── requirements.txt             锁定的 Python 依赖
├── pyproject.toml               项目元数据、打包与 pytest 配置
│
├── traj_data/                   MuJoCo 数据采集阶段
│   ├── collect_data_controlled.py
│   ├── collect_data_uncontrolled.py
│   ├── mujoco_cdsm.py
│   ├── references.py
│   ├── data_io.py
│   ├── assets/
│   ├── outputs/
│   └── PROGRAMS.md
│
├── prediction/                  Koopman 预测和模型辨识阶段
│   ├── edmd_prediction.py
│   ├── dkuc_prediction.py
│   ├── dkac_prediction.py
│   ├── dkn_prediction.py
│   ├── dktv_prediction.py
│   ├── otvdkl_prediction.py
│   ├── common.py
│   ├── dataset_selections.json
│   └── outputs/
│
├── control/                     闭环控制阶段
│   ├── lqr_control.py
│   ├── mpc_control.py
│   ├── kilc_control.py
│   ├── model_artifacts.py
│   ├── model_selections.json
│   ├── references.py
│   ├── cable_interface.py
│   ├── plotting.py
│   ├── io_utils.py
│   └── outputs/
│
├── visualization/               绘图、报告和 MuJoCo 动画阶段
│   ├── entrypoints/
│   ├── mujoco/
│   ├── plots/
│   ├── reports/
│   ├── legacy/
│   ├── path_setup.py
│   ├── outputs/
│   └── README.md
│
├── common/                      跨阶段共享运行代码和静态资源
│   ├── prediction_utils.py
│   ├── model_artifacts.py
│   ├── artifacts.py
│   ├── references.py
│   ├── cable_interface.py
│   ├── control_metrics.py
│   ├── control_plotting.py
│   ├── kilc_model.py
│   ├── io_utils.py
│   ├── assets/
│   ├── packages/
│   ├── README.md
│   └── DEPENDENCY_AUDIT.md
│
├── tests/                       自动化测试
│   └── time_varying/
│
├── docs/                        计划、公式映射、实验报告和评审材料
├── others/                      非主流程诊断和报告工具
└── legacy_system/               历史代码、资料和迁移证据
```

`outputs/` 目录只保存生成产物，不是应用源码目录。新实验产物不应提交到 Git；仓库中已经跟踪的历史产物属于已有研究证据，不应在未确认的情况下覆盖或删除。

## 3. `traj_data/`：数据采集

`traj_data/` 负责生成供预测模型训练和评估使用的 CDSM 轨迹数据。

### 3.1 可运行入口

- `collect_data_controlled.py`：通过 PD 控制跟踪随机平滑关节参考，生成受控训练数据。
- `collect_data_uncontrolled.py`：生成随机开环力矩数据或被动自由响应数据，支持 `random` 和 `passive` 模式。

两个入口共享以下实现：

- `mujoco_cdsm.py`：MuJoCo 状态读写、关节限位、绳索雅可比及力矩—张力分配。
- `references.py`：随机关节参考、软限位保护等轨迹工具。
- `data_io.py`：数据集校验、NPZ 和 JSON 保存。
- `assets/multi_joint_cable_driven_space_robot.xml`：数据采集使用的 MuJoCo 模型。

### 3.2 数据产物

推荐输出结构：

```text
traj_data/outputs/<run_type>/<data_run_id>/
├── dataset.npz
├── metadata.json
└── summary.json
```

`dataset.npz` 的主要字段为：

| 字段 | 典型形状 | 含义 |
| --- | --- | --- |
| `t` | `(steps,)` | 控制步时间，单位 s |
| `states` | `(traj, steps+1, 4)` | `[qa, qb, dqa, dqb]` |
| `inputs` | `(traj, steps, 2)` | `[tau_a, tau_b]`，单位 Nm |
| `q_ref` | `(traj, steps, 2)` | 参考关节角 |
| `dq_ref` | `(traj, steps, 2)` | 参考关节角速度 |
| `cable_ctrl` | `(traj, steps, 8)` | 8 根绳索的控制张力，单位 N |

预测训练主要读取 `states` 和 `inputs`。进入预测阶段前，应检查 `summary.json` 中的有限值、状态范围、峰值力矩和峰值绳索张力。

更详细的采集方法和参数见 [traj_data/PROGRAMS.md](traj_data/PROGRAMS.md)。

## 4. `prediction/`：模型辨识与预测

`prediction/` 中每种方法都有独立入口，负责训练或在线更新、one-step 评估、rollout 评估及产物保存。

### 4.1 方法与用途

| 方法 | 入口 | 当前用途 |
| --- | --- | --- |
| EDMD | `edmd_prediction.py` | RBF-EDMD 训练与预测，可供 LQR 使用 |
| DKUC | `dkuc_prediction.py` | Deep Koopman with control，可供 LQR 和在线更新方法使用 |
| DKAC | `dkac_prediction.py` | Deep Koopman affine control，可供 LQR 和 MPC 使用 |
| DKN | `dkn_prediction.py` | Deep Koopman prediction，当前仅用于预测评估 |
| DKTV | `dktv_prediction.py` | Hao 等人的累积式在线更新，当前仅用于预测评估 |
| OTVDKL | `otvdkl_prediction.py` | Zhang 等人的滑动窗口在线更新，当前仅用于预测评估 |

DKTV 和 OTVDKL 是两个不同的方法，不能使用 DKTV 作为二者的统称。它们都复用冻结的 DKUC artifact 和编码器：

```text
冻结 DKUC artifact + 历史数据 + 时间有序 stream dataset
  ├── DKTV：累积统计更新
  └── OTVDKL/OTVDKL*：滑动窗口更新
```

当前仓库尚未提供直接消费 DKTV 或 OTVDKL 结果的稳定性保证控制器。

### 4.2 数据集选择

最稳妥的方式是显式传递：

```text
--train_dataset traj_data/outputs/<run_type>/<data_run_id>/dataset.npz
```

`dataset_selections.json` 提供可复用的数据集别名，可通过 `--dataset_key` 使用。正式实验应确认别名指向的文件确实是预期数据集。

### 4.3 普通预测 artifact

输出结构形如：

```text
prediction/outputs/<run_type>/<method>/<prediction_run_id>/
├── 模型权重或模型矩阵
├── model_config.json
├── normalizers.json
├── dataset_train.npz
├── dataset_val.npz
├── one_step_metrics.json
├── rollout_metrics.json
├── one_step_prediction_rollouts.npz
├── rollout_prediction_rollouts.npz
└── run_summary.json
```

展示图片单独保存在：

```text
prediction/outputs/<run_type>/figures/<prediction_run_id>/
```

不同模型的核心文件分别为：

- EDMD：`edmd_model.npz`
- DKUC：`best_dkuc.pt`
- DKAC：`best_dkac.pt`
- DKN：`best_dkn.pt`

下游控制阶段需要完整 artifact 目录，不能只复制一个权重文件。神经模型加载还依赖 `model_config.json` 和 `normalizers.json`。

### 4.4 在线时变预测产物

DKTV 通常输出：

```text
prediction_arrays.npz
final_dktv_state.npz
metrics.json
update_history.json
manifest.json
```

OTVDKL 通常输出：

```text
prediction_arrays.npz
metrics.json
otvdkl_update_history.json
otvdkl_star_update_history.json
final_otvdkl_state.npz
final_otvdkl_star_state.npz
manifest.json
```

## 5. `control/`：闭环控制

`control/` 读取固定预测模型 artifact，生成参考轨迹，执行 MuJoCo 闭环控制并保存数值证据。

### 5.1 控制器兼容关系

| 控制器 | 入口 | 可接受的模型 |
| --- | --- | --- |
| LQR | `lqr_control.py` | EDMD、DKUC、DKAC |
| MPC | `mpc_control.py` | DKAC |
| KILC | `kilc_control.py` | 专用 continuous-DKUC artifact |

DKN、DKTV 和 OTVDKL 当前不能直接作为现有控制入口的模型。

KILC 依赖专门的 continuous-DKUC artifact，不能假定普通 `dkuc_prediction.py` 生成的离散模型可以直接使用。

### 5.2 模型选择

临时实验推荐显式传递：

```text
--artifact_dir prediction/outputs/<run_type>/<method>/<prediction_run_id>
```

`model_selections.json` 用于给固定基准模型定义别名，控制入口可以通过 `--model_key` 读取。`model_artifacts.py` 负责识别并加载 EDMD、DKUC 和 DKAC artifact。

### 5.3 控制结果

LQR 和 MPC 的标准输出为：

```text
control/outputs/<run_type>/<controller>/<control_run_id>/
├── manifest.json
├── arrays/
│   ├── reference.npz
│   └── closed_loop_<model>.npz
├── metrics/
│   └── tracking_metrics.json
└── logs/
```

控制图保存在：

```text
control/outputs/<run_type>/figures/<control_run_id>/
```

`manifest.json` 记录入口、参数、Python 解释器、Git 分支和提交、模型来源及控制配置。`closed_loop_<model>.npz` 是后续重绘和动画回放的核心原始数组。

验收控制结果时，应检查关节与笛卡尔跟踪误差、关节限位、控制饱和、绳索张力、张力分配残差、非有限值以及优化器状态。

## 6. `visualization/`：结果展示

推荐入口为：

- `entrypoints/render_animation.py`：从单个控制结果生成 MuJoCo GIF。
- `entrypoints/render_combined_animation.py`：生成多模型合并对比动画。
- `plots/control_result_plots.py`：从闭环 NPZ 重新生成静态图。
- `reports/dkac_circle_tracking_figures.py`：生成 DKAC 圆轨迹报告图。

实际渲染实现位于 `mujoco/`，通用绘图位于 `plots/`。`path_setup.py` 负责加入共享包路径。新实验不应以 `visualization/legacy/` 中的历史脚本作为主入口。

单模型动画主要读取：

```text
control/outputs/<run_type>/<controller>/<control_run_id>/arrays/
  closed_loop_<model>.npz
```

推荐输出到：

```text
visualization/outputs/<run_type>/media/<media_run_id>/
├── *.gif
└── *_metadata_*.json
```

多模型合并动画要求结果目录中存在多个兼容的 `closed_loop_<model>.npz`，普通单模型控制运行不会自动形成这种比较目录。

可视化模块详情见 [visualization/README.md](visualization/README.md)。

## 7. `common/`：共享实现

`common/` 不是实验阶段入口，主要保存多个阶段共用的稳定接口和静态资源：

- `prediction_utils.py`：数据集加载、训练/验证切分、归一化、预测评估、输出目录和图像生成。
- `model_artifacts.py`：控制阶段的模型 artifact 解析和适配。
- `artifacts.py`：通用 artifact 读写辅助。
- `references.py`：共享参考轨迹和 IK 相关接口。
- `cable_interface.py`：等效关节力矩到绳索张力的接口。
- `control_metrics.py`：关节和笛卡尔控制指标。
- `control_plotting.py`：闭环控制结果绘图。
- `kilc_model.py`：continuous-DKUC/KILC 专用模型加载。
- `io_utils.py`：输出目录、manifest 和 JSON 辅助函数。
- `assets/`：各阶段共享的 MuJoCo XML 静态资源。
- `packages/`：`cable_robotics` 和 `cdsm` 共享运行包。

部分 `control/` 文件是共享实现的阶段内适配或兼容副本。新增公共能力前，应确认至少有两个独立工作流确实需要它，避免再次形成平行实现。

详细依赖关系见 [common/README.md](common/README.md) 和 [common/DEPENDENCY_AUDIT.md](common/DEPENDENCY_AUDIT.md)。

## 8. 其他目录

### `tests/`

自动化测试目录。目前主要覆盖 DKTV 和 OTVDKL 的时变更新核心逻辑：

```text
tests/time_varying/test_core_updates.py
```

### `docs/`

用于保存研究计划、公式映射、实验执行报告和评审材料。代码行为应以当前源码和测试为准，历史报告用于解释设计背景与实验证据。

### `others/`

保存不属于五个主阶段的诊断和报告工具：

- `diagnostics/`：绳索力矩映射、MuJoCo Jacobian、运动学规划和 PD 跟踪诊断。
- `reporting/`：已有实验结果的专项报告生成。
- `assets/`：这些工具使用的兼容静态资源。

这些程序可以独立运行，但不属于标准训练—控制主链。详情见 [others/README.md](others/README.md)。

### `legacy_system/`

历史程序、旧目录结构、文献资料和迁移证据。该目录用于复现和追溯，不用于开发新功能。需要保留的能力应整理到当前五个主目录，而不是继续扩展历史实现。

## 9. Python 环境

项目要求使用 Conda 环境 `env_dk_cdsm`。

### Linux

```text
PYTHON_EXE=/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

Linux 主机可能从 ROS 2 继承不兼容的 `PYTHONPATH`，所有项目 Python 命令都应先清除它：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -c "import platform, sys; print(sys.executable); print(platform.system())"
```

### Windows

```text
PYTHON_EXE=D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe
```

验证解释器：

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' -c "import sys; print(sys.executable)"
```

不得使用裸 `python`、`pip` 或 `pytest`，也不得静默切换到其他环境。Linux 依赖安装方式见 `AGENTS.md`；锁定依赖记录在 `requirements.txt`。

## 10. 快速开始

以下示例展示 Linux 下的一套手工 DKAC-MPC 工作流。运行每一步后，将占位路径替换为终端实际输出的目录。

### 10.1 验证环境

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -c "import platform, sys; print(sys.executable); print(platform.system())"
```

### 10.2 采集受控数据

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  traj_data/collect_data_controlled.py \
  --out_dir traj_data/outputs/full_run \
  --tag example
```

### 10.3 训练并评估 DKAC

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  prediction/dkac_prediction.py \
  --train_dataset traj_data/outputs/full_run/<data_run_id>/dataset.npz \
  --run_type full_run \
  --pred_mode both \
  --device cuda \
  --tag example
```

### 10.4 执行 DKAC-MPC 控制

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  control/mpc_control.py \
  --artifact_dir prediction/outputs/full_run/dkac/<prediction_run_id> \
  --run_type full_run \
  --device cuda \
  --trajectory star \
  --period 20 \
  --num_cycles 1 \
  --start_hold 0 \
  --radius 0.45 \
  --inner_radius_ratio 0.382 \
  --tag example
```

### 10.5 渲染 MuJoCo GIF

在支持 EGL 的无界面 Linux 主机上：

```bash
env -u PYTHONPATH MUJOCO_GL=egl \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  visualization/entrypoints/render_animation.py \
  --result_dir control/outputs/full_run/mpc/<control_run_id> \
  --models dkac \
  --trajectory star \
  --out_dir visualization/outputs/full_run/media/<media_run_id> \
  --tag example
```

## 11. 测试与检查

运行全部测试：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python -m pytest
```

常用的低成本检查包括：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  prediction/dkac_prediction.py --help

env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  control/mpc_control.py --help
```

非交互式 Matplotlib 运行应设置 `MPLBACKEND=Agg`。如果 Matplotlib 配置目录不可写，可将 `MPLCONFIGDIR` 指向 `/tmp` 下的任务专用目录。无界面 MuJoCo 渲染优先使用 `MUJOCO_GL=egl`。

## 12. 产物与复现规范

各阶段统一使用局部输出根目录：

```text
traj_data/outputs/
prediction/outputs/
control/outputs/
visualization/outputs/
```

一次有意义的实验至少应保存并核对：

- 入口脚本和完整命令行参数；
- 随机种子；
- 源数据集和过滤规则；
- 模型 artifact 路径；
- Python 解释器和环境；
- Git 分支和提交哈希；
- one-step 与 rollout 预测指标；
- 闭环关节和笛卡尔跟踪指标；
- 可重绘图像的原始 NPZ 数组；
- 图像、动画及其输出路径。

数值证据和展示产物应分开保存。不要仅凭命令返回码判断实验成功，也不要只保留 PNG、PDF 或 GIF 而丢弃原始数组与指标。

## 13. 阅读顺序

新参与者建议按以下顺序了解仓库：

1. 本文档：理解目录职责、方法关系和产物衔接。
2. [FIVE_FOLDER_RUN_GUIDE.md](FIVE_FOLDER_RUN_GUIDE.md)：查看各阶段完整命令和参数示例。
3. [traj_data/PROGRAMS.md](traj_data/PROGRAMS.md)：理解数据生成和字段定义。
4. 目标预测方法入口，例如 `prediction/dkac_prediction.py`。
5. 目标控制入口，例如 `control/mpc_control.py`。
6. [visualization/README.md](visualization/README.md)：理解图像和动画生成方式。
7. [common/README.md](common/README.md)：理解共享运行依赖。
8. `tests/`：查看当前自动化验证覆盖的行为。

开始修改前还应阅读 [AGENTS.md](AGENTS.md)，其中包含环境、目录所有权、测试、Git 和实验复现的强制规则。
