# DK_CDSM 项目总体架构与论文实验工作流

> 文档状态：架构提案 v0.2
>
> 更新日期：2026-08-24
>
> 本文描述建议采用的轻量目标架构。核心目标不是把仓库建设成通用软件平台，而是让数据采集、模型预测、闭环控制和结果可视化能够分别独立运行，并能公平、可复现地完成论文对比实验。

## 1. 项目定位

DK_CDSM 是面向绳驱空间机械臂的研究仓库，当前主要研究链路为：

```text
MuJoCo 数据采集
  -> EDMD / DKUC / DKAC / DKN 模型训练与预测评估
  -> LQR / MPC / KILC 闭环控制
  -> 指标、论文图和 MuJoCo 动画
```

论文实验需要重点回答三类问题：

1. 在相同数据集和评估协议下，不同预测模型的能力有什么差异？
2. 在相同模型、plant 和参考任务下，不同控制方法的能力有什么差异？
3. 不同模型与控制方法组合后，预测性能和闭环性能之间有什么关系？

因此，本仓库的架构优先级是：

```text
公平对比 > 实验复现 > 独立运行 > 代码复用 > 平台化扩展
```

## 2. 总体结论

当前的五目录主线方向是正确的，不需要整体迁移成复杂的多包或多项目架构：

```text
traj_data/       数据采集
prediction/      模型训练与预测比较
control/         控制实验与控制器比较
common/          共享的 CDSM、绳索和 artifact 基础
visualization/   只读结果并生成图表和动画
```

建议保留这些顶层目录，并进行以下改进：

- 保证每个阶段有自己的入口、配置、测试和输出；
- 保证每种预测方法和控制方法可以单独运行；
- 增加统一的预测模型比较入口和控制器比较入口；
- 固定数据划分、模型 artifact、场景和评价协议；
- 统一单次实验与对比实验的输出格式；
- 清理 `common/` 与业务目录之间的重复代码；
- 将可视化从训练和控制循环中分离；
- 用明确的 run id 和 manifest 连接论文图与原始实验；
- 保留 `hardware/`、`paper/` 和 `legacy_system/` 为独立支线。

本文确认采用后，应同步更新根目录 `AGENTS.md`、`pyproject.toml`、`.gitignore` 和运行指南，使执行规则与五目录架构一致；在这些文件正式更新前，实际操作仍遵守当前 `AGENTS.md`。

## 3. “独立运行”的定义

本项目需要三层独立性。

### 3.1 阶段独立

各阶段可以从仓库根目录单独启动，不要求先运行一键全流程：

```powershell
# 数据采集
& $PY .\traj_data\collect_data_controlled.py ...

# 单模型训练与预测
& $PY .\prediction\dkac_prediction.py --train_dataset <dataset> ...

# 单控制器实验
& $PY .\control\mpc_control.py --artifact_dir <model-dir> ...

# 已有结果可视化
& $PY .\visualization\entrypoints\render_animation.py --result_dir <control-run> ...
```

这里的“独立”是指阶段能够显式读取上游 artifact 后运行，不表示每个目录必须完全复制一套 MuJoCo、IK、绳索分配和保存工具。共享稳定能力仍由 `common/` 提供。

### 3.2 方法独立

每种方法保留单独入口：

```text
prediction/edmd_prediction.py
prediction/dkuc_prediction.py
prediction/dkac_prediction.py
prediction/dkn_prediction.py

control/lqr_control.py
control/mpc_control.py
control/kilc_control.py
```

独立入口必须能完成一次方法自身的训练、评估或控制实验，不依赖比较脚本才能工作。

### 3.3 比较独立

比较脚本负责组织公平实验，但不重新实现模型或控制器：

```text
prediction/compare_models.py
control/compare_controllers.py
control/compare_model_controller_matrix.py       # 可选，论文需要时增加
```

比较脚本应调用各方法已有 API 或独立入口，并统一传入数据、划分、场景和评估协议。

## 4. 建议的轻量目录结构

```text
DK_CDSM/
├── AGENTS.md
├── README.md
├── PROJECT_ARCHITECTURE.md
├── FIVE_FOLDER_RUN_GUIDE.md
├── requirements.txt
├── pyproject.toml
├── run_interactive_fullflow.bat
├── run_interactive_fullflow.ps1
│
├── common/                              # 非实验阶段；唯一共享基础
│   ├── README.md
│   ├── contracts.py                     # 数据、模型、控制结果 schema
│   ├── artifacts.py                     # manifest、JSON、路径和哈希
│   ├── control_metrics.py
│   ├── model_artifacts.py
│   ├── packages/
│   │   ├── cable_robotics/
│   │   └── cdsm/
│   └── assets/
│       └── multi_joint_cable_driven_space_robot.xml
│
├── traj_data/                           # 阶段 1：数据采集
│   ├── README.md
│   ├── configs/
│   │   ├── controlled.yaml
│   │   ├── uncontrolled_random.yaml
│   │   ├── uncontrolled_passive.yaml
│   │   └── validation.yaml
│   ├── collect_data_controlled.py
│   ├── collect_data_uncontrolled.py
│   ├── validate_dataset.py
│   ├── data_io.py
│   ├── mujoco_cdsm.py
│   ├── references.py
│   ├── tests/
│   └── outputs/
│       ├── smoke_test/
│       ├── full_run/
│       ├── datasets/                    # 验收通过的固定数据集
│       └── rejected/
│
├── prediction/                          # 阶段 2：预测模型
│   ├── README.md
│   ├── configs/
│   │   ├── methods/
│   │   │   ├── edmd.yaml
│   │   │   ├── dkuc.yaml
│   │   │   ├── dkac.yaml
│   │   │   └── dkn.yaml
│   │   ├── evaluation.yaml
│   │   └── comparisons/
│   │       └── default.yaml
│   ├── edmd_prediction.py
│   ├── dkuc_prediction.py
│   ├── dkac_prediction.py
│   ├── dkn_prediction.py
│   ├── compare_models.py
│   ├── data.py
│   ├── training.py
│   ├── evaluation.py
│   ├── plotting.py
│   ├── tests/
│   └── outputs/
│       ├── smoke_test/<method>/<run-id>/
│       ├── full_run/<method>/<run-id>/
│       └── comparisons/<comparison-id>/
│
├── control/                             # 阶段 3：闭环控制
│   ├── README.md
│   ├── configs/
│   │   ├── controllers/
│   │   │   ├── lqr.yaml
│   │   │   ├── mpc.yaml
│   │   │   └── kilc.yaml
│   │   ├── scenarios/
│   │   │   ├── joint.yaml
│   │   │   ├── circle.yaml
│   │   │   └── star.yaml
│   │   └── comparisons/
│   │       └── default.yaml
│   ├── lqr_control.py
│   ├── mpc_control.py
│   ├── kilc_control.py
│   ├── compare_controllers.py
│   ├── compare_model_controller_matrix.py
│   ├── model_artifacts.py
│   ├── cable_interface.py
│   ├── references.py
│   ├── tests/
│   └── outputs/
│       ├── smoke_test/<controller>/<run-id>/
│       ├── full_run/<controller>/<run-id>/
│       └── comparisons/<comparison-id>/
│
├── visualization/                       # 阶段 4：只读结果
│   ├── README.md
│   ├── entrypoints/
│   ├── prediction_comparison/
│   ├── control_comparison/
│   ├── paper_figures/
│   ├── animations/
│   ├── tests/
│   └── outputs/
│       ├── figures/<figure-set-id>/
│       └── media/<media-run-id>/
│
├── hardware/                            # 独立硬件实验支线
│   ├── README.md
│   ├── configs/
│   ├── common/
│   ├── scripts/
│   ├── tests/
│   └── outputs/
│
├── paper/                               # 论文、图索引和实验表
│   ├── experiments/
│   ├── figures/
│   └── manuscript/
│
├── others/                              # 迁移期保留，最终按职责清空
└── legacy_system/                       # 冻结历史，不作为新入口
```

目录只在有实际内容时创建。`compare_model_controller_matrix.py`、各阶段 `tests/` 等可以按迁移顺序逐步增加，不需要一次性建立所有空文件。

## 5. 各目录职责

| 目录 | 负责 | 不负责 |
| --- | --- | --- |
| `common/` | 稳定合同、CDSM plant、IK、绳索分配、artifact 基础 | 某个模型训练流程、某个控制器实验、论文图编排 |
| `traj_data/` | 采集、质量检查、数据集发布 | 模型训练和控制 |
| `prediction/` | 模型训练、one-step/rollout 评估、模型比较 | 闭环控制结论 |
| `control/` | 控制器运行、闭环指标、控制器和组合比较 | 重新训练预测模型 |
| `visualization/` | 只读已有 artifact，生成图、动画和论文展示产品 | 修改原始数组、训练或控制计算 |
| `hardware/` | XL330 通信、标定、低风险测试和硬件结果 | 论文模型训练和 MuJoCo 算法副本 |
| `paper/` | 实验索引、表格、LaTeX 和最终选图 | 算法源码和唯一实验数据 |
| `legacy_system/` | 历史追溯 | 新功能开发 |

## 6. `common/` 的边界与去重原则

`common/` 是共享基础，不是第六个实验阶段。它只保存两个及以上阶段真正共享、且职责稳定的内容。

建议保留：

- `common/packages/cdsm/`：plant、IK、关节/绳索合同、参考轨迹；
- `common/packages/cable_robotics/`：张力分配、安全和通用接口；
- `common/contracts.py`：dataset、model artifact、control result schema；
- `common/artifacts.py`：manifest、路径、哈希和 JSON 保存；
- `common/control_metrics.py`：统一闭环指标；
- `common/model_artifacts.py`：模型 artifact 的只读加载接口；
- `common/assets/`：唯一权威 MuJoCo XML。

不建议保留双份实现：

- `common/prediction_utils.py` 与 `prediction/common.py` 应合并后归 `prediction/` 内部；
- `common/control_plotting.py` 与 `control/plotting.py` 应只保留一份，长期建议由 `visualization/` 消费结果；
- `common/cable_interface.py` 与 `control/cable_interface.py` 应选择一个权威实现；
- `common/io_utils.py` 与 `control/io_utils.py` 应把通用部分放 `common`，控制专用部分留在 `control`；
- `common/references.py` 与 `control/references.py` 应按通用参考和控制场景明确归属。

去重时先增加测试，再切换 import，最后删除无人使用的副本，避免一次性改变所有入口。

## 7. 数据采集与固定数据集

### 7.1 单次采集输出

```text
traj_data/outputs/<run_type>/<run-id>/
├── manifest.json
├── resolved_config.yaml
├── dataset.npz
├── metadata.json
├── summary.json
└── logs/
```

`manifest.json` 至少记录：

- 采集入口和完整参数；
- seed、dt、轨迹数量和步数；
- MuJoCo XML 相对路径及 SHA-256；
- state、input、cable 顺序和单位；
- Python executable；
- Git branch、commit 和 dirty 状态；
- 数据检查结果和输出文件哈希。

### 7.2 数据验收

不能只检查数组是否有限。正式数据集至少检查：

- NaN/Inf；
- 状态和速度范围；
- 关节物理限位与安全限位违规；
- 等效力矩峰值和分布；
- 绳索张力最小值、最大值和分位数；
- 张力上下界违规次数；
- 张力分配残差；
- 饱和比例；
- 每条异常轨迹的编号和原因。

当前历史数据中出现过约 `226 kN` 的绳张力峰值。这类数据即使全部有限，也必须进入 `rejected/`，或者在明确的物理依据和实验说明下单独发布，不能直接作为所有模型的公平比较基准。

### 7.3 固定数据集

验收通过后发布为不可覆盖的数据集：

```text
traj_data/outputs/datasets/<dataset-id>/<version>/
├── dataset.npz
├── dataset_manifest.json
├── split_manifest.json
└── validation_report.json
```

所有预测方法通过同一个 `dataset_manifest.json` 读取数据。正式论文比较不直接读取某次临时采集 run。

## 8. 预测模型实验

### 8.1 单模型运行

每个模型脚本保持独立运行，并保存完整 artifact：

```text
prediction/outputs/<run_type>/<method>/<run-id>/
├── manifest.json
├── resolved_config.yaml
├── model_config.json
├── model.pt或model.npz
├── normalizers.json或npz
├── split_manifest.json
├── metrics/
│   ├── one_step.json
│   └── rollout.json
├── arrays/
│   ├── one_step_rollouts.npz
│   └── rollout_rollouts.npz
└── logs/
```

控制阶段读取整个模型目录或 `manifest.json`，不读取孤立的 `.pt` 文件。

### 8.2 公平预测比较协议

同一比较中的所有方法必须固定：

- 同一个 dataset id 和 version；
- 同一个 train/validation/test split；
- 同一组 rollout episode 和起始索引；
- 相同预测 horizon；
- 相同状态顺序和单位；
- 相同 one-step 与 rollout 指标实现；
- 相同 seed 集合，或明确记录各方法的 seed；
- 归一化只能拟合训练集，不能读取 validation/test 统计量。

网络结构、特征映射和训练策略可以不同，因为这些正是模型能力的一部分，但必须完整记录。

### 8.3 预测比较入口

建议入口：

```powershell
& $PY .\prediction\compare_models.py `
  --dataset_manifest <dataset_manifest.json> `
  --methods edmd,dkuc,dkac,dkn `
  --config .\prediction\configs\comparisons\default.yaml `
  --tag paper_prediction_v1
```

比较脚本不应复制训练算法，而应依次调用各模型已有训练/评估 API，并把运行目录汇总到：

```text
prediction/outputs/comparisons/<comparison-id>/
├── comparison_manifest.json
├── resolved_config.yaml
├── method_runs.json
├── metrics.json
├── metrics.csv
├── arrays/
└── figures/
```

`metrics.csv` 建议至少包含：

| method | parameters | train_time_s | one_step_rmse | rollout_rmse | rollout_horizon | seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

同时保存各状态的 RMSE、不同 horizon 的误差曲线和多 seed 的均值/标准差，避免论文只展示一个总 RMSE。

## 9. 控制实验

### 9.1 单控制器运行

每个控制器继续拥有独立入口：

```powershell
& $PY .\control\lqr_control.py --artifact_dir <model-run> ...
& $PY .\control\mpc_control.py --artifact_dir <model-run> ...
& $PY .\control\kilc_control.py --artifact_dir <model-run> ...
```

单次输出：

```text
control/outputs/<run_type>/<controller>/<run-id>/
├── manifest.json
├── resolved_config.yaml
├── metrics/tracking_metrics.json
├── diagnostics.json
├── arrays/
│   ├── reference.npz
│   └── closed_loop.npz
├── logs/
└── error.json                       # 失败或提前终止时存在
```

### 9.2 公平控制器比较协议

比较相同模型下不同控制器时必须固定：

- 同一个模型 artifact 和 normalizer；
- 同一个 MuJoCo XML 及参数；
- 同一个初始状态；
- 同一个参考轨迹数组，而不是分别重新生成；
- 相同 dt、实验时长和随机 seed；
- 相同关节、力矩和绳张力约束；
- 相同张力分配器；
- 相同提前终止条件；
- 相同指标实现。

控制器的 horizon、Q、R、solver 和内部状态属于方法自身参数，可以不同，但必须保存 resolved config。

### 9.3 模型 capability

不是所有控制器都能公平消费同一个模型：

- LQR 可消费满足 lifted-linear 接口的 EDMD、DKUC 或 DKAC artifact；
- 当前 MPC 主要消费 DKAC artifact；
- KILC 需要 continuous-DKUC artifact；
- DKN 当前保持 prediction-only。

比较脚本必须显式检查 capability。不可用组合应记录为 `unsupported` 并说明原因，不能静默更换模型、自动跳过或把不同 artifact 伪装成“相同模型比较”。

### 9.4 控制器比较入口

建议入口：

```powershell
& $PY .\control\compare_controllers.py `
  --artifact_dir <fixed-model-run> `
  --controllers lqr,mpc `
  --scenario_config .\control\configs\scenarios\circle.yaml `
  --comparison_config .\control\configs\comparisons\default.yaml `
  --tag paper_control_v1
```

输出：

```text
control/outputs/comparisons/<comparison-id>/
├── comparison_manifest.json
├── resolved_config.yaml
├── controller_runs.json
├── metrics.json
├── metrics.csv
├── arrays/
└── figures/
```

`metrics.csv` 建议至少包含：

| controller | model | rmse_q | rmse_dq | rmse_ee | max_ee_error | peak_tau | peak_tension | violations | mean_solve_ms | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## 10. 论文实验矩阵

论文工作流建议明确区分三类实验。

### 10.1 固定数据集，比较预测模型

```text
dataset D1 + split S1 + evaluation E1
  ├── EDMD
  ├── DKUC
  ├── DKAC
  └── DKN
```

结论范围：模型拟合、one-step、rollout、参数量和训练成本。不能直接据此宣称控制性能更好。

### 10.2 固定模型，比较控制器

```text
model M1 + plant P1 + scenario R1 + constraints C1
  ├── LQR
  └── MPC
```

只比较共同支持该模型 capability 的控制器。KILC 若使用不同模型合同，应作为另一组实验，不能混入“固定模型”的表格。

### 10.3 模型与控制器组合比较

```text
                    LQR        MPC        KILC
EDMD                 run    unsupported  unsupported
DKUC                 run    capability?  separate continuous artifact
DKAC                 run       run       unsupported
DKN              prediction-only
```

实际 capability 以实现和 artifact schema 为准。组合矩阵用于研究“预测指标是否能解释闭环指标”，但每个单元格必须明确使用的 artifact、配置和限制。

## 11. 配置组织

每个阶段拥有自己的配置，避免建立复杂的全局配置系统：

```text
traj_data/configs/...
prediction/configs/...
control/configs/...
hardware/configs/...
```

配置原则：

- 只保存稳定、可审查的实验参数；
- 不保存开发机器绝对路径；
- 输入数据和模型由 manifest 路径显式指定；
- CLI 可以覆盖配置，但覆盖值必须写入 resolved config；
- smoke-test 和 full-run 可以使用不同配置；
- 方法参数、评价协议和比较矩阵分开；
- 场景配置生成一次参考数组，比较中的所有控制器共享该数组。

当前 `prediction/dataset_selections.json` 和 `control/model_selections.json` 可以在迁移期继续使用，但正式论文实验优先通过显式 manifest 路径选择输入，不依赖硬编码的默认 run id。

## 12. Manifest 与可复现性

单次运行和对比运行都必须保存 manifest，至少包含：

- run id、阶段、方法和状态；
- 完整命令和 resolved config；
- seed；
- Python executable 和关键包版本；
- Git branch、commit、dirty 状态；
- 上游 dataset/model/control manifest 的相对路径与 SHA-256；
- state、input、joint、cable 顺序与单位；
- 数值指标、数组和展示产品路径；
- 失败、提前终止、饱和和约束违规信息。

禁止：

- 只保存临时 Windows 绝对路径；
- 通过目录修改时间选择“最新模型”；
- 只保存图片、不保存绘图原始数组；
- 只打印指标、不写 JSON/CSV；
- 用 prediction-only 结果代替闭环控制证据。

## 13. 可视化与论文输出

可视化是只读阶段：

```text
dataset/model/control comparison artifact
                  |
                  v
             visualization
          /          |          \
         v           v           v
 automatic report  paper figure  MuJoCo animation
```

建议分为：

- `prediction_comparison/`：one-step、rollout、误差随 horizon、模型复杂度；
- `control_comparison/`：关节、末端、力矩、张力、求解时间和约束；
- `paper_figures/`：论文尺寸、字体、配色和多方法布局模板；
- `animations/`：单方法或多方法 MuJoCo 回放。

论文目录只保存：

- LaTeX 源码；
- 论文实验配置或比较 ID；
- 图表生成命令；
- 最终选用的展示文件或稳定引用。

建议建立：

```text
paper/experiments/experiment_index.yaml
```

示例字段：

```yaml
prediction_table_1:
  comparison_id: <prediction-comparison-id>
  generator: visualization/paper_figures/plot_prediction_comparison.py

control_figure_3:
  comparison_id: <control-comparison-id>
  generator: visualization/paper_figures/plot_control_comparison.py
```

这样论文中的每张图和每个表都能追溯到具体 comparison artifact。

## 14. 测试策略

测试跟随阶段放置，保持每一部分可单独验证：

### `traj_data/tests/`

- state/input/cable 字段和形状；
- seed 确定性；
- 关节限位和软保护；
- 张力分配与上下界；
- dataset validation 和 rejected 行为。

### `prediction/tests/`

- 固定 split；
- 归一化只拟合训练集；
- 各模型最小训练与加载；
- one-step/rollout 指标；
- comparison 聚合和 CSV schema。

### `control/tests/`

- artifact capability；
- LQR/MPC/KILC 的最小数学接口；
- 场景参考数组一致性；
- 约束和张力违规统计；
- 失败运行保存真实轨迹前缀。

### `visualization/tests/`

- 读取 prediction/control artifact；
- 生成最小 PNG；
- 渲染短 GIF 并检查代表帧；
- 不修改上游 artifact。

根目录只需要一个小规模端到端 smoke test，不需要建设复杂的统一测试平台。

## 15. 当前架构的具体改进项

### P0：比较公平性和数据可信度

1. 增加数据 validation 和固定 dataset manifest；
2. 固定预测比较的 split 与 evaluation protocol；
3. 固定控制比较的参考数组、初始状态和约束；
4. 禁止交互脚本静默改变模型或控制器选择；
5. 明确处理历史数据的异常绳张力。

### P1：论文比较入口

1. 实现 `prediction/compare_models.py`；
2. 实现 `control/compare_controllers.py`；
3. 统一 `metrics.csv/json`；
4. 增加论文图生成模板和 experiment index。

### P2：代码和配置整理

1. 拆分 900+ 行的 `prediction/common.py`；
2. 清理 `common/` 与 `control/` 的重复文件；
3. 将 argparse 中的重要默认值迁入阶段内配置；
4. 选定唯一权威 XML；
5. 修正根 `pyproject.toml` 中已经不存在的 `src/`、`tests/` 假定；
6. 为每个阶段补充 README 和测试。

### P3：Git 与历史产物

1. 保持所有新增 `outputs/` 不进入 Git；
2. 为已跟踪的 192 个历史输出文件建立索引；
3. 在用户确认和备份后再决定是否取消跟踪；
4. 冻结 `legacy_system/`；
5. 将 `others/` 中仍有价值的脚本归入 diagnostics 或 visualization。

## 16. 轻量 Git 工作流

不需要复制 `orca_ws` 的大量长期分支。建议保持简单：

- 当前稳定分支保存已验证代码；
- 一个任务使用一个短生命周期分支；
- 大型并行任务才使用 worktree；
- 正式论文实验从干净提交运行；
- 生成物不提交 Git。

分支示例：

```text
feat/prediction-comparison
feat/control-comparison
feat/dataset-validation
refactor/common-dedup
fix/control-constraints
docs/paper-experiment-index
```

提交示例：

```text
feat(prediction): add fixed-split model comparison
feat(control): compare LQR and MPC on shared reference
fix(data): reject cable tension outliers
refactor(common): remove duplicate artifact loaders
docs(paper): index prediction comparison runs
```

正式论文实验建议：

1. 先提交并验证代码；
2. 确认 `git status --short` 干净；
3. 使用固定 dataset/model/comparison config；
4. 运行实验并记录 commit；
5. 检查 metrics、arrays 和 figures；
6. 必要时创建 annotated tag；
7. 只提交代码、配置和论文索引，不提交大体量输出。

## 17. 分阶段实施顺序

### 阶段 1：先建立实验合同

- 定义 dataset、model、control 和 comparison manifest；
- 增加数据 validation；
- 固定 prediction split 和 control reference；
- 不移动现有入口。

完成条件：现有脚本仍可运行，新运行能生成完整 manifest。

### 阶段 2：增加论文比较能力

- 实现预测模型比较入口；
- 实现控制器比较入口；
- 生成统一 JSON、CSV 和原始数组；
- 增加基础对比图。

完成条件：能够分别完成“固定数据集比较模型”和“固定模型比较控制器”。

### 阶段 3：去重和配置化

- 清理 `common` 重复实现；
- 拆分大型公共文件；
- 将关键默认值迁入阶段内配置；
- 选定唯一 XML；
- 补充阶段级测试。

完成条件：每个职责只有一个权威实现，每种方法仍能单独运行。

### 阶段 4：整理展示与论文索引

- 把算法运行与绘图彻底分开；
- 建立论文图模板；
- 建立 `experiment_index.yaml`；
- 将现有 arXiv 内容按确认后的方式整理到 `paper/`。

完成条件：论文图和表可以从 comparison ID 一键重绘。

### 阶段 5：处理历史目录与 Git 产物

- 索引旧数据、模型和结果；
- 标记 verified、unverified 和 rejected；
- 归档有价值的 diagnostics；
- 用户确认后再处理已跟踪生成物。

完成条件：Git 只承载代码、配置、文档和小型必要静态资源。

## 18. 最终验收标准

整理完成后应满足：

- 任意一个预测模型能通过自己的脚本单独训练和评估；
- 任意一个控制器能通过自己的脚本读取兼容模型后单独运行；
- 可视化能在不重新运行算法的情况下读取已有结果；
- 相同数据集的模型比较共享完全相同的 split 和 evaluation protocol；
- 相同模型的控制器比较共享完全相同的 plant、参考和约束；
- 不支持的模型-控制器组合会明确报告，而不是静默替换；
- 每个结果都能追溯到数据、模型、配置、seed 和 Git commit；
- one-step、rollout 和 closed-loop 结论明确分开；
- 论文中的图表可以从 comparison artifact 重绘；
- 所有新增生成物保持在对应阶段的 `outputs/` 中且不进入 Git；
- `legacy_system/` 不再参与新流程。

这套架构保留了当前仓库最有价值的“各阶段、各方法可独立运行”特征，同时补上论文对比实验真正需要的公平协议、比较入口和结果追溯能力。
