# legacy_system 归档清单

归档时间：2026-07-06

## 已移动到 `legacy_system/source_tree/`

| 原根目录路径 | 当前路径 | 说明 |
| --- | --- | --- |
| `src/` | `legacy_system/source_tree/src/` | 旧 reusable implementation，包括 `koopman_control`、`cdsm`、`cable_robotics`。 |
| `experiments/` | `legacy_system/source_tree/experiments/` | 旧 CLI 入口和实验组合层。 |
| `configs/` | `legacy_system/source_tree/configs/` | 旧模型、部署、项目配置。 |
| `assets/` | `legacy_system/source_tree/assets/` | 旧 MuJoCo XML、文献 PDF/TXT 等静态资源。 |
| `docs/` | `legacy_system/source_tree/docs/` | 旧研究文档和审计文档。 |
| `tests/` | `legacy_system/source_tree/tests/` | 旧单元/集成测试。 |
| `archive/` | `legacy_system/source_tree/archive/` | 旧历史程序和兼容脚本。 |
| `remark.md` | `legacy_system/notes/remark.md` | 旧根目录备注。 |

## 根目录保留

| 路径 | 说明 |
| --- | --- |
| `traj_data/` | 当前数据采集主线。 |
| `prediction/` | 当前 Koopman 预测主线。 |
| `control/` | 当前控制主线。 |
| `common/` | 当前公共模块和精简运行包。 |
| `visualization/` | 当前可视化主线。 |
| `others/` | 当前实用小工具和诊断脚本。 |
| `outputs/` | 历史和本地生成结果，仍由 `.gitignore` 忽略。 |
| `FIVE_FOLDER_RUN_GUIDE.md` | 当前五目录运行指南。 |
| `AGENTS.md` | 当前仓库工作规则。 |
| `requirements.txt`、`pyproject.toml`、`.gitignore` | 当前环境/项目元数据。 |

## 未移动的本地/工具目录

| 路径 | 说明 |
| --- | --- |
| `.git/` | Git 仓库元数据。 |
| `.agents/`、`.codex/` | 本地 agent/codex 配置。 |
| `.vscode/`、`.pytest_cache/` | 本地工具缓存或 IDE 状态，已由 `.gitignore` 忽略。 |
