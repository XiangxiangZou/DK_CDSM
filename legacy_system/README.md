# legacy_system 旧系统归档说明

本目录保存重构前的旧仓库系统。它不是当前推荐运行入口，只用于追溯历史实现、排查新系统问题、恢复旧实验逻辑或查阅旧论文/配置资料。

当前根目录主线为：

```text
traj_data/        数据采集
prediction/       Koopman 预测训练与评估
control/          LQR/MPC/KILC 控制
common/           公共模块和精简运行包
visualization/    绘图与 MuJoCo 动画
others/           实用小工具和诊断脚本
```

旧系统源码集中在：

```text
legacy_system/source_tree/
  archive/
  assets/
  configs/
  docs/
  experiments/
  src/
  tests/
```

旧的根目录 `remark.md` 已移动到：

```text
legacy_system/notes/remark.md
```

## 使用原则

- 新实验不从 `legacy_system/` 启动。
- 新流程的独立阶段入口和运行方式见根目录 `FIVE_FOLDER_RUN_GUIDE.md`。
- 如果新系统运行异常，可以到 `legacy_system/source_tree/` 对照旧实现。
- 如果需要恢复旧实验，先检查旧脚本中的相对路径；它们现在位于 `legacy_system/source_tree/` 下，直接运行可能需要临时调整工作目录或路径。
- 不要在 `legacy_system/` 里继续开发新功能；需要保留的新能力应整理到 `traj_data/`、`prediction/`、`control/`、`common/`、`visualization/` 或 `others/`。

## 旧结果目录

根目录 `outputs/` 没有移动。它包含历史生成结果，且被 `.gitignore` 忽略。为了避免把大量实验结果变成未跟踪文件，旧结果仍保留在根目录 `outputs/` 中。

如需以后进一步整理旧结果，建议新建单独的结果索引文档，而不是把大体量结果文件混入源码归档。
