# 旧系统归档后的依赖检查记录

检查时间：2026-07-06

## 归档动作

已将以下旧系统目录从仓库根目录移动到 `legacy_system/source_tree/`：

```text
archive/
assets/
configs/
docs/
experiments/
src/
tests/
```

旧根目录备注 `remark.md` 已移动到：

```text
legacy_system/notes/remark.md
```

根目录 `outputs/` 未移动，原因是它是生成结果目录，已由 `.gitignore` 忽略。移动到 `legacy_system/` 下反而可能让大量历史结果变成未跟踪文件。

## 新主线保留目录

```text
traj_data/
prediction/
control/
common/
visualization/
others/
```

## 依赖扫描结论

对新主线目录执行了旧路径引用扫描：

```text
traj_data/
prediction/
control/
common/
visualization/
others/
```

扫描重点：

```text
experiments
archive
configs
assets/models
src
koopman_control
cable_robotics
cdsm
```

结论：

- 旧路径 `experiments/archive/configs/assets/models/src` 的命中主要在 README 或审计文档中，用作来源说明，不是运行依赖。
- `cdsm` 和 `cable_robotics` 的 Python import 仍然存在，但它们来自 `common/packages/` 的精简运行包，不依赖根目录旧 `src/`。
- 未发现新主线入口需要根目录旧 `src/`、`experiments/`、`archive/`、`configs/`、`assets/models/` 才能完成 `--help` 解析。

## 已完成入口检查

使用解释器：

```text
D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe
```

已通过：

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\traj_data\collect_data_controlled.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\prediction\dkac_prediction.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\control\mpc_control.py --help
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\visualization\entrypoints\render_animation.py --help
powershell.exe -NoProfile -Command "[void][scriptblock]::Create((Get-Content -LiteralPath '.\run_interactive_fullflow.ps1' -Raw)); Write-Host 'PowerShell syntax OK'"
```

同时 `git diff --check` 通过。

## 未完成项

本次未跑完整 smoke/full 流程，只做了入口级检查。后续正式提交归档前，建议至少用 `run_interactive_fullflow.bat` 跑一次 `smoke_test`，验证：

```text
数据采集 -> 预测训练 -> 控制 -> 可视化
```

全部能在旧目录归档后完成。
