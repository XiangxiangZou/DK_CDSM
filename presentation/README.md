# DK_CDSM Presentation Assets

`presentation/` 集中管理 PPT 和学术汇报使用的最终插图、动画、原始素材、
绘图脚本与提示词。训练、控制和可视化程序仍由仓库原有目录负责；本目录只保存
经过筛选的展示材料及其必要复现信息。

## 目录约定

- `figures/`：按汇报主题分类的最终静态图，优先保留可编辑的 PDF/SVG 和预览 PNG。
- `animations/`：MuJoCo 或 Koopman 动画。最终运动示意默认同时保存 GIF 和
  1920×1080 MP4；每个重要动画可使用独立运行目录，保留 `manifest.json`、
  指标 JSON、原始数组和渲染 metadata。
- `scripts/`：仅保存汇报专用的绘图或导出入口；公共算法继续复用 `common/`、
  `control/`、`prediction/` 和 `visualization/`。
- `assets/`：Logo、截图、图标等不由实验直接生成的素材。
- `prompts/`：可复用的绘图需求和视觉规范。
- `drafts/`：尚未定稿的临时版本，不得覆盖最终图。

## 当前动画

- `animations/mujoco/20260829_110539_cdsm_joint_sweep_circle/`：基于仓库
  MuJoCo XML 正运动学生成的 CDSM 关节反向扫动动画。`joint1` 从 −90° 运动到
  90°，`joint3` 从 90° 运动到 −90°；红色实线实时记录末端轨迹，无红色末端
  marker。该运行包同时保留 GIF 和 1080p MP4。

## 文件命名

最终材料使用能表达“系统/任务/方法/版本”的稳定名称。若产物来自具体实验运行，
保留原运行目录名和 sidecar 文件，避免图像与数值证据失去对应关系。
