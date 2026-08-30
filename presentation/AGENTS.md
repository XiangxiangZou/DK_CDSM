# DK_CDSM Presentation Asset Rules

本目录专门用于 DK_CDSM 学术汇报和论文插图制作。

## General principles

1. 不修改 traj_data、prediction、control 中的科研算法逻辑，
   除非任务明确要求。

2. 绘图所需科研数据优先读取已有 outputs，
   不允许为了画图伪造实验结果。

3. 所有科研示意图应优先采用矢量格式：
   SVG / PDF。

4. 同时输出适合 PowerPoint 的 PNG：
   - transparent background when appropriate
   - 1920×1080 or sufficiently high resolution
   - minimum 300 dpi for publication figures

5. 学术插图总体风格：
   - white / transparent background
   - HIT blue as primary visual language
   - red only for emphasis / trajectory / error
   - avoid decorative gradients
   - avoid cartoon style
   - avoid unnecessary 3D effects

6. 数学符号使用 LaTeX conventions：
   x_k, u_k, z_k, ψ(x), K, A, B, C

7. 图中术语保持统一：
   Cable-Driven Space Manipulator (CDSM)
   Koopman operator
   Observable / lifting function
   Lifted state
   One-step prediction
   Rollout prediction
   Model Predictive Control

8. 每个绘图脚本应支持重复运行，
   不应依赖临时手工修改文件。

9. 最终图和脚本必须使用对应文件名，例如：

   scripts/koopman/draw_koopman_framework.py
   figures/koopman/koopman_framework.svg
   figures/koopman/koopman_framework.png

10. 不覆盖已有科研实验数据。
