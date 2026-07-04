# 五目录隔离全流程依赖审计

运行时间：2026-07-04 12:18-12:44

隔离副本：

```text
C:\Users\ZouXiangxiang\AppData\Local\Temp\DK_CDSM_fullflow_20260704_121823
```

## 隔离范围

隔离副本中只复制并运行以下五个目录：

- `visualization/`
- `traj_data/`
- `prediction/`
- `control/`
- `common/`

没有复制根目录下的 `src/`、`assets/`、`archive/`、`experiments/`、`configs/`、`docs/`、`tests/`。

## 运行链路

1. 受控数据采集：
   - `traj_data/collect_data_controlled.py --out_dir .\traj_data\outputs\full_run --tag fullflow_paper`
   - 输出：`traj_data/outputs/full_run/20260704_121925_controlled_pd_fullflow_paper/dataset.npz`
2. DKAC 训练与预测：
   - `prediction/dkac_prediction.py --train_dataset .\traj_data\outputs\full_run\20260704_121925_controlled_pd_fullflow_paper\dataset.npz --run_type full_run --pred_mode both --device cuda --tag fullflow_paper`
   - 输出：`prediction/outputs/full_run/dkac/20260704_121951_dkac_fullflow_paper/`
3. DKAC-MPC 五角星跟踪：
   - `control/mpc_control.py --run_type full_run --device cuda --artifact_dir .\prediction\outputs\full_run\dkac\20260704_121951_dkac_fullflow_paper --trajectory star --period 20 --num_cycles 1 --start_hold 0 --radius 0.45 --inner_radius_ratio 0.382 --tag fullflow_paper`
   - 输出：`control/outputs/full_run/mpc/20260704_124340_mpc_fullflow_paper/`
4. MuJoCo 动画：
   - `visualization/entrypoints/render_animation.py --result_dir .\control\outputs\full_run\mpc\20260704_124340_mpc_fullflow_paper --models dkac --trajectory star --actual_trail_color red --out_dir visualization\outputs\full_run\media\20260704_124340_mpc_dkac_star --tag fullflow_paper`
   - 输出：`visualization/outputs/full_run/media/20260704_124340_mpc_dkac_star/`

## 外部依赖结论

未发现需要从五个目标目录之外补入的 repo-local 文件。

本次隔离运行中唯一暴露的问题是可视化脚本默认只在结果目录根部查找 `closed_loop_dkac.npz`，而当前控制输出规范将该文件保存到 `arrays/closed_loop_dkac.npz`。该问题已经在 `visualization/mujoco/mujoco_animation.py` 中修复，不属于五目录之外遗漏依赖。

## 关键结果

- CUDA 设备：`NVIDIA GeForce RTX 3060 Laptop GPU`
- 数据集规模：`states=(40, 501, 4)`，`inputs=(40, 500, 2)`，`cable_ctrl=(40, 500, 8)`
- 采集张力范围：`min=20.0 N`，`max=226001.22994260959 N`
- DKAC 训练：`best_epoch=67`，`best_val=0.00565188005566597`
- DKAC one-step RMSE：`0.022044502906610183`
- DKAC rollout RMSE：`2.509591407506612`
- MPC joint RMSE：`0.0035003359690128116`
- MPC end-effector RMSE：`0.014993723648310758 m`
- MPC OSQP 状态：`solved=2000`
- MPC 张力范围：`20.0 N` 到 `156.87261997835262 N`
- GIF：`visualization/outputs/full_run/media/20260704_124340_mpc_dkac_star/20260704_124449_star_dkac_mujoco_motion_fullflow_paper.gif`
- GIF 帧数：`501`
- 动画样式：期望轨迹 `white dashed`，实际末端轨迹 `red solid`
