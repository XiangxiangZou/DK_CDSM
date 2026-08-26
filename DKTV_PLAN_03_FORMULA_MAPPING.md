# DKTV Plan 03 公式与实现映射

本文件冻结本仓库对 Zhang 滑动窗口方法的工程解释。encoder、normalizer、精确
`C0` 读出和坐标契约沿用 Plan 01/02，只更新 `A/B`。

对窗口回归量 `R=[z,u_norm]` 和目标 `Y=z_next`，维护：

```text
G = R.T R
H = R.T Y
P = (G + lambda I)^-1
Theta = P H
```

每批先加入 `R_new/Y_new`，再删除最旧的 `R_out/Y_out`：

```text
P_add = P - P R_new.T (I + R_new P R_new.T)^-1 R_new P
P_new = P_add + P_add R_out.T (I - R_out P_add R_out.T)^-1 R_out P_add
H_new = H + R_new.T Y_new - R_out.T Y_out
```

`Theta_new=P_new H_new` 与候选窗口 ridge direct refit 逐批比较。低维加/删系统、
窗口秩、最小奇异值、条件数、recursive `A/B/theta`、矩阵差与有限性均检查；递推
超出容差时记录明确的 `direct_refit_fallback`，并从候选 raw window 重新计算
`G/H/P` 以消除累计统计漂移。窗口秩亏或病态则为 `failed_numerical` 并保留当前状态。

计时口径分离为：

```text
recursive_candidate_time_s   # 加/删递推与候选矩阵，不含 oracle
direct_refit_oracle_time_s   # direct refit 数值参考
fallback_time_s              # 仅在 fallback 时重建 G/H/P
total_update_time_s          # 完整判断路径
```

部署递推效率只能使用第一项；direct oracle 不属于部署路径。

选择性方法始终在同一个 `S_new` 上计算 current/candidate latent RMSE：current 不高于
`epsilon` 时 `skipped_threshold`；candidate 不优于 current 时 `rejected`。阈值跳过
仍推进数据窗口而不更新模型。拒绝策略冻结为：

- `discard_on_reject`：消费新批次，但窗口和模型均不变；
- `retain_on_reject`：候选数据进入窗口，但模型保持不变。

样本 ID、移入/移出边界、window/model version 与实际策略逐批保存，因而可验证没有
错位或重复计数。这里不实现论文的在线 encoder、SDP-MPC 或稳定性证明。

选择性阈值由固定 seed `20260824` 的独立 MuJoCo nominal calibration stream 标定，
不再使用当前评价 stream。噪声消融同时保存 `states_observed` 和 `states_clean`：在线
更新只读取 observed，主评价对 clean truth，同时另存 observed-truth 一致性指标。
