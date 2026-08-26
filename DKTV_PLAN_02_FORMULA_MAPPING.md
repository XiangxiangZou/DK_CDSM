# DKTV Plan 02 公式—数组映射

## 文献依据

- Hao, Huang, Pan, Wu, Mou, “Deep Koopman Learning of Nonlinear Time-Varying
  Systems”, Automatica 159 (2024) 111372，DOI：
  `10.1016/j.automatica.2023.111372`，公开稿：
  `https://arxiv.org/abs/2210.06272`。
- Zhang, Ren, Duan, Sun, Chen, “Deep Koopman iterative learning and
  stability-guaranteed control for unknown nonlinear time-varying systems”,
  Automatica 190 (2026) 113054，DOI：
  `10.1016/j.automatica.2026.113054`。

Hao 的 DKTV 在线批次不断加入历史数据，不移除旧快照。其原始算法会在批次到达
后递推更新 lifted matrices，并继续求解 DNN 参数。Zhang 将它作为累积基线，区别
是 Zhang 的 OTVDKL 明确移除过时快照。

本仓库按 `DKTV_PLAN_02_HAO_ACCUMULATIVE.md` 的公平比较约束，冻结 Plan 01 的
encoder、normalizer 和 exact readout，只更新 `A/B`。这是对 Hao 累积矩阵更新
机制的受控工程复现，不宣称逐项复现其在线 DNN 优化。报告和图例采用
`Hao-style accumulative DKTV (fixed encoder)`，`dktv_accumulative` 仅作为代码标识。

## 坐标与数组

| 文献符号 | 本仓库数组/对象 | 形状 | 坐标 |
| --- | --- | --- | --- |
| `x_k` | `states[:, k]` | `(N, 4)` | physical state |
| `u_k` | `inputs[:, k]` / `applied_torque` | `(N, 2)` | physical torque |
| `g(x_k)` | `DKUCModel.lift(x_k)` | `(N, 16)` | normalized latent |
| 模型输入 | `u_normalizer.transform(u_k)` | `(N, 2)` | normalized torque |
| `A_tau` | updater `A` | `(16, 16)` | latent-to-latent |
| `B_tau` | updater `B` | `(16, 2)` | normalized input-to-latent |
| `C0` | `initial_model.npz/C0` | `(4, 16)` | latent-to-normalized-state |

固定公式：

```text
x_norm = (x_phys - x_mean) / x_std
u_norm = (u_phys - u_mean) / u_std
z_k = [x_norm, phi(x_norm), 1]
z_{k+1} = A_tau z_k + B_tau u_norm,k
x_phys,k+1 = (C0 z_{k+1}) * x_std + x_mean
```

## Ridge direct refit 与充分统计量

对每个快照定义行向量：

```text
r_k = [z_k, u_norm,k]              # dimension p = 16 + 2 = 18
y_k = z_{k+1}                      # dimension r = 16
```

堆叠为 `R in R^(M x 18)`、`Y in R^(M x 16)`。全历史 ridge oracle 为：

```text
Theta = solve(R.T @ R + lambda * I, R.T @ Y)
A = Theta[:16].T
B = Theta[16:].T
```

累积 updater 不保留或删除历史 raw snapshot，而只维护：

```text
Gram_tau  = sum(r_k.T @ r_k)
Cross_tau = sum(r_k.T @ y_k)
count_tau = 历史样本数
```

新批次到达时只做加法：

```text
Gram_{tau+1}  = Gram_tau  + R_new.T @ R_new
Cross_{tau+1} = Cross_tau + R_new.T @ Y_new
```

随后使用同一 ridge 解线性系统。它与重新拼接全部历史数据的 direct refit 数学
等价；每次更新都保存二者的 `A/B` 最大绝对差作为数值验收证据。

由于 affine lift 最后一维恒为 1，解出矩阵后统一施加：

```text
A[-1, :] = 0
A[-1, -1] = 1
B[-1, :] = 0
```

这与 Plan 01 初始模型的 affine constraint 一致。

## 因果评价顺序

对时间步 `k`：

1. 使用更新前的当前 `A_tau/B_tau` 预测所有轨迹的 `x_{k+1}`；
2. 观测真实 `x_{k+1}` 后，把 `(x_k,u_k,x_{k+1})` 放入 pending batch；
3. pending snapshot 数达到 `b` 时执行一次累积更新；
4. 新模型从下一个尚未预测的 snapshot 起生效。

因此预测指标不会使用当前目标状态提前更新模型。多步 rollout 在各窗口起点使用
当时可用的模型版本和共同的未来输入序列，用于离线评价模型质量。

## 并行轨迹 stream 与批大小

主实验的 validation stream 含 5 条同步仿真轨迹，并按
`time_major_then_trajectory` 排序：同一仿真时间步先产生 5 个 snapshot，再进入下
一个时间步。因此主实验中：

```text
b=5  -> 每 1 个仿真时间步更新一次
b=10 -> 每 2 个仿真时间步更新一次
b=20 -> 每 4 个仿真时间步更新一次
```

这是并行轨迹在线学习，不等同于单台真实机械臂每步只产生一个 snapshot。Plan 03
必须复用完全相同的轨迹数、排序和批语义；若后续增加 single-trajectory stream，
必须作为单独场景标记，不能与本结果直接混合聚合。

## 失败策略、审计与内存边界

数值异常批次采用 `discard_invalid_batch`：保留最近已接受模型和充分统计量，坏批次
不进入 recursive statistics，也不进入 direct-refit oracle history，不重试。每次尝试
的接受状态、原因和完整 recursive diagnostics 写入 `logs/update_history.jsonl`，数值
轨迹以 schema v2 写入 `arrays/update_history.npz`；两者均不需要 pickle。

部署 updater 只保存 `Gram/Cross`，其 `updater_statistics_memory_bytes` 为常数。验收
实验为了逐批验证等价性，另行保存不断增长的 raw oracle history，因此单独报告
`oracle_history_memory_bytes_initial/final`。recursive update 时间和 direct-refit oracle
时间也分别统计，不能把 oracle 的增长内存或耗时归入可部署方法。

## 当前实验覆盖边界

zero-noise、公共正弦扰动、中等变化速率和三阶段主预测实验已执行；low-noise、
slow/fast 变化速率消融尚未执行，MPC 仍为可选项。因此 Plan 02 的准确状态是“核心
实现及主预测实验完成，消融实验待补”。
