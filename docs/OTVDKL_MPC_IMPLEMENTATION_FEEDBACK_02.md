# OTVDKL-MPC 实施反馈 02

> 执行日期：2026-08-28  
> 前序报告：`docs/OTVDKL_MPC_IMPLEMENTATION_FEEDBACK_01.md`  
> 结论：完成论文式 (23) 纠偏、因果控制事务和 MuJoCo 安全降级 smoke；高维在线 SDP 仍阻止有效 MPC 闭环。

## 1. 本轮推进结果

本轮优先处理反馈 01 中风险最高的三个问题：

1. 对照论文公开预印本逐式核对式 (21)-(23) 和 Algorithm 2；
2. 建立可测试的 `过去转移 -> OTVDKL 提交 -> 不可变快照 -> SDP -> MPC -> fallback` 事务；
3. 接入 MuJoCo、动态绳驱力矩边界、实际执行输入、Koopman residual 和阶段本地输出。

论文来源：

- Automatica DOI：<https://doi.org/10.1016/j.automatica.2026.113054>
- arXiv HTML：<https://arxiv.org/html/2601.21230>

## 2. SDP 公式纠偏

反馈 01 中的 SDP 原型并非论文式 (23) 的逐式实现。本轮已替换为论文四块 LMI：

```text
[ P_bar, (A P_bar + B Y)^T, (Q^1/2 P_bar)^T, (R^1/2 Y)^T ]
[ A P_bar + B Y, P_bar, 0, 0                              ] >= 0
[ Q^1/2 P_bar, 0, gamma I, 0                               ]
[ R^1/2 Y, 0, 0, gamma I                                   ]
```

并修正为：

- 当前状态约束 `[[1, g^T], [g, P_bar]] >= 0`；
- 输入约束 `[[Diag(u_max), Y], [Y^T, P_bar]] >= 0`；
- 恢复 `P = gamma inv(P_bar)`、`K = Y inv(P_bar)`；
- 独立重算式 (22) 的
  `P-(A+BK)^T P(A+BK)-Q-K^T R K` 最小特征值；
- 独立重算输入 LMI、状态椭球、`P/P_bar` 正定裕度。

这次纠偏是实质性修改。反馈 01 的低维 SDP 数值结果不再作为当前实现证据。

## 3. 因果控制事务

新增 `OnlineOTVDKLControlTransaction`：

```text
接收截至 x_k 的已完成过去转移
  -> pending 满 b 后 propose/select/commit
  -> 读取完整 OTVDKLModelSnapshot
  -> 构造保守对称 SDP 输入盒
  -> 求解论文式 SDP
  -> 使用同一 model_version 求解 MPC
  -> SDP/MPC/输入盒失败时执行明确 fallback
```

事务测试确认：

- 只有完整过去批次会在当前求解前提交；
- 未满批次不改变版本；
- snapshot、SDP/MPC 使用同一 model version；
- 候选失败不会泄漏半更新矩阵；
- 非法输入盒确定性触发 fallback；
- 每步记录 deadline miss。

## 4. MuJoCo 接入

`control/otvdkl_control.py` 现可从仓库根目录运行，接入：

- DKUC artifact、normalizer 和 frozen encoder；
- 历史窗口初始化；
- MuJoCo CDSM plant；
- 姿态相关绳驱力矩上下界；
- normalized input 与实际物理力矩转换；
- 八绳张力分配与 residual；
- 实际执行输入形成下一条因果转移；
- Koopman residual；
- `manifest.json`、四类 metrics JSON 和闭环 NPZ。

新增 `--max_online_sdp_dimension` 明确防止高维同步 SDP 无限阻塞控制周期。超过门槛时进入记录完整的物理零力矩 fallback，不静默复用旧 SDP。

## 5. 高维 SDP 探针

当前唯一 DKUC artifact：

```text
prediction/outputs/smoke_test/dkuc/20260703_105217_dkuc_prediction_split
```

其配置仅训练 `1 epoch / 1 step_per_epoch`，latent dimension 为 68，不是控制验收级模型。

第一次探针关闭维数防护，目标仅运行 2 步。第一步 SDP 在 60 秒内没有返回，远超 10 ms 控制周期，随后人工终止；没有生成有效控制产物。

结论：当前 68 维 CVXPY 同步 SDP 不具备在线可运行性。该结果不等同于论文方法不可实时，仅说明当前 artifact、维数、求解器和实现组合不满足本仓库实时要求。

## 6. MuJoCo 安全降级 smoke

命令核心参数：

```text
control/otvdkl_control.py
--duration 1.0 --dt 0.01 --horizon 3
--window_size 100 --batch_size 10 --epsilon 0.01
--reference_amplitude 0.001
--max_online_sdp_dimension 32
--run_type smoke_test --tag plan02_safe_guard_final
```

产物：

```text
control/outputs/smoke_test/otvdkl/
  20260828_180948_otvdkl_plan02_safe_guard_final/
```

关键结果：

| 指标 | 结果 |
|---|---:|
| 仿真步数 | 100 |
| degraded steps | 100 |
| deadline misses | 0 |
| maximum transaction time | 2.294 ms |
| joint position RMSE | 7.071e-4 rad |
| maximum allocation residual | 0 |
| tension range | 60 N – 60 N |
| tension violations | 0 |
| torque-bound violations | 0 |
| maximum Koopman residual norm | 1.008e-2 |
| finite values | true |

全部步均因 `latent_dim=68 > 32` 进入明确降级，因此上述跟踪 RMSE 只来自极低幅参考和静止机械臂，不能作为 MPC 跟踪结果。

## 7. 测试变化

`tests/time_varying/test_otvdkl_control.py` 新增：

- 按论文式 (23) 求解并独立重算 residual；
- 因果批次提交和 model version 一致性；
- 非法 SDP 输入盒 fallback；
- constrained lifted MPC。

## 8. 当前阶段完成度

| 阶段 | 当前状态 |
|---|---|
| 阶段 1：OTVDKL 预测核心 | 大部分完成，paper-exact 未正则路径待补 |
| 阶段 2：在线控制事务 | 核心事务完成，artifact 诊断字段仍可继续扩展 |
| 阶段 3：论文式 SDP | 公式纠偏并通过低维测试；68 维在线性能失败 |
| 阶段 4：lifted MPC | 低维测试通过；尚无非降级 MuJoCo 证据 |
| 阶段 5：MuJoCo | 安全降级链路完成；有效 MPC smoke 未完成 |
| 阶段 6：稳定性与对比 | 未完成 |

## 9. 下一步建议

下一轮不能直接进行 30 s full run。建议顺序：

1. 训练控制验收级 DKUC artifact，并将 latent dimension 降至可在线求解的消融范围，例如 8/12/16；
2. 对各 latent dimension 测量 SDP 可行率和真实耗时；
3. 仅在非降级 1–2 s smoke 通过后进入 5 s 扰动实验；
4. 增加终端预测椭球、Assumption 5、模型跳变 residual 和最优值函数差分；
5. 最后执行 30 s、多 seed 和 fixed DKUC 对照。

在出现至少一个 `degraded_steps=0`、SDP/MPC residual 全通过的 MuJoCo run 前，仍只能称为论文式控制原型。
