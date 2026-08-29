# Zhang-OTVDKL 在线学习与 MPC 控制实施计划

> 状态：待执行
>
> 建立日期：2026-08-28
>
> 目标论文：H. Zhang, Y. Ren, Z. Duan, Z. Sun, and G. Chen,
> “Deep Koopman iterative learning and stability-guaranteed control for
> unknown nonlinear time-varying systems,” *Automatica*, 190:113054, 2026.
>
> DOI：<https://doi.org/10.1016/j.automatica.2026.113054>
>
> 作者代码：<https://github.com/wulidede/Online-time-varying-deep-Koopman-learning>

## 1. 实施目标

本计划的最终目标是在当前绳驱机械臂（CDSM）仓库中完成 Zhang 等人的完整
OTVDKL 工作流：

1. 以 DKUC 的状态拼接型 lifting 网络作为初始 Deep Koopman 表示；
2. 通过固定宽度滑动窗口在线更新时变 lifted 模型；
3. 实现论文式可行性检查、误差触发和负更新拒绝机制；
4. 基于实时更新的 lifted 模型实现论文式 MPC；
5. 通过在线 SDP 计算终端权重和局部反馈参数；
6. 在 MuJoCo 绳驱机械臂上完成闭环关节及末端轨迹跟踪；
7. 保存预测、控制、安全性和稳定性诊断证据，支持后续论文对比。

本计划把“模型预测正确”“普通在线 MPC 能闭环运行”和“满足 Zhang 论文稳定性
条件”视为三个不同等级。只有完成最后一级的公式、数值与闭环验收，才允许使用
“stability-guaranteed OTVDKL-MPC”这一表述。

## 2. 论文方法边界

### 2.1 OTVDKL 在线模型

论文考虑未知离散时变非线性系统：

```text
x_(k+1) = f(x_k, u_k, k)
```

利用 lifting 函数构造近似时变线性模型：

```text
g(x_(k+1), theta_tau) ~= A_tau g(x_k, theta_tau) + B_tau u_k
x_(k+1)               ~= C_tau g(x_(k+1), theta_tau)
```

在每个更新时刻使用：

```text
w    固定窗口宽度
b    更新批次大小，且 w >= b
S_cur  当前窗口
S_new  尚未消费的新批次
S_out  当前窗口最旧的 b 个样本
```

与 Hao-DKTV 的主要区别是：OTVDKL 在加入 `S_new` 的同时显式丢弃
`S_out`，从而避免累积历史数据逐渐过时。

### 2.2 选择性更新

论文提出两个顺序不可交换的判断：

1. 先用当前模型计算 `S_new` 上的拟合误差；误差不超过阈值
   `epsilon` 时不构造候选模型；
2. 只有当前误差超过阈值时才构造候选模型；候选模型在 `S_new` 上的物理状态
   预测误差不大于当前模型时才提交更新。

判断必须使用论文式物理状态预测：

```text
x_next_hat = C [A B] [g(x); u]
```

不能用 latent residual 代替论文式物理状态误差而仍声称复现了式 (17)、(18)。

根据论文 Algorithm 1 和作者公开 Duffing 代码，阈值跳过或候选拒绝时不移动
`S_cur`；只有候选被接受时才执行“删除最旧批次、加入新批次”。这一语义将作为
本项目的默认论文模式。

### 2.3 Zhang MPC

论文控制器不是现有固定 DKAC-QP MPC 的简单换模版本。每个控制时刻需要：

1. 使用当前 `A_tau/B_tau` 求解论文式 SDP（式 (23)）；
2. 得到 `gamma`、终端权重 `P` 和局部反馈增益 `K`；
3. 使用当前 lifted state 求解有限时域 MPC（式 (21)）；
4. 将最优控制序列的第一个输入施加到原始非线性系统；
5. 由新测量形成因果转移样本，供后续 OTVDKL 更新。

论文证明的是闭环系统相对于 Koopman 近似误差的输入到状态稳定性（ISS），不是
无条件渐近稳定。该结论依赖 SDP/MPC 可行、终端集合、输入约束、模型误差裕度
以及 lifting/readout 结构等条件。

## 3. 当前仓库审计

| 内容 | 当前状态 | 主要问题 |
|---|---|---|
| DKUC 网络与 artifact 加载 | 已有 | 可以作为 OTVDKL 初始模型 |
| 状态拼接 lifting `g(x)=[x_norm, phi(x)]` | 已有 | 符合论文 Corollary 2 所需结构基础 |
| 滑窗 `A/B` 加减递推 | 已有 | 需要继续对齐论文式状态空间记号和更新事务 |
| Woodbury 低维条件数诊断 | 已有 | 需要增加论文 Corollary 1 的明确验收字段 |
| 直接窗口 refit oracle | 已有 | 当前 ridge 正则属于工程增强，需与论文公式分开记录 |
| `C_tau` 及其递推状态 | 未完成 | 当前 OTVDKL 只更新和保存 `A/B` |
| 式 (17)、(18) | 部分完成 | 当前使用 latent RMSE；阈值跳过时还会推进窗口 |
| 在线 `theta` 契约 | 部分完成 | 当前冻结 encoder，但 manifest 尚未解释论文控制实验依据 |
| one-step/rollout 预测产物 | 基础已有 | 需补 `C`、选择事件和完整长时域证据 |
| OTVDKL 控制模型接口 | 未完成 | `control/model_artifacts.py` 不支持在线模型状态 |
| Zhang SDP 式 (23) | 未完成 | 现有 MPC 没有 SDP、终端椭球和 LMI 诊断 |
| Zhang MPC 式 (21) | 未完成 | 现有 `mpc_control.py` 只支持固定 DKAC |
| 在线更新与 MuJoCo 闭环组合 | 未完成 | 尚无 `control/otvdkl_control.py` |
| ISS 数值证据 | 未完成 | 尚未保存 LMI、椭球、误差裕度和 Lyapunov 诊断 |

作者当前公开代码只包含简单 NTVS 和 Duffing 的预测示例，没有公开论文机械臂
控制实现。因此控制部分必须以论文公式为主，并通过本仓库自己的公式级测试和
数值 oracle 验证，不能仅以“与作者代码运行相似”作为验收依据。

## 4. 仓库架构

保持当前五目录研究架构，不增加 `src/`、`experiments/` 或 `configs/`：

```text
prediction/
  dkuc_prediction.py               复用网络、归一化与初始 artifact
  otvdkl_prediction.py             完善 Algorithm 1 和独立预测入口

control/
  otvdkl_control.py                新增 Algorithm 2 独立闭环入口
  cable_interface.py               复用绳张力分配和可行关节力矩边界
  io_utils.py                      复用控制输出与 manifest
  references.py                    复用参考轨迹和 IK
  plotting.py                      复用闭环结果绘图

tests/time_varying/
  test_core_updates.py             保留基础递推 oracle
  test_otvdkl_algorithm.py         公式、选择机制和 checkpoint 测试
  test_otvdkl_control.py           SDP、MPC 和轻量闭环测试

docs/
  OTVDKL_MPC_IMPLEMENTATION_PLAN.md
```

架构原则：

- OTVDKL 更新状态只在 `prediction/otvdkl_prediction.py` 实现一次；控制入口直接
  复用该 API，不复制另一套滑窗算法。
- `control/otvdkl_control.py` 独立于 `control/mpc_control.py`。现有文件继续表示
  固定 DKAC-MPC，不能通过大量条件分支把两种控制方法混在一起。
- 不把在线变化的 OTVDKL 快照伪装成普通固定 prediction artifact 交给
  `load_prediction_control_model()`。
- Zhang 专用 SDP、终端集合和 MPC 实现在规模可控时先保留于
  `otvdkl_control.py`；只有出现第二个真实消费者后再提取共享模块。
- 数据采集和预测入口保持独立；闭环控制过程中产生的在线转移只保存在
  `control/outputs/`，不反写历史训练数据。

## 5. 必须先固定的数学与坐标契约

### 5.1 状态、输入和 lifting

CDSM 当前使用：

```text
x = [q1, q2, dq1, dq2]
u = [tau1, tau2]
```

模型内部使用训练 artifact 的归一化坐标：

```text
x_norm = (x - mean_x) / std_x
u_norm = (u - mean_u) / std_u
z      = g(x) = [x_norm, phi(x_norm)]
```

必须验证 `z` 的前四维始终等于 `x_norm`。结构化 readout 固定为：

```text
C_struct = [I_4, 0]
```

这正是论文由 lifted-system ISS 推到原系统 ISS 的关键条件。若实现式 (12) 的
递推 `C_tau`，必须同时保存它与 `C_struct` 的偏差；控制和稳定性结论不得在两种
readout 之间无记录切换。

第一版控制实现采用 `C_struct` 作为物理状态 readout。式 (12) 的递推结果作为
论文公式诊断与选择机制候选进行保存。如果后续决定让控制器消费学习得到的
`C_tau`，必须重新审查 Corollary 2 的适用性。

### 5.2 在线 encoder 策略

论文 Algorithm 1 的文字/伪代码与公开 Duffing 代码允许在线更新 `theta`，但论文
机械臂控制实验明确说明在线阶段固定 DNN 参数；同时 Proposition 1 的递推统计
要求历史窗口处于同一 latent 坐标系。

因此 CDSM 的第一版 Zhang 控制复现采用：

```text
offline：训练/加载 theta_0
online ：冻结 theta_0，只更新 A_tau/B_tau/C_tau
```

这与论文机械臂控制实验一致，也避免 encoder 改变导致旧窗口统计失效。在线更新
`theta` 只能作为后续扩展：更新后必须对完整 `S_cur` 重新 lifting 和 refit，且不能
再宣称具有 Proposition 1 的 `O(b r^2)` 递推开销。

### 5.3 正则化与论文条件

论文公式基于满行秩和未正则化逆矩阵；当前实现使用 ridge 和直接 refit fallback
增强数值稳定性。实现时需要同时区分：

```text
paper_exact       满秩条件成立时验证式 (10)-(13)
regularized_safe  CDSM 默认数值模式，明确记录 lambda 和 fallback
```

正则化模式可以用于实验，但不能把正则化 Woodbury 更新直接描述为论文定理的
逐式等价实现。manifest 必须记录模式、`ridge_lambda`、rank、condition number、
fallback 原因及与直接 refit 的最大矩阵偏差。

### 5.4 跟踪误差坐标

论文式 (21)写成原点稳定问题，而机械臂实验是轨迹跟踪。实现前必须把跟踪问题
明确写成误差系统：

```text
e_z(k) = z(k) - z_ref(k)
v(k)   = u(k) - u_ref(k)
```

第一版允许 `u_ref=0`，但必须在 manifest 中记录。参考状态应由同一 encoder lifting，
代价函数至少对 lifted state 的前四个物理状态分量施加正权重。若 lifted 其余维度
使用零权重，应明确这是论文机械臂表格中的实验选择，但不满足“Q 正定”的理论
文字条件；稳定性验收应另用小正数正则化后的正定 `Q`。

### 5.5 绳驱约束与 SDP 输入边界

论文 SDP 使用对称输入约束 `|u_i| <= u_max,i`。CDSM 的可行关节力矩范围由当前
姿态和绳张力边界决定，通常是时变且不对称的。

每个控制时刻应：

1. 计算物理力矩上下界；
2. 转换到模型的 normalized input 坐标；
3. 验证零输入位于区间内；
4. 使用上下界绝对值的较小者构造保守对称 `u_max` 供 SDP 使用；
5. 在 MPC QP 中继续施加完整的非对称上下界；
6. 执行后再次检查绳张力、分配残差和实际施加力矩。

若无法构造非空对称边界，当前时刻不得声称满足论文输入约束条件，应进入明确的
安全降级状态，而不是继续使用上一次不可验证的 SDP 结果。

## 6. 分阶段实施

### 阶段 0：建立论文公式映射和基线快照

目标：在改代码前固定术语、坐标、基线和验收口径。

- [ ] 在计划执行反馈中列出论文式 (6)、(7)、(10)-(13)、(17)、(18)、
  (21)-(23) 对应的代码对象。
- [ ] 保存当前 OTVDKL smoke 的预测结果作为整改前基线；若当前没有可用 run，
  先运行最小时间有序 replay。
- [ ] 确认选用的 DKUC artifact 的状态维数、输入维数、latent 维数、归一化器、
  `include_constant` 和 `C_struct`。
- [ ] 确认初始窗口只来自历史数据或闭环开始前已获得的数据，不使用控制运行的
  未来状态。
- [ ] 固定第一轮默认模式：冻结 encoder、选择性 OTVDKL、regularized-safe。

验收条件：公式—代码映射无歧义；所有维数和坐标转换均有断言；不存在未来数据
泄漏。

### 阶段 1：完成论文对齐的 OTVDKL 预测核心

目标：先让 Algorithm 1 在离线时间有序 stream 上独立可信，再接控制。

- [ ] 扩展 `SlidingWindowKoopmanUpdater`，保存 `A/B/C`、两个逆统计量以及式
  (10)-(13) 所需状态。
- [ ] 对 `A/B` 和 `C` 分别实现 add/remove 递推，并保存低维矩阵可逆性与条件数。
- [ ] 逐次与当前窗口直接 refit 比较 `A/B/C`，形成公式级 oracle。
- [ ] 把选择条件改为物理状态预测误差，而不是 latent RMSE。
- [ ] 严格实现“阈值跳过不移动窗口、候选拒绝不移动窗口、接受才移动窗口”。
- [ ] 区分 `otvdkl`（无选择机制的每批滑窗更新）和 `otvdkl_star`（式 (17)、
  (18)）；最终 Zhang Algorithm 2 默认使用 `otvdkl_star`。
- [ ] 预测必须发生在吸收当前真实转移之前，继续保持 one-step 因果顺序。
- [ ] 保存 checkpoint：encoder 指纹、`A/B/C`、递推统计、窗口原始样本、样本 ID、
  pending buffer、模型/窗口版本和选择历史。
- [ ] 支持 checkpoint 恢复后继续消费 stream，并与连续运行结果比较。

阶段 1 测试：

- [ ] 式 (10)-(13) 与直接 refit 在指定容差内一致；
- [ ] 人工构造阈值跳过、候选接受、候选拒绝和数值失败四种路径；
- [ ] 每种路径检查模型版本、窗口版本、样本 ID 和 buffer；
- [ ] Causality 测试确认修改未来观测不会改变此前预测；
- [ ] 连续运行与 checkpoint 恢复的最终状态一致；
- [ ] 全部保存数组可重算 one-step 和 rollout 指标。

验收条件：Algorithm 1 独立通过单元测试和时间有序 smoke；不依赖控制代码即可
生成完整 prediction artifact。

### 阶段 2：定义在线控制模型事务

目标：把“测量—更新—求解—执行—记录”的顺序固定成可测试 API。

每个控制步采用以下因果顺序：

```text
读取 x_k
  -> 用上一控制步得到的 (x_(k-1), u_(k-1), x_k) 填充 S_new
  -> 满 b 个样本时执行 OTVDKL propose/select/commit
  -> 使用已提交的当前 A_tau/B_tau 求解 SDP
  -> 求解 MPC 得到 u_k
  -> 通过绳张力分配施加 u_k
  -> MuJoCo 前进一步得到 x_(k+1)
  -> 保存本步完整事务
```

论文 Algorithm 2 的排版容易被理解为“模型更新步不执行控制”。CDSM 实现不能
跳过物理控制周期，因此采用上述同周期先更新、后控制的因果解释，并在文档和
manifest 中明确记录。

- [ ] 定义只读 `OTVDKLModelSnapshot`，包含版本一致的 `A/B/C/theta` 与坐标元数据。
- [ ] 模型更新采用事务提交；失败或拒绝时控制器只能看到上一完整快照。
- [ ] 规定更新耗时跨越采样周期时的策略：第一版同步运行并记录 deadline miss，
  不虚构实时性；后续再考虑异步双缓冲。
- [ ] 规定 SDP 或 MPC 不可行时的安全降级：保持可行预载、限幅零/PD 安全输入，
  标记失败并终止或受控降级，禁止静默沿用非法控制。

验收条件：用确定性假 plant 验证整个控制步没有未来状态泄漏、没有半更新矩阵，
失败状态可重现。

### 阶段 3：实现论文式 SDP 终端设计

目标：逐式实现论文式 (22)、(23)，输出可验证的 `P/K/gamma`。

- [ ] 在 `control/otvdkl_control.py` 中实现 Zhang SDP 数据类和求解器。
- [ ] 使用仓库已有 `cvxpy`，优先验证 CLARABEL；必要时以 SCS 作为显式 fallback。
- [ ] 决策变量至少包括 `gamma`、`P_bar`、`Y`；求解后恢复：

  ```text
  P = gamma * inv(P_bar)
  K = Y * inv(P_bar)
  ```

- [ ] 每步保存 solver status、迭代/耗时、`gamma`、`P/K`、正定性和三组 LMI 的
  最小特征值裕度。
- [ ] 检查 `P_bar`、`P` 对称正定，恢复后的 `K` 满足输入边界。
- [ ] 检查当前 lifted error 是否位于椭球 `e_z^T P e_z <= gamma`。
- [ ] 在固定模型时允许 warm start，但第一轮不能因为缓存而跳过论文要求的可行性
  检查。
- [ ] 明确求解容差；只有 LMI 数值残差也通过，才能把 `optimal_inaccurate` 视为可用。

阶段 3 测试：

- [ ] 对稳定可控低维线性系统构造已知可行案例；
- [ ] 构造不可控、输入边界过小和当前状态超椭球的不可行案例；
- [ ] 从保存结果独立重算所有 LMI 和正定性裕度；
- [ ] 验证 `K` 下的候选一步状态保持于终端集合，并满足输入约束。

验收条件：SDP 不能只返回“solver success”；全部恢复矩阵与 LMI residual 必须可从
保存数组独立重算。

### 阶段 4：实现 Zhang lifted MPC

目标：用当前时变模型、在线 SDP 终端权重和 CDSM 输入边界形成闭环控制器。

- [ ] 实现式 (21) 的有限时域优化，决策变量为 normalized physical torque sequence。
- [ ] 每个预测步使用同一当前模型快照 `A_tau/B_tau`；窗口内不预知未来模型更新。
- [ ] 代价包含 lifted tracking error、控制输入和终端项；记录 `Q/R/P/H`。
- [ ] MPC QP 优先复用 OSQP，但不得复用 DKAC 的 state-dependent internal-control
  语义；OTVDKL/DKUC 的 `B` 直接对应 normalized physical torque。
- [ ] MPC 施加完整时变非对称力矩约束；SDP 使用第 5.5 节的保守对称约束。
- [ ] 施加第一项控制后重新计算实际 normalized input，并把实际执行值而不是未执行
  的优化值写入下一条转移。
- [ ] 保存预测状态序列、参考序列、控制序列、目标值、solver status、迭代数和耗时。

阶段 4 测试：

- [ ] 固定线性模型上的零参考稳定测试；
- [ ] 非零参考误差坐标和 `u_ref` 处理测试；
- [ ] 输入上下界、终端代价和首控制量与独立 QP oracle 一致；
- [ ] 模型版本变化后 MPC 矩阵同步刷新，且不复用旧维数或旧终端权重；
- [ ] SDP/MPC 失败时安全策略能够确定性触发。

验收条件：先在假 plant 上稳定运行，再进入 MuJoCo；现有 DKAC-MPC 行为和测试不受
影响。

### 阶段 5：接入 CDSM MuJoCo 闭环

目标：新增 `control/otvdkl_control.py` 并完成最小安全闭环。

- [ ] 复用现有 XML、plant、IK、参考轨迹、绳张力分配和控制指标。
- [ ] 从 DKUC artifact 和历史初始窗口构造 OTVDKL 初始状态。
- [ ] 第一轮使用关节空间低幅正弦参考，随后再运行圆形/星形末端轨迹。
- [ ] 先运行无额外时变扰动，验证控制链路；再加入与预测实验一致的可记录时变扰动。
- [ ] 扰动力矩只作为 plant 外部扰动和诊断字段，不进入模型控制输入 `u_k`。
- [ ] 先完成短时 smoke，再完成 30 s 完整运行。
- [ ] 每步检查有限值、关节限位、力矩限幅、绳张力上下界和分配残差。

推荐递进顺序：

```text
1. 1-2 s，固定 DKUC + Zhang MPC，低幅关节参考
2. 1-2 s，OTVDKL* + Zhang MPC，无附加扰动
3. 5 s，OTVDKL* + Zhang MPC，低幅时变扰动
4. 30 s，完整关节轨迹跟踪
5. 30 s，圆形/星形末端轨迹跟踪
```

进入下一档前，前一档必须满足安全验收条件。

### 阶段 6：稳定性诊断与论文级对比

目标：将“运行成功”升级为可支持论文结论的证据。

同一初始 artifact、初态、参考、扰动、随机种子和安全边界下至少比较：

```text
fixed_dkuc_mpc       固定 DKO/DKUC + 同一 Zhang MPC
dktv_mpc             Hao 累积更新 + 同一 Zhang MPC（后续比较项）
otvdkl_mpc           每批无条件滑窗更新 + Zhang MPC
otvdkl_star_mpc      式 (17)、(18) 选择性更新 + Zhang MPC
```

现有 `dkac_mpc` 可作为工程参考，但其模型结构、内部控制维度和控制矩阵不同，不应
直接作为 Zhang 论文同模型消融项。

- [ ] 至少使用 5 个 seed；论文级最终结果建议 10 个 seed。
- [ ] 同时报告 one-step、固定快照 rollout 和 closed-loop tracking，不用预测指标
  替代控制结论。
- [ ] 单因素扫描 `w`、`b`、`epsilon` 和 MPC horizon；保持其他参数不变。
- [ ] 报告更新次数、拒绝次数、跳过次数、数值失败次数和计算耗时。
- [ ] 报告 SDP/MPC 可行率、deadline miss 和安全降级次数。
- [ ] 绘制均值与标准差带，不以单次最好 run 作为最终结论。

验收条件：所有对比均能由保存的 manifest 和 raw arrays 重现；结果中明确区分
探索性结论和统计性结论。

## 7. 稳定性声明的验收门槛

以下条件全部满足前，只能称为“OTVDKL 接入 MPC”或“论文式控制原型”：

- [ ] lifting 明确包含原始 normalized state，且 readout 契约不变；
- [ ] `Q/R/P` 满足实现所声称的正定条件；
- [ ] 每个已接受控制步的 SDP 和 MPC 均可行；
- [ ] 三组 LMI 数值残差在预定容差内；
- [ ] 当前状态与终端预测满足椭球条件；
- [ ] 输入满足论文式边界和 CDSM 实际绳张力边界；
- [ ] 保存并检查 Koopman residual：

  ```text
  epsilon_k = g(x_(k+1)) - A_tau g(x_k) - B_tau u_k
  ```

- [ ] 检查论文 Assumption 5 对应的数值裕度；
- [ ] 更新时刻把模型跳变项计入有效 residual；
- [ ] 保存最优值函数或其可重算量，检查 ISS Lyapunov 差分不等式；
- [ ] 无 NaN/Inf、关节越界、张力越界或未记录的控制降级。

即使这些数值条件全部通过，也应使用“在已测试运行和数值容差内满足论文条件”
的表述，而不是把有限次数仿真当成普适数学证明。

## 8. 结果与产物契约

### 8.1 预测阶段

```text
prediction/outputs/<run_type>/otvdkl/<run_id>/
  manifest.json
  metrics.json
  prediction_arrays.npz
  otvdkl_update_history.json
  final_otvdkl_state.npz
```

至少保存：

- `A/B/C` by step；
- model/window/encoder version；
- window sample IDs 和边界；
- 当前/候选物理状态误差；
- 式 (17)、(18) 判断及原因；
- rank、condition number、inverse/LMI residual；
- one-step 和 rollout 原始数组。

### 8.2 控制阶段

```text
control/outputs/<run_type>/otvdkl/<run_id>/
  manifest.json
  metrics/
    tracking_metrics.json
    update_metrics.json
    stability_metrics.json
    timing_metrics.json
  arrays/
    reference.npz
    closed_loop_otvdkl.npz
    model_snapshots.npz
    solver_diagnostics.npz
  logs/

control/outputs/<run_type>/figures/<run_id>/
```

至少保存以下控制证据：

- 测量/参考关节状态与末端位置；
- 优化控制、实际执行力矩和 normalized input；
- 绳张力、分配残差、动态力矩边界；
- `A/B/C/P/K/gamma` 与对应模型版本；
- MPC 预测轨迹、目标值、状态和输入约束裕度；
- SDP/MPC status、迭代数和分项耗时；
- Koopman residual、模型跳变 residual、椭球裕度和 LMI 最小特征值；
- 更新、跳过、拒绝、fallback 和 deadline miss 事件。

图像至少包含：

1. 关节角/角速度跟踪；
2. 末端轨迹跟踪；
3. 跟踪误差随时间变化；
4. 力矩与动态上下界；
5. 八根绳张力与边界；
6. OTVDKL 更新事件和 `A/B` 变化；
7. Koopman residual 与选择阈值；
8. SDP 椭球/LMI 裕度；
9. SDP、MPC 和模型更新耗时。

所有图必须能够从保存的 NPZ 数组重新生成，PNG/PDF 只是展示产物。

## 9. 关键指标

### 9.1 预测指标

- one-step state RMSE/MAE；
- 0.5 s、1 s、2 s、5 s、10 s rollout RMSE；
- 每状态 RMSE；
- 更新前后新批次物理状态误差；
- update acceptance/skip/rejection/failure rate；
- `A/B/C` 范数、谱半径和条件数；
- 单次及总更新耗时。

### 9.2 控制指标

- joint position/velocity RMSE、MAE、maximum error；
- Cartesian RMSE、MAE、maximum error；
- control effort 和 control increment；
- 最小/最大绳张力、越界次数；
- 最大分配残差；
- 关节限位和力矩饱和次数；
- MPC/SDP 可行率、平均/最大求解耗时；
- 控制周期 deadline miss 数量。

### 9.3 稳定性诊断

- `min_eig(P_bar)`、`min_eig(P)`；
- 每组 LMI 的最小特征值；
- `gamma - e_z^T P e_z`；
- 终端状态椭球裕度；
- `||epsilon_k||` 及更新跳变后的有效 residual；
- Assumption 5 数值裕度；
- 最优值函数差分及 ISS 上界 residual。

## 10. 风险与处理策略

### 10.1 SDP 无法满足 10 ms 控制周期

论文机械臂报告控制求解约 0.2 ms，但使用 ACADOS，且不代表本项目的 CVXPY SDP
也能达到同样速度。第一轮必须完整计时，禁止提前声称实时。

处理顺序：先正确同步实现；再利用 warm start 和模型未更新时的结构缓存；最后才
考虑异步求解。任何缓存/异步策略都必须保存模型版本并防止使用过期终端权重。

### 10.2 当前 DKUC artifact 不满足控制理论条件

可能表现为 `A/B` 不可控、SDP 不可行、`Q` 非正定或当前状态不在终端椭球内。
此时应回到数据覆盖、latent 维数、正则化和初始训练，而不是放松所有数值阈值。

### 10.3 滑动窗口 rank/conditioning 不足

必须满足至少 `w >= r + m` 的必要维数条件，并通过实际 rank/condition 检查。
增大 `w` 有利于秩和数值稳定，但可能引入更多过时数据；应通过 `w/b` 消融选择，
而不是只追求矩阵可逆。

### 10.4 论文符号和实验设置存在不一致

论文正文要求 `Q` 正定，但机械臂表格对部分 lifted 维度使用零权重；Algorithm 1
对 `theta` 的描述与机械臂在线冻结设置也不同。所有这类选择必须在 manifest 中
记录为“theory-aligned”或“paper-experiment-aligned”，不能混为同一套保证。

### 10.5 控制激励不足导致在线辨识退化

跟踪控制可能使输入和状态缺少持续激励，造成窗口 rank 下降。第一轮只做诊断，
不擅自加入 dither。若需要辨识激励，应单独评估其跟踪、安全和论文公平性影响。

## 11. 最终完成定义

只有同时满足以下条件，Zhang-OTVDKL-MPC 才视为实现完成：

- [ ] Algorithm 1 的 `A/B/C`、滑窗、低维可行性检查和选择机制通过公式级测试；
- [ ] 独立预测入口通过 one-step 和长时 rollout 验收；
- [ ] Algorithm 2 的在线事务顺序无未来数据泄漏；
- [ ] SDP 式 (23) 与 MPC 式 (21) 均有独立数值 oracle；
- [ ] `control/otvdkl_control.py` 可从仓库根目录独立运行并提供 `--help`；
- [ ] MuJoCo smoke 无非有限值、关节越界、张力越界和未处理 solver failure；
- [ ] 30 s 完整闭环结果保存于约定的 `control/outputs/full_run/otvdkl/`；
- [ ] 所有指标和图像可由 raw arrays 重算；
- [ ] 至少完成固定 DKUC-MPC 与 OTVDKL*-MPC 的同条件比较；
- [ ] 稳定性表述与实际通过的理论/数值条件严格一致；
- [ ] 执行反馈和后续 review 文档记录命令、参数、结果、失败与残余风险。

## 12. 推荐执行优先级

```text
第一优先级：修正 OTVDKL 的 C、式 (17)/(18) 和窗口语义
第二优先级：完成公式级 oracle、checkpoint 和预测 smoke
第三优先级：实现并验证 SDP 式 (23)
第四优先级：实现 lifted MPC 式 (21) 和假 plant 闭环
第五优先级：接入 CDSM，完成短时安全 smoke
第六优先级：完成 30 s 完整运行和多 seed 对比
```

在第一、二优先级未验收前，不应直接把当前 `A/B` updater 接入 MPC；否则即使
闭环能够运动，也无法判断结果来自 Zhang-OTVDKL、普通滑窗 refit，还是当前选择
语义的工程变体。
