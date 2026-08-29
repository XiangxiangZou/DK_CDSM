# OTVDKL-MPC 实施审阅 01

> 审阅日期：2026-08-28  
> 对应计划：`docs/OTVDKL_MPC_IMPLEMENTATION_PLAN.md`  
> 对应反馈：`docs/OTVDKL_MPC_IMPLEMENTATION_FEEDBACK_01.md`  
> 审阅对象：Zhang 等（2026）OTVDKL/OTVDKL*、式 (21)—(23) 与当前仓库实现

## 1. 审阅结论

**可以继续推进代码整改，但暂不批准进入 MuJoCo 闭环、1—2 s 控制 smoke 或
30 s 完整运行。**

本轮实现已经建立了较好的工程骨架：OTVDKL/OTVDKL* 的滑窗状态、选择性更新
事务、预测 smoke、控制独立入口、低维 SDP/MPC 数值原型及基础测试均已出现。
全量测试也没有发现对现有仓库测试的回归。

但是，当前实现还存在四项会改变论文方法含义的一级阻断问题：

1. `solve_terminal_sdp()` 实现的不是 Zhang 论文式 (23)；
2. learned `C_tau` 参与选择误差，却没有用于保存的 one-step/rollout 预测，模型
   readout 契约不一致；
3. `C_tau` 的逆矩阵仍采用每次完整求逆，没有实现计划要求的式 (12)、(13)
   add/remove 递推；
4. 当前 68 维 lifted 模型上的 SDP 没有实时可用证据，现有原型一次求解超过
   90 s 仍未完成，远高于 10 ms 控制周期。

因此，本轮应判定为：

| 范围 | 结论 |
|---|---|
| 阶段 0：公式映射与基线 | 部分通过 |
| 阶段 1：OTVDKL 预测核心 | 有条件通过，仍需整改 |
| 阶段 2：在线控制事务 | 原型，不通过阶段验收 |
| 阶段 3：Zhang SDP | 公式不一致，不通过 |
| 阶段 4：lifted MPC | 低维原型，不通过阶段验收 |
| 阶段 5：MuJoCo 闭环 | 不放行 |
| 阶段 6：稳定性/论文对比 | 尚未开始 |

下一轮可以继续做“阶段 1—4 的公式与接口整改”，但在本审阅列出的 P0 条件通过
之前，不应运行物理闭环，也不得使用“stability-guaranteed OTVDKL-MPC 已实现”
这一表述。

## 2. 已核实的通过项

### 2.1 工程入口与回归测试

- `prediction/otvdkl_prediction.py` 保持独立预测入口；
- `control/otvdkl_control.py` 保持独立控制入口，没有混入现有 DKAC-MPC；
- 两个入口的 `--help` 均可正常执行；
- 控制入口 `--dry_run` 可完成配置检查，普通运行会明确停止，尚未伪装成已完成
  的物理闭环；
- 相关 10 个 focused tests 通过；
- 当前仓库全量测试为 `19 passed`，只有 2 条 OSQP API deprecation warning；
- 修改文件均通过 `py_compile`，`git diff --check` 无空白错误。

### 2.2 OTVDKL 更新事务

以下语义与当前计划一致：

- encoder 保持冻结；
- `A/B/C`、原始窗口、样本 ID、模型版本和窗口版本由一个 updater 维护；
- OTVDKL* 先检查当前模型误差，再决定是否构造候选；
- 阈值跳过、候选拒绝和数值失败均不移动窗口；
- 只有候选接受后才同时提交模型和窗口；
- checkpoint API 能保存 updater 的主要统计量和 pending 字段；
- A/B 的 Woodbury add/delete 候选会与当前窗口直接 refit 比较。

### 2.3 Prediction smoke 产物

已核查反馈中给出的目录：

```text
prediction/outputs/smoke_test/otvdkl/20260828_172840_otvdkl_plan01/
```

其中 30 个已保存数组均为有限值；主要数组形状包括：

```text
states                 (1, 201, 4)
inputs                 (1, 200, 2)
A_by_step              (200, 68, 68)
B_by_step              (200, 68, 2)
C_by_step              (200, 4, 68)
```

反馈报告中的 fixed DKUC、OTVDKL 和 OTVDKL* one-step RMSE 可从现有 metrics/arrays
中核对。该 smoke 能证明预测入口、更新循环和产物保存链路可运行；由于 OTVDKL
本次 RMSE 高于 fixed DKUC，它不能证明在线更新提升了预测性能，反馈报告对此表述
是谨慎且正确的。

## 3. P0 阻断问题

### P0-1：当前终端 SDP 不是论文式 (23)

`control/otvdkl_control.py:123`—`136` 当前构造的是：

- 二块 contraction LMI；
- current-state membership LMI；
- 按输入分量拆开的 scalar input LMI；
- 额外的 `trace(X) == 1`；
- 目标函数 `min gamma`。

Zhang 论文式 (23) 的第一组约束是包含以下对象的四块 LMI：

```text
P_bar
A P_bar + B Y
Q^(1/2) P_bar
R^(1/2) Y
gamma I
```

当前实现遗漏了 `Q^(1/2) P_bar`、`R^(1/2)Y` 和对应的 `gamma I` 块，也没有
把论文中的 `R` 引入 SDP。当前加入的 `trace(X) == 1` 也不是反馈报告或计划中已经
论证过的论文约束。

同样，`terminal_sdp_margins()` 在 `control/otvdkl_control.py:91` 检查的是：

```text
P - (A + BK)^T P (A + BK) - q I
```

而论文式 (22) 对应的下降条件还包含 `Q + K^T R K`。因此，当前 residual 重算
只能验证当前自定义 LMI，不能验证论文式 (22)、(23)。

独立低维复核还发现：现有测试案例被当前代码判为 `usable=True`，但把求解结果
代回论文四块 LMI 后，最小特征值为负；取 `R=I` 时约为 `-1.69e-1`。这说明当前
“usable”不能解释为论文 SDP 可行。

**整改要求：**

1. 按论文式 (23) 逐块实现 `Q/R/gamma/P_bar/Y`；
2. 删除或单独论证所有论文以外的归一化约束；
3. 从保存的纯 NumPy 数组独立重算三组 LMI；
4. 残差必须包含式 (22) 中的 `Q + K^T R K`；
5. 增加已知可行、不可控、边界过小、椭球不包含当前状态四类测试；
6. 在这些测试通过前，将当前实现明确标为 `prototype`，不得标为 Zhang SDP。

### P0-2：learned `C_tau` 的使用契约前后不一致

选择条件在 `prediction/otvdkl_prediction.py:723`—`749` 使用 learned `C_tau` 计算
normalized physical-state error；但是 replay 在
`prediction/otvdkl_prediction.py:897` 调用 `predict_dkuc_latent_batch()`，后者在
`prediction/common.py:327`—`343` 直接取 latent 前四维并反归一化，相当于使用
`C_struct=[I,0]`，没有使用 learned `C_tau`。

rollout 路径也沿用结构化 readout。于是当前产物存在以下混合：

```text
更新选择判据：learned C_tau
保存的预测数组：C_struct
计划中的第一版控制：C_struct
```

在已保存 smoke 上重新使用 learned `C_tau` 计算 one-step prediction，和现有数组
的最大绝对差约为 `2.41e-2`；同时 `max|C_tau-C_struct|` 分别约为 `0.2433`
和 `0.1690`。虽然总体 RMSE 数值接近，但两套输出并不相同，不能静默混用。

**整改要求：**依据计划第 5.1 节明确固定两类契约：

- 论文 Algorithm 1 的预测/式 (17)、(18) 使用 learned `C_tau`；
- 控制稳定性主线第一版使用 `C_struct`，并把 learned `C_tau` 作为诊断量；
- manifest、数组键名和函数名必须明确 readout 类型；
- 若对外报告 OTVDKL 预测指标，应由 `C_tau(A_tau z+B_tau u)` 生成；
- 若控制器改用 learned `C_tau`，必须重新审查 Corollary 2 的适用性。

### P0-3：`C_tau` 尚未实现计划要求的递推逆更新

`prediction/otvdkl_prediction.py:367`—`374` 对 `c_gram/c_cross` 做了窗口加减，但
在 `405`—`408` 又执行：

```text
inv(candidate_c_gram + lambda I)
```

即每次对 `r x r` 矩阵完整求逆。它能得到 regularized direct solution，但不是计划
阶段 1 要求的 C add/remove inverse recursion，也不能作为式 (13) 已递推实现的证据。

**整改要求：**

- 为 C 路径实现独立的 add/delete Woodbury 更新及低维条件数诊断；
- 与当前窗口 full refit oracle 比较 `C` 和逆矩阵；
- 数值 fallback 必须显式记录；
- 如果暂时保留完整求逆，应把它标为 `direct_inverse_refit` 工程基线，不得称为
  式 (12)、(13) 的递推实现。

### P0-4：68 维 SDP 实时性不可接受且尚无降维策略

使用当前 DKUC artifact 的实际 `A(68x68)`、`B(68x2)` 调用现有简化 SDP，运行
超过 90 s 仍未返回，已人工中止。当前控制周期计划为 10 ms，因此即使先不考虑
P0-1 的公式错误，这一 CVXPY 求解路径也没有作为同步在线控制器的可行证据。

**整改要求：**

1. 先实现正确式 (23)，再对 68 维真实快照做单次和连续多次 benchmark；
2. 分别记录建模时间、canonicalization 时间、solver 时间、状态和 LMI residual；
3. 明确论文要求的“每步求解”与工程求解调度之间的关系；
4. 若需要降低 latent 维度或重训控制专用 DKUC artifact，应形成独立实验，不得
   无记录地改变预测模型；
5. 任何缓存、降频或异步方案都必须在 manifest 中记录，并重新界定理论声明。

在获得数量级合理的 benchmark 前，不应把该 SDP 放进 MuJoCo 同步控制循环。

## 4. P1 完整性问题

### P1-1：CLI 最终状态并不是可恢复 checkpoint

`save_checkpoint()/load_checkpoint()` API 保存的字段较完整，但预测入口在
`prediction/otvdkl_prediction.py:1127`—`1143` 使用另一套手写
`final_<variant>_state.npz` 保存逻辑。实际 smoke 文件缺少：

- A/B `gram/cross/inverse_regularized_gram`；
- `attempt_index`；
- pending buffer；
- updater 配置和选择策略。

因此反馈中“checkpoint 已实现”只对底层 API 成立，对实际 CLI artifact 不成立。
应让 CLI 直接调用唯一 checkpoint API，并另存展示型摘要，而不是维护两套状态格式。

### P1-2：恢复测试没有验证继续运行等价性

`tests/time_varying/test_otvdkl_algorithm.py:62`—`74` 只验证保存后立即读取的若干
字段相等，没有执行计划要求的：

```text
连续消费完整 stream
vs.
消费前半段 -> 保存/恢复 -> 消费后半段
```

应比较最终 `A/B/C`、统计量、窗口样本、pending、版本、历史事件和预测数组。
OTVDKL* 的 `epsilon/improvement_tolerance/variant` 也需要进入恢复契约。

### P1-3：候选拒绝策略 CLI 已失去实际含义

`SelectiveWindowKoopmanUpdater` 仍暴露 `discard_on_reject` 与
`retain_on_reject` 两个 policy，但 `prediction/otvdkl_prediction.py:751`—`763`
在拒绝时固定不推进窗口，两种选项不会改变行为。

论文默认语义已经明确，应删除失效的旧选项，或把 buffer 策略放到真正管理 pending
数据的上层并写出差异测试，避免 CLI 给出虚假的可配置性。

### P1-4：oracle 汇总遗漏 C 与部分有限值检查

更新候选的 fallback 条件会比较 C 差异，但 `oracle_tolerance_passed` 在
`prediction/otvdkl_prediction.py:972`—`975` 只检查 A/B。`recursive_finite` 和
`differences_finite` 也没有完整纳入 C。

应把 A/B/C、逆矩阵和所有 reported differences 一并纳入 finite/tolerance 汇总，
防止 manifest 报告通过而 C 路径实际失效。

### P1-5：只读快照并非真正不可变

`OTVDKLModelSnapshot` 使用 `frozen=True`，只能阻止字段重新赋值；其中 NumPy 数组
仍可原位修改。进入控制事务前，应对快照数组设置只读标志，或由控制器消费隔离
副本，并添加“修改快照失败/不影响 updater”的测试。

### P1-6：MPC 仍缺计划中的终端与独立 oracle 验收

当前 `LiftedMPC.solve()` 已有 error-coordinate、非对称输入边界和终端代价，但没有
实现/验证终端集合约束，也没有保存完整 QP 数组供独立重算。现有测试只验证一维
系统中首控制量方向和 box bound，无法验证式 (21) 的完整矩阵、reference drift、
`u_ref`、终端条件和模型版本切换。

阶段 4 放行前至少应加入：

- 非零参考及非零 `u_ref`；
- 与独立 dense/CVXPY QP oracle 的首控制量对比；
- 终端集合/终端代价一致性；
- snapshot 版本变化后的矩阵刷新；
- SDP/MPC 不可行时的确定性 fallback；
- 保存目标值、约束裕度、迭代数和求解耗时。

## 5. 测试覆盖缺口

当前算法测试覆盖了接受、阈值跳过、基础 checkpoint 和物理误差坐标，但计划中
以下路径仍缺少可重复测试：

- 人工构造候选拒绝；
- 人工构造数值失败；
- 四种路径下分别核查 model/window version、sample IDs 和 pending；
- 修改未来观测不影响历史预测的 causality；
- 连续运行与 checkpoint-resume 的完整等价；
- 保存数组能够独立重算 one-step/rollout；
- SDP 不可行案例和论文 LMI 独立 oracle；
- 终端集合一步正不变性；
- 完整假 plant 的 measurement-update-SDP-MPC-actuation 事务。

这些不是要求在本轮全部完成，但它们是进入阶段 5 前的必要验收证据。

## 6. 下一轮建议执行顺序

### Gate A：先修正预测核心

1. 固定 learned `C_tau` 与 `C_struct` 的职责和产物命名；
2. 实现 C add/delete inverse recursion；
3. 统一 CLI final artifact 与 checkpoint API；
4. 补 reject/failure/causality/resume 等价测试；
5. 修正 A/B/C oracle 汇总。

**Gate A 通过条件：**阶段 1 所有测试通过，并能只靠保存数组重新计算报告指标。

### Gate B：重写论文式 SDP

1. 按式 (22)、(23) 重写 Q/R/gamma 四块 LMI；
2. 建立与 CVXPY 表达式相互独立的 NumPy residual oracle；
3. 完成可行/不可行低维测试；
4. 对 68 维实际快照做耗时和可行率 benchmark；
5. 根据 benchmark 决定是否需要控制专用低维 artifact。

**Gate B 通过条件：**求解器状态、三组 LMI、恢复后的 `P/K/gamma` 和输入约束均
从保存数组独立通过，且形成明确的实时执行策略。

### Gate C：完成假 plant 控制事务

1. 实现单步因果事务 API；
2. 加入动态 cable torque bounds 和安全 fallback；
3. 补独立 MPC QP oracle 与模型版本刷新测试；
4. 用确定性假 plant 验证失败、拒绝和 deadline miss。

**Gate C 通过条件：**没有未来数据泄漏、半更新快照或静默沿用失效解。

只有 Gate A—C 全部通过，才建议进入计划中的 1—2 s MuJoCo smoke。30 s 完整运行
仍需等待短时 smoke 的关节、力矩、绳张力、分配残差和 deadline 指标全部通过。

## 7. 本次复核命令与结果

使用解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

所有 Python 命令均在清除继承 `PYTHONPATH` 后运行。复核结果：

| 检查 | 结果 |
|---|---|
| `py_compile` | 通过 |
| OTVDKL focused tests | `10 passed`，2 warnings |
| 全量测试 | `19 passed`，2 warnings |
| prediction `--help` | 通过 |
| control `--help` / `--dry_run` | 通过 |
| smoke 数组 finite 检查 | 30/30 数组通过 |
| 当前 SDP 低维论文四块 LMI 回代 | 不通过 |
| 当前 SDP 68 维实际快照 | 超过 90 s 未完成，人工中止 |
| `git diff --check` | 通过 |

两条 warning 均来自 OSQP 旧 API，不影响本轮判断，但后续可以在不改变数值行为的
前提下更新接口。

## 8. 最终判定

反馈报告对“尚未完成稳定性保证、尚未进入物理闭环”的总体判断是正确的；但它把
式 (12)、(13) 和式 (22)、(23) 的当前实现程度估计得过高。

**允许继续推进的范围：**预测核心整改、论文 SDP 重写、MPC oracle 和假 plant
事务测试。  
**暂不允许推进的范围：**MuJoCo 控制 smoke、30 s full run、稳定性结果图和论文式
控制性能结论。

下一轮应优先完成 Gate A 与 Gate B，而不是继续堆叠闭环入口。这样可以避免在错误
SDP 和混合 readout 契约之上产生一批无法解释的控制结果。
