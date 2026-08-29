# Hao-DKTV 论文对齐与 CDSM 验证计划

> 状态：待执行
>
> 建立日期：2026-08-28
>
> 目标论文：W. Hao, B. Huang, W. Pan, D. Wu, and S. Mou,
> “Deep Koopman learning of nonlinear time-varying systems,” Automatica,
> 159:111372, 2024.
>
> DOI：<https://doi.org/10.1016/j.automatica.2023.111372>
>
> 作者代码：<https://github.com/wenjianhao/Deep-Koopman-learning-of-nonlinear-time-varying-systems>

## 1. 计划目标

本计划用于把当前仓库中的 DKTV 从“冻结 DKUC 编码器、仅累积更新
`A/B` 的工程基线”，逐步补齐为与 Hao 等人论文核心算法相符、能够在
绳驱机械臂（CDSM）时变动力学数据上独立运行和验证的方法。

最终需要回答以下问题：

1. Hao-DKTV 的递推矩阵更新是否按论文公式正确实现？
2. 观测网络参数 `theta` 是否在每个在线批次中真实更新？
3. `A_tau`、`B_tau`、`C_tau` 和 `theta_tau` 是否形成一致的时变
   Deep Koopman representation？
4. DKTV 是否在严格时间有序、无测试轨迹串扰的数据上运行？
5. 面对明确的时变动力学，完整 DKTV 是否优于固定 DKUC，以及是否优于
   冻结编码器的累积更新消融项？
6. 所有结论是否都有可复现的指标、数组、模型快照和运行元数据支持？

本计划只处理 Hao-DKTV 的数据、预测和辨识问题。Zhang 等人的 OTVDKL、
稳定性保证控制器以及 DKTV 与 OTVDKL 的综合对比不在本计划范围内。

## 2. 架构约束

后续实现必须继续遵守当前仓库的五目录研究架构，不新建 `src/`、
`experiments/` 或 `configs/`：

```text
traj_data/
  collect_data_time_varying.py    新增：独立生成 CDSM 时变数据流

prediction/
  dkuc_prediction.py              复用：DKUC 网络结构和 artifact 加载能力
  dktv_prediction.py              完善：Hao-DKTV 唯一主入口
  common.py                       仅存放确实被多个预测方法共享的工具

tests/time_varying/
  test_core_updates.py            保留已有累积/滑窗基础测试
  test_dktv_algorithm.py          新增：Hao 公式和在线网络更新测试
  test_dktv_pipeline.py           新增：数据到预测产物的轻量集成测试

docs/
  Plan.md                         本计划
```

实现原则：

- DKTV 继续复用 DKUC 的网络定义、归一化器和 artifact，不复制一套新的
  DKUC 网络实现。
- Hao 方法的专用递推状态、在线损失和训练逻辑保留在
  `prediction/dktv_prediction.py`。
- 数据采集只放在 `traj_data/`，预测阶段不直接操纵 MuJoCo。
- 不修改 `control/`；DKTV 本身的完成条件是预测/在线辨识完成，不以控制器
  是否实现作为验收条件。
- 当前 fixed-encoder 累积更新不得继续冒充完整 DKTV；它应被明确标记为
  `frozen_encoder` 消融模式。该模式固定 `theta`，但仍按 Hao 公式（18）、
  （19）更新 `A/B/C`。
- 不创建 `time_varying_comparison.py`。不同方法分别运行，比较阶段读取各自
  保存的结果。

## 3. 当前基线与已知缺口

### 3.1 当前已经具备

- `prediction/dktv_prediction.py` 可以独立执行并读取冻结 DKUC artifact。
- `AccumulativeKoopmanUpdater` 能够累积 Gram/Cross 充分统计量。
- 固定特征空间下，更新后的 `A/B` 与全历史 ridge refit 数值一致。
- one-step 预测在吸收当前真实转移之前完成，基本因果顺序正确。
- 已保存更新历史、矩阵序列、预测数组、指标和 manifest。
- 已有单元测试覆盖固定编码器下的累积矩阵更新。

### 3.2 尚未实现

- 每个在线批次更新观测网络参数 `theta_tau`。
- 解码矩阵 `C_tau` 的初始化、递推更新与保存。
- Hao 论文的 `w*L1 + (1-w)*L2` 在线优化目标。
- `A/B/C` 与 `theta` 的批次级交替更新。
- 论文式递推逆矩阵状态及公式级验证。
- 单条严格时间有序的 CDSM 时变数据采集入口。
- 多轨迹独立运行，当前多轨迹会在同一时刻共同更新一个模型。
- 批次满秩、条件数、矩阵范数和理论假设相关诊断。
- 包含在线神经网络训练时间在内的完整计算开销统计。
- 可从上一次 DKTV 状态继续在线运行的 CLI 契约。
- 完整 DKTV 的正式 CDSM 实验结果。

## 4. 论文公式与代码对象映射

统一使用论文的列样本记号：

```text
X_tau      = [x_k, ..., x_(k+beta-1)]
Xbar_tau   = [x_(k+1), ..., x_(k+beta)]
U_tau      = [u_k, ..., u_(k+beta-1)]
G_tau      = [g(x_k; theta_tau), ..., g(x_(k+beta-1); theta_tau)]
Gbar_tau   = [g(x_(k+1); theta_tau), ..., g(x_(k+beta); theta_tau)]
Chi_tau    = [G_tau; U_tau]
K_tau      = [A_tau, B_tau]
```

对应的近似模型为：

```text
g(x_(k+1); theta_tau) ~= A_tau g(x_k; theta_tau) + B_tau u_k
x_k                    ~= C_tau g(x_k; theta_tau)
```

初始化闭式解：

```text
[A_0, B_0] = Gbar_0 pinv(Chi_0)
C_0        = X_0 pinv(G_0)
```

工程实现允许加入明确记录的 ridge 正则项，但必须保证：

- `ridge_lambda=0` 或足够小时能与论文无正则公式对应；
- 正则化版本有独立的公式测试；
- manifest 明确记录实际使用的 `ridge_lambda`；
- 结果报告不能把正则化扩展描述成论文原始设置。

在线观测网络损失定义为：

```text
L1 = mean(||g(x_next; theta) - A_tau g(x; theta) - B_tau u||^2)
L2 = mean(||x_norm - C_tau g(x; theta)||^2)
L  = w * L1 + (1 - w) * L2
```

`x_norm` 使用 DKUC artifact 中冻结的状态归一化器。归一化器在在线阶段不
更新，避免同一物理状态的坐标定义随时间漂移。

CDSM 中使用的 `g` 继续采用 DKUC 结构：

```text
g(x; theta) = [x_norm, encoder(x_norm; theta)]
```

即使前若干 latent 坐标已经包含 `x_norm`，仍按论文计算和保存 `C_tau`，
以便检查重构误差并保持 `A/B/C/theta` 契约完整。后续可以把固定选择矩阵
作为消融项，但不能用它代替默认的论文对齐实现。

## 5. 实施阶段

### 阶段 0：锁定算法契约和基线证据

目标：在修改实现前，把“完整 DKTV”和“冻结编码器基线”区分清楚。

任务：

- [ ] 在 `dktv_prediction.py` 中定义两个明确模式：
  - `full`：更新 `A/B/C/theta`，作为最终默认 DKTV；
  - `frozen_encoder`：固定 `theta`，更新 `A/B/C`，作为固定观测函数消融。
- [ ] 明确 Hao 原论文没有 `frozen_encoder` 模式；该模式是项目为了判断在线
  `theta` 更新贡献而增加的工程消融，不得描述为论文原始 DKTV。
- [ ] 给当前 updater、输出字段和 CLI 参数建立迁移表。
- [ ] 记录当前固定 DKUC artifact、测试数据和 smoke 指标作为改动前基线。
- [ ] 明确所有更新采用“先预测、后观测、再更新”的顺序。
- [ ] 明确一次独立 trial 只能消费一条时间有序轨迹。

本阶段不应删除当前累积更新代码，也不应修改 OTVDKL。

退出条件：

- 文档和 CLI 名称不再把 frozen-encoder 结果描述为完整 DKTV。
- 当前聚焦测试继续通过。
- 基线 manifest 和指标可以定位到具体 DKUC artifact 与数据集。

### 阶段 1：建立可验证的 CDSM 时变数据流

目标：生成能够明确写成
`x_(k+1) = f(x_k, u_k, k)` 的数据，而不是在固定 MuJoCo 模型上重复拟合。

新增入口：

```text
traj_data/collect_data_time_varying.py
```

第一版只实现一种容易验证且不污染模型输入的时变来源：施加绝对时间相关的
外部关节扰动力矩。模型输入仍只保存缆索产生的等效关节力矩，外部扰动不作为
预测模型输入，因此系统转移规律显式依赖时间。

建议扰动形式：

```text
d_tau(t) = amplitude * sin(2*pi*frequency*t + phase)
```

任务：

- [ ] 复用 `MujocoCDSM.apply_joint_disturbance()`，不复制机器人适配器。
- [ ] 支持 `--traj`、`--steps`、`--dt`、`--seed`、扰动幅值、频率和相位。
- [ ] 支持无扰动 warm-up，使离线历史数据与在线时变 stream 清楚分开。
- [ ] 每条轨迹具有独立初始状态；每条轨迹的时间从零开始且内部严格递增。
- [ ] 数据集至少保存 `t`、`states`、`inputs` 和 `disturbance_torque`。
- [ ] `disturbance_torque` 只用于诊断，不得进入 DKTV 的 `u_k`。
- [ ] metadata 保存扰动公式、全部参数、随机种子、XML 路径和 Git 信息。
- [ ] summary 检查有限值、状态范围、关节限位、峰值力矩和峰值绳索张力。
- [ ] 生成一个无扰动对照数据集和一个时变扰动数据集。

数据输出：

```text
traj_data/outputs/<run_type>/<run_id>/
  dataset.npz
  metadata.json
  summary.json
```

退出条件：

- 同一 seed 能复现相同数据。
- `states/inputs/disturbance_torque` 全部有限。
- 时变扰动确实被施加，且不包含在模型输入中。
- 至少一条代表轨迹通过数值和可视化检查。
- 无明显关节限位越界或不可接受的绳索张力异常。

后续扩展（不阻塞第一版）：时变负载、质量或阻尼。只有外部扰动版本稳定后，
再评估是否需要更物理化的参数变化。

### 阶段 2：实现 `A/B/C` 递推更新

目标：把当前只更新 `A/B` 的充分统计实现扩展为论文所需的完整线性表示。

在 `prediction/dktv_prediction.py` 中建立一个 Hao 专用递推状态，至少包含：

```text
A, B, C
P_chi                 # regularized inverse of Chi*Chi^T
P_g                   # regularized inverse of G*G^T
sample_count
update_index
model_version
encoder_version
```

任务：

- [ ] 从初始历史批次计算 `A_0/B_0/C_0/P_chi/P_g`。
- [ ] 按论文公式（18）递推 `A/B`。
- [ ] 按论文公式（19）递推 `C`。
- [ ] 使用 Woodbury 形式同步更新 `P_chi` 和 `P_g`。
- [ ] 允许稳定的 ridge 初始化，禁止直接对病态矩阵裸求逆。
- [ ] 每次更新前检查样本形状、有限值、维度和 encoder version。
- [ ] 保存批次的秩、最小奇异值、条件数和正则化条件数。
- [ ] 记录 `||A||_2`、谱半径、`||B||_2` 和 `||C||_2`。
- [ ] 更新失败时保持上一个可用状态，保存失败原因，不产生半更新状态。
- [ ] 保留 `frozen_encoder` 模式，使新递推结果可与现有全历史 refit 比较。

公式验收：

- [ ] 固定 `theta` 时，递推 `A/B` 与同一累计数据的直接 ridge refit 一致。
- [ ] 固定 `theta` 时，递推 `C` 与直接 ridge refit 一致。
- [ ] 分批方式变化不能显著改变最终矩阵。
- [ ] save/load 后继续更新，与不中断连续更新得到相同结果。

退出条件：

- 公式级单元测试全部通过。
- 正常输入下所有矩阵与诊断值有限。
- 当前 `frozen_encoder` 测试没有退化。

### 阶段 3：实现在线 `theta_tau` 更新和 Hao 损失

目标：使 DKTV 从线性矩阵再拟合升级为真正的 Deep Koopman 在线更新。

任务：

- [ ] 从 DKUC artifact 加载网络后创建可训练副本，不能直接修改原 artifact。
- [ ] 复用 `dkuc_prediction.py` 的网络定义，禁止在 DKTV 中复制 MLP 类。
- [ ] 实现 `L1`、`L2` 和加权总损失 `L`。
- [ ] 新增 CLI 参数：`--loss_weight`、`--online_epochs`、
  `--online_lr`、`--online_weight_decay`、`--grad_clip`。
- [ ] 每批从上一版本 `theta_tau` warm start，而不是随机初始化。
- [ ] 按 Hao Algorithm 1 保持矩阵与观测网络参数一致：在线优化过程中
  `A(theta)/B(theta)/C(theta)` 受论文公式（18）、（19）约束；得到
  `theta_(tau+1)^*` 后，最终提交的矩阵必须是对应的
  `A(theta^*)/B(theta^*)/C(theta^*)`。
- [ ] 若工程上采用“固定当前 `A/B/C` 优化 `theta`”的块坐标近似，训练结束后
  必须使用 `theta^*` 重新计算 `A/B/C`，必要时迭代至约定收敛条件，并在
  manifest 中明确记录该近似，不能把只完成一次冻结矩阵梯度步描述为论文等价实现。
- [ ] 一个批次完成后原子保存相互匹配的最优 `theta` 和 `A/B/C`，更新
  `encoder_version/model_version`。
- [ ] 记录训练前后 `L1/L2/L`、梯度有限性和训练耗时。
- [ ] 如果训练产生非有限 loss 或参数，回滚到上一 encoder 版本。
- [ ] 明确 encoder 更新后旧递推统计属于历史特征坐标，这是 Hao 累积算法的
  方法特征，同时在 manifest 中记录这一近似。
- [ ] 可选加入 `||A||` 惩罚，但必须默认关闭，并与论文原始损失分开报告。

建议的单批在线顺序：

```text
使用 model_tau 预测当前批次
  -> 获得当前批次真实 x_next
  -> 使用 theta_tau 提取 G_new/Gbar_new
  -> 由公式 (18)/(19) 得到 A(theta), B(theta), C(theta)
  -> 在矩阵公式约束下优化 theta_(tau+1)^*
  -> 计算与 theta_(tau+1)^* 对应的 A/B/C
  -> 原子保存完整且坐标一致的 model_(tau+1)
```

退出条件：

- 在线批次后 encoder 参数校验和发生变化。
- 在可学习的合成样本上，总损失相对训练前下降。
- `A/B/C/theta` 任一部分失败时能够完整回滚。
- `full` 与 `frozen_encoder` 模式可以使用同一个入口独立运行。

### 阶段 4：修正时间顺序、轨迹隔离与恢复运行

目标：消除不同测试轨迹相互提供未来/旁路信息的风险。

任务：

- [ ] 禁止把同一 `time_step` 的多条独立轨迹拼成一个在线批次。
- [ ] 对多轨迹数据逐条运行；每条轨迹从同一初始 DKTV 状态独立开始。
- [ ] 输出数组增加明确的 `trial_id`、`time_index`、`update_index`。
- [ ] 更新边界严格按已观察转移数量确定，不按数组 reshape 后的位置确定。
- [ ] 最后不足一个 batch 的样本默认不更新，并记录 pending 数量。
- [ ] 增加 `--resume_state`，允许读取完整 DKTV checkpoint 继续运行。
- [ ] checkpoint 同时保存神经网络权重、`A/B/C`、递推逆矩阵、归一化器
  fingerprint、优化配置和版本号。
- [ ] 加入 causality 测试：修改未来样本不能改变更早的预测和模型版本。
- [ ] 加入 isolation 测试：修改 trial 2 不能改变 trial 1 的输出。

退出条件：

- 因果性和轨迹隔离测试通过。
- 中断恢复结果与连续运行结果在数值容差内一致。
- 每个预测样本能够追溯到当时实际使用的模型版本。

### 阶段 5：完善评估和研究产物

目标：同时提供工程验证和论文可用的预测证据。

必须区分三类指标：

1. `one_step`：每一步使用真实 `x_k`，评估局部模型误差；
2. `batch_rollout`：每个在线批次开始时用真实状态初始化，随后在该批次内
   递推预测，对齐 Hao 论文的估计方式；
3. `fixed_horizon_rollout`：在多个起点进行 10/25/50 等固定预测长度评估。

比较方法：

```text
fixed_dkuc          不进行在线更新
dktv_frozen_encoder 固定 theta，累积更新 A/B/C（工程消融）
dktv_full           更新 A/B/C/theta（Hao-DKTV）
```

TVDMD 或单 DNN 可以作为后续补充基线，但不阻塞 Hao-DKTV 的 CDSM 第一轮
完成验收。

任务：

- [ ] 三种方法读取完全相同的初始 artifact、历史数据和在线 trial。
- [ ] 分别保存状态维度 MAE/RMSE、总体 MAE/RMSE 和随时间误差范数。
- [ ] 保存每次更新前后的新批次预测误差。
- [ ] 分开记录编码、矩阵更新、神经网络训练和总更新时间。
- [ ] 统计更新次数、失败次数、平均/最大更新时间和内存占用。
- [ ] 至少使用多个独立 trial；正式结果报告 mean 和 standard deviation。
- [ ] 检查所有预测、矩阵、loss 和指标中的非有限值。
- [ ] 保存 `A/B/C` 演化、矩阵范数、条件数、模型版本和 loss 历史原始数组。
- [ ] 图像只由保存数组生成，不把图像当作唯一证据。

DKTV 结果至少包含：

```text
prediction/outputs/<run_type>/dktv/<run_id>/
  manifest.json
  metrics.json
  prediction_arrays.npz
  update_history.json
  training_history.json
  final_dktv_state.npz
  final_dktv_model.pt
```

manifest 至少记录：

- 完整命令行参数；
- Python 可执行文件和设备；
- Git 分支与 commit；
- 随机种子；
- 初始 DKUC artifact 和 fingerprint；
- 历史数据和 stream 数据路径；
- 状态、输入和 disturbance 字段顺序；
- `full/frozen_encoder` 模式；
- 网络和在线优化超参数；
- ridge 和 batch 设置；
- trial 隔离策略；
- 主要指标和产物路径。

退出条件：

- 三种方法均能独立运行并生成完整产物。
- 指标能从保存数组重新计算得到。
- full DKTV 的在线网络参数确实变化，而 fixed DKUC 和 frozen-encoder
  模式符合各自契约。
- 至少一个时变强度下，完整 DKTV 相对 fixed DKUC 展现可重复的改善；若未
  改善，必须保留结果并分析数据激励、批次长度和优化收敛，而不能直接宣称成功。

### 阶段 6：正式 CDSM 实验与方法审查

目标：判断 Hao-DKTV 是否适合进入论文对比，而不只确认代码能够运行。

建议实验变量：

- 时变扰动幅值：弱、中、强；
- 时变频率：慢、中、快；
- batch size：至少三档，并满足或明确偏离论文满秩要求；
- online epochs：至少两档；
- `loss_weight`：至少三档；
- trial/seed：正式结果不少于能够计算稳定均值和标准差的数量。

每次只改变一类因素。不要在第一轮同时搜索全部超参数。

审查问题：

- [ ] DKTV 的改善来自在线 `theta`，还是仅来自固定 `theta` 下的 `A/B/C`
  再拟合？
- [ ] 随历史不断累积，旧数据是否使快速时变场景性能下降？
- [ ] batch size 对跟踪变化速度和数值条件的影响是什么？
- [ ] 在线神经训练能否在采样周期内完成？若不能，应区分算法时间与仿真时间。
- [ ] `C_tau` 是否真实变化，还是退化为状态选择矩阵？
- [ ] `||A_tau||` 和 rollout 误差是否出现不稳定增长？
- [ ] 不同 trial 下结论是否一致？
- [ ] 当前结果是否足以支持“时变预测能力”，而不越界声称控制稳定性？

阶段完成后形成单独 review 文档，记录：

- 已实现内容；
- 与论文一致的部分；
- 有意采用的工程扩展；
- 未满足的论文假设；
- 失败实验及原因；
- 是否进入 OTVDKL 对比阶段的结论。

## 6. 测试清单

### 6.1 数学与数值测试

- [ ] `A/B` 初始化等价于直接最小二乘或 ridge 解。
- [ ] `C` 初始化等价于直接最小二乘或 ridge 解。
- [ ] 递推 `A/B` 等价于累计数据直接 refit。
- [ ] 递推 `C` 等价于累计数据直接 refit。
- [ ] Woodbury 更新后的逆矩阵与直接求逆一致。
- [ ] 病态、秩亏和非有限输入能够被诊断并安全拒绝。
- [ ] save/load 不改变预测结果和后续更新结果。

### 6.2 神经网络测试

- [ ] DKTV 使用的是 DKUC 网络定义，而不是复制网络。
- [ ] `full` 模式 encoder 权重会更新。
- [ ] `frozen_encoder` 模式 encoder 权重不会更新。
- [ ] 在线 loss 有限，失败时可以回滚。
- [ ] `C` 重构输出维度和物理状态恢复正确。

### 6.3 因果性与数据测试

- [ ] 未来数据变化不影响过去预测。
- [ ] 不同 trial 之间没有模型状态串扰。
- [ ] batch 边界和共享端点没有 off-by-one。
- [ ] 扰动没有误加入模型控制输入。
- [ ] 数据中的时间、状态和输入严格对齐。

### 6.4 集成与产物测试

- [ ] `prediction/dktv_prediction.py --help` 成功。
- [ ] 一个小型 CPU smoke run 成功。
- [ ] 一个短 CDSM 时变 stream 端到端运行成功。
- [ ] manifest、metrics、arrays、checkpoint 和图像全部存在且可读取。
- [ ] 指标可从 arrays 独立重算。
- [ ] 至少打开检查一张代表性图像。
- [ ] 完整测试、`git diff --check` 和 `git status --short` 完成。

## 7. 建议 CLI 契约

最终入口仍为：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  prediction/dktv_prediction.py \
  --artifact_dir prediction/outputs/full_run/dkuc/<run_id> \
  --history_dataset prediction/outputs/full_run/dkuc/<run_id>/dataset_train.npz \
  --stream_dataset traj_data/outputs/full_run/<stream_run_id>/dataset.npz \
  --mode full \
  --batch_size 80 \
  --ridge_lambda 1e-3 \
  --loss_weight 0.5 \
  --online_epochs 20 \
  --online_lr 1e-4 \
  --run_type full_run \
  --device cuda \
  --seed 50 \
  --tag hao_cdsm
```

该命令只是目标契约示例。参数默认值必须在完成阶段 1 至阶段 3 的 smoke
验证后确定，不能因为论文示例使用某一组参数就直接视为 CDSM 的稳定默认值。

## 8. 风险与处理原则

### 风险 1：在线更新 encoder 后历史充分统计失去统一坐标含义

这是 Hao 累积方法本身需要面对的问题。实现必须记录 encoder version，且不能
错误宣称更新后的矩阵等价于“使用最新 encoder 对全部历史数据重新拟合”。
`frozen_encoder` 模式中 `theta` 不变，因此累计更新后的 `A/B/C` 可作为直接
refit 的严格等价 oracle 和消融对照。

### 风险 2：DKUC latent 维度较高，论文满秩条件要求较大 batch

默认 DKUC latent 维度为状态维度加 lift 维度，可能使 `r+m` 远大于较小的
在线 batch。处理方式是记录批次秩和条件数、使用 ridge 保证工程可计算，并在
论文结果中明确哪些实验满足理论条件、哪些只属于正则化扩展。

### 风险 3：完整在线神经训练无法实时完成

必须分别统计矩阵更新和神经训练耗时。第一阶段只要求因果 replay 正确，不把
离线 replay 的成功描述为实时部署成功。

### 风险 4：外部正弦扰动不能充分代表实际参数时变

正弦扰动用于验证算法链路。若 DKTV 链路稳定，再增加负载、质量或阻尼变化，
而不是在基础实现尚未确认时同时引入多个复杂因素。

### 风险 5：完整 DKTV 不一定优于冻结编码器基线

性能改善不是代码正确性的先决条件。先用公式测试证明实现正确，再通过 loss、
激励范围、batch、时变速率和优化收敛分析性能。失败结果必须保留。

## 9. 总体验收标准

只有同时满足以下条件，才将 Hao-DKTV 标记为“核心实现完成”：

- [ ] 时变数据由独立入口生成并通过安全/有限值检查。
- [ ] `A/B/C` 递推公式通过直接 refit oracle 测试。
- [ ] `theta` 在每个接受的在线批次中按 Hao loss 更新。
- [ ] 每次 full 更新提交的 `A/B/C` 与同一批次最终 `theta^*` 相匹配，而不是
  停留在优化 `theta` 之前的旧特征坐标。
- [ ] full 和 frozen-encoder 两种模式名称、状态和输出明确分离。
- [ ] 多轨迹之间无数据和模型状态串扰。
- [ ] one-step、batch rollout 和 fixed-horizon rollout 分开评估。
- [ ] checkpoint 能恢复并继续在线运行。
- [ ] 完整计算时间、矩阵条件和非有限值均被检查。
- [ ] CDSM smoke 和 full run 都生成可复现产物。
- [ ] 正式结果至少比较 fixed DKUC、frozen-encoder DKTV 和 full DKTV。
- [ ] review 文档明确说明论文一致项、工程扩展和剩余风险。

## 10. 推荐推进顺序

严格按以下顺序执行，前一阶段未通过退出条件时不进入下一阶段：

```text
阶段 0：锁定契约和基线
  -> 阶段 1：时变 CDSM 数据
  -> 阶段 2：A/B/C 递推
  -> 阶段 3：在线 theta 更新
  -> 阶段 4：因果性、轨迹隔离和恢复
  -> 阶段 5：指标与产物
  -> 阶段 6：正式实验与 review
```

每完成一个阶段，都应在本文件中勾选对应任务，并单独记录执行命令、测试结果、
关键指标和输出路径。不得仅凭程序退出码或生成图片判断阶段完成。
