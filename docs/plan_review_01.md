# Hao-DKTV 计划执行审阅 01

> 审阅日期：2026-08-28  
> 对照计划：`docs/Plan.md`  
> 执行反馈：`docs/plan_feedback_01.md`  
> 审阅范围：实际代码、自动化测试、时变数据 smoke 产物、DKTV smoke 产物  
> 审阅结论：**已经形成 Hao-DKTV 的可运行原型，但尚不能标记为“核心实现完成”。阶段 2、3 的主体算法已有初步证据；阶段 0、1、4、5 仍存在契约或验收缺口，阶段 6 尚未开始。**

## 1. 总体判断

本轮不是“没有实现”，而是已经完成了以下关键骨架：

- `HaoDKTVState` 已包含 `A/B/C/P_chi/P_g` 及累计统计量；
- 固定 encoder 条件下，`A/B/C` 的累计 ridge 结果能够与直接 refit 对齐；
- `full` 模式能够按 `L1/L2` 更新复用的 DKUC encoder；
- replay 的单步顺序基本满足“先预测、再观察、后更新”；
- 时变数据和 DKTV 均已有可读取的 smoke 产物；
- 保存数组中的 one-step 指标可以独立重算得到相同结果。

但是，当前实现中固定 DKUC 基线并未真正保持冻结，时变数据入口也不能按仓库规定的模块方式启动。这两个问题会分别影响科研比较的有效性和项目入口契约。因此，反馈报告中的“核心链路已实现”可以理解为原型链路已经贯通，但不应进一步表述为“Plan 核心验收已完成”。

建议当前状态标记为：

```text
Hao-DKTV prototype implemented / core acceptance pending
```

## 2. 需要优先处理的问题

### P0：`fixed_dkuc` 基线受到在线 encoder 更新污染

位置：`prediction/dktv_prediction.py:654-657`、`prediction/dktv_prediction.py:675-680`、`prediction/dktv_prediction.py:797-801`。

当前 `fixed_dkuc_one_step` 使用 `model.lift()` 生成特征。`full` 模式每个批次又会原地更新同一个 `model.model.encoder`，因此第一次在线更新之后，“fixed DKUC”虽然继续使用原始 `A/B`，却已经使用了变化后的 encoder。fixed-horizon rollout 也在 replay 完成后使用最后一次在线训练留下的 encoder，存在相同问题。

本次用同一 smoke 数据重新加载原始 DKUC artifact 复算真正冻结的 one-step 结果：

```text
保存的 fixed_dkuc RMSE：       0.1203046291394887
重新加载原始 DKUC 的 RMSE：   0.1203049680574463
更新后预测最大绝对差异：       3.7557773984175924e-06
```

差异较小只是因为 smoke 使用了 `online_lr=1e-6`，不能说明实现正确。正式实验提高学习率或增加更新批次后，该偏差可能明显增大。

影响：

- 当前 fixed DKUC one-step 和 fixed-horizon 指标不能作为严格冻结基线证据；
- `fixed_dkuc` 与 `dktv_full` 的科研对比存在方法串扰；
- 反馈报告中的两个 RMSE 可用于检查程序出数，但不能用于方法优劣判断。

整改要求：固定 DKUC 必须持有独立且不可变的网络副本，或预先缓存由原始 encoder 得到的固定特征。新增测试应在高学习率、多批次条件下断言 fixed 结果与重新加载原始 artifact 的结果一致。

### P1：时变数据入口不符合仓库模块运行契约

位置：`traj_data/collect_data_time_varying.py:12-14`。

按照 `AGENTS.md` 规定执行：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m traj_data.collect_data_time_varying --help
```

实际失败：

```text
ModuleNotFoundError: No module named 'data_io'
```

原因是采集器只使用了面向文件路径执行的顶层导入，没有像 DKTV 入口一样兼容包内相对导入。反馈报告中的采集命令通过直接执行文件工作，但不能证明标准模块入口可用。

整改要求：使该入口同时支持仓库要求的 `-m traj_data.collect_data_time_varying`；至少补充 `--help` 和一个短 smoke 集成测试。

### P1：`full` 模式更新 `theta` 后没有使 `A/B/C` 与 `theta^*` 重新一致

Hao 原论文 Algorithm 1 的输出是同一更新批次下的
`g(·, theta_tau)、A_tau、B_tau、C_tau`。论文先按公式（18）、（19）更新
`A(theta)/B(theta)/C(theta)`，再优化 `theta`，最后提交的是与
`theta^*` 对应的 `A(theta^*)/B(theta^*)/C(theta^*)`。因此论文结论是明确的：

```text
Hao-DKTV full：A/B/C/theta 全部更新并保持同一特征坐标
frozen_encoder 消融：theta 固定，A/B/C 仍按公式（18）、（19）更新
```

Hao 论文没有 `frozen_encoder` 模式；它是本项目为分离在线网络更新贡献而增加
的工程消融。既然冻结的只是 encoder，当前代码在该模式更新 `A/B/C` 是正确的，
不需要冻结 `C`，也不需要因为更新 `C` 而重命名。

真正需要整改的是 `full`：当前代码先用旧 `theta` 提取批次特征并更新
`A/B/C`，随后固定这些矩阵训练 encoder，但训练完成后没有用最终 `theta^*`
重新计算 `A/B/C`。提交的矩阵和 encoder 因而处于不同特征坐标，这只能视为
一次未闭环的块坐标近似，不能直接称为论文等价实现。

整改要求：在得到 `theta^*` 后，按论文公式重新获得与其一致的 `A/B/C`；若
采用交替优化近似，应规定迭代和收敛条件，并将相互匹配的 `A/B/C/theta` 原子
提交及保存。

### P1：在线训练的“最佳 loss—最佳权重”对应关系偏移一个优化步

位置：`prediction/dktv_prediction.py:444-458`。

循环先在当前权重上计算并记录 loss，随后执行 `optimizer.step()`，最后却把更新后的权重保存为刚才那个更新前 loss 对应的 `best_state`。因此保存的权重不严格对应记录的最低 loss，训练 history 也不是逐步更新后的 loss。

此外，`epochs=0` 当前被允许，并会返回 accepted；这样 `full` 模式可能在没有发生 theta 更新时仍被视为一次成功训练。

整改要求：每次参数更新后重新评估 loss，再决定是否保存权重；明确 `full` 模式是否要求 `online_epochs > 0`；测试应验证加载的 best state 能复现记录的 best loss。

### P1：阶段 4 的恢复与独立性仍未达到验收条件

现有代码确实在每条 trial 开始时重置网络和递推状态，这是正确方向。但仍缺少以下关键证据：

- 没有 causality 自动化测试；
- 没有 trial isolation 自动化测试；
- 没有“连续运行 vs 中断恢复”的等价测试；
- `resume_state` 与 `resume_model` 可单独提供，没有强制成对及 fingerprint/config/version 校验；
- 多 trial 运行最终只保存最后一个 trial 的 `final_dktv_state.npz` 和 `final_dktv_model.pt`，无法代表其他独立 trial 的最终状态。

因此“代码路径已具备”可以保留，但阶段 4 不能判定完成。

### P1：时变 smoke 数据尚不能通过安全退出条件

本轮数据摘要满足：必需数组有限、时间严格递增、没有关节限位越界、外部扰动已保存且未并入 `inputs`。但代表性 smoke 同时给出：

```text
peak_abs_tau          = 1331.0948588941503
peak_cable_tension    = 3030.391572992118
```

当前代码没有定义力矩和绳索张力的可接受上限，也没有自动拒绝或标记异常数据，因此只能确认“数值有限”，不能确认“物理上可接受”。计划要求的无扰动对照数据、同 seed 确定性测试和代表轨迹可视化也尚未提供。

整改要求：先根据 MuJoCo 模型和执行器设定给出明确安全阈值，再决定该数据应进入 `raw`、`processed` 还是 `rejected`；同时生成同参数无扰动对照。

### P2：矩阵诊断和公式测试覆盖仍不完整

当前递推公式测试是有价值的，但距离 Plan 清单仍缺少：

- 未保存正则化条件数；
- 当前 rank/condition 诊断针对新批次矩阵，不是累计正则系统；
- 没有验证不同 batch 划分得到相同最终状态；
- save/load 测试只检查了部分字段，没有验证恢复后继续更新与连续运行一致；
- 非有限回滚只断言了 `A/C/sample_count`，没有覆盖全部统计量、逆矩阵和版本号；
- 没有病态但有限输入的稳定性测试。

smoke 每批只有 4 个样本，而 `chi` 的特征维度约为 70，批次秩记录为 4，显然不满足满秩设定。ridge 使计算可以继续，但正式报告必须把它标记为正则化工程扩展。两次更新的 `A` 谱范数约为 `15.13`，尽管谱半径约为 `1.0`，仍需要通过 rollout 检查非正规放大和长时预测稳定性。

### P2：阶段 5 评估只完成了 one-step 的一部分

当前产物包含 one-step 指标和 fixed DKUC 的 horizon-4 rollout；DKTV 的 `batch_rollout` 与 fixed-horizon rollout 仍为空。当前保存数组均为有限值，且 one-step 指标可以从数组精确重算，这是有效的局部完成证据。

不过还缺少：

- full DKTV 的 batch rollout；
- full DKTV 的多个 fixed-horizon rollout；
- 三种模式在同一数据上的独立结果；
- 多 trial 均值和标准差；
- 编码、矩阵更新、神经训练和总更新时间的分项统计；
- 正式内存与实时性证据。

## 3. 分阶段完成度审阅

| 阶段 | 审阅状态 | 已验证内容 | 未通过的主要退出条件 |
|---|---|---|---|
| 阶段 0：契约与基线 | 部分完成 | `full/frozen_encoder` CLI、先预测后更新、trial 重置；frozen 模式更新 `A/B/C` 符合固定 `theta` 消融定义 | fixed DKUC 不完全冻结；缺少可靠改动前基线 |
| 阶段 1：时变数据 | 部分完成 | 独立文件、正弦外扰、字段分离、有限值和时间检查 | 标准模块入口失败；无无扰动对照；无确定性测试；安全阈值和可视化验收缺失 |
| 阶段 2：A/B/C 递推 | 主体已实现，验收未完成 | A/B/C、Woodbury、ridge、原子拒绝、基础 save/load、直接 refit 对齐 | batch 划分、恢复续算、病态输入、正则条件数和完整回滚测试缺失 |
| 阶段 3：在线 theta | 主体已实现，验收未完成 | 复用 DKUC 网络、L1/L2、warm start、checksum 变化、smoke loss 下降 | 最终 `A/B/C` 未与 `theta^*` 重新一致；best-state 对齐错误；`epochs=0` 契约；失败时四部分整体回滚测试缺失 |
| 阶段 4：因果/隔离/恢复 | 代码路径部分完成 | 单 trial 顺序合理；trial 开始时重置 | 三类核心测试缺失；checkpoint 未配对校验；多 trial 最终 checkpoint 不完整 |
| 阶段 5：评估产物 | 部分完成 | manifest/metrics/arrays/history/checkpoint/图像存在；one-step 可重算 | 固定基线被污染；两类 DKTV rollout 缺失；多 trial 与三方法正式比较缺失 |
| 阶段 6：正式实验 | 未开始 | 无 | 幅值/频率/batch/loss/seed 扫描及论文级结论均未完成 |

## 4. 本次复核得到的有效证据

### 4.1 自动化测试

指定环境：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

聚焦测试及全仓库测试均为：

```text
6 passed in 1.73s
```

需要注意：当前全仓库测试总数就是 6 个，尚没有反馈报告中承认缺失的 pipeline、causality、isolation 和 resume 测试。

### 4.2 smoke 数值与产物

审阅的 DKTV 产物：

```text
/tmp/dktv_plan_smoke2/dktv/20260828_102610_dktv_plan/
```

确认结果：

- `prediction_arrays.npz` 中全部数组有限；
- `states_true` 形状为 `(1, 9, 4)`；
- 接受 `A/B/C` 更新 2 次，encoder checksum 改变 2 次；
- DKTV one-step RMSE 独立重算为 `0.04955806680843792`，与 JSON 一致；
- 保存的 fixed one-step RMSE 独立重算为 `0.1203046291394887`，与 JSON 一致；
- 代表性 one-step 图像可以正常打开，位置和速度曲线均可辨识；
- 以上仅证明产物内部一致，不消除固定基线污染，也不构成论文性能结论。

反馈报告将 DKTV RMSE 写为“约 `0.0503`”，实际 manifest 为约 `0.04956`。这是轻微记录误差，后续应统一从 `metrics.json` 自动生成报告数值。

## 5. 建议的下一轮整改顺序

### 第一步：先修科研比较契约

1. 分离不可变 fixed DKUC 和可训练 DKTV 网络实例；
2. 保持唯一契约：`frozen_encoder` 固定 `theta` 并更新 `A/B/C`；
3. 在 full 模式得到 `theta^*` 后，重新获得与其匹配的 `A/B/C`；
4. 修正在线 best-state 与 loss 的对应关系；
5. 为上述各项补充单元测试。

验收标准：无论 DKTV 训练多少批次，fixed DKUC 的 one-step 和 rollout 均与重新加载原始 artifact 的结果一致。

### 第二步：补齐入口与阶段 2/4 测试

1. 修复时变采集器的模块导入；
2. 增加 batch 划分等价和 save/load 续算测试；
3. 增加 causality、trial isolation、resume equivalence 测试；
4. 将 state/model、normalizer fingerprint、配置和版本绑定为一个 checkpoint 契约；
5. 明确多 trial 是否保存逐 trial checkpoint，或明确只输出评估数组、不提供含糊的单一 final checkpoint。

### 第三步：完成阶段 1 和阶段 5

1. 生成无扰动与有扰动的成对数据；
2. 定义力矩和绳索张力安全阈值，补充确定性和数据对齐测试；
3. 完成 DKTV batch rollout 与多个 fixed-horizon rollout；
4. 确保三种方法独立运行、读取同一数据、保存可重算数组；
5. 完成后再进行多 trial 和参数扫描。

## 6. 下一轮最低验收清单

- [ ] `python -m traj_data.collect_data_time_varying --help` 在指定环境成功；
- [ ] full 模式下 fixed DKUC 与新加载原始 artifact 的预测一致；
- [ ] `frozen_encoder` 固定 `theta`，并按论文公式更新 `A/B/C`；
- [ ] full 模式每批最终保存的 `A/B/C` 与该批次 `theta^*` 属于同一特征坐标；
- [ ] 保存的 best encoder 能复现记录的 best loss；
- [ ] batch 划分等价、因果性、trial 隔离和恢复续算测试通过；
- [ ] checkpoint 成对校验，错误组合能够明确拒绝；
- [ ] 无扰动/有扰动数据均通过确定性、有限值、限位和执行器阈值检查；
- [ ] one-step、batch rollout、fixed-horizon rollout 均能从保存数组重算；
- [ ] fixed DKUC、frozen-encoder DKTV、full DKTV 三种方法分别生成独立产物；
- [ ] 全部修复后重新生成 smoke，不沿用当前受基线污染的比较指标。

## 7. 最终审阅结论

本轮最值得保留的是 `A/B/C` 累计递推和在线 encoder 训练的主体实现，它们已经让项目从“固定特征空间在线更新”推进到了“可运行的 Hao-DKTV 原型”。当前主要问题不在于核心公式完全缺失，而在于实验契约和验证证据还没有闭环。

因此本次审阅结论为：

```text
阶段 2/3：允许进入修正与加强测试；
阶段 0/1/4/5：继续整改，不通过退出验收；
阶段 6：暂缓；
Hao-DKTV：暂不标记为核心实现完成，暂不用于论文结论。
```
