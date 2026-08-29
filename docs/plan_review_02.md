# Hao-DKTV 整改执行审阅 02

> 审阅日期：2026-08-28  
> 对照审阅：`docs/plan_review_01.md`  
> 执行反馈：`docs/plan_feedback_02.md`  
> 审阅范围：实际代码、全仓库测试、CLI 模块入口、review02 smoke 数组、指标、模型和递推状态  
> 审阅结论：**第二轮整改取得了明显进展，主干算法已经具备进入探索性完整运行的条件。fixed-horizon 固定基线串扰会直接影响比较数值，应先修复或在本轮禁用该指标；resume、checkpoint 身份、异常回滚、精细耗时等工程完备项可后置，只要完整运行采用单次不中断、单 trial 独立执行，并且不宣称恢复能力或实时性。**

## 1. 总体判断

`plan_feedback_02.md` 中大部分已完成项能够在代码和产物中找到对应实现，本轮不是只补报告或标志位。经复核，以下内容可以接受：

- fixed DKUC one-step 已使用独立冻结网络；
- full smoke 的最终 `A/B/C` 与最终 encoder 对全部离线历史和已消费 stream 的直接 ridge refit 完全一致；
- online best-state 已改为在 `optimizer.step()` 后重新计算 loss；
- `online_epochs <= 0` 在 full 模式下会被拒绝；
- 时变数据入口可以按仓库要求使用 `python -m`；
- 累计矩阵诊断增加了正则系统 rank 和 condition number；
- 非有限回滚、batch 划分、递推状态续算和 trial 隔离测试得到扩充；
- DKTV one-step、snapshot fixed-horizon 和 batch rollout 均已保存原始数组及指标；
- review02 smoke 中全部数组有限，所有 JSON RMSE 都能从保存数组精确重算。

但是，反馈报告把“fixed DKUC 完全冻结”和 resume 风险描述得偏乐观：fixed-horizon 仍使用在线训练后的 encoder；full resume 在下一批更新后会抛弃 checkpoint 中此前累计的在线样本。前者会直接污染相应比较结果；后者只在使用 resume 时触发。

因此当前不必等待全部工程验收项完成，可以进入受约束的探索性完整运行：

```text
允许：单次不中断、单 trial 独立执行的 exploratory full_run
暂不允许：使用 resume、报告 fixed-horizon DKUC、宣称实时性或形成论文定论
```

### 1.1 两级准入原则

后续不再把全部验收条件都设为完整运行的前置门槛，而是分为两级：

1. **完整运行启动门槛**：只保留会直接改变本轮预测结果、方法身份或数据对齐的条件；
2. **论文级最终验收**：包含恢复运行、异常事务、完整资源统计、自动化因果证明和物理安全边界等增强条件。

在当前状态下，满足以下约束即可先进入完整运行：

- 每个 trial 使用独立命令从初始 artifact 开始，不使用 `--resume_state` 和
  `--resume_model`；
- fixed-horizon DKUC 串扰先修复；若暂不修，则该指标不生成、不比较、不写入
  结论，只保留已经正确冻结的 fixed DKUC one-step；
- full DKTV、frozen encoder 和 fixed DKUC 分别运行，不能在同一进程中共享
  在线更新状态；
- 运行结束检查有限值、joint limit、最终坐标一致性、指标可重算和产物完整性；
- 结果统一归档到 `prediction/outputs/full_run/`，并标记为 exploratory，暂不
  作为论文最终表格。

## 2. 上轮问题逐项复核

| `plan_review_01.md` 项目 | 本轮状态 | 复核结论 |
|---|---|---|
| fixed DKUC 基线污染 | 部分解决 | one-step 已完全冻结；fixed-horizon 仍使用在线 encoder |
| 时变采集模块入口 | 已解决 | `-m traj_data.collect_data_time_varying --help` 成功 |
| full 的 `A/B/C/theta` 坐标一致性 | 主路径已解决 | final smoke 与最终 encoder 的直接 refit 完全一致；仍缺自动化 oracle、原子回滚和正确的 post-refit 诊断 |
| best loss 与 best-state 错位 | 已解决 | 更新后 loss 决定权重，final loss 可复现 best loss |
| `online_epochs=0` | 已解决 | 训练函数和 full replay 均拒绝零 epoch |
| 因果性、隔离和恢复 | 部分解决 | trial isolation 有测试；未来篡改 causality 测试仍缺；full resume 存在实际数据丢失 |
| 数据安全与成对数据 | 未解决 | 反馈报告已如实保留该缺口 |
| 递推公式测试覆盖 | 大部分解决 | batch 划分、固定坐标续算、完整非有限回滚已补；病态有限输入和 full 坐标重建 oracle 仍缺 |
| DKTV rollout | 主体已解决 | DKTV 两类 rollout 已生成且可重算；fixed-horizon 基线仍有串扰 |
| 三方法正式比较与阶段 6 | 未开始 | 与反馈报告一致 |

## 3. 仍需优先整改的问题

### P1（可后置）：full resume 会丢失 checkpoint 中已经累计的在线样本

位置：`prediction/dktv_prediction.py:713-715`、`prediction/dktv_prediction.py:737-739`、`prediction/dktv_prediction.py:780-787`。

full 模式为了使最终矩阵和 `theta*` 坐标一致，会从物理状态重新 lifting 并直接重建统计量。正常单次运行时，`coordinate_x/y/u` 包含离线历史和当前 trial 已接受的所有批次，因此结果正确。

恢复运行时，`resume_state` 虽然包含此前在线样本对应的累计统计量，但新的 `coordinate_x/y/u` 只从离线 `history_dataset` 初始化。checkpoint 又没有保存此前在线物理转移。下一次 encoder 更新后执行全量坐标重建时，此前在线样本便从新统计量中消失。

本次用 review02 的 final checkpoint 继续运行一个 4-sample 批次，得到：

```text
resume 前 sample_count：       16008
正确的下一批 sample_count：   16012
实际重建后的 sample_count：   16004
update_index：                 2 -> 3
```

这说明当前问题不只是“尚未校验 fingerprint”，而是 resume 的数值语义已经错误：版本号继续增加，但累计样本反而减少。连续运行与中断恢复必然不等价。

整改要求：full checkpoint 必须能够恢复用于坐标重建的全部物理转移，或者保存可验证的数据源、过滤规则和精确消费位置并在恢复时重放。完成前不应把 `--resume_state/--resume_model` 描述为可继续 Hao-DKTV 的正式接口。

该问题不阻塞单次不中断的完整运行，但完整运行期间必须明确禁用 resume。后续
若需要长任务断点恢复，再将其提升为前置修复项。

最终验收测试：将同一 stream 分成两段，比较连续运行与 checkpoint 恢复运行的 `A/B/C/P/statistics/theta/version/prediction`，必须在明确容差内一致。

### P0：fixed-horizon DKUC 仍受到在线 encoder 污染

位置：`prediction/dktv_prediction.py:702-707`、`prediction/dktv_prediction.py:745-748`、`prediction/dktv_prediction.py:907-914`。

replay 内部创建的 `fixed_network` 正确修复了 one-step。但该冻结副本没有返回给主程序。主程序生成 fixed-horizon rollout 时仍调用：

```text
snapshot_rollout_predictions(model, ...)
```

其中 `model` 已经过 full 在线训练，`snapshot_rollout_predictions()` 又通过 `model.lift()` 计算窗口起点。因此 fixed-horizon 使用了更新后的 encoder，而不是原始 DKUC encoder。

用同一 artifact 重新加载真正固定的 DKUC 复算 review02 smoke：

```text
horizon 2 保存数组最大绝对偏差：7.291377933454912e-06
horizon 4 保存数组最大绝对偏差：1.4677373089105883e-05
```

smoke 偏差小不改变其结构性错误。`plan_feedback_02.md` 中“fixed DKUC 基线完全冻结”的结论应改为“fixed one-step 已冻结，fixed-horizon 待修复”。

整改要求：固定基线的 one-step 和所有 rollout 必须共享同一个不可变 DKUC 对象或缓存的原始 lift；新增测试同时覆盖 one-step 与至少一个 horizon rollout。

### P2（可后置）：post-refit 诊断与实际提交的坐标一致模型不对应

位置：`prediction/dktv_prediction.py:762-789`。

`state.update()` 先产生旧 encoder 坐标下的诊断并写入 `record`；训练 encoder 后，代码用 `_consistent_state_from_physical_history()` 创建了新的 `state`，却没有重算 `record["diagnostics"]`。随后仅增加：

```text
coordinate_consistent = true
coordinate_refit_sample_count = ...
```

因此同一条记录中的“坐标一致”标志描述 post-refit state，而矩阵范数和条件数描述 pre-refit state。review02 最后一批即可看到差异：

```text
record A spectral norm：15.110106329538505
final  A spectral norm：15.097883405828693

record regularized condition：10244255.411199158
final  regularized condition：10243788.298897967
```

该问题不改变实际提交模型和预测数组，可以在探索性完整运行后整改。但当前
condition number 和矩阵范数只能作为近似诊断，不能作为最终模型的精确证据。

最终整改要求：分开保存 `pre_theta_update` 与 `post_coordinate_refit` 两套诊断，主要模型证据必须引用实际提交的 post-refit state，不能只设置布尔标志。

### P2（可后置）：full 坐标重建尚未形成原子事务

位置：`prediction/dktv_prediction.py:766-789`。

训练失败时已有 encoder/state 回滚；但训练成功后调用 `_consistent_state_from_physical_history()` 没有放在回滚保护中。如果重新 lifting、求解 ridge 或创建状态时失败，encoder 已经改变，函数直接抛异常退出，无法记录 rejected update，也没有恢复上一完整模型。

成功完成且产物通过有限值检查的运行不受该缺口影响。探索性完整运行中若出现
relift/refit 异常，应将该 run 标记为失败并重新运行，不能接受部分产物。

最终整改要求：把“矩阵候选更新、encoder 训练、物理历史重建、post-refit 有限值检查”作为一个事务；任一步失败都恢复 batch 前 `A/B/C/statistics/theta/version` 并记录失败原因。

### P2（可后置）：当前耗时统计没有覆盖 full 坐标重建

反馈报告已经说明 full 模式每批会重新 lifting 全部已消费历史。在 review02 smoke 中，每批重建约 16000 个样本，这很可能是主要计算开销。但目前：

- `update_time_s` 只统计训练前的 Woodbury/matrix update；
- `training_time_s` 只统计当前小批次 encoder 训练；
- 全历史 lifting 和 direct ridge refit 没有计时；
- `mean_training_time_s` 不能代表一次 full DKTV 更新的总时间。

这套实现保证了坐标一致性，但不能再直接使用当前计时支撑 Hao 递推效率或实时性结论。

该问题不影响预测精度比较，可以先运行并保存墙钟总耗时，但本轮结果不能用于
“满足实时性”或“计算效率优于其他方法”的结论。

最终整改要求：至少分开记录 `matrix_preupdate_time`、`encoder_training_time`、`coordinate_relift_time`、`coordinate_refit_time` 和 `total_transaction_time`，并保存最大值与均值。

### P2（可后置）：阶段 4 的因果性和 checkpoint 身份仍未闭环

反馈报告已经如实承认：

- 尚无“篡改未来观测不影响过去输出”的 causality 测试；
- state/model 虽强制同时提供，但没有同源 fingerprint；
- normalizer、配置、数据位置和消费进度未绑定；
- 多 trial 仍只保存最后一个 trial 的 final state/model。

其中成对 CLI 只能防止漏传一个文件，不能防止两个文件来自不同 run。阶段 4
仍不能通过最终验收，但只要本轮不使用 resume、每个 trial 独立运行，该部分不
阻塞探索性完整运行。

### P1（探索运行可放宽）：时变数据安全与确定性验收仍未完成

该项与反馈报告一致，上一轮 smoke 的峰值仍为：

```text
peak_abs_tau       = 1331.0948588941503
peak_cable_tension = 3030.391572992118
```

没有执行器/缆索允许范围，就不能把“有限且未越关节限位”等同于“物理上已
验收”。但当前为 MuJoCo 探索性预测实验，可以先以“数组有限、无 joint limit
越界、无仿真崩溃、峰值完整记录”为最低门槛进行 full run。此类结果只能用于
筛选参数和观察趋势；无扰动/有扰动成对数据、确定性测试及执行器阈值应在论文
正式结果前补齐。

### P2：代码注释和实际 full 策略存在残留不一致

`HaoDKTVState` 的 docstring 仍称 full 模式保留旧特征坐标的历史统计，而当前 full 主路径实际会对全部物理历史重新 lifting，并在 manifest 中记录 `historical_coordinate_approximation=false`。这不会改变数值结果，但会误导后续维护者。

整改要求：把“固定 encoder 的递推状态”和“full 模式的全历史坐标重建策略”分别描述清楚，并明确后者的计算复杂度属于当前工程实现。

## 4. 已通过的复核证据

### 4.1 环境、入口和测试

指定解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

复核结果：

```text
py_compile：通过
prediction DKTV 模块 --help：通过
时变数据采集模块 --help：通过
全仓库 pytest：9 passed in 1.79s
```

9 个测试比上一轮的 6 个有实质扩充。不过仍没有 causality、pipeline resume、fixed-horizon 冻结基线和 full post-refit oracle 测试。

### 4.2 full 最终坐标一致性

审阅产物：

```text
/tmp/dktv_review02b/dktv/20260828_113944_dktv_review02/
```

重新加载 `final_dktv_model.pt`，对离线历史和完整 8-step stream 重新 lifting、直接 ridge refit 后，以下对象与 `final_dktv_state.npz` 的最大绝对偏差均为 0：

```text
A, B, C,
gram_chi, cross_chi, gram_g, cross_g,
P_chi, P_g
```

`sample_count=16008`、`update_index=2`、`encoder_version=2` 也与本次单段 smoke 一致。这证明 full 单次连续运行的最终坐标重建确实生效。

### 4.3 best-state、数组与指标

- 两次 encoder 更新的 final loss 均与保存的 `best_loss` 一致；
- 全部 23 个保存数组均为有限值；
- one-step、fixed-horizon h2/h4、DKTV snapshot h2/h4 和 batch h4 的 RMSE 均可从数组精确重算；
- 反馈报告列出的 DKTV 指标与 `metrics.json` 一致；
- 代表性 one-step PNG 可以正常打开，四个状态曲线清晰可辨。

这些证据证明产物内部一致，但 fixed-horizon 数组内部可重算不等于其 fixed DKUC 方法身份正确。

## 5. 当前阶段状态

| 阶段 | 审阅状态 | 本轮变化 | 剩余阻塞项 |
|---|---|---|---|
| 阶段 0：契约与基线 | 部分完成 | one-step fixed 基线已修；模式契约清楚 | fixed-horizon 仍受在线 encoder 串扰 |
| 阶段 1：时变数据 | 部分完成 | 标准模块入口已修 | 安全阈值、成对数据、确定性和可视化验收缺失 |
| 阶段 2：A/B/C 递推 | 基础验收基本完成 | batch 划分、固定坐标续算、完整非有限回滚和正则诊断已补 | 病态有限输入；full post-refit 诊断/oracle 尚缺 |
| 阶段 3：在线 theta | 主路径基本完成 | best-state、正 epoch、连续 smoke 最终坐标一致性已修 | refit 原子回滚和完整耗时缺失 |
| 阶段 4：因果/隔离/恢复 | 最终验收未通过，不阻塞受约束 full run | trial isolation 测试、CLI 成对参数已补 | 本轮禁用 resume；causality、身份绑定、连续/恢复等价可后置 |
| 阶段 5：评估产物 | 部分完成 | DKTV batch 与 fixed-horizon rollout 已补且可重算 | fixed rollout 身份错误；三模式、多 trial、完整资源统计缺失 |
| 阶段 6：完整运行 | 可开始探索性运行 | 单段主链路、坐标一致性和主要指标已具备 | 修复或禁用错误 fixed-horizon 指标；论文定论仍等待最终验收 |

### 5.1 DKTV 结果归档要求

为了与 EDMD、DKAC、DKUC 等预测方法保持一致，并方便后续快速查找，DKTV
通过验收的冒烟测试和后续完整运行结果必须分类保存在仓库内的
`prediction/outputs/`，不得只保留在 `/tmp` 或另建一套输出树。

统一目录为：

```text
prediction/outputs/
  smoke_test/
    dktv/
      <run_id>/
        manifest.json
        metrics.json
        prediction_arrays.npz
        update_history.json
        training_history.json
        final_dktv_state.npz
        final_dktv_model.pt
    figures/
      <same_run_id>/
        *.png
        *.pdf

  full_run/
    dktv/
      <run_id>/
        manifest.json
        metrics.json
        prediction_arrays.npz
        update_history.json
        training_history.json
        final_dktv_state.npz
        final_dktv_model.pt
    figures/
      <same_run_id>/
        *.png
        *.pdf
```

该结构已经由 `create_prediction_run_paths()` 支持。正式保留的运行应使用默认
输出根目录，并通过 `--run_type smoke_test` 或 `--run_type full_run` 分类；不要
再使用 `--out_dir /tmp/...` 作为最终产物位置。

归档规则：

- `/tmp` 仅用于开发中的一次性诊断，允许随时消失，不得作为执行报告中的最终
  可追溯结果；
- 每次通过验收的 smoke 必须在
  `prediction/outputs/smoke_test/dktv/<run_id>/` 重新生成；
- 后续论文参数或多 seed 正式运行必须进入
  `prediction/outputs/full_run/dktv/<run_id>/`；
- artifact 与 figures 使用完全相同的 `<run_id>`，通过 manifest 互相定位；
- `full` 和 `frozen_encoder` 仍分别独立运行，建议在 `--tag` 中包含模式、场景
  和 seed，例如 `full_sine_mid_seed50`、`frozen_sine_mid_seed50`；
- 不把多次运行覆盖到同一个目录，也不手工复制同一图像到多个结果目录；
- `prediction/outputs/` 继续保持 Git 忽略，目录用于本地实验归档，不提交生成
  模型、数组和图像。

本次审阅使用的
`/tmp/dktv_review02b/dktv/20260828_113944_dktv_review02/` 可以保留为临时核查
证据，但它不满足长期归档要求。fixed-horizon 串扰修复或明确禁用该指标后，
应在上述 smoke 目录重新运行一次，随后即可进入 exploratory full run，并在
下一份 feedback 中记录准确的 run 目录。

## 6. 下一轮整改顺序

### 第一优先级：完成完整运行启动门槛

1. 让 fixed one-step 和 fixed-horizon 共用不可变原始 DKUC；
2. 增加 fixed-horizon 冻结测试；若本轮不修，则禁用和排除该指标；
3. 用默认输出目录重新生成一次验收 smoke；
4. 以单次不中断、单 trial 独立命令开始 exploratory full run。

启动门槛不要求先完成 resume、checkpoint manifest、异常事务和精细耗时。

### 第二优先级：完整运行期间的最低检查

1. 检查 states、predictions、loss、`A/B/C/P` 和全部指标有限；
2. 检查 joint limit、峰值力矩和峰值缆索张力并原样记录；
3. 从数组重算 one-step、batch rollout 和 DKTV fixed-horizon 指标；
4. 检查最终 encoder checksum 和 final coordinate consistency；
5. 记录完整命令、seed、数据、artifact、Git 和墙钟总耗时。

### 第三优先级：论文最终验收前补齐

1. 修复 full resume 并增加连续/恢复等价测试；
2. 增加未来观测篡改 causality 测试；
3. 建立 checkpoint manifest、错误配对拒绝和逐 trial checkpoint；
4. 完善 post-refit 诊断、事务回滚和分项耗时；
5. 推导执行器和缆索安全阈值，完成确定性与成对数据测试；
6. 根据探索性 full run 选定参数后，再生成多 seed 论文正式结果。

## 7. 探索性完整运行最低清单

- [ ] fixed DKUC one-step 与新加载原始 artifact 一致；fixed-horizon 则在“修复
  并验证一致”与“本轮禁用且不报告”两种处理方式中明确选择一种；
- [ ] 每个 trial 从初始 artifact 独立、不中断运行，不传入任何 resume 参数；
- [ ] 输入数据有限、无 joint limit 越界，力矩和缆索张力峰值已记录；
- [ ] 运行后全部预测、矩阵、loss 和指标有限；
- [ ] DKTV 指标可从保存数组重算，最终模型通过坐标一致性检查；
- [ ] 命令、seed、数据路径、artifact、Git 信息和墙钟总耗时已保存；
- [ ] 验收 smoke 的 artifact 和同名 figures 已归档到
  `prediction/outputs/smoke_test/` 对应分类目录；
- [ ] 探索性完整运行已归档到 `prediction/outputs/full_run/dktv/<run_id>/`，而非
  `/tmp` 或其他临时目录；
- [ ] manifest 或报告明确标记 `exploratory`，不宣称 resume、实时性、物理安全
  或论文最终结论。

以下条件移至论文最终验收，不再阻塞本轮完整运行：resume 等价、checkpoint
身份绑定、post-refit 精确诊断、异常原子回滚、分项耗时、自动化 causality、
逐 trial checkpoint、执行器物理阈值和确定性成对数据。

## 8. 最终结论

第二轮整改已经把 Hao-DKTV 从“能运行的原型”推进到“主要单段在线链路基本可信”：特别是 one-step 固定基线、best encoder 和最终坐标一致性都有了比上一轮更强的证据。

当前可以先进入探索性完整运行，不需要等待所有工程完备项。唯一会直接污染本轮比较数值的是 fixed-horizon 固定基线串扰：优先修复最稳妥；若暂时不修，就必须从本轮指标和结论中排除。full resume 的问题通过“不使用 resume、每个 trial 独立运行”即可隔离，其余诊断、事务、耗时和安全增强项安排在论文最终验收前完成。
