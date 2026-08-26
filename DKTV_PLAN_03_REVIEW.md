# DKTV Plan 03 执行审查报告

> 审查对象：`DKTV_PLAN_03_EXECUTION_FEEDBACK.md`
>
> 对照计划：`DKTV_PLAN_03_ZHANG_SLIDING_WINDOW.md`
>
> 审查日期：2026-08-26
>
> 分支：`OTVDKL`
>
> 审查结论：**算法实现与 noncanonical 工程实验有条件通过；预测控制 Gate 正确判定为未通过。修复产物哈希、聚合 provenance 和计时合同后，方可重跑 canonical 结果。**

## 1. 总体结论

本轮核查没有发现会推翻当前主要数值结论的错误。固定窗口、选择性状态机、因果
重放、10-seed 配对统计以及长 rollout 负结果，均能由当前代码和保存产物复核。

当前可以接受的结论是：

- `otvdkl_window` 在当前五轨迹同步数据流上显著改善 one-step 和 10/20-step
  rollout；
- 该优势没有延伸至 50/100-step rollout，因此不能进入控制实验，也不能宣称
  长时域稳定；
- selective 方法总体优于累积方法的 one-step，但弱于基础 window，且阈值与拒绝
  策略证据仍需谨慎解释；
- 当前运行全部是 `accepted_noncanonical`，只能作为开发阶段工程证据。

需要修改的核心问题有三个：

1. 10 个 full run 的 `logs/run.log` 哈希均与 manifest 不一致；
2. Plan 03 聚合器的 canonical 判定忽略聚合时自身 dirty 状态，也没有保存聚合器源码
   provenance；
3. window 的 `update_time_s` 同时包含 Woodbury 递推和 direct-refit oracle，尚未完成
   计划要求的递推与直接拟合耗时对比。

## 2. 独立验证结果

### 2.1 环境与自动化检查

使用仓库指定解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
Linux
```

独立执行结果：

| 检查 | 结果 |
| --- | --- |
| Plan 03 相关文件 `py_compile` | 通过 |
| `experiments.dktv.plan_03 --help` | 通过 |
| `experiments.dktv.aggregate_plan_03 --help` | 通过 |
| Plan 03 + Plan 02 回归测试 | `25 passed in 10.19s` |
| 全仓测试 | `38 passed in 14.83s` |
| final 聚合少于 10 seed | 按预期拒绝，且未创建输出目录 |
| `git diff --check` | 通过 |

测试数量和通过状态与执行反馈一致；运行时长差异属于正常主机负载波动。

### 2.2 10-seed 产物与 provenance

复核的 source runs 为：

```text
20260826_plan03_full_gatefinal_seed20260825
...
20260826_plan03_full_gatefinal_seed20260834
```

独立检查结果：

- 10 个 Plan 01 seed 均不同；
- 10/10 Plan 03 engineering acceptance 为 true；
- 150/150 个 source provenance 哈希与当前仓库文件一致；
- aggregate 中记录的 10/10 个 source manifest 哈希一致；
- 结果文件哈希为 170/180 匹配；10 个失配项全部是各 run 的
  `logs/run.log`；
- aggregate NPZ 含 38 个数组，全部 finite，无 object 数组，不需要 pickle；
- 独立重新聚合得到的 metrics 与原 aggregate 完全一致。

独立重聚合输出：

```text
outputs/results/dktv/plan_03/20260826_plan03_aggregate_final_review_round1
```

已打开检查 `paired_one_step_difference_ci.png`，散点、均值、95% Student-t CI、零线
和标签均正常。

### 2.3 更新状态、窗口和递推

10 个 seed 的主方法合计：

| 方法 | accepted | rejected | skipped | failed |
| --- | ---: | ---: | ---: | ---: |
| window `w100,b10` | 1200 | 0 | 0 | 0 |
| selective | 250 | 5 | 945 | 0 |

retain 策略另有 2 次真实 rejection。主 window 和主 selective 各发生 42 次
`direct_refit_fallback`；全部 base window/selective 消融合计 378 次，与执行反馈一致。

进一步确认：

- 主方法固定窗口内存检查全部通过；
- 最大 raw recursive `A` 差为 `2.021539e-8`；
- fallback 后最大 candidate `A` 差为 `9.983799e-9`；
- 所有预测 Gate 均为 false；
- 四状态均由单元测试覆盖，真实 full stream 中没有 numerical failure。

## 3. 主要数值结论复核

独立重算的 one-step 均值与 aggregate 完全相同：

| 方法 | one-step RMSE mean |
| --- | ---: |
| fixed DKO | `0.0047948954` |
| accumulative | `0.0033174370` |
| window | `0.0023809613` |
| selective | `0.0030025736` |

window 相对 accumulative 的配对改善为：

```text
mean difference = +0.000936476
95% CI          = [+0.000707550, +0.001165401]
wins             = 10/10
```

selective 相对 accumulative：

```text
mean difference = +0.000314863
95% CI          = [+0.000147971, +0.000481756]
wins             = 9/10
```

rollout 结论也与反馈一致：window 在 horizon 10/20 较好，在 horizon 50/100 明显
恶化，且长时域为 0/10 seed 优于 accumulative。因此不执行可选 MPC 是正确决定，
不是 Plan 03 的漏做项。

## 4. 待修改问题

### P1-01：full run 的 `run.log` 哈希在 manifest 保存后失效

`experiments/dktv/plan_03.py` 先在 `result_files` 中计算 logs 哈希并保存 manifest，
之后又调用 `log(...)` 追加最终状态。结果是 10 个 gatefinal run 的 `run.log` 全部与
manifest 中记录的 SHA-256 不一致。

影响：

- 数值 JSON/NPZ/figure 哈希均未受影响；
- 但 manifest 无法完整验证自己的结果树，不能作为 canonical 可追溯证据。

建议：先写入最终日志，再生成 `result_files` 快照和 manifest；增加集成测试，逐个
重算 manifest 中所有结果文件哈希。修复后应重新生成正式 full runs，不能直接修改
旧 manifest 来伪装原运行完整。

### P1-02：aggregate canonical 判定和自身 provenance 不完整

`experiments/dktv/aggregate_plan_03.py` 当前只依据 source runs 的 `canonical` 字段
决定 aggregate 是否 canonical；虽然记录了聚合时 Git 状态，却没有把
`git.dirty` 纳入判定。同时 aggregate manifest 没有保存
`aggregate_plan_03.py` 自身及相关统计实现的源码哈希。

当前 source runs 本身均 noncanonical，所以本次 aggregate 状态仍然正确；但若未来
输入均 canonical、聚合器工作树有未提交修改，它会错误地产生
`accepted_canonical`。

建议：

- `canonical = all(source canonical) and not aggregate_git.dirty and acceptance passed`；
- 保存 aggregate source provenance，至少包含聚合器、统计依赖、配置与 AGENTS；
- 显式保存 `canonical_blockers`；
- 聚合时校验 source manifest 所记录的关键结果哈希，而不只校验 manifest 文件本身。

### P1-03：尚未形成 recursive 与 direct-refit 的独立耗时证据

`SlidingWindowKoopmanUpdater.propose()` 每个候选都会先执行 direct refit，再比较
Woodbury 结果；外层 `update_time_s` 覆盖整个过程。因此反馈中的约 `0.540 ms`
是“递推 + direct oracle + 判定/可能 fallback”的总耗时，不能解释为纯递推部署耗时。

这不影响矩阵一致性和预测指标，但 Plan 03 Step 02 中“比较递推与 direct refit 的
耗时”目前只完成了一部分。

建议分别记录：

```text
recursive_candidate_time_s
direct_refit_oracle_time_s
fallback_time_s
total_update_time_s
```

论文中若讨论在线计算效率，应使用分离后的口径，并明确 oracle 是否属于部署路径。

### P2-01：fallback 应重新锚定完整窗口统计量

fallback 当前将候选模型替换为 direct-refit 结果，但 `candidate_gram` 和
`candidate_cross` 仍来自连续加减的累计统计量，inverse 也由该累计 Gram 重新求逆。
这能保证当前输出模型通过 oracle 容差，却没有完全消除长期浮点漂移。

建议 fallback 时从 `candidate_z/candidate_u/candidate_next` 重新计算 `G/H/P`，并新增
长重放测试，验证 fallback 后统计量与 raw-window direct statistics 同步。此项属于
数值稳健性强化，不推翻当前结果。

### P2-02：论文级 selective 阈值应避免同流标定与评价

当前 epsilon 使用同一 validation stream 的 nominal 段真实下一状态误差标定，随后又
在该 stream 上评价 selective。该做法符合当前工程配置，但用于论文时存在调参与评价
数据重叠。

建议把 epsilon 冻结在独立 calibration split/seed 上，再在未参与标定的 test seeds
上评价。基础 window 不依赖 epsilon，因此其主结论不受此项影响。

### P2-03：噪声消融应同时保留 clean truth

当前噪声场景直接覆盖 `stream["states"]`，训练输入和评价 truth 都变成 noisy state。
这可以描述“对观测序列的一致性”，但不足以单独回答“面对测量噪声时对真实物理状态
的预测能力”。

建议同时保存 `states_observed` 与 `states_clean`：在线更新使用 observed，最终
prediction RMSE 同时对 observed 和 clean truth 计算。多 seed aggregate 也应补充
噪声/速率场景的 rollout，而不只汇总 one-step。

### P2-04：递推 finite 检查可进一步收紧

递推路径当前主要检查 direct diagnostics、candidate inverse 和差值阈值。建议显式
检查 recursive `A/B/theta` 及差值本身均 finite，避免 NaN 在比较表达式中绕过
`> tolerance`。增加极端尺度和溢出输入测试即可。

## 5. 计划验收项对照

| Plan 03 验收项 | 结论 | 备注 |
| --- | --- | --- |
| recursive 与 direct-refit 容差 | 通过 | 允许并记录 fallback；需强化统计重锚定 |
| 精确移入/移出与固定窗口 | 通过 | 单元测试和真实 replay 均通过 |
| 固定内存 | 通过 | 主方法及消融均通过 |
| 四状态测试覆盖 | 通过 | full stream 无 numerical failure，失败状态由测试覆盖 |
| 两种 reject policy | 通过 | 参数化测试，且 full runs 有真实 rejection |
| 四方法共享数据/初始模型/评价器 | 通过 | manifest 合同和源码复核一致 |
| one-step/rollout/分段/延迟/耗时/内存 | 有条件通过 | 递推与 oracle 耗时尚未拆分 |
| `w/b/epsilon`、noise、rate 消融 | 工程通过 | 论文级 clean-truth 与跨 seed rollout 汇总待补 |
| 不少于 10 seed | 通过 | 10 个 distinct Plan 01 seeds |
| 结果可追溯性 | 未完全通过 | 10 个 `run.log` 哈希失配 |
| 预测 Gate 后可选控制 | 正确跳过 | 50/100-step Gate 失败 |
| 测试与 `git diff --check` | 通过 | `38 passed`，diff check 通过 |

## 6. 建议修改顺序

1. 修复 `run.log` 写入顺序和全结果树哈希集成测试；
2. 修复 aggregate canonical/provenance 合同；
3. 拆分 recursive/direct/fallback 计时；
4. fallback 时从 raw candidate window 重建 `G/H/P`，补极端 finite 测试；
5. 用 smoke/development runs 验证以上合同；
6. 固定 Git 版本并重新生成 Plan 01、Plan 02、Plan 03 canonical runs；
7. 在解决 50/100-step 恶化前，不进入 MPC；优先检查谱半径、长 rollout 稳定性、
   window/ridge 选择和多步目标；
8. 论文实验阶段再引入独立 epsilon calibration 与 clean-truth noise evaluation。

## 7. 最终 Gate

当前允许：

- 保留本批结果作为 noncanonical 开发证据；
- 基于 one-step 与短 rollout 结论继续诊断窗口长度、ridge 和长时域稳定性；
- 修改并补齐 provenance、计时和数值稳健性合同。

当前不允许：

- 将 gatefinal runs 描述为 canonical 或论文最终结果；
- 将 `0.540 ms` 解释为纯 Woodbury 递推耗时；
- 依据 one-step 优势进入 MPC 或宣称 Zhang 方法已获得稳定性保证；
- 把当前噪声表直接解释为相对 clean physical truth 的鲁棒性结论。
