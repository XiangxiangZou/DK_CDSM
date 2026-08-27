# DKTV Plan 02 Review 修正反馈

> 对应审核：`DKTV_PLAN_02_REVIEW.md`
>
> 修正日期：2026-08-25
> 结论：**审核提出的代码与审计缺口已修正；核心实现及主预测实验完成，消融实验待补；结果仍为 noncanonical。**

## 1. 修正结论

本轮没有改写或删除原有 Plan 02 结果，而是以新 run-id 重跑 smoke、10 个 full run
和 final aggregate。递推公式与主要预测数值未发生变化，新增的是完整失败保护、更新
历史 schema、聚合比较 Gate、provenance、paired statistics 和实验语义披露。

| Review 项 | 状态 | 本轮处理 |
| --- | --- | --- |
| P0-01 canonical 基线 | 未关闭 | 未经授权不执行 commit；新结果继续标记 `accepted_noncanonical`，阻断项为 dirty 工作树和上游 Plan 01 noncanonical |
| P0-02 aggregate Gate | 已修正 | 强制校验 config/source hash、batch、坐标、stream、horizon、stage、Plan 01 schema/scenario、history schema 和 finite metrics；记录 aggregate Git 与源码 hash；final 至少 10 seed |
| P1-01 rejected replay | 已修正 | 坏批次丢弃、不重试、不加入统计量或 oracle history；保留最近模型并继续回放；增加真实模型 NaN replay 测试 |
| P1-02 update history | 已修正 | schema v2 NPZ 保存 accepted 和完整 recursive 数值诊断；JSONL 保存 reason、决策和嵌套 diagnostics；`allow_pickle=False` 可读 |
| P1-03 “Three seeds” | 已修正 | 新增 `development/final` profile；解释文本按 profile 和实际 seed 数动态生成 |
| P1-04 batch 物理语义 | 已修正 | manifest、公式映射均注明 5 条同步仿真轨迹；`b=5/10/20` 对应 1/2/4 个仿真步 |
| P1-05 内存口径 | 已修正 | 分别报告常量 updater statistics memory 与增长的 oracle history memory；recursive/oracle 时间分开 |
| P1-06 实验矩阵 | 部分完成 | zero-noise、中速正弦扰动和三阶段主实验完成；low-noise、slow/fast 消融待补；MPC 仍为可选项 |
| P2-01 paired statistics | 已修正 | 保存逐 seed difference/ratio、均值、样本标准差、95% Student-t CI、win/tie count 及 NPZ 数组 |
| P2-02 方法命名 | 已修正 | 图例与 manifest 使用 `Hao-style accumulative DKTV (fixed encoder, b=...)` |

## 2. 关键实现变化

- `run_accumulative_replay()` 对 non-finite batch 执行
  `discard_invalid_batch`。rejected update 的 model/statistics/sample count 不变，oracle
  不接收坏样本，后续合法批次继续更新。
- 每条更新增加 `attempt_index`、`accepted`、`reason`、batch disposition、recursive
  rank/condition/spectral radius/finite、oracle 是否执行以及两类内存字段。
- `update_summary()` 可汇总 `diagnostics=None` 的 rejected record，并分开报告 recursive
  update 与 direct-refit oracle 时间。
- aggregate 在写结果前执行 `comparison_contract`；任一契约项不一致或出现非有限指标
  就拒绝聚合。`final` profile 少于 10 个不同 seed 也会直接失败。
- Plan 01 scenario contract 固化 manifest/artifact schema、profile、状态和输入、采样周期、
  lift/readout、坐标及扰动定义，供跨 seed 比较。

## 3. 新产物

主 full run（seed `20260825`）：

```text
outputs/results/dktv/plan_02/20260825_plan02_full_review2b_seed20260825/
```

其余 full runs 使用同一前缀，seed 为 `20260826` 至 `20260834`。当前源码对应的 smoke
run 为：

```text
outputs/results/dktv/plan_02/20260825_plan02_smoke_review_round2b/
```

最终 10-seed 聚合：

```text
outputs/results/dktv/plan_02/20260825_plan02_aggregate_final_review_round2b/
```

其中重点文件为：

```text
manifest.json
metrics/multiseed_summary.json
arrays/multiseed_metrics.npz
figures/multiseed_one_step_rmse.png
```

单次 full run 新增的更新审计文件为：

```text
arrays/update_history.npz
arrays/update_history_schema.json
logs/update_history.jsonl
```

## 4. 数值结果

10 个 seed、每 seed 420 次更新，共 4200 次更新全部接受。recursive 与 full-history
direct refit 的全 seed 最大 `A` 绝对差为：

| batch | 最大 `A` 差 |
| --- | ---: |
| `b=5` | `2.007867483388992e-11` |
| `b=10` | `1.5817014364927218e-11` |
| `b=20` | `1.4827361560776353e-11` |

均远小于 `1e-8` oracle 容差。one-step RMSE 的 10-seed 结果为：

| 方法 | mean ± sample std | paired mean `fixed-method` | 95% Student-t CI | wins |
| --- | ---: | ---: | ---: | ---: |
| Fixed DKO | `0.00479490 ± 0.00131659` | — | — | — |
| Hao-style, `b=5` | `0.00328081 ± 0.00035692` | `0.00151408` | `[0.00079626, 0.00223191]` | `10/10` |
| Hao-style, `b=10` | `0.00331744 ± 0.00036164` | `0.00147746` | `[0.00076397, 0.00219095]` | `10/10` |
| Hao-style, `b=20` | `0.00336360 ± 0.00037039` | `0.00143129` | `[0.00072459, 0.00213799]` | `10/10` |

负面结果仍保留：50-step rollout 的 fixed mean 为 `0.06609152`，而 `b=5/10/20`
分别为 `0.06911809`、`0.06936093`、`0.06973346`，累积方法在该窗口略差。

主 full run 每个 updater 的统计量内存为常量 `4896 bytes`。验收用 oracle history
由 `391680 bytes` 增至 `718080 bytes`，未再把它描述为部署方法的常量内存。

## 5. 验证记录

使用解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

完成的检查：

- 配置解释器验证：通过；
- `py_compile`：通过；
- Plan 02 和 aggregate CLI `--help`：通过；
- Plan 02 focused tests：`12 passed`；
- 全量测试：`25 passed in 8.97s`；
- 真实 Plan 01 模型的 non-finite replay 注入测试：通过，坏批次被记录并丢弃，后续更新继续；
- final profile 仅给 2 seed：按预期拒绝，错误为至少需要 10 个不同 seed；
- final 10-seed `comparison_contract`：11 项检查全部为 `true`；
- 当前源码 hash 与 aggregate reference source hash：全部一致；
- NPZ 使用 `allow_pickle=False` 读取：通过；
- 聚合 PNG：`2160 x 900`，已检查内容；
- `git diff --check`：通过（见最终交付检查）。

## 6. 剩余风险与后续条件

1. 当前不能作为 canonical 论文基线。需要先由用户决定固化/提交版本，再从 canonical
   Plan 01 多 seed artifact 重跑；本轮未执行 commit、push 或覆盖旧产物。
2. low-noise 与 slow/fast rate 消融尚未完成，不能把当前结果表述为完整实验矩阵。
3. 当前 stream 是 5 条同步仿真轨迹，不是单机器人流。Plan 03 公平对比必须复用同一
   ordering 和 batch 语义；单轨迹实验应另建场景。
4. fixed encoder 仅复现 Hao-style 累积矩阵更新机制，不代表完整在线 DNN 优化复现。

综上，本轮关闭了 Review 中所有可在当前授权范围内完成的代码、审计和统计问题；
Plan 02 核心算法 Gate 保持通过，可以开展 Plan 03 开发，但正式 canonical 数值对比
仍应等待 canonical Plan 01/02 重跑。
