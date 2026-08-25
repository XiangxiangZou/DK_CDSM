# DKTV Plan 01 第二轮执行反馈报告

> 对应复审：`DKTV_PLAN_01_REVIEW.md`
>
> 执行日期：2026-08-25
>
> 本轮结论：**复审代码项已修正，工程验收通过；canonical 基线仍待授权提交源码后重跑**

## 1. 结论

本轮已处理复审中的 P0-02、全部 P1 和 P2 表述问题，并为 P0-01 建立了强制
canonical gate、Git dirty 记录、关键源码 hash 和全链路制品 hash。新 smoke/full
run 的数据质量、时变性证明、坐标契约、artifact 重载和 fixed DKO 退化工程门槛
全部通过。

新 full run 没有被标记为 canonical。运行时工作树仍包含未提交的 Plan 01 实现，
manifest 因此正确记录：

```text
status = accepted_noncanonical_dirty
canonical = false
canonical_blockers = [git_worktree_dirty]
```

仓库规则禁止 agent 在未获请求时 commit。故本轮没有擅自提交代码，也没有伪造
canonical 结论。待用户确认并形成明确 Git 版本后，应在 clean worktree 上以同一
配置再次运行 full，届时 canonical gate 才会放行。

## 2. 新运行记录

| 类型 | Run ID | 状态 | Canonical |
| --- | --- | --- | --- |
| smoke | `20260825_145551_plan01_smoke_baseline_reviewed` | `accepted_noncanonical_dirty` | 否 |
| full | `20260825_145620_plan01_full_baseline_reviewed` | `accepted_noncanonical_dirty` | 否 |

旧的 `20260825_112618_plan01_smoke_acceptance` 和
`20260825_112709_plan01_full_baseline` 均被保留，未覆盖、未删除。

## 3. 复审问题闭环情况

### P0-01：源码和制品追溯

已完成：

- manifest schema 升级到 v2，记录 branch、HEAD、`git.dirty` 和完整 porcelain 状态；
- dirty worktree 下即使工程 acceptance 全部通过，也只能生成
  `accepted_noncanonical_dirty`；
- full manifest 保存 12 个关键源码/计划/配置文件的 SHA-256 和字节数；
- 保存 base config、raw、raw metadata/quality、3 个 processed dataset、split 的
  路径、SHA-256 和字节数；
- artifact manifest schema 升级到 v2，保存 8 个组成文件的 SHA-256 和字节数；
- 保存评价 metrics、arrays、figures 的逐文件 SHA-256；
- 独立复核 47 条 hash 记录，全部匹配。

未完成项只有“在明确 Git 版本上生成 canonical full-run”。原因不是实现缺失，而是
当前任务没有授权 commit，且仓库规则明确禁止擅自 commit。

### P0-02：归一化坐标契约

artifact 和公共配置已冻结以下语义：

```text
x_norm = (x_phys - x_mean) / x_std
u_norm = (applied_torque_phys - u_mean) / u_std
z = [x_norm, phi(x_norm), 1]
z_next = A0 @ z + B0 @ u_norm
x_norm_next = C0 @ z_next
x_phys_next = x_norm_next * x_std + x_mean
```

`initial_model.npz` 现在同时保存 `A0/B0/C0`、`x_mean/x_std` 和
`u_mean/u_std`。artifact contract 明确 `C0` 输出 normalized state，所有在线方法、
direct-refit oracle 和 fixed DKO 必须共用固定 normalizer。Plan 02 文档也已同步，
明确禁止把物理力矩直接送入 `B_tau`。

新增坐标契约测试和实际 artifact 检查，二者均通过。

### P1-01：质量拒绝路径

`assess_data_quality()` 现在按以下顺序处理：

1. 必需字段检查；
2. 数组形状检查；
3. finite、限位、饱和、张力和 allocation residual 检查；
4. 返回统一结构的接受或拒绝结果。

缺少 `states` 时不再抛出 `KeyError`。新增并通过四类复审指定测试：缺字段、错
shape、NaN、allocation residual 超限。

### P1-02：文件清单和治理文件

本报告在第 4 节分别列出本轮实现文件、继承的 Plan 01 文件和治理文件。
`AGENTS.md` 的 Linux/Conda 规则来自用户此前的环境兼容任务，本轮保留但没有再次
改写；它不作为 Plan 01 算法实现隐式隐藏。

### P1-03：未跟踪文件文本检查

已移除 5 个 `__init__.py` 的多余 EOF 空行。最终检查覆盖：

- tracked diff：通过；
- 18 个未跟踪文件：逐文件等价 `--check`；
- whitespace error：0。

### P1-04：审计日志和端到端测试

每次新 run 自动保存：

- `logs/command.json`：入口、argv 和工作目录；
- `logs/environment.json`：解释器、Conda、PYTHONPATH、平台；
- `logs/reproducibility.json`：seed、16 个 raw 字段名和过滤规则；
- `logs/run.log`：采集、切分、训练、重载、坐标检查和最终状态。

full run 另保存 `py_compile.log`、`pytest.log`、`reproducibility_check.log`、
`hash_verification.log`、`legacy_artifact_compatibility.log`、`diff_check.log`、
`cli_help.log` 和 `acceptance_summary.json`。

新增 smoke CLI 端到端测试，实际执行采集、训练、评价，随后检查 artifact 重载、
预测数组、metrics、manifest acceptance 和 figure 相对路径。

### P1-05：绝对图像路径

`manifest.metrics.figures` 现全部为 result-relative 路径，例如：

```text
figures/rollout_prediction_error.png
```

本次 full manifest 中 8 个图像路径均不以 `/` 开头。

### P1-06：raw 字段数

更正为 16 个字段。重放日志保存了全部字段名，不再只写总数。相同 seed 重放的
16 个字段全部 `array_equal=True`。

### P2-01：退化比解释边界

manifest 和本报告均只把退化比称为单一配置 stream 的“工程门槛”。不把
`3.554727` 表述为多 seed 统计结论或时变扰动的因果结论。多 seed、paired control
和置信区间留给后续统一论文实验。

## 4. 文件变更清单

### 本轮实现与契约修正

- `configs/dktv/base.json`
- `src/cdsm/dktv_data.py`
- `src/koopman_control/dktv/config.py`
- `src/koopman_control/dktv/foundation.py`
- `experiments/dktv/plan_01.py`
- `tests/test_dktv_foundation.py`
- `DKTV_PLAN_02_HAO_ACCUMULATIVE.md`
- `experiments/__init__.py`
- `experiments/dktv/__init__.py`
- `src/cdsm/__init__.py`
- `src/koopman_control/__init__.py`
- `src/koopman_control/dktv/__init__.py`
- `DKTV_PLAN_01_REVIEW_FEEDBACK.md`（本报告）

### 上一轮 Plan 01 已实现、当前仍在工作树中的文件

- `prediction/dkuc_prediction.py`
- `traj_data/mujoco_cdsm.py`
- `DKTV_PLAN_01_EXECUTION_REPORT.md`
- `DKTV_PLAN_01_FOUNDATION.md`

### 治理与用户提供的计划/审查文件

- `AGENTS.md`：此前 Linux 环境兼容任务的治理变更，本轮未再修改；
- `DKTV_PLAN_01_REVIEW.md`：用户提供的复审输入，本轮未修改；
- `DKTV_IMPLEMENTATION_PLAN.md`、`DKTV_PLAN_03_ZHANG_SLIDING_WINDOW.md`：保留，
  本轮未修改。

## 5. 执行命令与检查

使用的解释器：

```text
/home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python
```

主要命令：

```bash
env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m experiments.dktv.plan_01 --run-type smoke --device cpu \
  --tag baseline_reviewed

env -u PYTHONPATH MPLCONFIGDIR=/tmp/dktv-matplotlib \
  /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m experiments.dktv.plan_01 --run-type full --device cpu \
  --tag baseline_reviewed

env -u PYTHONPATH /home/zouxx/Apps/miniconda3/envs/env_dk_cdsm/bin/python \
  -m pytest -q
```

检查结果：

| 检查 | 结果 |
| --- | --- |
| 5 个修改 Python 文件 compile | 通过 |
| 全部自动化测试 | `13 passed in 7.62s` |
| smoke CLI 端到端 | 通过 |
| full 工程 acceptance | 全部通过 |
| 同 seed raw 重放 | 16/16 字段 `array_equal=True` |
| manifest/artifact/result/source hash | 47/47 匹配 |
| 旧两份 Plan 01 artifact 重载 | 通过，`A=(16,16)`、`B=(16,2)` |
| tracked/untracked whitespace 检查 | 0 错误 |
| 代表性 PNG 打开检查 | 通过 |

## 6. Full-run 关键指标

### 数据和时变性

- 数据规模：20 条轨迹 × 240 steps；
- non-finite、joint-limit、saturation、tension outlier：均为 0；
- 最大 allocation residual：`1.338385438032219e-07 N·m`；
- 最大绝对 applied torque：`20.2488471395619 N·m`；
- 最大 effective tension：`43.988520230957405 N`；
- 固定 plant 同状态/输入跨时间差：`3.469446951953614e-18`；
- 时变 plant 同状态/输入跨时间差：`0.019682015896010277`。

### 模型和预测

- `A0=(16,16)`、`B0=(16,2)`、`C0=(4,16)`；
- one-step RMSE：`0.0037884481068249766`；
- full-stream rollout RMSE：`0.09150945844885967`；
- horizon 10/20/50/100 RMSE：
  `0.01984287342152731 / 0.035247038403943894 /
  0.06328443289920438 / 0.06489908752954186`；
- nominal/transition/time-varying stage RMSE：
  `0.01594877716125733 / 0.03216437538209753 / 0.056693550691015865`；
- time-varying / nominal：`3.5547271190631142`，工程门槛通过。

## 7. 输出位置

Full run：

```text
outputs/results/dktv/plan_01/20260825_145620_plan01_full_baseline_reviewed/
outputs/data/raw/20260825_145620_plan01_full_baseline_reviewed/
outputs/data/processed/20260825_145620_plan01_full_baseline_reviewed/
outputs/models/dktv/20260825_145620_plan01_full_baseline_reviewed/
```

Smoke run：

```text
outputs/results/dktv/plan_01/20260825_145551_plan01_smoke_baseline_reviewed/
outputs/data/raw/20260825_145551_plan01_smoke_baseline_reviewed/
outputs/data/processed/20260825_145551_plan01_smoke_baseline_reviewed/
outputs/models/dktv/20260825_145551_plan01_smoke_baseline_reviewed/
```

## 8. 剩余风险与下一步 Gate

唯一阻止 Plan 01 无条件通过的事项是 canonical Git 版本尚未形成。建议后续顺序：

1. 用户审阅本轮文件变更；
2. 用户明确授权后形成 Git commit；
3. 确认 worktree clean；
4. 使用 `--run-type full --tag baseline_reviewed` 再运行；
5. 核对 manifest 为 `accepted_canonical` 且 `canonical=true`；
6. 将该新 run 作为 Plan 02 正式数值对照的唯一 canonical 基线。

在此之前，可以继续 Plan 02 的接口、公式映射和单元测试开发，但不应引用本报告的
noncanonical full run 作为最终论文基线。
