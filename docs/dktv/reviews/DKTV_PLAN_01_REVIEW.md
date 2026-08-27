# DKTV Plan 01 执行情况 Review

> 审查对象：`DKTV_PLAN_01_FOUNDATION.md`、`DKTV_PLAN_01_EXECUTION_REPORT.md`
>
> 审查日期：2026-08-25
>
> 审查结论：**有条件通过（Conditional Pass）**

## 1. 总体判断

Plan 01 的核心实验链路已经形成：公共配置、时变数据采集、数据质量检查、唯一
初始模型、fixed DKO 评价和标准输出目录均有对应实现与实际产物。执行报告中的
主要数据规模、质量指标、时变性结果和预测指标都能在 full-run 的 JSON/NPZ
产物中找到对应证据，未发现指标抄写错误。

当前结果足以开始 Plan 02 的接口设计和算法开发，但还不宜把现有 full-run 直接
视为已经冻结的论文基线。进入 Plan 02 的正式对比实验前，应先解决两个 P0 问题：

1. 让基线产物能够精确追溯到生成它的源码状态；
2. 明确在线更新使用的归一化坐标、`A0/B0/C0` 含义和输入变换。

完成这两项并重新生成一次 canonical full-run 后，可将 Plan 01 改为无条件通过。

## 2. 本次审查范围与证据

本次检查了：

- Plan 01 计划与执行报告；
- `prediction/dktv_base_config.json`；
- `traj_data/mujoco_cdsm.py` 与 `prediction/dkuc_prediction.py` 的改动；
- `traj_data/dktv_data.py`、`prediction/dktv/`；
- `prediction/dktv_foundation_prediction.py` 与 `tests/dktv/test_foundation.py`；
- smoke/full 的 manifest、metrics、模型 metadata、文件清单和代表性 PNG；
- 当前 Git 状态、忽略规则和文本差异检查。

独立核对得到：

- full raw 数据 SHA-256 与 manifest 一致：
  `edec9b0b5b4a94d4e4176433cd95113eef573226858c82911d9a9e98e18acf9c`；
- `initial_model.npz` SHA-256 与 manifest 一致：
  `105257f75463cdb6db1058e1b108fa35182c59c80c68bb998a1eae18757de877`；
- 当前 `base.json` SHA-256 与 manifest 一致：
  `4ca588df226dd1a1aca71b6c7e44eb14f02f731e38818d4ecbfcf19d3467660d`；
- 报告中的 `20 × 240`、质量指标、时变性差异、one-step/rollout RMSE 和
  `3.554727` 退化比均与保存的 JSON 一致；
- raw NPZ 包含报告所列 11 个必需字段和兼容字段；预测 NPZ 包含真值、预测、
  各 horizon、各阶段及窗口起点，可用于重新绘图和复算；
- `rollout_prediction_error.png` 可以正常打开，曲线与误差增大趋势可见；
- `outputs/`、缓存和测试缓存均被根 `.gitignore` 正确忽略。

本次没有重新执行 Python 测试或实验。当前任务说明指定的 Windows Python
可执行文件在本 Linux 会话中不可访问，且没有可用的 PowerShell；按照仓库规则
没有切换到其他解释器。因而报告中的 `7 passed`、数据重放和旧 artifact 兼容性
属于“已有报告证据”，不是本次独立复跑结果。

## 3. 验收条目复核

| Plan 01 验收条目 | Review 结论 | 说明 |
| --- | --- | --- |
| 不批量搬迁或重复实现五部分工作流 | 通过 | 新代码边界基本符合 `src/`、`experiments/`、`configs/`、`tests/` 约定 |
| 一份配置驱动数据、模型和评价 | 通过 | smoke/full 共用 `base.json`，profile 只物化运行规模 |
| 相同 `x/u`、不同 `t` 能证明时变性 | 通过 | 固定 plant 差异约 `3.47e-18`，加时变外扰后约 `1.97e-2` |
| 数据质量契约 | 当前数据通过 | full 数据无非有限值、限位、饱和、张力异常或超限残差；检查器仍有缺字段健壮性问题 |
| 唯一初始模型可独立加载 | 基本通过 | artifact 文件和 `A/B/C` 形状齐全；归一化坐标语义需在 Plan 02 前补明 |
| one-step、rollout、分阶段结果齐全 | 通过 | JSON 指标和可重绘数组均存在 |
| 相同 seed 可复现 | 部分通过 | split 和报告中的重放结果一致，但本次未独立复跑，且源码状态未被完整记录 |
| 自动化检查与文本检查通过 | 部分通过 | 已报告 7 个测试通过；`git diff --check` 未覆盖未跟踪文件 |
| fixed DKO 出现可识别退化 | 工程验收通过 | 单次实验比值为 `3.554727`；不能直接升级为论文统计结论 |

## 4. 发现的问题

### P0-01：基线不能精确追溯到生成源码

full manifest 记录的 Git commit 为
`9dfc7056932b20929b42279d8caa5854493eef08`，但生成实验时的 Plan 01 实现包含
多个未跟踪文件及已修改文件。仅 checkout 该 commit 无法还原本次实验源码，
manifest 也没有记录 `git_dirty`、源码 diff 或关键实现文件 hash。

这不会否定现有指标，但会阻断“冻结论文基线”的可复现性要求。

建议：

1. 先完成本 review 中必须修正的代码和契约；
2. 将 Plan 01 源码形成一个明确的 Git 版本，再生成 canonical full-run；
3. manifest 增加 `git.dirty`，dirty 时默认不得标记为 canonical；
4. 至少记录配置、raw/processed 数据、模型文件和关键评价数组的 hash；
5. artifact manifest 记录其自身 schema 版本和所有组成文件 hash。

### P0-02：`A0/B0/C0` 的坐标语义必须在 Plan 02 前冻结

当前实际模型使用：

```text
z = [x_normalized, phi(x_normalized), 1]
u_model = u_normalized
z_next = A0 z + B0 u_model
C0 z = x_normalized
```

因此 `C0` 前四列为单位阵表示的是“精确读取归一化状态”，不是直接输出物理单位
的关节角和角速度。物理状态还需经过 `x_normalizer.inverse(...)`；物理力矩在进入
`B0` 前也必须用固定 `u_normalizer` 变换。

执行报告目前只写了 `C0` 的形状和单位阵结构，Plan 02 也只写
`applied_torque`，容易让累积式更新错误地把物理力矩直接送入 `B0`。

建议把以下内容加入 artifact contract 和 Plan 02：

- `state_coordinate = normalized`；
- `input_coordinate = normalized_applied_torque`；
- `lift_definition = [x_norm, phi(x_norm), 1]`；
- `C0` 的输出空间及物理状态恢复公式；
- online update、direct refit oracle 和 fixed DKO 必须共用同一 normalizer；
- 增加一次“物理输入 → 归一化输入 → latent step → 物理状态恢复”的契约测试。

### P1-01：缺字段数据不能按设计进入 rejected 流程

`assess_data_quality()` 会先统计 `missing_fields`，随后仍直接读取
`states`、`applied_torque`、`effective_tensions` 等字段。只要这些字段真的缺失，
函数会抛出 `KeyError`，无法返回带 `missing_fields:*` 的拒绝原因，也无法让入口把
数据保存到 `outputs/data/rejected/`。

建议先在缺字段时返回结构化拒绝结果，或在访问数组前完成形状和必需字段校验，
并新增缺字段、错 shape、NaN、残差超限四类测试。

### P1-02：执行报告遗漏了治理文件和包文件变更

当前工作区还修改了 `AGENTS.md`，并新增多个 `__init__.py`，但报告第 3 节没有
列出这些文件。特别是 `AGENTS.md` 新增了 Linux 环境和依赖安装规则，属于仓库
治理契约变化，不应作为普通实验实现被隐式带入。

建议明确决定这部分是否保留：若保留，单独说明修改原因并经过确认；若不属于
Plan 01，则从 Plan 01 变更中分离。执行报告应列出所有任务相关文件。

### P1-03：`git diff --check` 的“通过”范围被高估

普通 `git diff --check` 只检查已跟踪差异，不会检查未跟踪源码。本次对未跟踪
文件逐个执行等价检查后，以下文件存在 `new blank line at EOF`：

- `experiments/__init__.py`；
- `experiments/dktv/__init__.py`；
- `src/cdsm/__init__.py`；
- `src/koopman_control/__init__.py`；
- `prediction/dktv/__init__.py`。

这不是运行错误，但报告中的“所有文本检查通过”目前不准确。建议修正空行，并在
未提交阶段使用 `git diff --no-index --check /dev/null <file>` 或等价方式覆盖新文件。

### P1-04：自动检查没有形成完整、可审计日志

full result 的 `logs/run.log` 只保存了六条流程摘要，没有保存完整命令、测试输出、
`py_compile`、旧 artifact 兼容检查和指标复算结果。执行报告虽然列出了这些检查，
后续人员无法仅依靠 run 目录复核它们。

建议将 acceptance 命令、stdout/stderr 和检查摘要保存到 `logs/`，并增加一个小型
端到端测试，至少验证 smoke CLI、artifact 重载、数组/指标文件存在和 manifest
状态。当前 7 个测试主要覆盖配置、单函数和 MuJoCo 时变性，没有覆盖完整入口。

### P1-05：manifest 中仍保存了机器相关绝对路径

`manifest.metrics.figures` 中的图像路径是
`/home/zouxx/PhD_Projects/DK_CDSM/...` 形式，而同一 manifest 的数据集和模型路径
已经使用仓库相对路径。绝对路径降低跨机器复现能力，也与当前 `AGENTS.md` 的路径
可移植性要求不一致。

建议所有产物路径相对 `PROJECT_ROOT` 或 `result_dir` 保存；运行时再解析为绝对路径。

### P1-06：报告中的 raw 字段数量多写了一个

执行报告第 10 节称“17 个保存字段逐数组 `array_equal=True`”，但当前 full raw
NPZ 实际包含 16 个数组：11 个必需字段、`stage_id` 和 4 个兼容字段。字段内容与
计划契约相符，问题只在报告计数。建议将“17 个”改为“16 个”，并在复现日志中
直接保存参与比较的字段名列表，避免只记录总数。

### P2-01：退化比只能作为进入在线更新实验的工程门槛

当前 `3.554727` 来自一个 seed、一个扰动幅频参数和一组按时间阶段切分的闭环
轨迹。阶段间不仅扰动幅值变化，状态、参考和控制输入分布也随轨迹进程变化，因此
该比值能证明“当前 stream 上 fixed DKO 后段明显变差”，但不能单独量化时变扰动
的因果贡献，也不能作为论文中的统计结论。

这不阻断 Plan 02。正式论文实验应补充多 seed、扰动关闭对照、相同初始条件/参考
的 paired run，以及均值、标准差或置信区间。建议不要在 Plan 01 中提前扩展这些
实验，把它们放入后续统一对比阶段即可。

## 5. 建议的最小修正顺序

1. 明确并记录 Windows/Linux 环境规则，处理 `AGENTS.md` 的归属；
2. 修正 5 个包文件的尾部空行；
3. 冻结 normalized state/input/latent/readout 契约，并补契约测试；
4. 修复数据质量检查器的缺字段路径并补测试；
5. 给 manifest 增加 dirty 状态、关键 hash 和相对路径；
6. 保存 smoke、full、pytest 和复算检查日志；
7. 在明确的源码版本上重新运行 smoke，再运行 canonical full；
8. 更新执行报告中的结论、文件清单、测试证据和新 run id。

建议 canonical run 使用新的明确标签，例如 `baseline_reviewed`，保留当前
`20260825_112709_plan01_full_baseline` 作为 review 前证据，不覆盖、不删除。

## 6. 进入 Plan 02 的 Gate

可以立即开展：

- Hao 累积式更新的公式—数组映射；
- updater API、充分统计量和 direct-refit oracle 的单元测试设计；
- 输出目录及评价复用设计。

在 P0-01、P0-02 未完成前，不建议：

- 宣称 Plan 01 已形成最终论文基线；
- 固化 Plan 02 的 `B` 更新输入而不说明是否归一化；
- 生成并引用正式的 Plan 02 与 fixed DKO 对比数字；
- 删除或覆盖当前 full-run。

## 7. 建议更新执行报告的结论

在上述修正完成前，建议把报告开头的结论调整为：

> Plan 01 核心功能与工程验收已完成；当前为有条件通过。完成源码追溯和归一化
> 坐标契约冻结后，可生成 canonical 基线并正式进入 Plan 02 对比实验。

这样既保留当前已经完成的工作，也不会把尚未闭合的可复现性问题带入后续论文
实验。
