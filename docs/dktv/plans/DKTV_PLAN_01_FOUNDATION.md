# Plan 01：基础设置与实验契约

> 状态：待实施
>
> 前置依赖：无
>
> 输出基线：`fixed_dko`

## 1. 目标

在不重构当前仓库的前提下，为后续两种在线更新算法准备一套最小、统一、可复现
的运行基础。Plan 01 完成后，应能用当前仓库已有的数据采集、DKUC 预测、控制
和可视化能力，生成一份固定初始模型及其基线预测结果。

本阶段不实现任何在线更新算法。

## 2. 与当前仓库的对应关系

继续沿用当前仓库的五部分工作流：

| 当前目录 | Plan 01 中的用途 |
| --- | --- |
| `traj_data/` | 复用 MuJoCo plant 和数据采集流程，只补最小的时变扰动与必要字段 |
| `prediction/` | 复用 DKUC 的 `g(x)=[x,phi(x)]`、训练和预测逻辑 |
| `control/` | 本阶段只确认模型接口可被后续 MPC 使用，不修改控制算法 |
| `common/` | 仅放两个及以上模块共用的数据、模型或产物工具 |
| `visualization/` | 复用现有绘图入口，不建立第二套绘图系统 |

遵循 `AGENTS.md`，后续新增的可复用算法代码放入规定的源码边界；Plan 01 不搬迁
上述旧目录，也不为了目录形式重写现有程序。

## 3. 最小工作范围

Plan 01 只完成以下五件事：

1. 冻结方法名称和一份公共基础配置；
2. 生成一份可验证的最小时变数据集；
3. 冻结固定 encoder、normalizer 和初始 `A0/B0/C0`；
4. 统一 one-step、rollout 和分时段评价口径；
5. 统一新实验的 manifest 和输出目录。

明确不做：累积更新、滑动窗口、选择性更新、在线 encoder、复杂模型生命周期、
SDP-MPC、载荷/绳索退化等多场景扩展，以及旧代码批量迁移。

## 4. 公共基础配置

第一版只维护一份 `prediction/dktv_base_config.json`，建议默认值为：

```text
state:                 [qa, qb, dqa, dqb]
state_dim:             4
input:                 applied_torque
input_dim:             2
sample_dt_s:           0.01
lift:                  [x, phi(x)]
lifted_dim:            16
encoder_update:        false
model_form:            affine
state_readout:         exact
ridge_lambda:          1.0e-3
seed:                  显式给出
```

`affine` 通过在 lifted state 中加入常数 observable `1` 实现。默认不把时间或
已知扰动参数输入 Koopman 网络。

Plan 02 和 Plan 03 只增加各自算法参数，不复制或改写这些公共字段。

## 5. 最小时变数据契约

第一版只加入正弦外部关节扰动力矩：

```math
\tau_d(t)=a\sin(\omega t+\phi).
```

先不实现载荷变化和绳索效率退化。必须用“相同状态、相同输入、不同绝对时刻”
的测试证明下一状态确实发生变化，避免把时变参考轨迹误称为时变动力学。

在现有数据字段基础上，至少保证保存：

```text
t
states
commanded_torque
applied_torque
commanded_tensions
effective_tensions
allocation_residual
disturbance_torque
reference_state          # 受控数据需要
saturation_flags
joint_limit_flags
```

在线辨识默认使用 `applied_torque`。数据接受前只做必要检查：有限值、关节限位、
力矩/张力饱和、分配残差和状态/输入范围。异常数据保存在
`outputs/data/rejected/` 并记录原因。

## 6. 初始模型契约

Plan 01 复用 `prediction/dkuc_prediction.py` 的固定升维思路，输出供三个方法
共同加载的初始 artifact：

```text
encoder
normalizer
A0
B0
C0
model_form
state_readout
training_dataset
validation_dataset
seed
```

四种方法必须从同一 artifact 开始：

```text
fixed_dko
dktv_accumulative
otvdkl_window
otvdkl_selective
```

不允许 Plan 02 或 Plan 03 为了获得更好结果重新训练各自的初始 encoder。

## 7. 统一评价和输出

最少保存：

- one-step 总 RMSE 和各状态 RMSE；
- 10/20/50/100 步 rollout RMSE；
- 按时变阶段划分的 rollout RMSE；
- 最大误差和非有限预测数量；
- 预测、真值、时间轴和扰动真值原始数组。

所有新产物统一放在：

```text
outputs/data/{raw,processed,rejected}/
outputs/models/dktv/
outputs/results/dktv/plan_01/
```

每次结果至少包含 `manifest.json`、`metrics/`、`arrays/`、`figures/` 和 `logs/`。
旧目录中的历史结果不迁移、不删除。

## 8. 实施顺序

1. 检查并复用当前数据采集和 DKUC 入口；
2. 增加公共基础配置及参数校验；
3. 给现有 plant 接入正弦扰动并补齐数据字段；
4. 完成时变性与数据质量检查；
5. 生成 smoke 数据，通过后再生成 full-run 数据；
6. 训练并冻结唯一初始模型 artifact；
7. 运行 `fixed_dko` 的统一 one-step 和 rollout 基线；
8. 保存完整 manifest、指标和可重绘数组。

## 9. 验收标准

- 未对当前五部分工作流做批量搬迁或重复实现；
- 一份基础配置可驱动数据、初始模型和固定基线评价；
- 相同 `x/u`、不同 `t` 的测试能区分固定 plant 与时变 plant；
- 数据无未记录的非有限值、限位、饱和或张力异常；
- 初始模型可被独立加载，并包含固定 encoder 和 normalizer；
- `fixed_dko` 同时输出 one-step、rollout 和分时段结果；
- 相同 seed 能复现数据划分与评价；
- 相关检查通过，`git diff --check` 无错误。

只有 Plan 01 验收通过，且 fixed DKO 在时变阶段表现出可识别的预测退化，才进入
Plan 02。
