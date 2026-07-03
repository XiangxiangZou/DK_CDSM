# traj_data 程序功能说明

本文档说明 `traj_data/` 文件夹中每个程序和配套资产的作用。该文件夹是一套相对独立的 MuJoCo 轨迹数据采集工具，核心目标是为 CDSM 绳驱空间机械臂生成可用于 Koopman 模型预测训练的数据。

## 总体数据约定

采集脚本生成的 `dataset.npz` 通常包含以下数组：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `t` | `(steps,)` | 每个控制步对应的时间，单位 s |
| `states` | `(traj, steps+1, 4)` | 状态序列，顺序为 `[qa, qb, dqa, dqb]` |
| `inputs` | `(traj, steps, 2)` | 等效关节力矩输入，顺序为 `[tau_a, tau_b]`，单位 Nm |
| `q_ref` | `(traj, steps, 2)` | 参考关节角；受控 PD 采集时有效，非受控采集时为 `NaN` |
| `dq_ref` | `(traj, steps, 2)` | 参考关节角速度；受控 PD 采集时有效，非受控采集时为 `NaN` |
| `cable_ctrl` | `(traj, steps, 8)` | 实际写入 8 个绳索电机的张力控制量，单位 N |

其中状态只记录两个主动自由度：

- `qa`: 主动关节 `joint1` 的角度。
- `qb`: 主动关节 `joint3` 的角度。
- `dqa`: `joint1` 的角速度。
- `dqb`: `joint3` 的角速度。

XML 中 `joint2` 跟随 `joint1`，`joint4` 跟随 `joint3`，所以采集数据用 4 维状态表示这个等效 2 自由度系统。

## 文件关系

```text
collect_data_controlled.py
  -> mujoco_cdsm.py
  -> references.py
  -> data_io.py
  -> assets/multi_joint_cable_driven_space_robot.xml

collect_data_uncontrolled.py
  -> mujoco_cdsm.py
  -> references.py
  -> data_io.py
  -> assets/multi_joint_cable_driven_space_robot.xml
```

两个采集入口共享同一套 MuJoCo 封装、参考轨迹/限位工具、保存工具和 XML 模型。

## `collect_data_controlled.py`

这是受控数据采集入口，用 PD 控制器跟踪随机生成的平滑关节空间参考轨迹。

### 主要功能

- 从 XML 模型加载 CDSM 绳驱机械臂。
- 读取 XML 中主动关节的物理限位，并按 `--limit_ratio` 收缩出安全采集范围。
- 每条轨迹随机采样初始关节角和角速度。
- 调用 `references.random_joint_reference(...)` 生成大范围、平滑的随机关节参考轨迹。
- 用 PD 控制律计算期望等效关节力矩：
  - `tau = kp * (q_ref - q) + kd * (dq_ref - dq)`
- 加入 `references.soft_limit_guard(...)` 生成的软限位保护力矩，避免轨迹靠近边界。
- 采集阶段不对等效关节力矩 `tau` 做最终限幅或裁剪。
- 将 `tau=[tau_a,tau_b]` 通过 `MujocoCDSM.torque_to_tensions(...)` 转换成 8 根绳索张力。
- 绳索张力只限制下限为 20 N，不限制上限。
- 将张力写入 MuJoCo actuator，并逐步仿真。
- 保存完整轨迹数据、元数据和摘要。

### 入口与默认输出

直接运行该脚本会进入 `main()`：

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\traj_data\collect_data_controlled.py
```

默认输出目录是：

```text
traj_data/outputs/<timestamp>_controlled_pd<tag>/
```

输出文件：

- `dataset.npz`: 主数据数组。
- `metadata.json`: XML 路径、采集模式、关节限位、字段顺序、参数等元数据。
- `summary.json`: 数据有限性、状态范围、峰值力矩、峰值绳张力等摘要。

### 关键参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--traj` | `40` | 轨迹条数 |
| `--steps` | `500` | 每条轨迹控制步数 |
| `--dt` | `0.01` | MuJoCo 仿真步长/控制周期 |
| `--seed` | `42` | 随机种子 |
| `--limit_ratio` | `0.92` | 使用 XML 物理关节限位的比例 |
| `--q_init_ratio` | `0.65` | 初始角度采样范围比例 |
| `--dq_init_range` | `0.4` | 初始角速度采样范围 |
| `--kp` | `(80.0, 70.0)` | 两个主动关节的 PD 比例增益 |
| `--kd` | `(8.0, 7.0)` | 两个主动关节的 PD 微分增益 |
| `--reference_waypoints` | `9` | 每条随机参考轨迹的路径点数量 |
| `--reference_range_ratio` | `0.95` | 参考路径点使用安全限位范围的比例 |
| `--guard_kp` | `80.0` | 软限位保护比例增益 |
| `--guard_kd` | `6.0` | 软限位保护微分增益 |

### 适用场景

该脚本适合生成带参考轨迹的闭环受控数据，覆盖“机器人在 PD 控制下跟踪大范围平滑参考”的状态-输入分布。保存的 `q_ref/dq_ref` 可以用于诊断控制误差，但当前 Koopman 预测训练主要使用 `states` 和 `inputs`。

## `collect_data_uncontrolled.py`

这是非受控数据采集入口，支持随机开环力矩激励和被动自由响应两种模式。

### 主要功能

- 从 XML 模型加载 CDSM 绳驱机械臂。
- 读取并收缩主动关节限位，定义安全采集范围。
- 每条轨迹随机采样初始状态。
- 根据 `--mode` 选择力矩生成方式：
  - `random`: 分段常值随机等效关节力矩，并叠加速度阻尼。
  - `passive`: 目标等效关节力矩为零，记录随机初始状态下的自由响应。
- 在 `random` 模式下加入软限位保护力矩。
- 采集阶段不对等效关节力矩 `tau` 做最终限幅或裁剪。
- 将等效关节力矩转换成 8 根绳索张力并驱动 MuJoCo 仿真。
- 绳索张力只限制下限为 20 N，不限制上限。
- 保存与受控采集一致的数据字段。

### 入口与默认输出

```powershell
& 'D:\Apps\Anaconda3\envs\env_dk_cdsm\python.exe' .\traj_data\collect_data_uncontrolled.py --mode random
```

默认输出目录是：

```text
traj_data/outputs/<timestamp>_uncontrolled_<mode><tag>/
```

输出文件同样包括：

- `dataset.npz`
- `metadata.json`
- `summary.json`

### 关键参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--mode` | `random` | `random` 为随机开环激励，`passive` 为零目标力矩自由响应 |
| `--traj` | `40` | 轨迹条数 |
| `--steps` | `500` | 每条轨迹控制步数 |
| `--dt` | `0.01` | MuJoCo 仿真步长/控制周期 |
| `--seed` | `42` | 随机种子 |
| `--limit_ratio` | `0.92` | 使用 XML 物理关节限位的比例 |
| `--q_init_ratio` | `0.65` | 初始角度采样范围比例 |
| `--dq_init_range` | `0.4` | 初始角速度采样范围 |
| `--random_tau` | `35.0` | 随机力矩采样幅值，单位 Nm；不是采集后的最终限幅 |
| `--random_hold_steps` | `8` | 随机力矩保持步数 |
| `--random_damping` | `0.8` | 随机模式下叠加的速度阻尼系数 |
| `--guard_kp` | `80.0` | 软限位保护比例增益 |
| `--guard_kd` | `6.0` | 软限位保护微分增益 |

### 适用场景

- `random` 模式适合增加状态-输入空间覆盖度，得到不依赖参考轨迹的开环激励数据。
- `passive` 模式适合观察系统在预紧绳索和随机初始状态下的自然响应，但输入激励信息较弱，通常不适合作为唯一的受控 Koopman 训练数据。

## `mujoco_cdsm.py`

这是 `traj_data/` 的底层 MuJoCo 适配器，封装 CDSM 绳驱机械臂的状态读写、关节限位、绳索雅可比和力矩到张力分配。

### 常量定义

- `ACTIVE_JOINTS = ("joint1", "joint3")`
  表示数据采集只把 `joint1` 和 `joint3` 当作独立主动关节。

- `MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}`
  表示 `joint2` 跟随 `joint1`，`joint4` 跟随 `joint3`。

- `CABLE_NAMES`
  8 根绳索顺序：
  `cable11, cable12, cable13, cable14, cable21, cable22, cable23, cable24`。

- `ACTUATOR_NAMES`
  与绳索一一对应的 MuJoCo motor actuator：
  `winch_c11, ..., winch_c24`。

- `STATE_ORDER = ("qa", "qb", "dqa", "dqb")`

- `INPUT_ORDER = ("tau_a", "tau_b")`

### 主要类：`MujocoCDSM`

#### 初始化

`MujocoCDSM(xml_path, dt)` 会：

- 延迟导入 `mujoco`。
- 从 XML 加载 `MjModel`。
- 设置 `model.opt.timestep = dt`。
- 创建主仿真数据 `self.data` 和用于有限差分的 `self.scratch`。
- 缓存关节、执行器、绳索 tendon 的 MuJoCo 索引。
- 读取主动关节物理限位。

#### 状态接口

- `set_state(q, dq)`
  清零 MuJoCo 状态，将两个主动关节的位置和速度写入模型，并同步 mimic 关节位置。

- `read_state()`
  返回 `[qa, qb, dqa, dqb]`。

#### 绳索雅可比

- `compute_tendon_jacobian(eps=1e-6)`
  用中心差分计算 8 根 tendon 长度对 MuJoCo 全部速度自由度的雅可比：

  ```text
  J[i, j] = d L_i / d q_j
  ```

  这个雅可比随后用于把绳索张力映射到等效广义力矩。

#### 等效力矩到绳索张力

- `torque_to_tensions(tau)`
  将 2 维等效关节力矩 `[tau_a, tau_b]` 转换为 8 维绳索张力。

  分配逻辑：

  - 对 `tau_a`，使用第一级绳索：
    - 正向组：`cable11 + cable13`
    - 反向组：`cable12 + cable14`
  - 对 `tau_b`，使用第二级绳索：
    - 正向组：`cable21 + cable23`
    - 反向组：`cable22 + cable24`
  - 所有绳索至少保持 `CABLE_TENSION_LOWER_BOUND = 20.0 N`。
  - 不限制单根绳最大张力。

内部 `_solve_pair(...)` 会在正向组、反向组和仅 20 N 下限张力之间选择最接近期望力矩且增量较小的张力方案。

#### 执行器接口

- `apply_cable_tensions(tensions)`
  将 8 维张力写入 MuJoCo 的 `data.ctrl`，顺序与 `ACTUATOR_NAMES` 一致。

- `step()`
  调用 `mujoco.mj_step(...)` 前进一个仿真步。

### 适用场景

该文件是采集脚本和 XML 模型之间的适配层。上层脚本只需要处理等效关节状态和力矩，底层真实执行则通过 8 根绳张力作用到 MuJoCo 模型上。

## `references.py`

这是随机关节参考轨迹、初始状态采样和软限位保护工具模块。

### 主要常量

- `SOFT_LIMIT_MARGIN = 0.08`
  软限位保护区宽度比例。默认在安全限位内侧各收缩 8% 作为边界保护区。

### 主要函数

#### `shrink_limits(q_limits, ratio)`

将形状 `(2, 2)` 的关节限位按中心向内收缩。常用于从 XML 物理限位得到更保守的安全采集范围。

输入：

- `q_limits`: 每个关节的 `[lower, upper]`。
- `ratio`: 收缩比例，范围 `(0, 1]`。

输出：

- 收缩后的 `(2, 2)` 安全限位。

#### `sample_initial_state(rng, safe_limits, q_init_ratio, dq_init_range)`

在安全限位内随机采样初始状态。

输出：

- `q0`: 两个主动关节初始角度。
- `dq0`: 两个主动关节初始角速度。

采样逻辑：

- 先用 `q_init_ratio` 进一步收缩 `safe_limits`。
- 在收缩后的角度范围内均匀采样 `q0`。
- 在 `[-dq_init_range, dq_init_range]` 内均匀采样 `dq0`。

#### `random_joint_reference(...)`

生成平滑、大范围的随机关节参考轨迹。

实现方式：

- 在安全限位内部随机采样多个路径点。
- 第一个路径点固定为当前初始位置 `q_start`。
- 在相邻路径点之间使用 smoothstep 三次平滑插值：

  ```text
  s(alpha) = 3 alpha^2 - 2 alpha^3
  ```

- 对生成的 `q_ref` 做数值微分，得到 `dq_ref`。

输出：

- `q_ref`: `(steps, 2)` 参考关节角。
- `dq_ref`: `(steps, 2)` 参考关节角速度。
- `t`: `(steps,)` 时间向量。

#### `soft_limit_guard(q, dq, safe_limits, kp, kd, margin=0.08)`

计算边界保护力矩。若关节位置进入安全限位内侧的边界缓冲区，则用类似虚拟弹簧-阻尼的形式把关节推回安全区间。

输出：

- 2 维保护力矩，安全区间内通常为 `[0, 0]`。

### 适用场景

`references.py` 不直接运行 MuJoCo，也不保存数据。它为两个采集脚本提供可复现的随机初始状态、参考轨迹和限位保护逻辑。

## `data_io.py`

这是数据保存和数据摘要工具模块。

### 主要函数

#### `jsonable(value)`

递归地把 Python/NumPy 对象转换成 JSON 可序列化对象：

- `Path` 转成字符串。
- `np.ndarray` 转成 list。
- `np.generic` 转成 Python 标量。
- dict/list/tuple 递归转换。

#### `save_json(path, payload)`

将元数据或摘要保存成 UTF-8 JSON 文件，自动创建父目录。

#### `validate_dataset(arrays)`

验证采集数组是否满足基本数据契约：

- `states` 必须是 `(traj, steps+1, 4)`。
- `inputs` 必须是 `(traj, steps, 2)`。
- `states` 与 `inputs` 的轨迹数一致。
- `states` 时间长度必须比 `inputs` 多 1。

同时计算摘要：

- `finite_required_arrays`
- `finite_by_array`
- `state_min`
- `state_max`
- `peak_abs_tau`
- `peak_cable_tension`
- `trajectory_count`
- `steps`

注意：非受控数据中 `q_ref/dq_ref` 故意填 `NaN`，所以 `finite_by_array` 可能显示它们不是有限值；摘要中的 `finite_required_arrays` 只要求 `states`、`inputs` 和 `cable_ctrl` 有限。

#### `save_dataset(path, arrays)`

先调用 `validate_dataset(...)` 生成摘要，再用 `np.savez_compressed(...)` 保存 `.npz` 数据。

### 适用场景

该文件为采集入口提供统一的数据契约检查、压缩保存和 JSON 元数据保存能力。

## `assets/multi_joint_cable_driven_space_robot.xml`

这是 `traj_data/` 使用的 MuJoCo 绳驱 CDSM 模型，不是 Python 程序，但两个采集脚本都依赖它。

### 主要内容

- 定义 4 个 hinge 关节：
  - `joint1`
  - `joint2`
  - `joint3`
  - `joint4`

- 定义 mimic/equality 关系：
  - `joint2 == joint1`
  - `joint4 == joint3`

- 定义 8 根 spatial tendon：
  - `cable11`
  - `cable12`
  - `cable13`
  - `cable14`
  - `cable21`
  - `cable22`
  - `cable23`
  - `cable24`

- 定义 8 个 tendon motor：
  - `winch_c11`
  - `winch_c12`
  - `winch_c13`
  - `winch_c14`
  - `winch_c21`
  - `winch_c22`
  - `winch_c23`
  - `winch_c24`

- `gear=1`，所以上层写入的 `ctrl` 数值可以按绳索张力 N 理解。

- XML 中当前 8 个 motor 没有启用 `ctrllimited/ctrlrange`；Python 采集层也不再提供单根绳张力上限，只保留 20 N 下限。

- 定义传感器：
  - `actuatorfrc`: 8 根绳索张力传感器。
  - `jointpos`: 4 个关节角传感器。

### 与 Python 代码的关系

`mujoco_cdsm.py` 通过名称查找 XML 中的 joint、tendon 和 actuator。如果 XML 中名称变化，`ACTIVE_JOINTS`、`CABLE_NAMES` 或 `ACTUATOR_NAMES` 也必须同步更新，否则模型加载或采集会失败。

## 推荐阅读顺序

如果要理解这套采集工具，建议按以下顺序阅读：

1. `collect_data_controlled.py` 或 `collect_data_uncontrolled.py`：先看采集入口和输出字段。
2. `mujoco_cdsm.py`：理解状态读写、力矩到绳张力映射和 MuJoCo 调用。
3. `references.py`：理解初始状态、参考轨迹和软限位保护。
4. `data_io.py`：理解保存格式和摘要检查。
5. `assets/multi_joint_cable_driven_space_robot.xml`：核对关节、绳索、执行器和物理限位名称。
