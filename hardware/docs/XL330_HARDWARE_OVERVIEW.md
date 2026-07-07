# XL330 舵机硬件说明与 hardware 程序参数梳理

本文档面向当前 `hardware/` 文件夹中的 DYNAMIXEL XL330 双舵机实验程序。当前硬件约定是：

- 舵机型号：DYNAMIXEL XL330 系列，当前脚本按 XL330-M288-T / X-series Protocol 2.0 控制表组织。
- 通信端口：默认 `COM3`。
- 通信波特率：默认 `57600`。
- 舵机 ID：默认 `10` 和 `20`。
- 控制目标：两个 XL330 分别映射到 MuJoCo 两个主动关节 `joint1` 和 `joint3`，从动关节由 MuJoCo equality/mimic 约束同步。

官方手册参考：

- XL330-M288-T e-Manual: https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- DYNAMIXEL Protocol 2.0: https://emanual.robotis.com/docs/en/dxl/protocol2/

## 1. XL330 舵机整体情况

XL330 是 ROBOTIS DYNAMIXEL X 系列小型智能舵机。它不是普通 PWM 舵机，而是一个带通信协议、控制表、内部控制器和状态反馈的总线舵机。PC 或控制板通过串口总线读写寄存器来控制它。

在当前项目里，程序主要使用这些能力：

| 功能 | 硬件含义 | 当前用途 |
|---|---|---|
| 总线通信 | Protocol 2.0，通过串口读写控制表 | 在 `COM3` 上控制 ID10、ID20 |
| 工作模式切换 | 写 `Operating Mode(11)` | 在电流模式、扩展位置模式之间切换 |
| 扭矩使能 | 写 `Torque Enable(64)` | 运动前 enable，退出或安全停止时 disable |
| 位置反馈 | 读 `Present Position(132)` | 编码器角度反馈，换算成关节角 |
| 目标位置 | 写 `Goal Position(116)` | 位置模式/扩展位置模式下发送目标 tick |
| 目标电流 | 写 `Goal Current(102)` | 电流模式下发送外部控制器算出的电流命令 |
| 当前电流 | 读 `Present Current(126)` | 记录输入电流 proxy，用于分析负载/控制强度 |
| 当前 PWM | 读 `Present PWM(124)` | 记录驱动输出 proxy |
| 当前速度 | 读 `Present Velocity(128)` 或编码器差分 | 用于速度估计和安全检查 |
| 温度/电压/错误 | 读 `Present Temperature(146)`、`Present Input Voltage(144)`、`Hardware Error Status(70)` | 运行前/运行中安全检查 |

## 2. 编码器与角度换算

XL330-M288-T 的位置传感器是 12-bit 绝对编码器，分辨率为：

```text
4096 ticks / revolution
```

因此：

```text
1 tick = 2*pi / 4096 rad
       = 0.00153398 rad
       ≈ 0.08789 deg
```

当前共享换算常量定义在 `hardware/common/xl330_mujoco.py`：

```python
TICKS_PER_REVOLUTION = 4096
RAD_PER_TICK = 2.0 * math.pi / TICKS_PER_REVOLUTION
```

当前程序把舵机编码器值换算成关节角时使用：

```text
q = sign * (present_tick - zero_tick) * RAD_PER_TICK
```

其中：

- `present_tick`：从 `Present Position(132)` 读到的当前编码器 tick。
- `zero_tick`：程序启动时记录的零点，或用户通过 `--zero-ticks` 指定的零点。
- `sign`：方向符号，`+1` 表示舵机正方向与关节正方向一致，`-1` 表示反向。

这意味着当前项目大多数硬件脚本使用的是“相对启动零点”的关节角定义。启动姿态不同，得到的 `q=0` 物理姿态也会不同；如果需要可重复实验，应固定机械臂初始姿态，或显式记录/指定 `--zero-ticks`。

## 3. XL330 工作模式概览

XL330-M288-T 的 `Operating Mode(11)` 支持以下常用模式：

| Mode 值 | 名称 | 主要命令寄存器 | 功能 |
|---:|---|---|---|
| `0` | Current Control Mode | `Goal Current(102)` | 电流/近似力矩控制。位置闭环由外部 PC 程序完成。 |
| `1` | Velocity Control Mode | `Goal Velocity(104)` | 速度控制，适合轮式/连续转速场景。 |
| `3` | Position Control Mode | `Goal Position(116)` | 单圈位置控制，适合普通有限角度关节。 |
| `4` | Extended Position Control Mode | `Goal Position(116)` | 多圈累计位置控制，适合卷线轮、多圈机构或需要超过 360 deg 的位置控制。 |
| `5` | Current-based Position Control Mode | `Goal Position(116)` + `Goal Current(102)` | 位置控制，同时使用电流限制/电流目标约束输出。 |
| `16` | PWM Control Mode | `Goal PWM(100)` | 直接 PWM 输出控制，低层但风险更高。 |

当前 `hardware/` 里实际主要用了两类模式：

- `Operating Mode = 4`：位置/多圈位置类程序使用。
- `Operating Mode = 0`：当前电流模式外部角度闭环/正弦轨迹跟踪程序使用。

## 4. 通用硬件寄存器

当前脚本反复使用的寄存器如下：

| 地址 | 名称 | 字节 | 含义 |
|---:|---|---:|---|
| `0` | `Model Number` | 2 | 型号编号，用于确认 ping 型号与寄存器型号一致。 |
| `6` | `Firmware Version` | 1 | 固件版本。 |
| `11` | `Operating Mode` | 1 | 工作模式。切换模式前必须先关闭扭矩。 |
| `64` | `Torque Enable` | 1 | `0` 为脱力，`1` 为使能。 |
| `65` | `LED` | 1 | LED 开关，主要用于测试可视化。 |
| `70` | `Hardware Error Status` | 1 | 硬件错误状态，`0x00` 表示无错误。 |
| `100` | `Goal PWM` | 2 | PWM 模式下的目标 PWM，当前程序只记录 PWM，不直接用它控制。 |
| `102` | `Goal Current` | 2 | 电流模式下的目标电流，单位按脚本记为 mA。 |
| `108` | `Profile Acceleration` | 4 | 位置模式的运动规划加速度档位。 |
| `112` | `Profile Velocity` | 4 | 位置模式的运动规划速度档位。 |
| `116` | `Goal Position` | 4 | 位置/扩展位置模式下的目标位置 tick。 |
| `122` | `Moving` | 1 | 是否仍在运动。 |
| `124` | `Present PWM` | 2 | 当前 PWM 输出；脚本中换算为百分比，系数 `0.113`。 |
| `126` | `Present Current` | 2 | 当前输入电流 proxy；脚本中按 `1 mA` 单位记录。 |
| `128` | `Present Velocity` | 4 | 当前速度；脚本中有时使用编码器差分代替。 |
| `132` | `Present Position` | 4 | 当前编码器位置 tick。 |
| `144` | `Present Input Voltage` | 2 | 当前输入电压，单位 `0.1 V`。 |
| `146` | `Present Temperature` | 1 | 当前温度，单位 deg C。 |

## 5. `hardware/scripts/check_two_xl330_servos.py`

### 作用

基础硬件连通和一圈运动测试。它逐个检测 ID10、ID20，并让每个舵机在扩展位置模式下正向转一圈再回到起点。这个脚本用于确认：

- PC 能打开 `COM3`。
- 两个舵机能被 ping 到。
- `Goal Position(116)` 写入有效。
- `Present Position(132)` 反馈正常。
- 舵机能完成一圈运动且误差在阈值内。

### 使用的工作模式

```text
Operating Mode = 4  Extended Position Control Mode
```

初始化流程：

1. 写 `Torque Enable(64)=0`，先关闭扭矩。
2. 写 `Operating Mode(11)=4`。
3. 写 `Profile Acceleration(108)` 和 `Profile Velocity(112)`。
4. 可选写 `LED(65)=1`。
5. 写 `Torque Enable(64)=1`。
6. 读取 `Present Position(132)` 作为起点。
7. 写 `Goal Position(116)=start_position + rev_ticks`。
8. 等待 `Present Position(132)` 接近目标，读取 `Moving(122)` 辅助判断。
9. 写 `Goal Position(116)=start_position` 回到起点。

### 默认硬件参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--port` | `COM3` | 串口名。 |
| `--baud` | `57600` | XL330 默认波特率。 |
| `--ids` | `[10, 20]` | 两个舵机 ID。 |
| `--rev-ticks` | `4096` | 一圈 tick 数。 |
| `--profile-velocity` | `80` | 位置模式速度档位；越小越慢。 |
| `--profile-acceleration` | `20` | 位置模式加速度档位；越小越柔和。 |
| `--position-threshold` | `20` ticks | 到位阈值，约 `1.76 deg`。 |
| `--timeout` | `20.0 s` | 单段运动超时。 |
| `--pause` | `1.0 s` | 正转和回零之间的暂停。 |
| `--keep-torque` | 关闭 | 默认退出时关闭扭矩。 |
| `--skip-led` | 关闭 | 默认会开关 LED。 |

### 硬件层面注意点

- 该脚本会真实驱动舵机转动一圈，运行前必须保证电机轴或机构不会撞限位。
- 它使用 Mode 4，多圈位置值不会在一圈内回绕。
- 如果舵机已经挂在机械臂或绳驱结构上，不建议直接跑默认一圈测试。

## 6. `hardware/scripts/mirror_two_servos_to_mujoco.py`

### 作用

把两个离线/可手拧的 XL330 编码器角度同步到 MuJoCo 两关节模型中。这个脚本更像“硬件角度采集 + MuJoCo 镜像”，不是主动轨迹控制脚本。

### 使用的工作模式

```text
Operating Mode = 4  Extended Position Control Mode
```

初始化时会：

1. ping 舵机。
2. 写 `Torque Enable(64)=0`，使舵机脱力，允许手动旋转。
3. 写 `Operating Mode(11)=4`，使 `Present Position(132)` 可作为多圈位置反馈。
4. 读取型号、固件、位置、电压、温度、硬件错误。

如果 `--enforce-servo-limits` 开启，当手拧越过有效关节限位时，脚本会短暂使能扭矩并写 `Goal Position(116)` 把舵机推回边界。

### 默认硬件参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--port` | `COM3` | 串口名。 |
| `--baud` | `57600` | 通信波特率。 |
| `--ids` | `[10, 20]` | 两个舵机。 |
| `--rate` | `30.0 Hz` | 读取编码器和刷新 MuJoCo 的频率。 |
| `--duration` | `0.0` | `0` 表示无限运行，直到关闭窗口或 Ctrl+C。 |
| `--zero-current` | 开启 | 启动瞬间当前 tick 作为 `q=[0,0]`。 |
| `--zero-ticks` | `None` | 手动指定零点，优先级高于 `--zero-current`。 |
| `--signs` | `[1, 1]` | 舵机方向到关节方向的符号。 |
| `--limit-ratio` | `1.0` | 使用 XML 关节限位的比例。 |
| `--limit` | `None` | 可选对称绝对角限位。 |
| `--enforce-servo-limits` | 开启 | 越界时主动推回限位边界。 |
| `--profile-velocity` | `80` | 软限位推回时的速度档位。 |
| `--profile-acceleration` | `20` | 软限位推回时的加速度档位。 |
| `--print-every` | `0.5 s` | 控制台状态打印间隔。 |

### 主要换算

```text
q = sign * (PresentPosition - zero_tick) * 2*pi/4096
```

然后把 `q=[qa,qb]` 写入 MuJoCo 的：

- `joint1` 和 `joint2`：由 ID10 映射。
- `joint3` 和 `joint4`：由 ID20 映射。

### 硬件层面注意点

- 默认是脱力镜像，适合手动拖动/校准。
- 软限位功能会在越界时主动使能扭矩，仍需保证机构安全。
- 这是后续其他脚本复用的共享硬件通信层，包含串口打开、读写寄存器、sync read/write、tick-rad 换算和 MuJoCo 写入函数。

## 7. `hardware/scripts/track_sine_position_mode.py`

### 作用

位置模式下的正弦轨迹跟踪脚本。它不是电流模式，而是在 Mode 4 下通过 `Goal Position(116)` 控制舵机，同时把测得的关节角镜像到 MuJoCo。

控制结构是：

```text
q_ref(t), dq_ref(t)
        ↓
外部 PD 算出下一步 q_goal
        ↓
q_goal 转换成 goal_ticks
        ↓
写 Goal Position(116)
```

### 使用的工作模式

```text
Operating Mode = 4  Extended Position Control Mode
```

底层写入：

- `Goal Position(116)`：目标位置 tick。
- `Profile Velocity(112)`：位置模式速度档位。
- `Profile Acceleration(108)`：位置模式加速度档位。

底层读取：

- `Present Position(132)`：编码器反馈。
- `Present Current(126)`：输入电流 proxy。
- `Present PWM(124)`：PWM 输出 proxy。

### 默认硬件和轨迹参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--port` | `COM3` | 串口。 |
| `--baud` | `57600` | 波特率。 |
| `--ids` | `[10, 20]` | 两个舵机。 |
| `--duration` | `20.0 s` | 跟踪时长。 |
| `--rate` | `30.0 Hz` | 外部控制循环频率。 |
| `--signs` | `[1.0, 1.0]` | 方向符号。 |
| `--zero-ticks` | `None` | 默认启动 tick 作为零点。 |
| `--limit-ratio` | `0.80` | 只用 XML 限位的 80%，留安全余量。 |
| `--limit` | `None` | 可选对称角限位。 |
| `--biases` | `[0.0, 0.0]` rad | 正弦中心角。 |
| `--amplitudes` | `[1.0, 1.0]` rad | 正弦幅值。 |
| `--frequencies` | `[0.10, 0.10]` Hz | 正弦频率。 |
| `--phases` | `[0.0, 0.0]` rad | 正弦相位。 |
| `--kp` | `40.0` | 外部位置增量比例增益。 |
| `--kd` | `1.0` | 外部位置增量微分增益。 |
| `--max-step` | `0.10 rad/cycle` | 每个控制周期最大目标角变化量。 |
| `--profile-velocity` | `400` | 舵机内部位置规划速度档位。 |
| `--profile-acceleration` | `100` | 舵机内部位置规划加速度档位。 |
| `--feedback-stride` | `1` | 每 N 个周期读取一次电流/PWM。 |
| `--torque-constant-nm-per-a` | `0.354` | 把 `Present Current` 估算成扭矩 proxy 的系数。 |

### 轨迹公式

```text
q_ref(t)  = bias + amplitude * sin(2*pi*frequency*t + phase)
dq_ref(t) = amplitude * 2*pi*frequency * cos(2*pi*frequency*t + phase)
```

外部 PD 算的是目标位置增量：

```text
q_step = dt * (kp * (q_ref - q_meas) + kd * (dq_ref - dq_meas))
q_goal = clip(q_meas + q_step, lower, upper)
```

然后转换为：

```text
goal_tick = zero_tick + sign * q_goal / RAD_PER_TICK
```

### 硬件层面注意点

- 该脚本仍然依赖舵机内部位置控制环，PC 只是在外部持续更新 `Goal Position`。
- 它适合“让电机稳定跟踪正弦角度曲线”，但不是纯电流/力矩输入实验。
- 若目标是收集“电流输入 -> 角度反馈”的控制数据，应优先看 `hardware/scripts/track_sine_current_mode.py`。

## 8. `hardware/scripts/track_sine_current_mode.py`

### 作用

当前主力的 Mode 0 电流模式外部闭环脚本。它已经从原始点到点控制扩展为：

- `--reference sine`：默认正弦轨迹跟踪。
- `--reference point`：保留点到点目标角测试。

控制结构是：

```text
q_ref(t), dq_ref(t)
        ↓
读取 Present Position(132)
        ↓
编码器 tick -> q_meas
        ↓
外部 PD:
  Goal Current = kp_current * (q_ref - q_meas)
               + kd_current * (dq_ref - dq_meas)
        ↓
限幅到 ±max_current_ma
        ↓
写 Goal Current(102)
```

### 使用的工作模式

```text
Operating Mode = 0  Current Control Mode
```

初始化流程：

1. ping 舵机。
2. 写 `Torque Enable(64)=0`。
3. 写 `Operating Mode(11)=0`。
4. 读取型号、固件、当前位置、电压、温度、硬件错误。
5. 启动 tick 作为 `zero_ticks`。
6. 先写 `Goal Current(102)=0`。
7. 确认初始角度在有效限位内。
8. 用户输入 `RUN` 后，写 `Torque Enable(64)=1`。
9. 循环 sync read 位置/电流/PWM，计算并写入新的 `Goal Current(102)`。
10. 退出时写 0 电流并关闭扭矩。

### 默认硬件和控制参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--port` | `COM3` | 串口。 |
| `--baud` | `57600` | 波特率。 |
| `--ids` | `[10, 20]` | 两个舵机。 |
| `--reference` | `sine` | 默认跑正弦轨迹；可设为 `point`。 |
| `--targets` | `[1.0, 1.0]` rad | `point` 模式目标角。 |
| `--biases` | `[0.0, 0.0]` rad | 正弦中心角。 |
| `--amplitudes` | `[0.6, 0.6]` rad | 正弦幅值。 |
| `--frequencies` | `[0.10, 0.10]` Hz | 正弦频率。 |
| `--phases` | `[0.0, 0.0]` rad | 正弦相位。 |
| `--duration` | `20.0 s` | 跟踪时长。 |
| `--rate` | `30.0 Hz` | 外部电流闭环频率。 |
| `--signs` | `[1.0, 1.0]` | 方向符号。 |
| `--limit-ratio` | `0.80` | 使用 XML 关节限位的 80%。 |
| `--limit` | `None` | 可选对称绝对角限位。 |
| `--kp-current` | `200.0` | 位置误差到电流的比例增益，约 `mA/rad`。 |
| `--kd-current` | `15.0` | 速度误差到电流的微分增益，约 `mA/(rad/s)`。 |
| `--max-current-ma` | `80.0 mA` | 写入 `Goal Current` 的对称限幅。 |
| `--position-tolerance` | `0.03 rad` | 点到点收敛判据。 |
| `--settle-time` | `0.5 s` | 点到点模式稳定时间。 |
| `--feedback-stride` | `5` | 每 5 个周期读取一次温度/硬件错误。 |
| `--velocity-filter-alpha` | `0.35` | 编码器差分速度的一阶滤波系数。 |
| `--max-abs-velocity` | `4.0 rad/s` | 速度安全阈值。 |
| `--max-temperature-c` | `60.0 deg C` | 温度安全阈值。 |
| `--comm-retries` | `3` | 通信重试次数。 |
| `--comm-retry-delay` | `0.02 s` | 通信重试间隔。 |

### 读写的核心寄存器

| 类型 | 寄存器 | 说明 |
|---|---|---|
| 写 | `Operating Mode(11)=0` | 切到电流控制模式。 |
| 写 | `Torque Enable(64)` | 运行前使能，退出时关闭。 |
| 写 | `Goal Current(102)` | 外部 PD 输出的电流命令。 |
| 读 | `Present Position(132)` | 编码器位置反馈。 |
| 读 | `Present Current(126)` | 当前输入电流 proxy。 |
| 读 | `Present PWM(124)` | 当前 PWM proxy。 |
| 读 | `Present Temperature(146)` | 温度安全检查。 |
| 读 | `Hardware Error Status(70)` | 硬件错误安全检查。 |

### 电流模式的重点

Mode 0 下，舵机内部不负责“去某个角度”。它只接受 `Goal Current(102)`。真正的角度闭环在 PC 程序中：

```text
error = q_ref - q_meas
derror = dq_ref - dq_filtered
goal_current = kp_current * error + kd_current * derror
goal_current = clip(goal_current, -max_current_ma, +max_current_ma)
```

因此：

- `kp-current` 太小：跟踪慢、误差大。
- `kp-current` 太大：容易振荡、撞限位、触发速度保护。
- `kd-current` 太小：阻尼不足，正弦跟踪可能滞后/振荡。
- `kd-current` 太大：会放大编码器速度噪声，电流命令抖动。
- `max-current-ma` 太小：力不够，跟不上轨迹。
- `max-current-ma` 太大：硬件风险上升。

### 输出文件

默认输出到：

```text
hardware/outputs/current_mode_tracking/<timestamp>/
```

主要保存：

- `manifest.json`：运行参数、模式、限位、参考轨迹。
- `arrays/current_mode_log.npz`：时间序列原始数据。
- `metrics/current_mode_metrics.json`：RMSE、最大误差、峰值电流等。
- `metrics/reference_summary.json`：参考轨迹摘要。
- `figures/*.png`：角度跟踪、电流、速度、误差曲线。

## 9. 程序之间的硬件层面区别

| 程序 | 工作模式 | 写入命令 | 主要反馈 | 是否主动驱动 | 主要用途 |
|---|---:|---|---|---|---|
| `hardware/scripts/check_two_xl330_servos.py` | `4` | `Goal Position(116)` | `Present Position(132)`、`Moving(122)` | 是 | 基础联通和一圈运动测试 |
| `hardware/scripts/mirror_two_servos_to_mujoco.py` | `4` | 通常不写运动命令；越界时写 `Goal Position(116)` | `Present Position(132)` | 默认否；软限位时是 | 手动旋转舵机并同步到 MuJoCo |
| `hardware/scripts/track_sine_position_mode.py` | `4` | `Goal Position(116)` | `Present Position(132)`、`Present Current(126)`、`Present PWM(124)` | 是 | 位置模式正弦轨迹跟踪 |
| `hardware/scripts/track_sine_current_mode.py` | `0` | `Goal Current(102)` | `Present Position(132)`、`Present Current(126)`、`Present PWM(124)`、温度、错误状态 | 是 | 电流模式外部角度闭环/正弦轨迹跟踪 |

如果目标是“让关节角稳定跟踪轨迹”，位置模式脚本更容易调；如果目标是“研究外部控制器输出电流/力矩，编码器角度作为反馈”，应使用 `hardware/scripts/track_sine_current_mode.py` 的 Mode 0 路线。

## 10. 安全使用建议

1. 第一次运行任何主动驱动脚本时，先使用低幅值、低速度、低电流参数。
2. Mode 4 一圈测试不要在机械臂/绳驱结构连接后直接运行，除非确认不会撞限位。
3. Mode 0 电流模式必须限制 `--max-current-ma`，并保留温度、速度、硬件错误检查。
4. 每次实验记录 `zero_ticks`、`signs`、`limit-ratio`、`amplitudes`、`frequencies`、`kp/kd` 或 `kp-current/kd-current`。
5. `Present Current(126)` 在 XL330 上更适合当作输入电流 proxy，不应直接等同于精确关节力矩。
6. 若要保证可复现实验，固定初始机械姿态，或显式指定 `--zero-ticks`。
