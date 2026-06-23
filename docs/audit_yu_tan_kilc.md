# Yu-Tan KiLC 全套程序审计报告

> 审计日期: 2026-06-23
> 审计范围: `yu_tan_kilc.py`, `kilc_tracking.py`, `run_kilc.py`, `runtime.py` (ContinuousDKUCModel), `continuous_dkuc.py`, `networks.py` (ContinuousDKUCNetwork), `registry.py`, 相关 config/`__init__.py`

---

## 涉及文件一览

| 文件 | 角色 |
|------|------|
| `src/koopman_control/control/yu_tan_kilc.py` | 核心算法：`YuTanKILCController` |
| `src/koopman_control/models/networks.py` | `ContinuousDKUCNetwork`（ż = A z + B u） |
| `src/koopman_control/models/runtime.py` | `ContinuousDKUCModel`（artifact 加载与推理） |
| `src/koopman_control/models/registry.py` | `load_continuous_dkuc_model` |
| `src/koopman_control/training/continuous_dkuc.py` | 连续时间 DKUC 训练流程 |
| `src/cdsm/runtime/kilc_tracking.py` | 多轮试验编排 + 单轮执行 |
| `experiments/deployment_pipeline/run_kilc.py` | 笛卡尔圆轨迹实验入口 |
| `configs/deployment/kilc_cartesian_circle.json` | 实验默认配置 |
| `src/koopman_control/control/__init__.py` | control 包导出 |

---

## 问题清单

### 🔴 P0 — Critical：ILC 时序对齐错误 **[✅ 已修复 2026-06-23]**

**涉及文件**: `src/cdsm/runtime/kilc_tracking.py`

**问题描述**:

`_run_trial` 在 `plant.step()` **之前**录制 `z_meas` 和 `z_ref`，导致 ILC 误差信号与控制量错位一个时间步。

当前代码（L89–L108）：
```python
for k in range(len(t) - 1):
    measured = plant.read_state()      # 状态 x[k]
    z_meas = model.lift(measured)      # z[k] ← step 之前
    z_ref = model.lift(x_ref[k])       # z_ref[k] ← 应该用 x_ref[k+1]
    raw_cmd = u_total_phys[k]          # 控制 u[k]，影响的是 x[k+1]
    ...
    plant.step()                       # x[k] → x[k+1]

    records["z_meas"].append(z_meas)   # 录的是 z[k]
    records["z_ref"].append(z_ref)     # 录的是 z_ref[k]
```

同时 `run_yu_tan_kilc_tracking`（L158）构造参考：
```python
z_ref_all = _lift_sequence(model, x_ref[:-1])   # z_ref[0..N-2]
```

而 `controller.update()`（L195–200）中：
```python
update = controller.update(
    controls_norm["u_total"],   # u_prev[0..N-2]
    z_ref_all,                  # z_ref[0..N-2]
    log["z_meas"],              # z_meas[0..N-2]
    log["t"],
)
# 内部: u_ilc[k] = u_prev[k] + ρ * (e_z[k] @ L.T)
# e_z[k] = z_ref[k] - z_meas[k]
```

**根因**: `u[k]` 的效应体现在 `x[k+1]`（即 `z[k+1]`），但 ILC 用 `e_z[k]` 来修正 `u[k]`。导致：
- `u[0]` 几乎不被修正（`x[0]=x_ref[0]` → `e_z[0]≈0`）
- `u[k]` 的修正来自 `e_z[k]`，但 `e_z[k]` 反映的是 `u[k-1]` 的效果
- 整个控制序列的修正系统性地偏移了一个时间步

**修复方案**:

1. `_run_trial` 中改为 `plant.step()` **之后**录 `z`：
```python
plant.step()
measured_after = plant.read_state()
z_meas = model.lift(measured_after)    # z[k+1]
z_ref = model.lift(x_ref[k + 1])       # z_ref[k+1]
```

2. `run_yu_tan_kilc_tracking` 中 `z_ref_all` 同步改为 `x_ref[1:]`：
```python
z_ref_all = _lift_sequence(model, x_ref[1:])   # z_ref[1..N-1]
```

---

### 🟡 P1 — `control_limit` 在归一化空间做裁剪，量纲错误 **[✅ 已修复 2026-06-23]**

**涉及文件**: `src/koopman_control/control/yu_tan_kilc.py`

**问题描述**:

`YuTanKILCConfig.control_limit` 默认值 `120.0` 是物理单位（Nm），但 `update()` 方法中用它裁剪的 `u_total` 是归一化量（量级 ~O(1)）：

```python
# yu_tan_kilc.py L111–L113
limit = float(self.config.control_limit)   # = 120.0
u_total = np.clip(u_total, -limit, limit)  # 对归一化量做 ±120 → 几乎不裁剪
```

真正的物理裁剪在 `_run_trial`（kilc_tracking.py L94）：
```python
cmd = np.clip(raw_cmd, -tau_limit, tau_limit)   # 物理裁剪
```

**后果**:
- 归一化空间的 ±120 裁剪形同虚设（`u_norm * u_std ≈ u_norm * 50`，120 对应 ~6000 Nm）
- ILC 更新出的 `u_total`（已归一化空间裁剪）与 `_run_trial` 实际执行的物理裁剪不一致
- ILC 认为执行了值 A，实际执行了 clip(A*std, ±tau_limit)，下次更新基于错误的前提

**修复方案**:

删除 `YuTanKILCConfig` 中的 `control_limit` 字段（以及 `update()` 中 L111–L113 的裁剪代码）。所有物理裁剪统一由 `_run_trial` 的 `tau_limit` 负责。

涉及改动：
- `yu_tan_kilc.py`: 从 `YuTanKILCConfig` 删除 `control_limit: float = 120.0`
- `yu_tan_kilc.py`: 从 `update()` 删除 L111–L113
- `configs/deployment/kilc_cartesian_circle.json`: 从 `kilc` 块删除 `"control_limit": 120.0`

---

### 🟡 P1 — `_control_delta_to_physical` 缺失均值偏移 **[✅ 已修复 2026-06-23]**

**涉及文件**: `src/cdsm/runtime/kilc_tracking.py`

**问题描述**:

```python
# kilc_tracking.py L52–L54
def _control_delta_to_physical(model, values_norm: np.ndarray) -> np.ndarray:
    std = np.asarray(model.u_normer.std, dtype=np.float64).reshape(1, -1)
    return np.asarray(values_norm, dtype=np.float64) * std
    # 缺少: + model.u_normer.mean
```

标准逆归一化公式是 `u_phys = u_norm * std + mean`。当前实现假设 `u_mean ≈ 0`。

- 第 0 轮：`u_total_norm = zeros` → `u_phys = 0 * std = 0` ✓（零力矩启动）
- 第 j 轮：`u_total_norm = u_prev_norm + Δu` → `u_phys` 永远缺一个常值偏置 `mean`
- 如果训练数据中 `u_mean ≠ 0`，ILC 永远学不到正确的偏置

**修复方案**:

方案 A（推荐）：如果训练数据的 `u_mean ≈ 0` 确实成立，改名并加断言：
```python
def _control_norm_to_physical(model, values_norm):
    assert np.allclose(model.u_normer.mean, 0, atol=1e-6), \
        "KiLC assumes zero-centered control normalization"
    return values_norm * model.u_normer.std
```

方案 B：做完整逆归一化：
```python
def _control_norm_to_physical(model, values_norm):
    mean = np.asarray(model.u_normer.mean, dtype=np.float64).reshape(1, -1)
    std = np.asarray(model.u_normer.std, dtype=np.float64).reshape(1, -1)
    return values_norm * std + mean
```

同时更新所有调用点（`_run_trial` 中 4 处调用）。

---

### 🟡 P2 — `ContinuousDKUCModel` 缺少 `recover_control` 方法

**涉及文件**: `src/koopman_control/models/runtime.py`

**问题描述**:

`ControlReadyModel` 协议（`models/base.py`）要求 `recover_control(x_phys, internal_control) -> u_phys`。`ContinuousDKUCModel` 只定义了 `recover_control_delta`，缺少通用的 `recover_control`。

当前 KILC 代码路径不走 `recover_control`（直接通过 `_control_delta_to_physical` 做转换），所以不会触发崩溃。但如果有其他代码调用了 `model.recover_control(...)` 会直接 `AttributeError`。

**修复方案**:

在 `ContinuousDKUCModel` 中添加：
```python
def recover_control(
    self,
    x_phys: np.ndarray,
    internal_control: np.ndarray,
) -> np.ndarray:
    """Map normalized control delta back to physical torque."""
    del x_phys
    return self.recover_control_delta(internal_control)
```

---

### 🟡 P2 — 缺少收敛检测和早停机制

**涉及文件**: `src/cdsm/runtime/kilc_tracking.py`

**问题描述**:

`run_yu_tan_kilc_tracking` 永远跑满 `max_trials`：
```python
for trial in range(int(max_trials)):    # L174
```

`_summarize_trial` 计算了 `lifted_error_rms`，但没有用于判断收敛。之前版本的 `converged()` 和 `record_trial()` 在重构中被移除。

**修复方案**:

在 `run_yu_tan_kilc_tracking` 中添加早停逻辑：
```python
convergence_window = 3
convergence_tol = 1e-4   # lifted_error_rms 的相对变化阈值

for trial in range(int(max_trials)):
    ...
    summaries.append(summary)

    # 早停检查
    if trial >= convergence_window:
        recent = [s.lifted_error_rms for s in summaries[-convergence_window:]]
        spread = max(recent) - min(recent)
        if spread < convergence_tol:
            if show_progress:
                print(f"[KILC] converged at trial {trial}")
            break
```

---

### 🟢 P3 — 函数命名误导

**涉及文件**: `src/cdsm/runtime/kilc_tracking.py`

**问题**: `_control_delta_to_physical` 处理的是**绝对控制量**（`u_total`），不是增量（delta）。

**修复**: 重命名为 `_control_norm_to_physical`，或 `_scale_normalized_control_to_physical`。

---

### 🟢 P3 — `__init__.py` 缺少 `build_ramp_reference` 导出

**涉及文件**: `src/koopman_control/control/__init__.py`

**问题**: `build_ramp_reference` 在 `finite_horizon_lqr.py` 中定义，是通用工具函数，但不在 `control/__init__.py` 的 `__all__` 列表中。

**修复**: 从 `finite_horizon_lqr` 导入 `build_ramp_reference` 并加入 `__all__`。

---

### 🟢 P3 — Config 中 `artifact_dir` 是占位路径

**涉及文件**: `configs/deployment/kilc_cartesian_circle.json`

**问题**: 
```json
"artifact_dir": "outputs/models/deployment_pipeline/latest_continuous_dkuc_placeholder"
```
该路径不存在。实际已训练好的连续时间 DKUC 模型在：
```
outputs/models/deployment_pipeline/20260623_154959_train_dkuc_continuous_yu_tan_kilc_firstflow
```

**修复**: 更新为实际路径，或通过 CLI `--artifact_dir` 参数传入。

---

### 🟢 P3 — 记录字段 `solve_ms` 恒为零

**涉及文件**: `src/cdsm/runtime/kilc_tracking.py` L115

**问题**: KiLC 是纯前馈无需求解优化问题，`solve_ms` 恒为 0。保留此字段是为了与 LQR/MPC 日志格式兼容，但可以考虑在 KiLC 特有的日志中移除以避免混淆。

**建议**: 保留以维持接口兼容性，加注释说明。

---

## 修复汇总表

| # | 优先级 | 问题 | 涉及文件 | 改动量 |
|---|--------|------|----------|--------|
| 1 | 🔴 P0 | ILC 时序对齐 | `kilc_tracking.py` | ~5 行 |
| 2 | 🟡 P1 | `control_limit` 量纲错误 | `yu_tan_kilc.py`, `kilc_cartesian_circle.json` | 删 ~6 行 |
| 3 | 🟡 P1 | 缺失 mean 偏移 | `kilc_tracking.py` | ~3 行 |
| 4 | 🟡 P2 | 缺 `recover_control` | `runtime.py` | 加 ~6 行 |
| 5 | 🟡 P2 | 无早停 | `kilc_tracking.py` | 加 ~8 行 |
| 6 | 🟢 P3 | 函数命名 | `kilc_tracking.py` | 改 5 处调用 |
| 7 | 🟢 P3 | 缺少导出 | `control/__init__.py` | 加 2 行 |
| 8 | 🟢 P3 | config 占位路径 | `kilc_cartesian_circle.json` | 改 1 行 |

---

## 已确认正确的部分

以下方面审查通过，无需修改：

- ✅ **连续时间训练流程**: 四项损失函数（物理导数 + 提升导数 + 读出 + rollout）设计合理
- ✅ **`A_c` 初始化**: `ContinuousDKUCNetwork._init_linear()` 将 `A` 初始化为零矩阵（对应连续时间平衡点），正确
- ✅ **增益矩阵**: `L = B_c^T / diag(||B_c columns||²)` 列归一化转置，比纯 pinv 鲁棒
- ✅ **三通道控制结构**: ILC（P 型）+ Adaptive（积分型）+ Robust（tanh 型）完整复现 Yu & Tan 控制律
- ✅ **Q-filter**: `scipy.signal.filtfilt` 零相位应用，顺序正确（滤波 → 裁剪）
- ✅ **模型校验**: `kilc_tracking.py:151` 的 `control_mode == "zdot=A_c z+B_c u_norm"` 检查确保了模型兼容性
- ✅ **文件清理**: 无残留旧文件（`iterative_learning.py`、`kilc_runtime.py` 等均已删除）
- ✅ **导入链**: control 包 → yu_tan_kilc / cdsm.runtime → kilc_tracking → registry 全链路可解析
