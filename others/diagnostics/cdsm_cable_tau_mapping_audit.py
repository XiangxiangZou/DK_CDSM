"""
cdsm_cable_tau_mapping_audit.py
=================================
独立诊断：拮抗绳驱映射 tau_cmd -> F_cable 是否能在 MuJoCo 上产生目标关节力矩 tau_act.

对比量
------
    tau_cmd : PD (或其它) 给出的期望关节力矩 [tau_a, tau_b]
    tau_act : 下发 F_cable 后, 在 x_k 处 mj_forward 读取
              tau_a = qfrc_actuator[j1] + qfrc_actuator[j2]
              tau_b = qfrc_actuator[j3] + qfrc_actuator[j4]

映射器内部残差 res_a, res_b 也会统计 (拮抗子问题 |tau_des - tau_achieved|).

运行
----
    python cdsm_cable_tau_mapping_audit.py

快速试跑 (少轨迹):
    python cdsm_cable_tau_mapping_audit.py --traj 10 --steps 100

指定构型网格 + 随机力矩 (不跑 PD 轨迹):
    python cdsm_cable_tau_mapping_audit.py --mode grid --grid_q_deg -60 -30 0 30 60
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils_plot import get_save_dir, save_figure

XML_DEFAULT = str(
    Path(__file__).resolve().parents[1]
    / "assets"
    / "multi_joint_cable_driven_space_robot.xml"
)
ACTIVE_JOINTS = ("joint1", "joint3")
MIMIC_JOINTS = {"joint2": "joint1", "joint4": "joint3"}
CABLE_NAMES = [
    "cable11", "cable12", "cable13", "cable14",
    "cable21", "cable22", "cable23", "cable24",
]
ACTUATOR_NAMES = ["winch_c" + n[len("cable"):] for n in CABLE_NAMES]
IDX_F1P = [0, 2]
IDX_F1M = [1, 3]
IDX_F2P = [4, 6]
IDX_F2M = [5, 7]
F_PRELOAD = 20.0


def _require_mujoco() -> object:
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "需要安装 mujoco: pip install mujoco  (或 pip install -r requirements.txt)"
        ) from e
    return mujoco


def name_to_joint_id(model: object, name: str) -> int:
    mujoco = _require_mujoco()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return int(jid)


def name_to_actuator_id(model: object, name: str) -> int:
    mujoco = _require_mujoco()
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found: {name}")
    return int(aid)


def name_to_tendon_id(model: object, name: str) -> int:
    mujoco = _require_mujoco()
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if tid < 0:
        raise ValueError(f"Tendon not found: {name}")
    return int(tid)


def load_cable_model(xml_path: str, dt: float) -> Tuple[object, object, object, Dict[str, np.ndarray]]:
    mujoco = _require_mujoco()
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(dt)
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)

    active_joint_ids = np.array([name_to_joint_id(model, n) for n in ACTIVE_JOINTS], dtype=int)
    active_qpos = np.array([model.jnt_qposadr[j] for j in active_joint_ids], dtype=int)
    active_dof = np.array([model.jnt_dofadr[j] for j in active_joint_ids], dtype=int)

    jnt_id = {n: name_to_joint_id(model, n) for n in ("joint1", "joint2", "joint3", "joint4")}
    dof_all = {n: int(model.jnt_dofadr[jnt_id[n]]) for n in jnt_id}

    mimic_pairs = []
    for mimic, source in MIMIC_JOINTS.items():
        mimic_jid = name_to_joint_id(model, mimic)
        source_jid = name_to_joint_id(model, source)
        mimic_pairs.append((model.jnt_qposadr[mimic_jid], model.jnt_qposadr[source_jid]))

    indices = {
        "active_qpos": active_qpos,
        "active_dof": active_dof,
        "mimic_pairs": np.array(mimic_pairs, dtype=int),
        "actuator_ids": np.array([name_to_actuator_id(model, n) for n in ACTUATOR_NAMES], dtype=int),
        "tendon_ids": np.array([name_to_tendon_id(model, n) for n in CABLE_NAMES], dtype=int),
        "dof_j1": dof_all["joint1"],
        "dof_j2": dof_all["joint2"],
        "dof_j3": dof_all["joint3"],
        "dof_j4": dof_all["joint4"],
    }
    return model, data, scratch, indices


def set_active_state(
    model: object, data: object, indices: Dict[str, np.ndarray], q: np.ndarray, dq: np.ndarray
) -> None:
    mujoco = _require_mujoco()
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[indices["active_qpos"]] = q
    data.qvel[indices["active_dof"]] = dq
    for mimic_qpos, source_qpos in indices["mimic_pairs"]:
        data.qpos[mimic_qpos] = data.qpos[source_qpos]
    mujoco.mj_forward(model, data)


def read_actual_joint_torque(data: object, indices: Dict[str, np.ndarray]) -> np.ndarray:
    qf = np.asarray(data.qfrc_actuator, dtype=np.float64)
    j1, j2, j3, j4 = (int(indices[k]) for k in ("dof_j1", "dof_j2", "dof_j3", "dof_j4"))
    return np.array([qf[j1] + qf[j2], qf[j3] + qf[j4]], dtype=np.float64)


def compute_tendon_jacobian_fd(
    model: object,
    scratch: object,
    q_ref: np.ndarray,
    tendon_ids_ordered: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    mujoco = _require_mujoco()
    nv = model.nv
    J = np.zeros((len(tendon_ids_ordered), nv), dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64).copy()
    for j in range(nv):
        scratch.qpos[:] = q_ref
        scratch.qpos[j] = q_ref[j] + eps
        mujoco.mj_fwdPosition(model, scratch)
        L_plus = np.array(scratch.ten_length, dtype=np.float64)[tendon_ids_ordered]
        scratch.qpos[:] = q_ref
        scratch.qpos[j] = q_ref[j] - eps
        mujoco.mj_fwdPosition(model, scratch)
        L_minus = np.array(scratch.ten_length, dtype=np.float64)[tendon_ids_ordered]
        J[:, j] = (L_plus - L_minus) / (2.0 * eps)
    return J


def _solve_antagonistic_pair(
    m_p: float, m_m: float, tau_des: float, f_pre: float, f_max: float
) -> Tuple[float, float, float]:
    u_max = max(f_max - f_pre, 0.0) if f_max > 0 else float("inf")
    tau_base = (m_p + m_m) * f_pre
    tau_eff = tau_des - tau_base
    eps = 1e-12
    candidates: List[Tuple[float, float, float, float]] = [(0.0, 0.0, abs(tau_eff), 0.0)]
    if abs(m_p) > eps:
        u = max(tau_eff / m_p, 0.0)
        u_clip = min(u, u_max) if np.isfinite(u_max) else u
        candidates.append((u_clip, 0.0, abs(tau_eff - m_p * u_clip), u_clip))
    if abs(m_m) > eps:
        u = max(tau_eff / m_m, 0.0)
        u_clip = min(u, u_max) if np.isfinite(u_max) else u
        candidates.append((0.0, u_clip, abs(tau_eff - m_m * u_clip), u_clip))
    u_p, u_m, res, _ = min(candidates, key=lambda c: (c[2], c[3]))
    return f_pre + u_p, f_pre + u_m, res


def cable_antagonistic_map(
    tau_a_des: float,
    tau_b_des: float,
    J: np.ndarray,
    dof_j1: int,
    dof_j2: int,
    dof_j3: int,
    dof_j4: int,
    f_pre: float = F_PRELOAD,
    f_max: float = 0.0,
) -> Tuple[np.ndarray, float, float]:
    """返回 (F_cable[8], map_res_a, map_res_b). f_max<=0 表示不裁切张力上限."""
    a = J[:, dof_j1] + J[:, dof_j2]
    b = J[:, dof_j3] + J[:, dof_j4]
    m_p1 = a[IDX_F1P[0]] + a[IDX_F1P[1]]
    m_m1 = a[IDX_F1M[0]] + a[IDX_F1M[1]]
    m_p2 = b[IDX_F2P[0]] + b[IDX_F2P[1]]
    m_m2 = b[IDX_F2M[0]] + b[IDX_F2M[1]]
    f1p, f1m, res_a = _solve_antagonistic_pair(m_p1, m_m1, tau_a_des, f_pre, f_max)
    f2p, f2m, res_b = _solve_antagonistic_pair(m_p2, m_m2, tau_b_des, f_pre, f_max)
    F = np.empty(8, dtype=np.float64)
    F[IDX_F1P] = f1p
    F[IDX_F1M] = f1m
    F[IDX_F2P] = f2p
    F[IDX_F2M] = f2m
    return F, res_a, res_b


def measure_mapping_at_state(
    model: object,
    data: object,
    scratch: object,
    indices: Dict[str, np.ndarray],
    q: np.ndarray,
    dq: np.ndarray,
    tau_cmd: np.ndarray,
) -> Dict[str, float]:
    """在给定 (q,dq) 下测试一次映射."""
    mujoco = _require_mujoco()
    set_active_state(model, data, indices, q, dq)
    dof = tuple(int(indices[k]) for k in ("dof_j1", "dof_j2", "dof_j3", "dof_j4"))
    J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
    F, res_a, res_b = cable_antagonistic_map(
        float(tau_cmd[0]), float(tau_cmd[1]), J, *dof
    )
    data.ctrl[indices["actuator_ids"]] = F
    mujoco.mj_forward(model, data)
    tau_act = read_actual_joint_torque(data, indices)
    e = tau_act - tau_cmd
    return {
        "tau_cmd_a": float(tau_cmd[0]),
        "tau_cmd_b": float(tau_cmd[1]),
        "tau_act_a": float(tau_act[0]),
        "tau_act_b": float(tau_act[1]),
        "err_a": float(e[0]),
        "err_b": float(e[1]),
        "err_norm": float(np.linalg.norm(e)),
        "map_res_a": float(res_a),
        "map_res_b": float(res_b),
        "q_a": float(q[0]),
        "q_b": float(q[1]),
    }


# ---------------------------------------------------------------------------
# PD-style random trajectory audit (same protocol as training collection)
# ---------------------------------------------------------------------------
@dataclass
class AuditConfig:
    traj_count: int
    steps: int
    dt: float
    seed: int
    q_init_range: float
    dq_init_range: float
    amp_range: Tuple[float, float]
    omega_range: Tuple[float, float]
    kp: Tuple[float, float]
    kd: Tuple[float, float]


@dataclass
class SineRefParams:
    A1: float
    A2: float
    w1: float
    w2: float
    phi1: float
    phi2: float


def _sample_sine_params(rng: np.random.RandomState, cfg: AuditConfig) -> SineRefParams:
    amp_lo, amp_hi = cfg.amp_range
    w_lo, w_hi = cfg.omega_range
    return SineRefParams(
        A1=rng.uniform(amp_lo, amp_hi),
        A2=rng.uniform(amp_lo * 0.5, amp_hi * 0.8),
        w1=rng.uniform(w_lo, w_hi),
        w2=rng.uniform(w_lo * 1.3, w_hi * 1.8),
        phi1=rng.uniform(0.0, 2.0 * np.pi),
        phi2=rng.uniform(0.0, 2.0 * np.pi),
    )


def eval_sine_ref(params: SineRefParams, t: float) -> Tuple[float, float]:
    q = params.A1 * np.sin(params.w1 * t + params.phi1) + params.A2 * np.sin(params.w2 * t + params.phi2)
    dq = (
        params.A1 * params.w1 * np.cos(params.w1 * t + params.phi1)
        + params.A2 * params.w2 * np.cos(params.w2 * t + params.phi2)
    )
    return float(q), float(dq)


def pd_torque(
    q: np.ndarray,
    dq: np.ndarray,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return kp * (q_ref - q) + kd * (dq_ref - dq)


def run_pd_trajectory_audit(
    xml: str,
    cfg: AuditConfig,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    mujoco = _require_mujoco()
    model, data, scratch, indices = load_cable_model(xml, cfg.dt)
    rng = np.random.RandomState(int(cfg.seed))

    kp = np.array(cfg.kp, dtype=np.float64)
    kd = np.array(cfg.kd, dtype=np.float64)
    dof = tuple(int(indices[k]) for k in ("dof_j1", "dof_j2", "dof_j3", "dof_j4"))

    records: List[Dict[str, float]] = []

    for _ in range(int(cfg.traj_count)):
        q0 = rng.uniform(-cfg.q_init_range, cfg.q_init_range, size=2)
        dq0 = rng.uniform(-cfg.dq_init_range, cfg.dq_init_range, size=2)
        set_active_state(model, data, indices, q0, dq0)
        ref_a = _sample_sine_params(rng, cfg)
        ref_b = _sample_sine_params(rng, cfg)

        for k in range(int(cfg.steps)):
            t = k * float(cfg.dt)
            qa_ref, dqa_ref = eval_sine_ref(ref_a, t)
            qb_ref, dqb_ref = eval_sine_ref(ref_b, t)
            q_ref = np.array([qa_ref, qb_ref])
            dq_ref = np.array([dqa_ref, dqb_ref])

            q = np.array(data.qpos[indices["active_qpos"]], dtype=np.float64)
            dq = np.array(data.qvel[indices["active_dof"]], dtype=np.float64)
            tau_cmd = pd_torque(q, dq, q_ref, dq_ref, kp, kd)

            rec = measure_mapping_at_state(model, data, scratch, indices, q, dq, tau_cmd)
            records.append(rec)
            mujoco.mj_step(model, data)

    return _summarize_records(records, cfg), _records_to_arrays(records)


def run_grid_audit(
    xml: str,
    dt: float,
    q_deg_list: List[float],
    tau_samples_per_q: int,
    tau_range: float,
    seed: int,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """在构型网格上随机采样 tau_cmd, 静止 dq=0."""
    mujoco = _require_mujoco()
    model, data, scratch, indices = load_cable_model(xml, dt)
    rng = np.random.RandomState(seed)

    records: List[Dict[str, float]] = []
    for qa_deg in q_deg_list:
        for qb_deg in q_deg_list:
            q = np.deg2rad([qa_deg, qb_deg])
            dq = np.zeros(2)
            for _ in range(tau_samples_per_q):
                tau_cmd = rng.uniform(-tau_range, tau_range, size=2)
                rec = measure_mapping_at_state(model, data, scratch, indices, q, dq, tau_cmd)
                records.append(rec)

    meta = {"mode": "grid", "q_deg_list": q_deg_list, "tau_samples_per_q": tau_samples_per_q}
    stats = _summarize_records(records, meta)
    return stats, _records_to_arrays(records)


def _records_to_arrays(records: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    if not records:
        return {}
    keys = records[0].keys()
    return {k: np.array([r[k] for r in records], dtype=np.float64) for k in keys}


def _summarize_records(records: List[Dict[str, float]], meta: object) -> Dict[str, object]:
    if not records:
        raise RuntimeError("No samples collected.")

    err_norm = np.array([r["err_norm"] for r in records], dtype=np.float64)
    err_a = np.array([r["err_a"] for r in records], dtype=np.float64)
    err_b = np.array([r["err_b"] for r in records], dtype=np.float64)
    map_res_a = np.array([r["map_res_a"] for r in records], dtype=np.float64)
    map_res_b = np.array([r["map_res_b"] for r in records], dtype=np.float64)
    tau_cmd = np.array([[r["tau_cmd_a"], r["tau_cmd_b"]] for r in records], dtype=np.float64)
    rel = err_norm / np.maximum(np.linalg.norm(tau_cmd, axis=1), 1e-9)

    def pct(x: np.ndarray, p: float) -> float:
        return float(np.percentile(x, p))

    return {
        "samples": len(records),
        "meta": meta if isinstance(meta, dict) else {
            "traj_count": getattr(meta, "traj_count", None),
            "steps": getattr(meta, "steps", None),
            "q_init_range": getattr(meta, "q_init_range", None),
            "amp_range": list(getattr(meta, "amp_range", ())),
            "kp": list(getattr(meta, "kp", ())),
            "kd": list(getattr(meta, "kd", ())),
        },
        "err_norm": {
            "mean": float(err_norm.mean()),
            "std": float(err_norm.std()),
            "max": float(err_norm.max()),
            "p50": pct(err_norm, 50),
            "p90": pct(err_norm, 90),
            "p95": pct(err_norm, 95),
            "p99": pct(err_norm, 99),
        },
        "err_joint_a": {
            "mean_abs": float(np.mean(np.abs(err_a))),
            "max_abs": float(np.max(np.abs(err_a))),
            "rmse": float(np.sqrt(np.mean(err_a * err_a))),
        },
        "err_joint_b": {
            "mean_abs": float(np.mean(np.abs(err_b))),
            "max_abs": float(np.max(np.abs(err_b))),
            "rmse": float(np.sqrt(np.mean(err_b * err_b))),
        },
        "relative_err_norm": {
            "mean": float(rel.mean()),
            "p50": pct(rel, 50),
            "p95": pct(rel, 95),
            "max": float(rel.max()),
        },
        "map_residual": {
            "res_a_mean": float(map_res_a.mean()),
            "res_a_p95": pct(map_res_a, 95),
            "res_a_max": float(map_res_a.max()),
            "res_b_mean": float(map_res_b.mean()),
            "res_b_p95": pct(map_res_b, 95),
            "res_b_max": float(map_res_b.max()),
            "frac_res_a_gt_1Nm": float(np.mean(map_res_a > 1.0)),
            "frac_res_b_gt_1Nm": float(np.mean(map_res_b > 1.0)),
        },
        "frac_err_norm_gt": {
            "1Nm": float(np.mean(err_norm > 1.0)),
            "5Nm": float(np.mean(err_norm > 5.0)),
            "10Nm": float(np.mean(err_norm > 10.0)),
        },
    }


def plot_audit(arrays: Dict[str, np.ndarray], prefix: str = "cable_map") -> None:
    err_norm = arrays["err_norm"]
    tau_cmd_a = arrays["tau_cmd_a"]
    tau_act_a = arrays["tau_act_a"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(err_norm, bins=50, density=True, alpha=0.75, color="steelblue")
    axes[0].set_xlabel("|tau_act - tau_cmd| (Nm)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Mapping error norm")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(tau_cmd_a, tau_act_a, s=6, alpha=0.35)
    lim = max(np.max(np.abs(tau_cmd_a)), np.max(np.abs(tau_act_a)), 1.0) * 1.05
    axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    axes[1].set_xlabel("tau_cmd_a (Nm)")
    axes[1].set_ylabel("tau_act_a (Nm)")
    axes[1].set_title("Joint qa: command vs actual")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="box")
    fig.tight_layout()
    save_figure(f"{prefix}_audit")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(arrays["map_res_a"], err_norm, s=6, alpha=0.35, label="qa")
    ax.scatter(arrays["map_res_b"], err_norm, s=6, alpha=0.35, label="qb")
    ax.set_xlabel("Antagonistic map residual (Nm)")
    ax.set_ylabel("|tau_act - tau_cmd| (Nm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    save_figure(f"{prefix}_err_vs_map_res")
    plt.close(fig2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit cable antagonistic map: tau_cmd vs MuJoCo tau_act.")
    p.add_argument("--xml", default=XML_DEFAULT)
    p.add_argument("--mode", choices=["pd", "grid"], default="pd", help="pd: PD轨迹采样式; grid: 构型网格+随机力矩")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--traj", type=int, default=30, help="PD mode: number of trajectories")
    p.add_argument("--steps", type=int, default=250, help="PD mode: steps per trajectory")
    p.add_argument("--q_init_range", type=float, default=1.5)
    p.add_argument("--dq_init_range", type=float, default=0.5)
    p.add_argument("--amp_min", type=float, default=0.4)
    p.add_argument("--amp_max", type=float, default=1.5)
    p.add_argument("--omega_min", type=float, default=0.4)
    p.add_argument("--omega_max", type=float, default=1.2)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)

    p.add_argument(
        "--grid_q_deg",
        type=float,
        nargs="+",
        default=[-60.0, -30.0, 0.0, 30.0, 60.0],
        help="Grid mode: joint angles (deg) for qa and qb",
    )
    p.add_argument("--grid_tau_samples", type=int, default=40, help="Grid mode: random tau per (qa,qb)")
    p.add_argument("--grid_tau_range", type=float, default=80.0, help="Grid mode: |tau| sample range (Nm)")
    return p


def print_stats(stats: Dict[str, object]) -> None:
    en = stats["err_norm"]
    mr = stats["map_residual"]
    print("-" * 60)
    print(f"  samples              : {stats['samples']}")
    print(f"  |tau_act-tau_cmd| mean : {en['mean']:.4g} Nm")
    print(f"  |e| p50 / p95 / max    : {en['p50']:.4g} / {en['p95']:.4g} / {en['max']:.4g} Nm")
    print(f"  RMSE qa / qb           : {stats['err_joint_a']['rmse']:.4g} / {stats['err_joint_b']['rmse']:.4g} Nm")
    print(f"  rel |e| p50 / p95      : {stats['relative_err_norm']['p50']:.4g} / {stats['relative_err_norm']['p95']:.4g}")
    print(f"  map res_a p95 / max    : {mr['res_a_p95']:.4g} / {mr['res_a_max']:.4g} Nm")
    print(f"  map res_b p95 / max    : {mr['res_b_p95']:.4g} / {mr['res_b_max']:.4g} Nm")
    print(f"  frac |e|>5Nm           : {stats['frac_err_norm_gt']['5Nm']:.3f}")
    print(f"  frac map_res>1Nm (qa)  : {mr['frac_res_a_gt_1Nm']:.3f}")


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    out_dir = Path(get_save_dir())

    if args.mode == "pd":
        cfg = AuditConfig(
            traj_count=args.traj,
            steps=args.steps,
            dt=args.dt,
            seed=args.seed,
            q_init_range=args.q_init_range,
            dq_init_range=args.dq_init_range,
            amp_range=(args.amp_min, args.amp_max),
            omega_range=(args.omega_min, args.omega_max),
            kp=(args.kp_a, args.kp_b),
            kd=(args.kd_a, args.kd_b),
        )
        print(f"[audit] PD protocol: {args.traj} traj x {args.steps} steps")
        stats, arrays = run_pd_trajectory_audit(args.xml, cfg)
    else:
        print(f"[audit] Grid protocol: q in {args.grid_q_deg} deg, {args.grid_tau_samples} tau / cell")
        stats, arrays = run_grid_audit(
            args.xml,
            args.dt,
            list(args.grid_q_deg),
            args.grid_tau_samples,
            args.grid_tau_range,
            args.seed,
        )

    plot_audit(arrays)
    out_json = out_dir / "tau_cmd_vs_act_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    np.savez(out_dir / "tau_cmd_vs_act_audit_samples.npz", **arrays)

    print_stats(stats)
    print(f"[done] json -> {out_json}")
    print(f"[done] plots + npz -> {out_dir}")


if __name__ == "__main__":
    main()
