"""
cdsm_hybrid_smc_tracking.py
=============================
绳驱空间机械臂: 混合模型 (名义 + EDMD 残差) + 滑模鲁棒关节空间轨迹跟踪.

流程
----
1. MuJoCo 绳驱模型上 PD 采轨迹 -> 训练残差 EDMD (与 cdsm_hybrid_residual_edmd 一致);
2. 基于名义模型 M(q), C(q,dq) 设计滑模控制 (混合模型中的机理部分):
       s = e + λ ⊙ ė,   e = q - q_ref,  ė = dq - dq_ref
       qdd = qdd_ref - ė/λ - (K/λ)⊙sat(s/φ)
       τ   = M(q) qdd + C(q,dq) dq
3. 状态反馈来自 MuJoCo; τ 经拮抗映射为 8 绳张力后作用在真机模型上, 检验跟踪.

说明
----
- EDMD 残差用于离线训练混合一步预测器; 滑模律使用名义 M,C (连续动力学形式).
- 滑模面按用户指定: s = e + λ*ė (逐关节分量).

运行
----
    python cdsm_hybrid_smc_tracking.py
    python cdsm_hybrid_smc_tracking.py --skip_train --model_dir Figs/cdsm_hybrid_residual_edmd/<ts>
    python cdsm_hybrid_smc_tracking.py --train_traj 80 --steps 200 --T_track 5.0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from cdsm_rigid_nominal_model import CdsmRigidNominalModel, make_nominal_model
from cdsm_hybrid_residual_edmd import (
    PDCollectConfig,
    ResidualEdmdConfig,
    XML_PATH,
    STATE_LABELS,
    build_residual_dataset,
    cable_antagonistic_map,
    collect_pd_trajectories,
    compute_tendon_jacobian_fd,
    fit_residual_edmd,
    flatten_residual_data,
    get_active_state,
    load_cable_model,
    save_residual_model,
    set_active_state,
    set_seed,
)
from utils_plot import get_save_dir, save_figure


# ===================================================================
# 参考轨迹
# ===================================================================
def cosine_ramp(t: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray]:
    tau = np.clip(t / max(T, 1e-9), 0.0, 1.0)
    s = 0.5 * (1.0 - np.cos(np.pi * tau))
    ds = 0.5 * np.pi / max(T, 1e-9) * np.sin(np.pi * tau)
    ds = np.where((t > 0) & (t < T), ds, 0.0)
    return s, ds


def build_joint_reference(
    dt: float,
    T_total: float,
    qa0: float,
    qa1: float,
    qb0: float,
    qb1: float,
    T_ramp: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """构造关节空间参考 q_ref, dq_ref, qdd_ref."""
    T_ramp = T_total if T_ramp is None else T_ramp
    t = np.arange(0.0, T_total + 0.5 * dt, dt)
    s, ds = cosine_ramp(t, T_ramp)

    qa = qa0 + (qa1 - qa0) * s
    qb = qb0 + (qb1 - qb0) * s
    dqa = (qa1 - qa0) * ds
    dqb = (qb1 - qb0) * ds

    q_ref = np.column_stack([qa, qb])
    dq_ref = np.column_stack([dqa, dqb])

    qdd_ref = np.zeros_like(q_ref)
    qdd_ref[1:, 0] = np.diff(dqa) / dt
    qdd_ref[1:, 1] = np.diff(dqb) / dt
    qdd_ref[0] = qdd_ref[1]

    return {"t": t, "q_ref": q_ref, "dq_ref": dq_ref, "qdd_ref": qdd_ref}


# ===================================================================
# 滑模控制 (基于名义 M, C)
# ===================================================================
def sat_vec(x: np.ndarray) -> np.ndarray:
    """边界层饱和 sat(x)."""
    return np.clip(x, -1.0, 1.0)


def sliding_mode_torque(
    nominal: CdsmRigidNominalModel,
    q: np.ndarray,
    dq: np.ndarray,
    q_ref: np.ndarray,
    dq_ref: np.ndarray,
    qdd_ref: np.ndarray,
    lam: np.ndarray,
    K: np.ndarray,
    phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    滑模鲁棒控制 (计算力矩 + 边界层), 显式分解为等效项与切换项.

    滑模面 (逐关节):
        s = e + λ ⊙ ė,    e = q - q_ref,   ė = dq - dq_ref

    期望滑模动力学:
        ṡ = -K ⊙ sat(s / φ)

    由 ṡ = ė + λ ⊙ (qdd - qdd_ref) 得期望加速度分解:
        qdd_eq = qdd_ref - ė / λ                    (等价控制: 若模型精确, 使 ṡ = 0)
        qdd_sw = -(K / λ) ⊙ sat(s / φ)              (切换控制: 压 s 到零)

        qdd    = qdd_eq + qdd_sw

    关节力矩 (名义 M, C):
        τ_eq = M(q) qdd_eq + C(q, dq) dq
        τ_sw = M(q) qdd_sw
        τ    = τ_eq + τ_sw = M(q) qdd + C(q, dq) dq

    Returns
    -------
    tau, e, edot, s
    """
    q = np.asarray(q, dtype=np.float64).reshape(2)
    dq = np.asarray(dq, dtype=np.float64).reshape(2)
    q_ref = np.asarray(q_ref, dtype=np.float64).reshape(2)
    dq_ref = np.asarray(dq_ref, dtype=np.float64).reshape(2)
    qdd_ref = np.asarray(qdd_ref, dtype=np.float64).reshape(2)
    lam = np.asarray(lam, dtype=np.float64).reshape(2)
    K = np.asarray(K, dtype=np.float64).reshape(2)
    phi = np.asarray(phi, dtype=np.float64).reshape(2)

    e = q - q_ref
    edot = dq - dq_ref
    s = e + lam * edot

    sat_s = sat_vec(s / phi)

    # --- 加速度层分解 ---
    qdd_eq = qdd_ref - edot / lam
    qdd_sw = -(K / lam) * sat_s
    qdd = qdd_eq + qdd_sw

    # --- 力矩层分解 (τ = τ_eq + τ_sw) ---
    M = nominal.mass_matrix(q)
    C = nominal.coriolis_matrix(q, dq)
    tau_eq = M @ qdd_eq + C @ dq
    tau_sw = M @ qdd_sw
    tau = tau_eq + tau_sw

    return tau, e, edot, s


# ===================================================================
# 训练混合残差 EDMD
# ===================================================================
def train_hybrid_edmd(
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[object, CdsmRigidNominalModel, Dict[str, object]]:
    """PD 采数 + 残差 EDMD 训练."""
    pd_cfg = PDCollectConfig(
        traj_count=args.train_traj,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        q_init_range=args.q_init_range,
        dq_init_range=args.dq_init_range,
        amp_range=(args.amp_min, args.amp_max),
        omega_range=(args.omega_min, args.omega_max),
        kp=(args.kp_a, args.kp_b),
        kd=(args.kd_a, args.kd_b),
        tau_max=args.tau_max,
    )

    print("[train] Loading MuJoCo cable model and collecting PD data...")
    model, data, scratch, indices = load_cable_model(args.xml, args.dt)
    train_raw, meta_train = collect_pd_trajectories(model, data, scratch, indices, pd_cfg)
    print(f"        dataset states {train_raw['states'].shape}")

    np.savez(out_dir / "dataset_train.npz", **train_raw)

    nominal = make_nominal_model(dt=args.dt)
    res_train = build_residual_dataset(train_raw, nominal, args.dt)
    x_tr, u_tr, xp_tr, r_tr = flatten_residual_data(res_train)

    edmd_cfg = ResidualEdmdConfig(
        dictionary=args.dictionary,
        ridge=args.ridge,
        rbf_centers=args.rbf_centers,
        rbf_sigma=args.rbf_sigma,
        rbf_seed=args.rbf_seed,
    )
    edmd_model = fit_residual_edmd(x_tr, u_tr, r_tr, edmd_cfg)
    save_residual_model(out_dir / "residual_edmd_model.npz", edmd_model, edmd_cfg)

    train_info = {
        "pd_config": asdict(pd_cfg),
        "collection_meta": meta_train,
        "edmd_config": asdict(edmd_cfg),
        "mean_residual_l2": float(np.linalg.norm(r_tr, axis=1).mean()),
        "feature_dim": int(edmd_model.feature_dim),
    }
    with open(out_dir / "train_info.json", "w", encoding="utf-8") as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)

    print(f"        EDMD feature_dim={edmd_model.feature_dim}, mean|r|={train_info['mean_residual_l2']:.4g}")
    return edmd_model, nominal, train_info


def load_trained_edmd(model_dir: Path, dt: float) -> Tuple[object, CdsmRigidNominalModel, Dict]:
    """从 cdsm_hybrid_residual_edmd 输出目录加载 EDMD 模型."""
    from cdsm_hybrid_residual_edmd import Normalizer, ResidualEdmdModel

    npz_path = model_dir / "residual_edmd_model.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Model not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    dictionary = str(data["dictionary"][0])
    z_norm = Normalizer(data["z_mean"], data["z_std"])
    r_norm = Normalizer(data["r_mean"], data["r_std"])
    centers = data["centers"] if "centers" in data else None
    sigma = float(data["sigma"][0]) if "sigma" in data else None
    edmd_model = ResidualEdmdModel(
        weights=data["weights"],
        z_norm=z_norm,
        r_norm=r_norm,
        dictionary=dictionary,
        centers=centers,
        sigma=sigma,
        cond_number=float(data["cond_number"][0]),
        feature_dim=int(data["feature_dim"][0]),
    )
    nominal = make_nominal_model(dt=dt)
    train_info_path = model_dir / "train_info.json"
    train_info = {}
    if train_info_path.exists():
        with open(train_info_path, encoding="utf-8") as f:
            train_info = json.load(f)
    return edmd_model, nominal, train_info


# ===================================================================
# 绳驱 MuJoCo 闭环滑模跟踪
# ===================================================================
def run_smc_tracking(
    nominal: CdsmRigidNominalModel,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scratch: mujoco.MjData,
    indices: Dict[str, np.ndarray],
    ref: Dict[str, np.ndarray],
    lam: np.ndarray,
    K: np.ndarray,
    phi: np.ndarray,
    edmd_model: Optional[object] = None,
) -> Dict[str, np.ndarray]:
    """
    滑模跟踪主循环: 读 MuJoCo 状态 -> 算 τ -> 绳驱映射 -> mj_step.
    """
    t = ref["t"]
    dt = float(model.opt.timestep)
    n_step = len(t) - 1

    q0_ref = ref["q_ref"][0]
    dq0_ref = ref["dq_ref"][0]
    set_active_state(model, data, indices, q0_ref, dq0_ref)
    mujoco.mj_forward(model, data)

    dof_j1 = int(indices["dof_j1"])
    dof_j2 = int(indices["dof_j2"])
    dof_j3 = int(indices["dof_j3"])
    dof_j4 = int(indices["dof_j4"])

    rec_t = []
    rec_q = []
    rec_dq = []
    rec_q_ref = []
    rec_tau = []
    rec_s = []
    rec_e = []
    rec_hyb_err_norm = []

    for k in range(n_step):
        x = get_active_state(data, indices)
        q = x[:2]
        dq = x[2:]

        q_ref_k = ref["q_ref"][k]
        dq_ref_k = ref["dq_ref"][k]
        qdd_ref_k = ref["qdd_ref"][k]

        tau, e, edot, s = sliding_mode_torque(
            nominal, q, dq, q_ref_k, dq_ref_k, qdd_ref_k, lam, K, phi
        )

        J = compute_tendon_jacobian_fd(model, scratch, data.qpos.copy(), indices["tendon_ids"])
        F_cable = cable_antagonistic_map(
            float(tau[0]), float(tau[1]), J, dof_j1, dof_j2, dof_j3, dof_j4
        )
        data.ctrl[indices["actuator_ids"]] = F_cable
        mujoco.mj_step(model, data)

        rec_t.append(data.time)
        rec_q.append(q.copy())
        rec_dq.append(dq.copy())
        rec_q_ref.append(q_ref_k.copy())
        rec_tau.append(tau.copy())
        rec_s.append(s.copy())
        rec_e.append(e.copy())

        if edmd_model is not None:
            x_next_true = get_active_state(data, indices)
            x_pred_hyb = edmd_model.predict_hybrid_next(nominal, x, tau, dt)
            rec_hyb_err_norm.append(float(np.linalg.norm(x_pred_hyb - x_next_true)))

    rec_q = np.array(rec_q)
    rec_dq = np.array(rec_dq)
    rec_q_ref = np.array(rec_q_ref)
    rec_tau = np.array(rec_tau)
    rec_s = np.array(rec_s)
    rec_e = np.array(rec_e)

    return {
        "t": np.array(rec_t),
        "q": rec_q,
        "dq": rec_dq,
        "q_ref": rec_q_ref,
        "tau": rec_tau,
        "s": rec_s,
        "e": rec_e,
        "hyb_one_step_err": np.array(rec_hyb_err_norm) if rec_hyb_err_norm else np.array([]),
    }


# ===================================================================
# 绘图与指标
# ===================================================================
def tracking_metrics(log: Dict[str, np.ndarray]) -> Dict[str, float]:
    e = log["q"] - log["q_ref"]
    rmse_q = float(np.sqrt(np.mean(e * e)))
    mae_q = float(np.mean(np.abs(e)))
    max_q = float(np.max(np.abs(e)))
    rmse_a = float(np.sqrt(np.mean(log["s"] * log["s"])))
    return {
        "rmse_q": rmse_q,
        "mae_q": mae_q,
        "max_abs_q": max_q,
        "rmse_sliding_surface": rmse_a,
    }


def plot_tracking(log: Dict[str, np.ndarray], prefix: str) -> None:
    t = log["t"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)

    for i, label in enumerate(STATE_LABELS[:2]):
        ax = axes[0, i]
        ax.plot(t, log["q_ref"][:, i], "k--", lw=1.5, label="ref")
        ax.plot(t, log["q"][:, i], lw=1.5, label="actual")
        ax.set_title(f"{label} tracking")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i, label in enumerate(STATE_LABELS[2:]):
        ax = axes[1, i]
        ax.plot(t, log["dq"][:, i], lw=1.5, label="dq")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Joint-space SMC tracking")
    fig.tight_layout()
    save_figure(f"{prefix}_joint_tracking")
    plt.close(fig)

    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax2[0].plot(t, log["tau"][:, 0], lw=1.6, label="tau_a")
    ax2[0].plot(t, log["tau"][:, 1], lw=1.6, label="tau_b")
    ax2[0].set_ylabel("Torque (Nm)")
    ax2[0].legend(fontsize=8)
    ax2[0].grid(True, alpha=0.3)
    ax2[1].plot(t, log["s"][:, 0], label="s_a")
    ax2[1].plot(t, log["s"][:, 1], label="s_b")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_ylabel("Sliding variable")
    ax2[1].legend()
    ax2[1].grid(True, alpha=0.3)
    fig2.suptitle("SMC joint torque and sliding surface s")
    fig2.tight_layout()
    save_figure(f"{prefix}_tau_s")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t, log["e"][:, 0], label="e_qa")
    ax3.plot(t, log["e"][:, 1], label="e_qb")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Tracking error e")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    save_figure(f"{prefix}_error")
    plt.close(fig3)


# ===================================================================
# Main
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CDSM hybrid model + sliding mode tracking on MuJoCo cable plant.")
    p.add_argument("--xml", default=XML_PATH)

    # 训练数据
    p.add_argument("--skip_train", action="store_true", help="Skip training; load --model_dir instead.")
    p.add_argument("--model_dir", type=str, default=None, help="Directory with residual_edmd_model.npz")
    p.add_argument("--train_traj", type=int, default=120)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--q_init_range", type=float, default=0.3)
    p.add_argument("--dq_init_range", type=float, default=0.2)
    p.add_argument("--amp_min", type=float, default=0.25)
    p.add_argument("--amp_max", type=float, default=0.65)
    p.add_argument("--omega_min", type=float, default=0.4)
    p.add_argument("--omega_max", type=float, default=1.2)
    p.add_argument("--kp_a", type=float, default=120.0)
    p.add_argument("--kp_b", type=float, default=80.0)
    p.add_argument("--kd_a", type=float, default=25.0)
    p.add_argument("--kd_b", type=float, default=18.0)
    p.add_argument("--tau_max", type=float, default=45.0)
    p.add_argument("--dictionary", choices=["hermite", "rbf"], default="hermite")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--rbf_centers", type=int, default=200)
    p.add_argument("--rbf_sigma", type=float, default=None)
    p.add_argument("--rbf_seed", type=int, default=2007)

    # 跟踪参考轨迹 (默认较保守, 远离绳驱几何退化区)
    p.add_argument("--T_track", type=float, default=6.0, help="Tracking duration (s).")
    p.add_argument("--T_ramp", type=float, default=5.0, help="Cosine ramp duration (s).")
    p.add_argument("--qa0", type=float, default=-0.5)
    p.add_argument("--qa1", type=float, default=0.5)
    p.add_argument("--qb0", type=float, default=0.4)
    p.add_argument("--qb1", type=float, default=-0.4)

    # 滑模参数 s = e + λ⊙ė
    p.add_argument("--lam_a", type=float, default=2.0, help="λ for joint qa.")
    p.add_argument("--lam_b", type=float, default=2.0, help="λ for joint qb.")
    p.add_argument("--K_a", type=float, default=8.0, help="Switching gain K for qa.")
    p.add_argument("--K_b", type=float, default=8.0, help="Switching gain K for qb.")
    p.add_argument("--phi_a", type=float, default=0.15, help="Boundary layer φ for sat(s/φ), qa.")
    p.add_argument("--phi_b", type=float, default=0.15, help="Boundary layer φ for qb.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    out_dir = Path(get_save_dir())
    t0 = time.time()

    if args.skip_train:
        if not args.model_dir:
            raise ValueError("--skip_train requires --model_dir")
        print(f"[train] Skipped. Loading model from {args.model_dir}")
        edmd_model, nominal, train_info = load_trained_edmd(Path(args.model_dir), args.dt)
    else:
        edmd_model, nominal, train_info = train_hybrid_edmd(args, out_dir)

    lam = np.array([args.lam_a, args.lam_b])
    K = np.array([args.K_a, args.K_b])
    phi = np.array([args.phi_a, args.phi_b])

    ref = build_joint_reference(
        dt=args.dt,
        T_total=args.T_track,
        qa0=args.qa0,
        qa1=args.qa1,
        qb0=args.qb0,
        qb1=args.qb1,
        T_ramp=args.T_ramp,
    )

    print("[smc] Running sliding-mode tracking on MuJoCo cable plant...")
    model, data, scratch, indices = load_cable_model(args.xml, args.dt)
    log = run_smc_tracking(
        nominal, model, data, scratch, indices, ref, lam, K, phi, edmd_model=edmd_model
    )
    metrics = tracking_metrics(log)

    smc_cfg = {
        "sliding_surface": "s = e + lam * edot",
        "e_definition": "e = q - q_ref",
        "torque_decomposition": {
            "qdd_eq": "qdd_ref - edot / lam",
            "qdd_sw": "-(K / lam) * sat(s / phi)",
            "tau_eq": "M * qdd_eq + C * dq",
            "tau_sw": "M * qdd_sw",
            "tau": "tau_eq + tau_sw",
        },
        "lam": lam.tolist(),
        "K": K.tolist(),
        "phi": phi.tolist(),
        "reference": {
            "qa0": args.qa0, "qa1": args.qa1,
            "qb0": args.qb0, "qb1": args.qb1,
            "T_track": args.T_track, "T_ramp": args.T_ramp,
        },
    }
    summary = {
        "xml": args.xml,
        "train_info": train_info,
        "smc_config": smc_cfg,
        "tracking_metrics": metrics,
        "hyb_one_step_err_mean": float(log["hyb_one_step_err"].mean()) if log["hyb_one_step_err"].size else None,
        "elapsed_sec": time.time() - t0,
    }
    with open(out_dir / "smc_tracking_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    np.savez(out_dir / "smc_tracking_log.npz", **log)
    plot_tracking(log, "smc")

    print("-" * 60)
    print(f"  Tracking RMSE (q)     : {metrics['rmse_q']:.6g} rad")
    print(f"  Tracking MAE (q)      : {metrics['mae_q']:.6g} rad")
    print(f"  Max |q - q_ref|      : {metrics['max_abs_q']:.6g} rad")
    print(f"  RMSE sliding surface  : {metrics['rmse_sliding_surface']:.6g}")
    if log["hyb_one_step_err"].size:
        print(f"  Hybrid 1-step |dx| mean: {log['hyb_one_step_err'].mean():.6g} (monitor only)")
    print(f"[done] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
