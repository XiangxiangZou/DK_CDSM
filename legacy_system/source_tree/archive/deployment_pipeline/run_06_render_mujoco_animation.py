"""第六部分：把闭环跟踪日志回放成 MuJoCo 运动动画。

本脚本不重新运行控制器，只读取 `run_05_cartesian_ik_tracking_compare.py` 已保存的
闭环日志，例如：

```text
results/cartesian_tracking/<timestamp>_.../circle/
  closed_loop_edmd.npz
  closed_loop_dkuc.npz
  closed_loop_dkac.npz
  cartesian_tracking_metrics.json
```

然后把每个方法记录的真实状态 `x_meas=[qa,qb,dqa,dqb]` 逐帧写回 MuJoCo 模型，
用 MuJoCo 离屏 Renderer 保存机械臂运动 GIF。这样动画展示的是控制器已经作用到
“MuJoCo 真实机械臂”后的实际运动，而不是模型预测轨迹。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

try:
    from .model_registry import CONTROL_MODELS
    from .mujoco_plant import MujocoCablePlant
except ImportError:  # pragma: no cover - 支持直接运行本文件
    from model_registry import CONTROL_MODELS
    from mujoco_plant import MujocoCablePlant

ACTUAL_TRAIL_COLORS = {
    "red": np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "green": np.array([0.0, 0.85, 0.2, 1.0], dtype=np.float32),
    "blue": np.array([0.1, 0.45, 1.0, 1.0], dtype=np.float32),
    "orange": np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32),
}


def _require_mujoco():
    """延迟导入 MuJoCo，避免只查看帮助信息时提前创建渲染上下文。"""
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required to render motion animations.") from exc
    return mujoco


def build_parser() -> argparse.ArgumentParser:
    """构造 MuJoCo 动画渲染命令行参数。"""
    parser = argparse.ArgumentParser(
        description="读取闭环跟踪结果，回放 q,dq 并保存三种方法的 MuJoCo 运动 GIF。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result_dir",
        default="",
        help=(
            "某一次轨迹结果目录，通常是包含 closed_loop_edmd.npz 的 circle 子目录；"
            "留空时自动寻找最新的 circle 结果。"
        ),
    )
    parser.add_argument(
        "--trajectory",
        default="circle",
        help="自动寻找 result_dir 时使用的轨迹子目录名；本次需求为 circle。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["edmd", "dkuc", "dkac"],
        choices=CONTROL_MODELS,
        help="需要保存动画的控制方法。",
    )
    parser.add_argument(
        "--xml",
        default=str(
            Path(__file__).resolve().parents[2]
            / "assets"
            / "models"
            / "multi_joint_cable_driven_space_robot.xml"
        ),
        help="MuJoCo XML 路径。",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="闭环日志对应的控制周期，单位 s。")
    parser.add_argument("--width", type=int, default=960, help="动画宽度，单位像素。")
    parser.add_argument("--height", type=int, default=720, help="动画高度，单位像素。")
    parser.add_argument(
        "--render_stride",
        type=int,
        default=4,
        help="每隔多少个闭环采样点渲染一帧；0.01s 日志配合 stride=4 对应约 25 FPS。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="GIF 播放帧率；0 表示根据日志 dt 和 render_stride 保持近似真实时间。",
    )
    parser.add_argument("--camera_lookat_x", type=float, default=3.2, help="自由相机 lookat x。")
    parser.add_argument("--camera_lookat_y", type=float, default=0.0, help="自由相机 lookat y。")
    parser.add_argument("--camera_lookat_z", type=float, default=0.0, help="自由相机 lookat z。")
    parser.add_argument("--camera_distance", type=float, default=6.4, help="自由相机距离。")
    parser.add_argument("--camera_azimuth", type=float, default=90.0, help="自由相机方位角，单位 deg。")
    parser.add_argument("--camera_elevation", type=float, default=-90.0, help="自由相机俯仰角，单位 deg。")
    overlay_group = parser.add_argument_group("末端轨迹叠加显示参数")
    overlay_group.add_argument(
        "--no_desired_path",
        action="store_true",
        help="不绘制白色虚线期望末端轨迹。",
    )
    overlay_group.add_argument(
        "--no_actual_trail",
        action="store_true",
        help="不绘制实际末端运动轨迹。",
    )
    overlay_group.add_argument(
        "--actual_trail_color",
        choices=tuple(ACTUAL_TRAIL_COLORS),
        default="red",
        help="实际末端运动轨迹颜色。",
    )
    overlay_group.add_argument("--desired_path_stride", type=int, default=3, help="期望轨迹绘制下采样步长。")
    overlay_group.add_argument("--actual_trail_stride", type=int, default=2, help="实际轨迹绘制下采样步长。")
    overlay_group.add_argument("--desired_dash_on", type=int, default=3, help="白色虚线每周期绘制的线段数。")
    overlay_group.add_argument("--desired_dash_off", type=int, default=2, help="白色虚线每周期跳过的线段数。")
    overlay_group.add_argument("--desired_line_width", type=float, default=0.025, help="期望轨迹虚线宽度，单位 m。")
    overlay_group.add_argument("--actual_line_width", type=float, default=0.035, help="实际轨迹红线宽度，单位 m。")
    overlay_group.add_argument(
        "--trajectory_overlay_z",
        type=float,
        default=0.08,
        help="末端轨迹叠加线在 MuJoCo 世界系中的 z 坐标，单位 m。",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="动画输出目录；留空时保存到 result_dir/animations。",
    )
    parser.add_argument("--tag", default="", help="输出文件名附加标签。")
    return parser


def find_latest_result_dir(root: str | Path, trajectory: str) -> Path:
    """寻找最新一次包含指定轨迹闭环日志的结果目录。

    参数:
        root: `results/cartesian_tracking` 根目录。
        trajectory: 轨迹子目录名，例如 `circle`。

    返回:
        包含 `closed_loop_edmd.npz` 等日志文件的轨迹目录。
    """
    root_path = Path(root)
    candidates: List[Path] = []
    if root_path.exists():
        for run_dir in root_path.iterdir():
            traj_dir = run_dir / trajectory
            if (traj_dir / "closed_loop_edmd.npz").exists():
                candidates.append(traj_dir)
    if not candidates:
        raise FileNotFoundError(f"No {trajectory!r} tracking result found under {root_path}")
    return sorted(candidates, key=lambda p: p.parent.name)[-1]


def load_log(result_dir: str | Path, model_name: str) -> Dict[str, np.ndarray]:
    """读取单个模型的闭环日志。"""
    path = Path(result_dir) / f"closed_loop_{model_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Closed-loop log not found: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _build_camera(args: argparse.Namespace):
    """构造俯视自由相机。"""
    mujoco = _require_mujoco()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = np.array(
        [args.camera_lookat_x, args.camera_lookat_y, args.camera_lookat_z],
        dtype=np.float64,
    )
    camera.distance = float(args.camera_distance)
    camera.azimuth = float(args.camera_azimuth)
    camera.elevation = float(args.camera_elevation)
    return camera


def _frame_indices(n_frame_source: int, stride: int) -> np.ndarray:
    """根据日志长度和 stride 生成需要渲染的帧索引，确保包含最后一帧。"""
    step = max(int(stride), 1)
    indices = np.arange(0, int(n_frame_source), step, dtype=np.int64)
    if indices.size == 0 or indices[-1] != n_frame_source - 1:
        indices = np.append(indices, n_frame_source - 1)
    return indices


def _gif_duration_ms(dt: float, stride: int, fps: float) -> int:
    """计算 GIF 单帧持续时间。"""
    if fps and fps > 0.0:
        return max(1, int(round(1000.0 / float(fps))))
    return max(1, int(round(1000.0 * float(dt) * max(int(stride), 1))))


def _xy_to_xyz(points_xy: np.ndarray, z: float) -> np.ndarray:
    """把末端平面 xy 轨迹转换为 MuJoCo 3D 线段点。"""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected xy points with shape (N,2), got {points.shape}")
    z_col = np.full((points.shape[0], 1), float(z), dtype=np.float64)
    return np.hstack([points, z_col])


def _add_capsule_segment(scene, p0: np.ndarray, p1: np.ndarray, *, width: float, rgba: np.ndarray) -> None:
    """向当前 MuJoCo scene 追加一段胶囊线段。"""
    if scene.ngeom >= scene.maxgeom:
        return
    if np.linalg.norm(np.asarray(p1) - np.asarray(p0)) < 1e-9:
        return
    mujoco = _require_mujoco()
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(width),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    geom.rgba[:] = np.asarray(rgba, dtype=np.float32)
    scene.ngeom += 1


def _add_polyline(
    scene,
    points_xy: np.ndarray,
    *,
    rgba: np.ndarray,
    width: float,
    z: float,
    stride: int,
    dashed: bool = False,
    dash_on: int = 3,
    dash_off: int = 2,
) -> None:
    """向 MuJoCo scene 追加一条末端轨迹折线。"""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[0] < 2:
        return
    step = max(int(stride), 1)
    sampled = points[::step]
    if sampled.shape[0] == 0 or not np.allclose(sampled[-1], points[-1]):
        sampled = np.vstack([sampled, points[-1]])
    xyz = _xy_to_xyz(sampled, z)
    dash_period = max(int(dash_on), 1) + max(int(dash_off), 0)
    for i in range(xyz.shape[0] - 1):
        if dashed and (i % dash_period) >= int(dash_on):
            continue
        _add_capsule_segment(scene, xyz[i], xyz[i + 1], width=width, rgba=rgba)


def _add_current_point(scene, point_xy: np.ndarray, *, rgba: np.ndarray, width: float, z: float) -> None:
    """在实际末端轨迹末尾追加一个同色小点。"""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco = _require_mujoco()
    pos = np.array([float(point_xy[0]), float(point_xy[1]), float(z)], dtype=np.float64)
    size = np.array([float(width) * 1.7, 0.0, 0.0], dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        size,
        pos,
        mat,
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_trajectory_overlays(
    scene,
    *,
    desired_xy: np.ndarray | None,
    actual_xy: np.ndarray | None,
    upto_index: int,
    args: argparse.Namespace,
) -> None:
    """向动画帧中叠加期望末端轨迹和实际末端运动轨迹。"""
    z = float(args.trajectory_overlay_z)
    if desired_xy is not None and not args.no_desired_path:
        _add_polyline(
            scene,
            desired_xy,
            rgba=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            width=float(args.desired_line_width),
            z=z,
            stride=int(args.desired_path_stride),
            dashed=True,
            dash_on=int(args.desired_dash_on),
            dash_off=int(args.desired_dash_off),
        )
    if actual_xy is not None and not args.no_actual_trail:
        actual_rgba = ACTUAL_TRAIL_COLORS[str(args.actual_trail_color)]
        trail = np.asarray(actual_xy, dtype=np.float64)[: max(int(upto_index) + 1, 1)]
        _add_polyline(
            scene,
            trail,
            rgba=actual_rgba,
            width=float(args.actual_line_width),
            z=z + 0.015,
            stride=int(args.actual_trail_stride),
            dashed=False,
        )
        if trail.shape[0] > 0:
            _add_current_point(
                scene,
                trail[-1],
                rgba=actual_rgba,
                width=float(args.actual_line_width),
                z=z + 0.02,
            )


def _annotate_frame(frame: np.ndarray, *, model_name: str, sim_time: float) -> Image.Image:
    """在 MuJoCo 渲染帧上叠加方法名和仿真时间。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    label = f"{model_name.upper()}   t={sim_time:5.2f}s"
    pad = 8
    bbox = draw.textbbox((pad, pad), label)
    draw.rectangle(
        [bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3],
        fill=(0, 0, 0),
    )
    draw.text((pad, pad), label, fill=(255, 255, 255))
    return image


def render_model_gif(
    *,
    model_name: str,
    log: Dict[str, np.ndarray],
    xml: str | Path,
    dt: float,
    out_path: str | Path,
    args: argparse.Namespace,
) -> Path:
    """把单个模型的闭环日志渲染为 GIF。

    参数:
        model_name: 控制方法名。
        log: `closed_loop_<model>.npz` 读取出的日志。
        xml: MuJoCo XML 路径。
        dt: 控制周期，单位 s。
        out_path: GIF 输出路径。
        args: 渲染参数集合。
    """
    mujoco = _require_mujoco()
    plant = MujocoCablePlant(xml, dt)
    renderer = mujoco.Renderer(plant.model, height=int(args.height), width=int(args.width))
    camera = _build_camera(args)
    states = np.asarray(log["x_meas"], dtype=np.float64)
    times = np.asarray(log.get("t", np.arange(states.shape[0]) * float(dt)), dtype=np.float64)
    desired_xy = np.asarray(log["ee_ref"], dtype=np.float64) if "ee_ref" in log else None
    actual_xy = np.asarray(log["ee_meas"], dtype=np.float64) if "ee_meas" in log else None
    indices = _frame_indices(states.shape[0], int(args.render_stride))
    duration = _gif_duration_ms(dt, int(args.render_stride), float(args.fps))

    frames: List[Image.Image] = []
    for idx in indices:
        q = states[idx, :2]
        dq = states[idx, 2:]
        plant.set_state(q, dq)
        renderer.update_scene(plant.data, camera=camera)
        _add_trajectory_overlays(
            renderer.scene,
            desired_xy=desired_xy,
            actual_xy=actual_xy,
            upto_index=int(idx),
            args=args,
        )
        frame = renderer.render()
        frames.append(_annotate_frame(frame, model_name=model_name, sim_time=float(times[idx])))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )
    renderer.close()
    return out


def _save_json(path: str | Path, payload: dict) -> None:
    """保存动画生成元数据。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    """命令行入口：为三种控制方法保存 MuJoCo GIF 动画。"""
    args = build_parser().parse_args()
    if args.result_dir:
        result_dir = Path(args.result_dir)
    else:
        result_dir = find_latest_result_dir(
            Path(__file__).resolve().parents[2]
            / "outputs"
            / "results"
            / "deployment_pipeline"
            / "cartesian_tracking",
            args.trajectory,
        )
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir does not exist: {result_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=== CDSM MuJoCo motion animation render ===")
    print(f"result_dir={result_dir}")
    print(f"models={args.models}")
    print(f"output={out_dir}")

    outputs = {}
    for model_name in args.models:
        log = load_log(result_dir, model_name)
        out_path = out_dir / f"{stamp}_{args.trajectory}_{model_name}_mujoco_motion{suffix}.gif"
        saved = render_model_gif(
            model_name=model_name,
            log=log,
            xml=args.xml,
            dt=args.dt,
            out_path=out_path,
            args=args,
        )
        outputs[model_name] = str(saved)
        print(f"[saved] {model_name}: {saved}")

    meta = {
        "result_dir": str(result_dir),
        "trajectory": args.trajectory,
        "models": list(args.models),
        "xml": str(args.xml),
        "dt": args.dt,
        "render_stride": args.render_stride,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "trajectory_overlay": {
            "desired_path": not args.no_desired_path,
            "actual_trail": not args.no_actual_trail,
            "desired_path_style": "white dashed",
            "actual_trail_style": f"{args.actual_trail_color} solid",
            "desired_path_stride": args.desired_path_stride,
            "actual_trail_stride": args.actual_trail_stride,
            "trajectory_overlay_z": args.trajectory_overlay_z,
        },
        "outputs": outputs,
    }
    _save_json(out_dir / f"{stamp}_{args.trajectory}_mujoco_motion_metadata{suffix}.json", meta)
    print(f"[done] animations -> {out_dir}")


if __name__ == "__main__":
    main()
