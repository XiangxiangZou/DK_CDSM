"""第八部分：把三种方法的末端轨迹叠加到一个 MuJoCo GIF 中。

本脚本是独立的合并展示脚本，不修改也不依赖 `run_06_render_mujoco_animation.py`
的内部实现。用途是把同一条末端轨迹下 EDMD、DKUC、DKAC 三种方法的末端实际
运动轨迹放在同一个 MuJoCo 动画中，便于直接横向比较。

默认颜色约定：
- 白色虚线：期望末端轨迹；
- 蓝色实线：EDMD 方法下机械臂末端实际运动轨迹；
- 橙色实线：DKUC 方法下机械臂末端实际运动轨迹；
- 绿色实线：DKAC 方法下机械臂末端实际运动轨迹。

说明：
为了保持 MuJoCo 场景中仍有真实机械臂本体运动，脚本默认用 DKAC 的闭环状态
回放机械臂本体；三种方法的比较由三条彩色末端轨迹和当前末端点体现。可通过
`--arm_model` 切换用于回放机械臂本体的方法。
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


MODEL_COLORS_RGBA = {
    "edmd": np.array([0.0, 0.35, 1.0, 1.0], dtype=np.float32),
    "dkuc": np.array([1.0, 0.55, 0.0, 1.0], dtype=np.float32),
    "dkac": np.array([0.0, 0.8, 0.2, 1.0], dtype=np.float32),
}
MODEL_COLORS_PIL = {
    "edmd": (40, 120, 255),
    "dkuc": (255, 150, 30),
    "dkac": (40, 220, 80),
}


def _require_mujoco():
    """延迟导入 MuJoCo，避免只查看帮助信息时提前创建渲染上下文。"""
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mujoco is required to render combined trajectory GIFs.") from exc
    return mujoco


def build_parser() -> argparse.ArgumentParser:
    """构造三方法合并 MuJoCo GIF 的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="在一个 MuJoCo GIF 中叠加 EDMD/DKUC/DKAC 三种方法的末端实际轨迹。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result_dir",
        default="",
        help=(
            "轨迹结果子目录，需直接包含 cartesian_ik_reference.npz 和 closed_loop_<model>.npz；"
            "留空时自动寻找最新的 figure8 结果。"
        ),
    )
    parser.add_argument(
        "--trajectory",
        default="figure8",
        help="自动寻找 result_dir 时使用的轨迹子目录名；本次需求为 figure8。",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["edmd", "dkuc", "dkac"],
        choices=CONTROL_MODELS,
        help="需要叠加绘制的控制方法。",
    )
    parser.add_argument(
        "--arm_model",
        choices=CONTROL_MODELS,
        default="dkac",
        help="MuJoCo 机械臂本体按哪个方法的闭环状态回放；彩色轨迹仍显示全部方法。",
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
    parser.add_argument("--desired_path_stride", type=int, default=3, help="期望轨迹绘制下采样步长。")
    parser.add_argument("--actual_trail_stride", type=int, default=2, help="实际轨迹绘制下采样步长。")
    parser.add_argument("--desired_dash_on", type=int, default=3, help="白色虚线每周期绘制的线段数。")
    parser.add_argument("--desired_dash_off", type=int, default=2, help="白色虚线每周期跳过的线段数。")
    parser.add_argument("--desired_line_width", type=float, default=0.025, help="期望轨迹虚线宽度，单位 m。")
    parser.add_argument("--actual_line_width", type=float, default=0.035, help="三种实际末端轨迹线宽，单位 m。")
    parser.add_argument("--trajectory_overlay_z", type=float, default=0.08, help="轨迹叠加线的 z 坐标，单位 m。")
    parser.add_argument(
        "--out_dir",
        default="",
        help="动画输出目录；留空时保存到 result_dir/animations。",
    )
    parser.add_argument("--tag", default="", help="输出文件名附加标签。")
    return parser


def find_latest_result_dir(root: str | Path, trajectory: str) -> Path:
    """自动寻找最新一次指定轨迹的结果子目录。"""
    root_path = Path(root)
    candidates: List[Path] = []
    if root_path.exists():
        for run_dir in root_path.iterdir():
            traj_dir = run_dir / trajectory
            if (traj_dir / "cartesian_ik_reference.npz").exists() and (traj_dir / "closed_loop_edmd.npz").exists():
                candidates.append(traj_dir)
    if not candidates:
        raise FileNotFoundError(f"No {trajectory!r} tracking result found under {root_path}")
    return sorted(candidates, key=lambda p: p.parent.name)[-1]


def load_reference(result_dir: str | Path) -> Dict[str, np.ndarray]:
    """读取期望末端轨迹和 IK 参考。"""
    path = Path(result_dir) / "cartesian_ik_reference.npz"
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


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


def _add_current_point(scene, point_xy: np.ndarray, *, rgba: np.ndarray, width: float, z: float) -> None:
    """在轨迹末尾追加当前末端位置彩色点。"""
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco = _require_mujoco()
    pos = np.array([float(point_xy[0]), float(point_xy[1]), float(z)], dtype=np.float64)
    size = np.array([float(width) * 1.8, 0.0, 0.0], dtype=np.float64)
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


def _add_trajectory_overlays(
    scene,
    *,
    desired_xy: np.ndarray,
    actual_logs: Dict[str, np.ndarray],
    upto_index: int,
    args: argparse.Namespace,
) -> None:
    """向当前帧叠加期望轨迹和三种方法的实际末端轨迹。"""
    z = float(args.trajectory_overlay_z)
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
    for model_name, actual_xy in actual_logs.items():
        trail = np.asarray(actual_xy, dtype=np.float64)[: max(int(upto_index) + 1, 1)]
        rgba = MODEL_COLORS_RGBA[model_name]
        # 不同颜色线条使用很小 z 偏移，避免重合时发生深度闪烁。
        z_offset = z + 0.015 + 0.008 * list(actual_logs.keys()).index(model_name)
        _add_polyline(
            scene,
            trail,
            rgba=rgba,
            width=float(args.actual_line_width),
            z=z_offset,
            stride=int(args.actual_trail_stride),
            dashed=False,
        )
        if trail.shape[0] > 0:
            _add_current_point(scene, trail[-1], rgba=rgba, width=float(args.actual_line_width), z=z_offset + 0.008)


def _annotate_frame(frame: np.ndarray, *, arm_model: str, sim_time: float, models: List[str]) -> Image.Image:
    """在渲染帧上叠加标题和颜色图例。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    pad = 8
    title = f"Combined {arm_model.upper()} arm playback   t={sim_time:5.2f}s"
    lines = [title, "White dashed: desired trajectory"]
    for model_name in models:
        lines.append(f"{model_name.upper()}: actual end-effector trail")
    line_height = 15
    width = 325
    height = pad * 2 + line_height * len(lines)
    draw.rectangle([pad - 4, pad - 4, pad + width, pad + height], fill=(0, 0, 0))
    draw.text((pad, pad), title, fill=(255, 255, 255))
    draw.text((pad, pad + line_height), "White dashed: desired trajectory", fill=(255, 255, 255))
    for i, model_name in enumerate(models):
        y = pad + line_height * (i + 2)
        color = MODEL_COLORS_PIL[model_name]
        draw.line([(pad, y + 7), (pad + 36, y + 7)], fill=color, width=3)
        draw.text((pad + 44, y), f"{model_name.upper()}: actual end-effector trail", fill=color)
    return image


def render_combined_gif(
    *,
    result_dir: Path,
    models: List[str],
    arm_model: str,
    xml: str | Path,
    dt: float,
    out_path: str | Path,
    args: argparse.Namespace,
) -> Path:
    """渲染包含三种方法末端轨迹的单个 MuJoCo GIF。"""
    mujoco = _require_mujoco()
    reference = load_reference(result_dir)
    desired_xy = np.asarray(reference["ee_ref"], dtype=np.float64)
    logs = {model_name: load_log(result_dir, model_name) for model_name in models}
    arm_log = logs[arm_model]
    actual_logs = {
        model_name: np.asarray(logs[model_name]["ee_meas"], dtype=np.float64)
        for model_name in models
    }

    plant = MujocoCablePlant(xml, dt)
    renderer = mujoco.Renderer(plant.model, height=int(args.height), width=int(args.width))
    camera = _build_camera(args)
    states = np.asarray(arm_log["x_meas"], dtype=np.float64)
    times = np.asarray(arm_log.get("t", np.arange(states.shape[0]) * float(dt)), dtype=np.float64)
    n_source = min([states.shape[0], *[xy.shape[0] for xy in actual_logs.values()]])
    indices = _frame_indices(n_source, int(args.render_stride))
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
            actual_logs=actual_logs,
            upto_index=int(idx),
            args=args,
        )
        frame = renderer.render()
        frames.append(
            _annotate_frame(
                frame,
                arm_model=arm_model,
                sim_time=float(times[idx]),
                models=models,
            )
        )

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
    """保存合并 GIF 的生成元数据。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    """命令行入口：生成三种方法合并展示的 MuJoCo GIF。"""
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
    models = list(args.models)
    if args.arm_model not in models:
        raise ValueError(f"--arm_model {args.arm_model!r} must be included in --models {models}")

    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "animations"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_{args.trajectory}_combined_mujoco_motion{suffix}.gif"

    print("=== CDSM combined MuJoCo trajectory GIF render ===")
    print(f"result_dir={result_dir}")
    print(f"models={models}")
    print(f"arm_model={args.arm_model}")
    print(f"output={out_path}")

    saved = render_combined_gif(
        result_dir=result_dir,
        models=models,
        arm_model=args.arm_model,
        xml=args.xml,
        dt=args.dt,
        out_path=out_path,
        args=args,
    )
    meta = {
        "result_dir": str(result_dir),
        "trajectory": args.trajectory,
        "models": models,
        "arm_model": args.arm_model,
        "xml": str(args.xml),
        "dt": args.dt,
        "render_stride": args.render_stride,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "desired_path_style": "white dashed",
        "actual_trail_styles": {
            "edmd": "blue solid",
            "dkuc": "orange solid",
            "dkac": "green solid",
        },
        "output": str(saved),
    }
    meta_path = out_dir / f"{stamp}_{args.trajectory}_combined_mujoco_motion_metadata{suffix}.json"
    _save_json(meta_path, meta)
    print(f"[saved] {saved}")
    print(f"[metadata] {meta_path}")


if __name__ == "__main__":
    main()
