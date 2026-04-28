"""
utils_mujoco_log.py
===================
MuJoCo 仿真 "原生可回放" 日志与 GIF 录像工具  (地位类似 utils_plot.py).

--------------------------------------------------------------------------
  保存的内容 (全部以 basename = <program_name>_<YYYYmmdd_HHMMSS> 命名)
--------------------------------------------------------------------------
    mj_sim_log/<program_name>/<basename>/
        <basename>.xml        ← XML 文本副本 (可被 MuJoCo `simulate` 直接 File→Open)
        <basename>.mjb        ← 二进制模型 (mj_saveModel 输出, 同样可被 `simulate` 加载)
        <basename>.npz        ← 完整轨迹 (time/qpos/qvel/ctrl/ten_length/actuator_force)
        <basename>.gif        ← GIF 预览 (offscreen 高清渲染 + 可选场景装饰器叠加)
        <basename>.meta.txt   ← 人类可读摘要 (步数, 形状, 文件清单, 回放命令)

    (可选) Figs/<program_name>/<utils_plot_timestamp>/<basename2>.gif
        ← GIF 副本, 命名与位置按 utils_plot.save_figure 的规范, 和静态图放一起

--------------------------------------------------------------------------
  如何在 MuJoCo 原生 `simulate` 里复现仿真
--------------------------------------------------------------------------
    方法 1 (静态: 查看当时的模型几何 / 相机 / tendon 布置):
        simulate.exe <basename>.mjb
        或  simulate.exe <basename>.xml
        ← 这两个文件都是 MuJoCo 原生, 保留了当时的 XML 常量 (range, timestep 等)

    方法 2 (动态: 跟着当时的轨迹回放):
        python utils_mujoco_log.py mj_sim_log/<program_name>/<basename>/
        ← 内置的 replay_from_dir 会在 passive viewer 里逐步覆盖 qpos/qvel/ctrl
          并 mj_forward, 等效于在 MuJoCo 里 "播放" 原仿真

    方法 3 (预览 / 汇报材料): 直接打开 <basename>.gif

--------------------------------------------------------------------------
  主要 API
--------------------------------------------------------------------------
    from utils_mujoco_log import MjSimLogger, add_line_to_scene

    # 场景装饰器: 每帧渲染前会被调用, 可以往 MjvScene 追加额外的 line / arrow
    # / sphere 等几何, 用来把 "期望轨迹" 或 "实时末端轨迹" 画到 GIF 里
    def scene_decorator(scene, data):
        add_line_to_scene(scene, p0, p1, rgba=(1,1,1,1), width=3.0)   # 白线
        ...

    logger = MjSimLogger(
        model=model,
        xml_text=xml_str,                # 用于保存 <basename>.xml
        enable_gif=True,
        gif_fps=30,
        gif_width=2560, gif_height=1440, # 2K 默认 (需要 XML offwidth/offheight 也 >=)
        camera_lookat=(3.0, 0.0, 0.0),
        camera_distance=12.0,
        camera_azimuth=90.0,
        camera_elevation=-90.0,
        dt=dt,
        scene_decorator=scene_decorator,
        extra_gif_save_dir=get_save_dir(),  # utils_plot 的 Figs 目录; None 则只保存一份
        extra_gif_basename="myrun_sim_playback",  # Figs 下的文件名 (不含 .gif)
    )
    while ...:
        mujoco.mj_step(model, data)      # 或 mj_forward (运动学)
        logger.record(data)              # 每步一次, 自动采状态 + 可能采 GIF 一帧
    logger.save_and_close()              # 结束时一次性写盘
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from typing import Callable, Optional, Sequence

import numpy as np

import mujoco

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# ============================================================================
# 全局缓存: 同一 Python 进程的所有 MjSimLogger 共享一个目录
# (与 utils_plot._GLOBAL_TIMESTAMP 相互独立, 但命名风格一致)
# ============================================================================
_GLOBAL_TIMESTAMP: Optional[str] = None
_GLOBAL_LOG_DIR:   Optional[str] = None
_PROGRAM_NAME:     Optional[str] = None


def _init_log_dir(custom_name: Optional[str] = None):
    global _GLOBAL_TIMESTAMP, _GLOBAL_LOG_DIR, _PROGRAM_NAME

    if _GLOBAL_LOG_DIR is not None:
        return _GLOBAL_LOG_DIR, _PROGRAM_NAME, _GLOBAL_TIMESTAMP

    try:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        if not script_name:
            script_name = "interactive"
    except Exception:
        script_name = "unknown_program"

    _PROGRAM_NAME = custom_name if custom_name else script_name
    _GLOBAL_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 目录: mj_sim_log/<program>/<program>_<ts>/
    _GLOBAL_LOG_DIR = os.path.join(
        "mj_sim_log",
        _PROGRAM_NAME,
        f"{_PROGRAM_NAME}_{_GLOBAL_TIMESTAMP}",
    )
    os.makedirs(_GLOBAL_LOG_DIR, exist_ok=True)
    return _GLOBAL_LOG_DIR, _PROGRAM_NAME, _GLOBAL_TIMESTAMP


def get_log_dir(custom_name: Optional[str] = None) -> str:
    """对外暴露: 获取本次运行的 MuJoCo 日志目录路径."""
    path, _, _ = _init_log_dir(custom_name)
    return path


# ============================================================================
# 场景装饰工具: 往 MjvScene 追加自定义几何 (供 GIF 离屏渲染叠加轨迹用)
# ============================================================================
def add_line_to_scene(
    scene: mujoco.MjvScene,
    p0: Sequence[float],
    p1: Sequence[float],
    rgba: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    width: float = 3.0,
) -> bool:
    """向 MjvScene 追加一条 3D 线段. 如果 scene.ngeom 已达 maxgeom 则返回 False.

    参数
    ----
    scene : mujoco.MjvScene (通常是 Renderer.scene 或 viewer.user_scn)
    p0, p1 : (3,) 3D 端点坐标 (世界系)
    rgba   : (4,) 颜色 + 透明度
    width  : 线宽 (像素)
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.asarray(rgba, dtype=np.float64),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        float(width),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


# ============================================================================
# MjSimLogger —— 主类
# ============================================================================
class MjSimLogger:
    """
    MuJoCo 仿真记录器: 每步同时采样状态, 可选离屏渲染 GIF 帧; 结束时写
    .xml / .mjb / .npz / .gif / .meta.txt 一整套, 文件名都统一为
    <basename>.<ext>  (basename = <program>_<timestamp>).

    构造参数
    --------
    model : mujoco.MjModel                  本次仿真所用模型 (必填)
    xml_text : str | None                   XML 文本 (保存为 .xml 可复现模型)
    enable_gif : bool                       是否开启离屏 GIF 录制 (默认 True)
    gif_fps  / gif_width / gif_height       GIF 帧率与分辨率 (默认 2560x1440 @ 30fps)
    camera_* : 相机参数                     与 passive viewer 的 MjvCamera 字段一致
    dt : float | None                       仿真步长 (默认读 model.opt.timestep)
    scene_decorator : callable(scene, data) 每帧渲染前回调, 可往 scene 追加线段
    extra_gif_save_dir : str | None         额外拷贝 GIF 的目录 (一般传 Figs/.../)
    extra_gif_basename : str | None         额外 GIF 的 basename (不含 .gif; 默认 self.basename)
    custom_name : str | None                覆盖自动推断的 program_name

    注意
    ----
    - 2K GIF (2560x1440) 需要 XML 的 <visual><global offwidth="2560" offheight="1440"/>
      足够大 (否则 Renderer 初始化会失败). 请在主脚本 load_model 里 regex 替换这两个数.
    - 高分辨率 + 长轨迹 GIF 文件可能上百 MB; 若超出需要, 可降 fps 或分辨率.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        xml_text: Optional[str] = None,
        enable_gif: bool = True,
        gif_fps: int = 30,
        gif_width: int = 2560,
        gif_height: int = 1440,
        camera_lookat: Sequence[float] = (0.0, 0.0, 0.0),
        camera_distance: float = 5.0,
        camera_azimuth: float = 90.0,
        camera_elevation: float = -30.0,
        dt: Optional[float] = None,
        scene_decorator: Optional[Callable[[mujoco.MjvScene, mujoco.MjData], None]] = None,
        extra_gif_save_dir: Optional[str] = None,
        extra_gif_basename: Optional[str] = None,
        custom_name: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dt = float(dt) if dt is not None else float(model.opt.timestep)

        self.save_dir, self.program_name, self.timestamp = _init_log_dir(custom_name)
        self.basename = f"{self.program_name}_{self.timestamp}"

        # ---- 文件路径 (全部统一为 <basename>.<ext>) ----
        self.xml_path  = os.path.join(self.save_dir, f"{self.basename}.xml")
        self.mjb_path  = os.path.join(self.save_dir, f"{self.basename}.mjb")
        self.npz_path  = os.path.join(self.save_dir, f"{self.basename}.npz")
        self.gif_path  = os.path.join(self.save_dir, f"{self.basename}.gif")
        self.meta_path = os.path.join(self.save_dir, f"{self.basename}.meta.txt")

        # ---- 立即保存 XML 副本 ----
        if xml_text is not None:
            try:
                with open(self.xml_path, "w", encoding="utf-8") as f:
                    f.write(xml_text)
                self._xml_saved = True
            except Exception as e:
                print(f"[MjSimLogger] ⚠ 保存 {self.xml_path} 失败: {e}")
                self._xml_saved = False
        else:
            self._xml_saved = False
            print("[MjSimLogger] ⚠ xml_text=None, 未保存 .xml; 回放 / simulate 加载不便")

        # ---- 立即保存二进制 MJB 模型 (原生 MuJoCo 格式, 可被 simulate.exe 直接加载) ----
        try:
            mujoco.mj_saveModel(model, self.mjb_path)
            self._mjb_saved = True
        except Exception as e:
            print(f"[MjSimLogger] ⚠ 保存 {self.mjb_path} 失败: {e}")
            self._mjb_saved = False

        # ---- 状态缓冲 ----
        self._t:          list[float]      = []
        self._qpos:       list[np.ndarray] = []
        self._qvel:       list[np.ndarray] = []
        self._ctrl:       list[np.ndarray] = []
        self._ten_length: list[np.ndarray] = []
        self._actfrc:     list[np.ndarray] = []

        # ---- GIF 采帧设置 ----
        self._want_gif = bool(enable_gif)
        if self._want_gif and not _HAS_PIL:
            print("[MjSimLogger] ⚠ Pillow (PIL) 未安装, GIF 录制已禁用")
            self._want_gif = False

        self._frames: list[np.ndarray] = []
        self._gif_fps = int(max(1, gif_fps))
        self._gif_stride = max(1, int(round(1.0 / (self._gif_fps * self.dt))))
        self._gif_width  = int(gif_width)
        self._gif_height = int(gif_height)
        self._step_ctr = 0

        self._scene_decorator = scene_decorator
        self._decorator_fail_count = 0

        self._extra_gif_dir = extra_gif_save_dir
        self._extra_gif_basename = (
            extra_gif_basename if extra_gif_basename is not None else self.basename
        )

        self._renderer: Optional[mujoco.Renderer] = None
        self._cam: Optional[mujoco.MjvCamera] = None

        if self._want_gif:
            try:
                self._renderer = mujoco.Renderer(
                    model, height=self._gif_height, width=self._gif_width
                )
                self._cam = mujoco.MjvCamera()
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._cam.lookat[:] = np.asarray(camera_lookat, dtype=float)
                self._cam.distance = float(camera_distance)
                self._cam.azimuth = float(camera_azimuth)
                self._cam.elevation = float(camera_elevation)
            except Exception as e:
                print(f"[MjSimLogger] ⚠ 离屏 renderer 初始化失败 ({e}); GIF 已禁用")
                print("    [提示] 如需 2K/QHD, 请在 XML 里设置:")
                print('           <visual><global offwidth="2560" offheight="1440"/></visual>')
                self._want_gif = False
                self._renderer = None
                self._cam = None

        # ---- 终端打印 ----
        print(f"[MjSimLogger] 日志目录: {self.save_dir}")
        print(f"[MjSimLogger] dt={self.dt}")
        print(f"[MjSimLogger] XML  -> {os.path.basename(self.xml_path)}   "
              f"{'OK' if self._xml_saved else 'SKIP'}")
        print(f"[MjSimLogger] MJB  -> {os.path.basename(self.mjb_path)}   "
              f"{'OK' if self._mjb_saved else 'SKIP'}")
        if self._want_gif:
            print(f"[MjSimLogger] GIF  -> {self._gif_width}x{self._gif_height} "
                  f"@ {self._gif_fps} fps  (每 {self._gif_stride} 仿真步采 1 帧)")
            if self._scene_decorator is not None:
                print(f"[MjSimLogger] GIF  -> 启用场景装饰器 (将叠加轨迹线到渲染帧里)")
            if self._extra_gif_dir is not None:
                print(f"[MjSimLogger] GIF  -> 额外拷贝到: {self._extra_gif_dir}")
        else:
            print(f"[MjSimLogger] GIF  -> OFF")

    # --------------------------------------------------------------------
    # 运行时: 每步调用一次
    # --------------------------------------------------------------------
    def record(self, data: mujoco.MjData) -> None:
        """主循环里每步调用 (建议在 mj_step / mj_forward 之后):
            - 采样 time/qpos/qvel/ctrl/ten_length/actuator_force
            - 如启用 GIF 且恰逢采帧 stride, 离屏渲染 1 帧 + 装饰器叠加
        """
        self._t.append(float(data.time))
        self._qpos.append(np.array(data.qpos, dtype=float))
        self._qvel.append(np.array(data.qvel, dtype=float))
        self._ctrl.append(np.array(data.ctrl, dtype=float))
        self._ten_length.append(np.array(data.ten_length, dtype=float))
        self._actfrc.append(np.array(data.actuator_force, dtype=float))

        if self._want_gif and self._renderer is not None:
            if self._step_ctr % self._gif_stride == 0:
                try:
                    self._renderer.update_scene(data, camera=self._cam)
                    if self._scene_decorator is not None:
                        try:
                            self._scene_decorator(self._renderer.scene, data)
                        except Exception as e:
                            self._decorator_fail_count += 1
                            if self._decorator_fail_count <= 3:
                                print(f"[MjSimLogger] ⚠ scene_decorator 抛异常 "
                                      f"(step={self._step_ctr}, #{self._decorator_fail_count}): {e}")
                    frame = self._renderer.render()
                    self._frames.append(np.asarray(frame).copy())
                except Exception as e:
                    if self._step_ctr < 5:
                        print(f"[MjSimLogger] ⚠ 渲染帧失败 (step={self._step_ctr}): {e}")

        self._step_ctr += 1

    # --------------------------------------------------------------------
    # 结束时: 一次性写盘
    # --------------------------------------------------------------------
    def save_and_close(self) -> None:
        if len(self._t) == 0:
            print("[MjSimLogger] ⚠ 没有任何记录, 跳过保存")
            self._close_renderer()
            return

        t_arr          = np.array(self._t, dtype=float)
        qpos_arr       = np.array(self._qpos, dtype=float)
        qvel_arr       = np.array(self._qvel, dtype=float)
        ctrl_arr       = np.array(self._ctrl, dtype=float)
        ten_length_arr = np.array(self._ten_length, dtype=float)
        actfrc_arr     = np.array(self._actfrc, dtype=float)

        # ---- 写 .npz ----
        try:
            np.savez_compressed(
                self.npz_path,
                time=t_arr,
                qpos=qpos_arr,
                qvel=qvel_arr,
                ctrl=ctrl_arr,
                ten_length=ten_length_arr,
                actuator_force=actfrc_arr,
                dt=np.array([self.dt]),
                xml_basename=np.array([os.path.basename(self.xml_path) if self._xml_saved else ""]),
                mjb_basename=np.array([os.path.basename(self.mjb_path) if self._mjb_saved else ""]),
            )
            size_kb = os.path.getsize(self.npz_path) / 1024.0
            print(f"[MjSimLogger] 轨迹 .npz 已保存 : {self.npz_path}  "
                  f"({len(t_arr)} steps, {size_kb:,.1f} KB)")
        except Exception as e:
            print(f"[MjSimLogger] ⚠ 保存 .npz 失败: {e}")

        # ---- 写 .gif (主目录) ----
        gif_written = False
        if self._want_gif and len(self._frames) > 0 and _HAS_PIL:
            try:
                imgs = [Image.fromarray(f) for f in self._frames]
                imgs[0].save(
                    self.gif_path,
                    save_all=True,
                    append_images=imgs[1:],
                    duration=int(round(1000.0 / self._gif_fps)),
                    loop=0,
                    optimize=False,     # 高分辨率下 optimize=True 会非常慢
                    disposal=2,
                )
                size_kb = os.path.getsize(self.gif_path) / 1024.0
                print(f"[MjSimLogger] GIF (主) 已保存 : {self.gif_path}  "
                      f"({len(self._frames)} 帧 @ {self._gif_fps} fps, "
                      f"{self._gif_width}x{self._gif_height}, {size_kb:,.1f} KB)")
                gif_written = True
            except Exception as e:
                print(f"[MjSimLogger] ⚠ GIF 保存失败: {e}")

        # ---- 拷贝 .gif 到 Figs (可选副本) ----
        extra_gif_path = None
        if gif_written and self._extra_gif_dir is not None:
            try:
                os.makedirs(self._extra_gif_dir, exist_ok=True)
                extra_gif_path = os.path.join(
                    self._extra_gif_dir, f"{self._extra_gif_basename}.gif"
                )
                shutil.copy2(self.gif_path, extra_gif_path)
                size_kb = os.path.getsize(extra_gif_path) / 1024.0
                print(f"[MjSimLogger] GIF (副) 已保存 : {extra_gif_path}  "
                      f"({size_kb:,.1f} KB)")
            except Exception as e:
                print(f"[MjSimLogger] ⚠ GIF 副本拷贝失败: {e}")
                extra_gif_path = None

        # ---- 写 .meta.txt ----
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                f.write(f"program_name : {self.program_name}\n")
                f.write(f"timestamp    : {self.timestamp}\n")
                f.write(f"basename     : {self.basename}\n")
                f.write(f"dt           : {self.dt}\n")
                f.write(f"n_steps      : {len(t_arr)}\n")
                f.write(f"t_start      : {t_arr[0]:.6f}\n")
                f.write(f"t_end        : {t_arr[-1]:.6f}\n")
                f.write(f"nq / nv / nu / ntendon : "
                        f"{self.model.nq} / {self.model.nv} / "
                        f"{self.model.nu} / {self.model.ntendon}\n")
                f.write(f"qpos shape   : {qpos_arr.shape}\n")
                f.write(f"ten_length shape : {ten_length_arr.shape}\n")
                f.write(f"xml_saved    : {self._xml_saved}  ({os.path.basename(self.xml_path)})\n")
                f.write(f"mjb_saved    : {self._mjb_saved}  ({os.path.basename(self.mjb_path)})\n")
                f.write(f"gif_frames   : {len(self._frames)}\n")
                f.write(f"gif_resolution : {self._gif_width}x{self._gif_height}\n")
                f.write(f"gif_fps      : {self._gif_fps}\n")
                if extra_gif_path:
                    f.write(f"gif_extra    : {extra_gif_path}\n")
                f.write("\n[在 MuJoCo 原生 simulate 里查看当时的模型]\n")
                f.write(f"    simulate.exe \"{self.mjb_path}\"\n")
                f.write(f"    # 或 simulate.exe \"{self.xml_path}\"\n")
                f.write("\n[用 Python 回放当时的仿真视频]\n")
                f.write(f"    python utils_mujoco_log.py \"{self.save_dir}\"\n")
            print(f"[MjSimLogger] 元信息   : {self.meta_path}")
        except Exception as e:
            print(f"[MjSimLogger] ⚠ 保存 meta.txt 失败: {e}")

        self._close_renderer()
        print(f"[MjSimLogger] 日志目录 : {self.save_dir}\n")

    def _close_renderer(self):
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None


# ============================================================================
# 回放工具: 从一个日志目录重建 MuJoCo 场景, 在 passive viewer 里按原步长回放.
# ============================================================================
def _find_one(log_dir: str, ext: str) -> Optional[str]:
    """在 log_dir 里找最新的 <...>.<ext> 文件, 没找到返回 None."""
    cands = [f for f in os.listdir(log_dir) if f.lower().endswith(ext.lower())]
    if not cands:
        return None
    return os.path.join(log_dir, sorted(cands)[-1])


def replay_from_dir(log_dir: str, speed_factor: float = 1.0) -> None:
    """载入 log_dir 下的 <basename>.xml + <basename>.npz, 在 MuJoCo passive viewer 里回放.

    参数
    ----
    log_dir      : 保存仿真产物的目录
    speed_factor : 回放速率 (1.0=原速, 2.0=两倍速, 0.5=慢放一半)
    """
    xml_path = _find_one(log_dir, ".xml")
    if xml_path is None:
        # 兼容旧命名
        xml_path = os.path.join(log_dir, "model.xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"{log_dir} 下既没有 *.xml 也没有 model.xml")

    npz_path = _find_one(log_dir, ".npz")
    if npz_path is None:
        raise FileNotFoundError(f"{log_dir} 下没有 *.npz")

    print(f"[replay] 载入 XML: {xml_path}")
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_str = f.read()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    print(f"[replay] 载入轨迹: {npz_path}")
    rollout = np.load(npz_path)
    t_arr    = rollout["time"]
    qpos_arr = rollout["qpos"]
    qvel_arr = rollout["qvel"]
    ctrl_arr = rollout["ctrl"]
    dt = float(rollout["dt"][0])
    N = len(t_arr)
    print(f"[replay] N={N} steps, dt={dt}s, "
          f"总时长 {t_arr[-1] - t_arr[0]:.2f}s  (speed_factor={speed_factor})")

    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        tick = dt / max(1e-6, speed_factor)
        for k in range(N):
            if not viewer.is_running():
                break
            t0 = time.time()
            data.qpos[:]   = qpos_arr[k]
            data.qvel[:]   = qvel_arr[k]
            data.ctrl[:]   = ctrl_arr[k]
            data.time      = float(t_arr[k])
            mujoco.mj_forward(model, data)
            viewer.sync()
            rest = tick - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)

    print("[replay] 回放完成")


# ============================================================================
# CLI: python utils_mujoco_log.py <log_dir> [speed_factor]
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python utils_mujoco_log.py <log_dir> [speed_factor=1.0]")
        sys.exit(1)
    log_dir = sys.argv[1]
    speed = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    replay_from_dir(log_dir, speed_factor=speed)
