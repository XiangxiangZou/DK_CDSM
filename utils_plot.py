import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================================
# 全局变量：确保在同一个 Python 脚本的一次运行中，
# 生成的所有图片（包含多张静态图和动图）
# 都被统一放入同一个带时间戳的文件夹中。
# ==========================================
_GLOBAL_TIMESTAMP = None
_GLOBAL_SAVE_DIR = None
_PROGRAM_NAME = None

def _init_save_dir(custom_name=None):
    """
    内部方法：初始化并获取全局的保存目录。
    只会在此脚本第一次被调用时生成时间戳和创建文件夹。
    """
    global _GLOBAL_TIMESTAMP, _GLOBAL_SAVE_DIR, _PROGRAM_NAME
    
    # 如果已经初始化过，直接返回之前的目录和时间戳
    if _GLOBAL_SAVE_DIR is not None:
        return _GLOBAL_SAVE_DIR, _PROGRAM_NAME, _GLOBAL_TIMESTAMP
        
    # 自动侦测当前正在运行的 Python 脚本名称 
    try:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        if not script_name:
            script_name = "interactive"
    except Exception:
        script_name = "unknown_program"
        
    _PROGRAM_NAME = custom_name if custom_name else script_name
    _GLOBAL_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建多级目录: Figs/程序名称/时间戳/
    _GLOBAL_SAVE_DIR = os.path.join("Figs", _PROGRAM_NAME, _GLOBAL_TIMESTAMP)
    
    if not os.path.exists(_GLOBAL_SAVE_DIR):
        os.makedirs(_GLOBAL_SAVE_DIR)
        
    return _GLOBAL_SAVE_DIR, _PROGRAM_NAME, _GLOBAL_TIMESTAMP

def get_save_dir(custom_name=None):
    """
    对外暴露的方法：获取本次运行统一的保存目录路径。
    专门用于保存那些不由 plt.savefig() 直接处理的文件（比如 GIF 动图）。
    """
    save_dir, _, _ = _init_save_dir(custom_name)
    return save_dir

def save_figure(fig_name=None, custom_name=None):
    """
    专业的科研绘图保存工具。
    一式三份 (PNG, SVG, PDF) 保存当前激活的 matplotlib 图像。
    
    参数:
    - fig_name: 选填。给当前图片加一个后缀名 (例如 'trajectory', 'joint_angles')。
    - custom_name: 选填。强制指定程序名，一般留空让它自动侦测即可。
    """
    save_dir, program_name, timestamp = _init_save_dir(custom_name)
    
    # 构建基础文件名: 程序名_时间戳
    img_name = f"{program_name}_{timestamp}"
    
    # 如果用户传了后缀名，就拼上去
    if fig_name:
        img_name = f"{img_name}_{fig_name}"
        
    base_path = os.path.join(save_dir, img_name)
    
    print(f"\n[图片保存] 正在输出图像至: {save_dir}")
    
    # 一式三份保存，dpi=1200保证极高清晰度，bbox_inches='tight'自动去除多余白边
    plt.savefig(f"{base_path}.png", dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{base_path}.svg", bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f"{base_path}.pdf", bbox_inches='tight', pad_inches=0.1)
    
    print(f"[图片保存] 成功！文件名前缀: {img_name}\n")