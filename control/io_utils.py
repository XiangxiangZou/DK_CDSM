"""Small IO and path helpers for control scripts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
CONTROL_ROOT = PROJECT_ROOT / "control"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_XML = (
    PROJECT_ROOT
    / "assets"
    / "models"
    / "multi_joint_cable_driven_space_robot.xml"
)


def make_output_dir(run_type: str, method: str, tag: str = "") -> Path:
    if run_type not in {"smoke_test", "full_run"}:
        raise ValueError("run_type must be smoke_test or full_run")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = CONTROL_ROOT / "outputs" / run_type / method / f"{stamp}_{method}{suffix}"
    for child in ("metrics", "arrays", "logs"):
        (path / child).mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def manifest(entry_script: str, argv: list[str]) -> dict[str, Any]:
    return {
        "entry_script": entry_script,
        "argv": argv,
        "python_executable": sys.executable,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
