"""Run existing experiment scripts through a stable organized entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ARCHIVE_ROOT = PROJECT_ROOT / "archive"
ARCHIVE_IMPORT_PATHS = (
    ARCHIVE_ROOT,
    ARCHIVE_ROOT / "support",
    ARCHIVE_ROOT / "legacy",
    ARCHIVE_ROOT / "baselines",
    ARCHIVE_ROOT / "control",
    ARCHIVE_ROOT / "hybrid_models",
    ARCHIVE_ROOT / "model_comparison",
    ARCHIVE_ROOT / "diagnostics",
    ARCHIVE_ROOT / "deployment_pipeline",
)


def run_project_script(relative_path: str) -> None:
    """Execute a project script as ``__main__`` with stable import paths."""
    script_path = PROJECT_ROOT / relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment target not found: {script_path}")
    for path in (PROJECT_ROOT, SRC_ROOT, *ARCHIVE_IMPORT_PATHS, script_path.parent):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    runpy.run_path(str(script_path), run_name="__main__")
