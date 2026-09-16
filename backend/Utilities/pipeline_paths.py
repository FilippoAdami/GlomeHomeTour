"""Puts every numbered pipeline-stage directory on sys.path.

Stage folders (00_ingestion, 01_poses_refinment, ...) can't be dotted Python
package names (leading digit), so stage modules import each other by bare
module name (e.g. ``from package_loader import PackageLoader``) and rely on
this bootstrap to make cross-stage names resolvable regardless of which
stage's script is the entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_STAGE_DIRS = (
    "00_ingestion",
    "01_poses_refinment",
    "02_depth_estimation",
    "03_2DGS_training",
    "05_2DGS_to_mesh",
)


def stage_paths() -> list[str]:
    """The same directories :func:`bootstrap` adds, for building a subprocess env.

    Steps that shell out to a stage script (the COLMAP converter, ``train.py``)
    must pass these through ``PYTHONPATH``: a fresh interpreter inherits none of
    the parent's ``sys.path``, and those scripts import bare stage module names.
    """
    return [str(_BACKEND_DIR / stage) for stage in _STAGE_DIRS]


def subprocess_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """``env`` (default ``os.environ``) with stage dirs in PYTHONPATH and ROCm in LD_LIBRARY_PATH."""
    import os

    env = dict(os.environ if env is None else env)
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(stage_paths() + ([existing_py] if existing_py else []))

    # Ensure ROCm libraries are discoverable by binaries like colmap
    rocm_dirs = [p for p in ["/opt/rocm-7.1.1/lib", "/opt/rocm/lib"] if Path(p).is_dir()]
    if rocm_dirs:
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        new_ld = [d for d in rocm_dirs if d not in existing_ld.split(os.pathsep)]
        if new_ld:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(new_ld + ([existing_ld] if existing_ld else []))
    return env


def bootstrap() -> None:
    for path in stage_paths():
        if path not in sys.path:
            sys.path.insert(0, path)
