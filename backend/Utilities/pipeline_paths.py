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

_BACKEND_DIR = Path(__file__).resolve().parent
_STAGE_DIRS = (
    "00_ingestion",
    "01_poses_refinment",
    "02_depth_estimation",
    "03_2DGS_training",
    "05_2DGS_to_mesh",
)


def bootstrap() -> None:
    for stage in _STAGE_DIRS:
        path = str(_BACKEND_DIR / stage)
        if path not in sys.path:
            sys.path.insert(0, path)
