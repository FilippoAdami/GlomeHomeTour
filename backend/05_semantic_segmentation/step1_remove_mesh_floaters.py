#!/usr/bin/env python3
"""Compatibility entry point for the mesh cleanup now owned by Stage 04."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from mesh_cleanup import *  # noqa: F401,F403
from mesh_cleanup import main

if __name__ == "__main__":
    raise SystemExit(main())
