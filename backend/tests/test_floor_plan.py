"""Unit tests for Step 05: LOD 300 Floor Plan Vectorization."""

import math
from pathlib import Path
import sys
import numpy as np
import pytest

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap
bootstrap()

from wall_extraction import (
    find_principal_axes_and_bounds,
    build_watertight_room_walls,
)
from opening_consensus import (
    ray_intersect_plane,
    unproject_camera_ray,
    cluster_intervals_1d,
)
from cad_export import (
    export_floorplan_svg,
    export_floorplan_json,
    export_floorplan_dxf,
)


def test_build_watertight_room_walls():
    """Verify 4 watertight perimeter walls are built with exact corners and 0.30m thickness."""
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 0.0, 1.0])
    bounds = {"u_min": -2.0, "u_max": 2.0, "v_min": -1.5, "v_max": 1.5}

    walls, metrics = build_watertight_room_walls(u_axis, v_axis, bounds, default_thickness=0.30)

    assert len(walls) == 4
    assert metrics["width_m"] == 4.0
    assert metrics["length_m"] == 3.0
    assert metrics["area_sqm"] == 12.0
    assert metrics["perimeter_m"] == 14.0

    # Verify closed loop: each wall starts where previous ended
    for i in range(4):
        curr_end = walls[i]["end"]
        next_start = walls[(i + 1) % 4]["start"]
        assert np.allclose(curr_end, next_start, atol=1e-3)

    # Verify thickness
    for w in walls:
        assert w["thickness"] == 0.30
        assert len(w["polygon"]) == 4


def test_cluster_intervals_1d():
    """Verify overlapping candidate opening intervals fuse into weighted median consensus."""
    candidates = [
        {"u_min": 1.18, "u_max": 2.08, "v_sill": 0.02, "v_head": 2.10, "confidence": 0.95, "class_name": "door"},
        {"u_min": 1.20, "u_max": 2.10, "v_sill": 0.01, "v_head": 2.12, "confidence": 0.98, "class_name": "door"},
        {"u_min": 1.22, "u_max": 2.11, "v_sill": 0.03, "v_head": 2.08, "confidence": 0.92, "class_name": "door"},
    ]
    openings = cluster_intervals_1d(candidates, iou_threshold=0.35)
    assert len(openings) == 1
    op = openings[0]
    assert op["type"] == "door"
    assert abs(op["u_min"] - 1.20) < 0.03
    assert abs(op["width"] - 0.90) < 0.03
    assert op["support_views"] == 3


def test_cad_exporters(tmp_path: Path):
    """Verify SVG, DXF, and JSON floor plans write valid files."""
    walls = [{
        "name": "North Wall",
        "start": [0.0, 0.0],
        "end": [4.0, 0.0],
        "thickness": 0.30,
        "polygon": [[0.0, 0.0], [4.0, 0.0], [4.0, 0.30], [0.0, 0.30]],
    }]
    openings = [{
        "type": "door",
        "wall_start": [0.0, 0.0],
        "wall_end": [4.0, 0.0],
        "u_min": 1.20,
        "u_max": 2.10,
        "width": 0.90,
    }]
    metrics = {"area_sqm": 12.0, "ceiling_height_m": 2.70}

    svg_file = tmp_path / "floorplan.svg"
    dxf_file = tmp_path / "floorplan.dxf"
    json_file = tmp_path / "floorplan.json"

    export_floorplan_svg(walls, openings, [], metrics, svg_file)
    export_floorplan_dxf(walls, openings, [], metrics, dxf_file)
    export_floorplan_json(walls, openings, [], metrics, json_file)

    assert svg_file.is_file() and svg_file.stat().st_size > 500
    assert dxf_file.is_file() and dxf_file.stat().st_size > 500
    assert json_file.is_file() and json_file.stat().st_size > 100
