#!/usr/bin/env python3
"""Tests for ARCore geometry-based keyframe budgeting and selection."""

import sys
from pathlib import Path
import numpy as np

_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from select_keyframes_arcore import (
    estimate_room_extent_from_arcore,
    compute_keyframe_budget,
    select_keyframes_arcore,
    rotation_angle_deg,
)


def _make_dummy_frame(idx: int, x: float, y: float, z: float, yaw_deg: float = 0.0):
    rad = np.radians(yaw_deg)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = [x, y, z]
    return {
        "file_path": f"images/{idx:06d}.jpg",
        "timestamp_ns": idx * 33333333,
        "transform_matrix": mat.tolist(),
        "fl_x": 1000.0, "fl_y": 1000.0, "cx": 500.0, "cy": 500.0,
    }


def test_standoff_geometry_calculation():
    frames = []
    xs = np.linspace(0, 3, 25)
    zs = np.linspace(0, 2, 25)
    i = 0
    for x in xs:
        frames.append(_make_dummy_frame(i, x, 1.4, 0.0))
        i += 1
    for z in zs:
        frames.append(_make_dummy_frame(i, 3.0, 1.4, z))
        i += 1
    for x in reversed(xs):
        frames.append(_make_dummy_frame(i, x, 1.4, 2.0))
        i += 1
    for z in reversed(zs):
        frames.append(_make_dummy_frame(i, 0.0, 1.4, z))
        i += 1

    extent = estimate_room_extent_from_arcore(frames, standoff_m=0.8)

    assert abs(extent["camera_span_x"] - 3.0) < 0.2
    assert abs(extent["camera_span_z"] - 2.0) < 0.2
    assert extent["room_dim_x"] >= extent["camera_span_x"] + 1.5
    assert extent["room_dim_z"] >= extent["camera_span_z"] + 1.5
    assert 14.0 <= extent["floor_area_m2"] <= 20.0

    lo, hi, target = compute_keyframe_budget(extent, 1000)
    assert 150 <= lo <= 230
    assert 180 <= hi <= 280
    assert lo <= target <= hi


def test_select_keyframes_pruning():
    frames = [_make_dummy_frame(i, i * 0.01, 1.4, 0.0, yaw_deg=i * 0.1) for i in range(500)]
    selected, meta = select_keyframes_arcore(frames)

    assert meta["pruned"] is True
    assert len(selected) < len(frames)
    assert meta["budget"][0] <= len(selected) <= meta["budget"][1] + 2
    assert selected[0]["file_path"] == frames[0]["file_path"]
    assert selected[-1]["file_path"] == frames[-1]["file_path"]


def test_edge_cases():
    f1 = [_make_dummy_frame(0, 0, 1.4, 0)]
    sel1, meta1 = select_keyframes_arcore(f1)
    assert len(sel1) == 1

    f_few = [_make_dummy_frame(i, i * 0.2, 1.4, 0) for i in range(15)]
    sel_few, meta_few = select_keyframes_arcore(f_few)
    assert len(sel_few) == 15
    assert meta_few["pruned"] is False


def test_sharpness_aware_swapping():
    # Sequence of 100 frames where frame 10 is very blurry and frame 11 is crisp and close
    frames = [_make_dummy_frame(i, i * 0.05, 1.4, 0.0) for i in range(100)]
    sharpness = np.ones(100) * 50.0
    sharpness[10] = 5.0   # very blurry
    sharpness[11] = 95.0  # very sharp

    # Target count that includes ~50 frames
    sel, meta = select_keyframes_arcore(frames, target_count=50, sharpness_scores=sharpness)
    sel_paths = [f["file_path"] for f in sel]
    # Blurry frame 10 is rejected
    assert "images/000010.jpg" not in sel_paths
    # Crisp frame 11 is chosen
    assert "images/000011.jpg" in sel_paths


def test_max_rotation_preserved_during_pruning():
    # Sequence with a sharp 30-degree turn in the middle:
    # Even if target_count requests very few frames, consecutive frames must never exceed 16 deg jump
    frames = []
    for i in range(30):
        # 1 deg per frame
        frames.append(_make_dummy_frame(i, i * 0.05, 1.4, 0.0, yaw_deg=i * 1.5))
    sel, meta = select_keyframes_arcore(frames, target_count=5)
    # Check max consecutive rotation between selected frames
    rots = [np.array(f["transform_matrix"])[:3, :3] for f in sel]
    from select_keyframes_arcore import rotation_angle_deg
    for k in range(len(rots) - 1):
        ang = rotation_angle_deg(rots[k], rots[k + 1])
        assert ang <= 16.5, f"Rotation jump {ang:.1f} deg between {sel[k]['file_path']} and {sel[k+1]['file_path']} exceeds 16 deg limit"

