#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement

import pytest
import sys

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

import importlib.util

_align_path = _backend_dir / "05_floor_plan" / "scene_alignment.py"
_spec = importlib.util.spec_from_file_location("scene_alignment", str(_align_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

find_floor_height = _mod.find_floor_height
align_cameras = _mod.align_cameras
align_point_cloud_and_cameras = _mod.align_point_cloud_and_cameras
from scene.colmap_loader import read_extrinsics_text, qvec2rotmat


def create_synthetic_room_ply(ply_path: Path, floor_y: float = -1.5, angle_deg: float = 30.0):
    """Create a synthetic rectangular room with floor at floor_y, rotated by angle_deg around Y."""
    rng = np.random.default_rng(42)
    w, h, d = 4.0, 2.5, 3.0

    # 1. Generate floor points (Y = floor_y, normal = [0, 1, 0])
    n_floor = 2000
    fx = rng.uniform(-w/2, w/2, n_floor)
    fz = rng.uniform(-d/2, d/2, n_floor)
    fy = np.full(n_floor, floor_y)
    fnx = np.zeros(n_floor)
    fny = np.ones(n_floor)
    fnz = np.zeros(n_floor)

    # 2. Generate 4 walls (normals along X and Z)
    n_wall = 1000
    # Wall 1: X = -w/2
    w1_x = np.full(n_wall, -w/2)
    w1_z = rng.uniform(-d/2, d/2, n_wall)
    w1_y = rng.uniform(floor_y, floor_y + h, n_wall)
    w1_nx = np.ones(n_wall)
    w1_ny = np.zeros(n_wall)
    w1_nz = np.zeros(n_wall)

    # Wall 2: X = w/2
    w2_x = np.full(n_wall, w/2)
    w2_z = rng.uniform(-d/2, d/2, n_wall)
    w2_y = rng.uniform(floor_y, floor_y + h, n_wall)
    w2_nx = -np.ones(n_wall)
    w2_ny = np.zeros(n_wall)
    w2_nz = np.zeros(n_wall)

    # Wall 3: Z = -d/2
    w3_x = rng.uniform(-w/2, w/2, n_wall)
    w3_z = np.full(n_wall, -d/2)
    w3_y = rng.uniform(floor_y, floor_y + h, n_wall)
    w3_nx = np.zeros(n_wall)
    w3_ny = np.zeros(n_wall)
    w3_nz = np.ones(n_wall)

    # Wall 4: Z = d/2
    w4_x = rng.uniform(-w/2, w/2, n_wall)
    w4_z = np.full(n_wall, d/2)
    w4_y = rng.uniform(floor_y, floor_y + h, n_wall)
    w4_nx = np.zeros(n_wall)
    w4_ny = np.zeros(n_wall)
    w4_nz = -np.ones(n_wall)

    x = np.concatenate([fx, w1_x, w2_x, w3_x, w4_x])
    y = np.concatenate([fy, w1_y, w2_y, w3_y, w4_y])
    z = np.concatenate([fz, w1_z, w2_z, w3_z, w4_z])
    nx = np.concatenate([fnx, w1_nx, w2_nx, w3_nx, w4_nx])
    ny = np.concatenate([fny, w1_ny, w2_ny, w3_ny, w4_ny])
    nz = np.concatenate([fnz, w1_nz, w2_nz, w3_nz, w4_nz])

    pts = np.stack([x, y, z], axis=1)
    normals = np.stack([nx, ny, nz], axis=1)

    # Rotate around Y by angle_deg
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R_rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    pts_rot = pts @ R_rot.T
    normals_rot = normals @ R_rot.T

    # Also shift to arbitrary world position
    pts_rot += np.array([-5.0, 0.0, 3.0])

    v_data = np.zeros(
        len(pts_rot),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
    )
    v_data["x"] = pts_rot[:, 0]
    v_data["y"] = pts_rot[:, 1]
    v_data["z"] = pts_rot[:, 2]
    v_data["nx"] = normals_rot[:, 0]
    v_data["ny"] = normals_rot[:, 1]
    v_data["nz"] = normals_rot[:, 2]
    v_data["red"] = 200
    v_data["green"] = 200
    v_data["blue"] = 200

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v_data, "vertex")], text=False).write(str(ply_path))
    return pts_rot, normals_rot


def test_find_floor_height(tmp_path):
    ply_file = tmp_path / "room.ply"
    pts, normals = create_synthetic_room_ply(ply_file, floor_y=-1.5, angle_deg=25.0)
    detected_y = find_floor_height(pts, normals)
    assert abs(detected_y - (-1.5)) < 0.05


def test_align_point_cloud_and_cameras(tmp_path):
    workspace = tmp_path / "scene"
    sparse_dir = workspace / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    input_ply = workspace / "02_depth_estimation" / "depth" / "points3D_depth.ply"
    pts_rot, normals_rot = create_synthetic_room_ply(input_ply, floor_y=-1.2, angle_deg=35.0)

    # Create dummy images.txt
    images_txt = sparse_dir / "images.txt"
    # Camera at (0, 0, 0) looking forward
    images_txt.write_text(
        "# Image list with two lines of data per image:\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "1 1.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1 frame_0001.jpg\n"
        "100.0 200.0 1\n"
        "2 1.000000000 0.000000000 0.000000000 0.000000000 1.000000000 0.500000000 -2.000000000 1 frame_0002.jpg\n"
        "150.0 250.0 2\n"
    )

    out_dir = workspace / "05_floor_plan"
    meta = align_point_cloud_and_cameras(input_ply, sparse_dir, out_dir)

    # 1. Verify original files are preserved
    assert input_ply.exists()
    assert images_txt.exists()
    assert (out_dir / "points3D_aligned.ply").exists()
    assert (out_dir / "images_aligned.txt").exists()
    assert (out_dir / "alignment_meta.json").exists()

    # 2. Check aligned point cloud
    aligned_ply = PlyData.read(str(out_dir / "points3D_aligned.ply"))
    v = aligned_ply["vertex"].data
    aligned_pts = np.stack([v["x"], v["y"], v["z"]], axis=1)

    # Floor should be at Y ~ 0
    ny = v["ny"]
    floor_pts_y = aligned_pts[np.abs(ny) > 0.7, 1]
    assert np.mean(floor_pts_y) == pytest.approx(0.0, abs=0.08)

    # Min X and Min Z should be >= 0 (up to small percentile tolerance)
    assert np.percentile(aligned_pts[:, 0], 0.5) == pytest.approx(0.0, abs=0.05)
    assert np.percentile(aligned_pts[:, 2], 0.5) == pytest.approx(0.0, abs=0.05)

    # 3. Check aligned camera poses
    aligned_cams = read_extrinsics_text(str(out_dir / "images_aligned.txt"))
    assert len(aligned_cams) == 2
