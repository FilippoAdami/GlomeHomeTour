#!/usr/bin/env python3
"""
Unit tests for backend/08_artifacts_generation/match_real_captured_frames.py.
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "08_artifacts_generation"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_FastGS_DNSplatter"))

from match_real_captured_frames import (
    find_closest_real_camera,
    create_side_by_side_image,
)


def test_find_closest_real_camera():
    # Setup 3 synthetic cameras
    cameras = []
    # Cam 0: at (0, 0, 5), looking along (0, 0, -1)
    # W2C matrix R, t where center = -R^T t = [0, 0, 5], look = R^T [0, 0, 1] = [0, 0, -1]
    # So R has row 2 = [0, 0, -1]
    R0 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    t0 = -R0 @ np.array([0, 0, 5], dtype=np.float32)
    vm0 = np.eye(4, dtype=np.float32)
    vm0[:3, :3] = R0
    vm0[:3, 3] = t0
    cameras.append({"image_path": Path("frame_000.jpg"), "viewmat": vm0})

    # Cam 1: at (10, 10, 10), looking away
    R1 = np.eye(3, dtype=np.float32)
    t1 = -R1 @ np.array([10, 10, 10], dtype=np.float32)
    vm1 = np.eye(4, dtype=np.float32)
    vm1[:3, :3] = R1
    vm1[:3, 3] = t1
    cameras.append({"image_path": Path("frame_001.jpg"), "viewmat": vm1})

    # Cam 2: close to (0.1, 0.1, 4.9), looking along (0, 0, -1)
    t2 = -R0 @ np.array([0.1, 0.1, 4.9], dtype=np.float32)
    vm2 = np.eye(4, dtype=np.float32)
    vm2[:3, :3] = R0
    vm2[:3, 3] = t2
    cameras.append({"image_path": Path("frame_002.jpg"), "viewmat": vm2})

    # Query ideal eye at (0.05, 0.05, 4.95), target at (0, 0, 0)
    ideal_eye = np.array([0.05, 0.05, 4.95], dtype=np.float32)
    ideal_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    match = find_closest_real_camera(ideal_eye, ideal_target, cameras)
    assert match["image_name"] in ("frame_000.jpg", "frame_002.jpg")
    assert match["position_distance_m"] < 0.20
    assert match["angle_diff_deg"] < 5.0


def test_create_side_by_side_image(tmp_path: Path):
    mesh_img = tmp_path / "mesh.png"
    real_img = tmp_path / "real.jpg"
    out_sbs = tmp_path / "sbs.png"

    Image.new("RGB", (800, 800), (200, 50, 50)).save(mesh_img)
    Image.new("RGB", (600, 1000), (50, 200, 50)).save(real_img)

    result = create_side_by_side_image(
        mesh_img_path=mesh_img,
        real_img_path=real_img,
        output_path=out_sbs,
        title_mesh="Mesh Ideal View",
        title_real="Real Video Frame",
        target_height=400,
    )

    assert result.is_file()
    with Image.open(result) as sbs:
        assert sbs.height > 400
        assert sbs.width > 400
