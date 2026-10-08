"""Tests for depth map cleaning filters and step 02 performance optimizations.

Tests:
1. clean_single_depth_gradient
2. clean_saturation_mask
3. clean_depth_ceiling
4. clean_grazing_angles
5. clean_all_depth_maps
6. diagnose_planes_quads with pre_extracted_planes
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import tempfile

from package_loader import CameraIntrinsics
from clean_depth_maps import (
    clean_single_depth_gradient,
    clean_saturation_mask,
    clean_depth_ceiling,
    clean_grazing_angles,
    clean_all_depth_maps,
)
from regularize_planes import diagnose_planes_quads


@pytest.fixture
def test_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=200.0,
        fl_y=200.0,
        cx=50.0,
        cy=50.0,
        w=100,
        h=100,
        camera_angle_x=0.5,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
    )


def test_clean_single_depth_gradient():
    h, w = 40, 40
    depth = np.ones((h, w), dtype=np.float32) * 2.0
    # Create step discontinuity
    depth[:, 20:] = 10.0
    cleaned, n_cleared = clean_single_depth_gradient(depth, max_depth_gradient=0.15)
    assert n_cleared > 0
    # Pixels near step should be 0.0
    assert cleaned[20, 20] == 0.0
    # Flat areas should remain unaffected
    assert cleaned[5, 5] == 2.0
    assert cleaned[5, 35] == 10.0


def test_clean_saturation_mask():
    h, w = 40, 40
    depth = np.ones((h, w), dtype=np.float32) * 2.5
    rgb = np.full((h, w, 3), 100, dtype=np.uint8)
    # Create overexposed blown out region
    rgb[15:25, 15:25] = 255
    cleaned, n_cleared = clean_saturation_mask(
        depth,
        rgb,
        min_threshold=250,
        max_threshold=255,
        percentile=99.0,
        max_chroma_diff=35.0,
        dilation_radius=0,
    )
    assert n_cleared > 0
    assert cleaned[20, 20] == 0.0
    assert cleaned[5, 5] == 2.5


def test_clean_depth_ceiling():
    depth = np.array([[2.0, 3.0], [4.5, 12.0]], dtype=np.float32)
    cleaned, n_cleared = clean_depth_ceiling(depth, max_depth_ceiling=5.0)
    assert n_cleared == 1
    assert cleaned[1, 1] == 0.0
    assert cleaned[0, 0] == 2.0


def test_clean_grazing_angles(test_intrinsics: CameraIntrinsics):
    h, w = test_intrinsics.h, test_intrinsics.w
    depth = np.full((h, w), 2.0, dtype=np.float32)
    normals_cam = np.zeros((h, w, 3), dtype=np.float32)
    normals_cam[..., 0] = 1.0  # perpendicular to optical axis
    cleaned, n_cleared = clean_grazing_angles(
        depth,
        normals_cam,
        test_intrinsics,
        max_grazing_angle_deg=85.0,
    )
    assert n_cleared > 0
    assert cleaned[50, 50] == 0.0


def test_clean_all_depth_maps_cpu(test_intrinsics: CameraIntrinsics):
    h, w = test_intrinsics.h, test_intrinsics.w
    d1 = np.full((h, w), 2.0, dtype=np.float32)
    d2 = np.full((h, w), 2.0, dtype=np.float32)
    img1 = np.full((h, w, 3), 128, dtype=np.uint8)
    img2 = np.full((h, w, 3), 128, dtype=np.uint8)
    c2w1 = np.eye(4, dtype=np.float32)
    c2w2 = np.eye(4, dtype=np.float32)
    c2w2[0, 3] = 0.5  # baseline

    cleaned_maps, n_cleared = clean_all_depth_maps(
        depth_maps=[d1, d2],
        images=[img1, img2],
        c2w_mats=np.stack([c2w1, c2w2]),
        intrinsics=test_intrinsics,
        normal_maps=None,
        enable_freespace_filter=False,
    )
    assert len(cleaned_maps) == 2
    assert cleaned_maps[0].shape == (h, w)


def test_diagnose_planes_quads_pre_extracted(tmp_path: Path):
    from plyfile import PlyData, PlyElement
    n_pts = 100
    x = np.random.uniform(-1, 1, n_pts).astype(np.float32)
    y = np.random.uniform(-1, 1, n_pts).astype(np.float32)
    z = np.zeros(n_pts, dtype=np.float32)
    vertex = np.array(
        list(zip(x, y, z, np.zeros(n_pts), np.zeros(n_pts), np.ones(n_pts))),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")],
    )
    el = PlyElement.describe(vertex, "vertex")
    ply_path = tmp_path / "test.ply"
    PlyData([el]).write(str(ply_path))

    pts_plane = np.stack([x, y, z], axis=1)
    mock_plane = {
        "idx": np.arange(n_pts),
        "centroid": np.mean(pts_plane, axis=0),
        "normal": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "kind": "horizontal",
        "bbox_area": 1.0,
        "area": 1.0,
    }

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    report = diagnose_planes_quads(
        depth_ply_path=ply_path,
        output_dir=out_dir,
        min_area=0.001,
        pre_extracted_planes=[mock_plane],
    )
    assert "planes" in report
    assert len(report["planes"]) == 1
    assert Path(report["report_path"]).exists()
