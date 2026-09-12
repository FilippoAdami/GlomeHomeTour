"""Unit tests for backend/reconstruction/ depth priors and surfel initialization."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ingestion.package_loader import CameraIntrinsics, Keyframe
from reconstruction.depth_priors import (
    DepthPriorEstimator,
    GlobalDepthGraphOptimizer,
    GlobalDepthGraphResult,
    MetricDepthAligner,
    compute_surface_normals,
)
from reconstruction.initialization import (
    SurfelCloud,
    SurfelCloudInitializer,
    build_orthonormal_tangent_frame,
)


@pytest.fixture
def mock_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=500.0,
        fl_y=500.0,
        cx=320.0,
        cy=240.0,
        w=640,
        h=480,
        camera_angle_x=0.64,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
    )


# ==============================================================================
# 1. Depth Prior Estimation & Surface Normals Tests
# ==============================================================================

def test_depth_prior_estimation_shape_and_range():
    estimator = DepthPriorEstimator()
    img_rgb = (np.random.RandomState(42).rand(240, 320, 3) * 255).astype(np.uint8)

    depth = estimator.estimate_depth(img_rgb)
    assert depth.shape == (240, 320)
    assert depth.dtype == np.float32
    assert np.all(depth >= 0.2)
    assert np.all(depth <= 15.0)


def test_compute_surface_normals_flat_wall(mock_intrinsics: CameraIntrinsics):
    # Flat wall at constant depth z = 2.0 meters
    h, w = 100, 100
    intr = CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=100.0, fl_y=100.0, cx=50.0, cy=50.0,
        w=w, h=h, camera_angle_x=0.8,
        k1=0.0, k2=0.0, p1=0.0, p2=0.0
    )
    depth = np.full((h, w), 2.0, dtype=np.float32)

    normals = compute_surface_normals(depth, intr)
    assert normals.shape == (h, w, 3)

    # In OpenGL camera convention, a flat wall in front of camera (z < 0) has normal pointing towards camera (+Z)
    center_normal = normals[50, 50]
    np.testing.assert_allclose(center_normal, [0.0, 0.0, 1.0], atol=1e-3)

    # Vector magnitudes should all be 1.0
    norms = np.linalg.norm(normals[10:-10, 10:-10], axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ==============================================================================
# 2. Metric Scale-Shift Alignment Tests
# ==============================================================================

def test_metric_depth_alignment_analytical(mock_intrinsics: CameraIntrinsics):
    aligner = MetricDepthAligner()
    h, w = mock_intrinsics.h, mock_intrinsics.w

    # Ground truth: linear depth ramp
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    depth_gt = 1.0 + 3.0 * (y_grid / float(h))  # 1.0m to 4.0m

    # Simulated monocular prediction with scale 1.4 and shift 0.5
    s_true = 1.4
    t_true = 0.5
    mono_depth = (depth_gt - t_true) / s_true

    # Sample 30 sparse 3D landmark points in OpenGL camera frame (z = -depth, y = -(v-cy)*d/fy)
    pts_3d = []
    rng = np.random.RandomState(42)
    for _ in range(30):
        px = rng.randint(50, w - 50)
        py = rng.randint(50, h - 50)
        d = depth_gt[py, px]
        x = (px - mock_intrinsics.cx) * d / mock_intrinsics.fl_x
        y = -(py - mock_intrinsics.cy) * d / mock_intrinsics.fl_y
        z = -d
        pts_3d.append([x, y, z])

    sparse_pts = np.array(pts_3d, dtype=np.float64)
    c2w = np.eye(4)

    aligned, s_est, t_est, rmse = aligner.align(mono_depth, sparse_pts, c2w, mock_intrinsics)

    # Should recover scale and shift accurately
    assert math.isclose(s_est, s_true, rel_tol=0.05)
    assert math.isclose(t_est, t_true, abs_tol=0.1)
    assert rmse < 0.05
    np.testing.assert_allclose(aligned, depth_gt, atol=0.08)


def test_global_depth_graph_optimizer(mock_intrinsics: CameraIntrinsics, tmp_path: Path):
    h, w = mock_intrinsics.h, mock_intrinsics.w
    # Create high-contrast textured pattern image
    rng = np.random.RandomState(42)
    tex = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)

    # Save 3 keyframe image files
    kfs = []
    depths = []
    # Ground truth: flat wall at distance 2.0m
    gt_depth = np.full((h, w), 2.0, dtype=np.float32)

    for i in range(3):
        im_path = tmp_path / f"frame_{i:03d}.jpg"
        Image.fromarray(tex).save(im_path)

        # Slight camera translation along X: [0.0, 0.05, 0.10]
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = i * 0.05

        kf = Keyframe(
            file_path=str(im_path),
            timestamp_ns=int(i * 33_333_333),
            fl_x=mock_intrinsics.fl_x,
            fl_y=mock_intrinsics.fl_y,
            cx=mock_intrinsics.cx,
            cy=mock_intrinsics.cy,
            transform_matrix=c2w,
            image_loader=lambda p=im_path: Image.open(p),
        )
        kfs.append(kf)

        # Distort depth: frame 0 is anchor, frame 1 has scale 1.2, frame 2 has shift 0.3
        if i == 0:
            depths.append(gt_depth.copy())
        elif i == 1:
            depths.append(gt_depth * 1.2)  # Raw mono depth overestimating
        else:
            depths.append(gt_depth + 0.3)

    optimizer = GlobalDepthGraphOptimizer()
    res = optimizer.optimize(kfs, depths, mock_intrinsics)

    assert len(res.scales) == 3
    assert len(res.shifts) == 3
    assert len(res.aligned_depth_maps) == 3
    # Anchor frame 0 should stay near 1.0, 0.0
    assert math.isclose(res.scales[0], 1.0, abs_tol=0.05)
    assert math.isclose(res.shifts[0], 0.0, abs_tol=0.05)
    # Optimized aligned depth should have lower discrepancy than before
    assert res.rmse_after_m <= res.rmse_before_m + 1e-4
    assert res.num_temporal_edges >= 1


# ==============================================================================
# 3. Tangent Frame & Surfel Initialization Tests
# ==============================================================================

def test_build_orthonormal_tangent_frame():
    # Test arbitrary unit normal vectors
    normals = np.array([
        [0.0, 0.0, 1.0],               # Z-axis
        [1.0, 0.0, 0.0],               # X-axis
        [0.0, 1.0, 0.0],               # Y-axis
        [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],  # Diagonal
    ], dtype=np.float32)

    tangent_u, tangent_v = build_orthonormal_tangent_frame(normals)

    # Verify orthogonality: u . n == 0, v . n == 0, u . v == 0
    dot_un = np.sum(tangent_u * normals, axis=-1)
    dot_vn = np.sum(tangent_v * normals, axis=-1)
    dot_uv = np.sum(tangent_u * tangent_v, axis=-1)

    np.testing.assert_allclose(dot_un, 0.0, atol=1e-5)
    np.testing.assert_allclose(dot_vn, 0.0, atol=1e-5)
    np.testing.assert_allclose(dot_uv, 0.0, atol=1e-5)

    # Verify unit vectors
    np.testing.assert_allclose(np.linalg.norm(tangent_u, axis=-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(tangent_v, axis=-1), 1.0, atol=1e-5)


def test_surfel_cloud_initialization_and_ply_export(mock_intrinsics: CameraIntrinsics, tmp_path: Path):
    h, w = mock_intrinsics.h, mock_intrinsics.w

    # Create 2 synthetic keyframes
    img_arr = (np.ones((h, w, 3)) * 180).astype(np.uint8)
    img = Image.fromarray(img_arr)

    kf0 = Keyframe("img0.jpg", 100, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, np.eye(4), lambda: img)

    mat1 = np.eye(4)
    mat1[0, 3] = 0.5  # Moved 50 cm
    kf1 = Keyframe("img1.jpg", 200, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, mat1, lambda: img)

    # Synthetic depth: flat wall at 2m
    depth0 = np.full((h, w), 2.0, dtype=np.float32)
    depth1 = np.full((h, w), 2.0, dtype=np.float32)

    initializer = SurfelCloudInitializer(
        target_surfels=5_000,
        voxel_downsample_m=0.05,
    )
    cloud = initializer.initialize_from_keyframes(
        [kf0, kf1],
        [depth0, depth1],
        mock_intrinsics,
    )

    assert isinstance(cloud, SurfelCloud)
    assert len(cloud) > 100
    assert cloud.positions.shape[1] == 3
    assert cloud.normals.shape[1] == 3
    assert cloud.tangent_u.shape[1] == 3
    assert cloud.tangent_v.shape[1] == 3
    assert cloud.scales_2d.shape[1] == 2
    assert cloud.colors_rgb.shape[1] == 3
    assert cloud.sh_degree_0.shape[1] == 3
    assert cloud.opacities.shape[0] == len(cloud)

    # Test PLY export
    ply_path = tmp_path / "test_surfels.ply"
    cloud.to_ply(ply_path)
    assert ply_path.exists()
    assert ply_path.stat().st_size > 1000

    # Verify PLY header
    with open(ply_path, "rb") as f:
        header = f.read(200).decode("ascii", errors="ignore")
        assert "ply" in header
        assert "format binary_little_endian 1.0" in header
        assert "element vertex" in header

    # Verify PLY loading roundtrip
    loaded_cloud = SurfelCloud.from_ply(ply_path)
    assert len(loaded_cloud) == len(cloud)
    np.testing.assert_allclose(loaded_cloud.positions, cloud.positions, atol=1e-5)
    np.testing.assert_allclose(loaded_cloud.normals, cloud.normals, atol=1e-5)
    np.testing.assert_allclose(loaded_cloud.scales_2d, cloud.scales_2d, atol=1e-5)
    np.testing.assert_allclose(loaded_cloud.colors_rgb, cloud.colors_rgb, atol=1e-2)
    np.testing.assert_allclose(loaded_cloud.opacities, cloud.opacities, atol=1e-5)
