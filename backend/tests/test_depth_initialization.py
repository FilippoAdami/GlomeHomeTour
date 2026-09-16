"""Unit tests for backend/02_depth_estimation/ depth priors and surfel initialization."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from package_loader import CameraIntrinsics, Keyframe
from depth_priors import (
    DepthPriorEstimator,
    GlobalDepthGraphOptimizer,
    GlobalDepthGraphResult,
    MetricDepthAligner,
    compute_surface_normals,
)
from initialization import (
    SurfelCloud,
    SurfelCloudInitializer,
    build_orthonormal_tangent_frame,
    compute_overexposed_mask,
    filter_multiview_consistency,
    global_cross_view_freespace_carving,
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


def test_surfel_cloud_from_random():
    """SurfelCloud.from_random produces a valid point cloud confined to bbox."""
    bbox_min = np.array([-2.0, -1.0, -3.0], dtype=np.float32)
    bbox_max = np.array([2.0, 3.0, 1.0], dtype=np.float32)
    cloud = SurfelCloud.from_random(bbox_min, bbox_max, num_points=5000, seed=0)

    assert len(cloud) == 5000
    assert np.all(cloud.positions >= bbox_min) and np.all(cloud.positions <= bbox_max)
    normal_lengths = np.linalg.norm(cloud.normals, axis=-1)
    np.testing.assert_allclose(normal_lengths, 1.0, atol=1e-4)
    assert cloud.opacities.shape == (5000,)
    assert np.all((cloud.opacities > 0.0) & (cloud.opacities <= 1.0))


# ==============================================================================
# 4. Saturated Pixel Masking & Epipolar Free-Space Filter Tests
# ==============================================================================

def test_compute_overexposed_mask():
    h, w = 60, 80
    # Create image with diffuse white wall (238, 238, 238)
    img = np.full((h, w, 3), 238, dtype=np.uint8)

    # Add saturated light fixture core (255, 255, 252)
    img[20:25, 20:25] = [255, 255, 252]

    # Add vibrant saturated red object (255, 20, 20) - should NOT be treated as optical bloom
    img[40:45, 40:45] = [255, 20, 20]

    mask_no_dilate = compute_overexposed_mask(img, min_threshold=250, dilation_kernel_size=0)
    # The light fixture core must be masked
    assert np.all(mask_no_dilate[20:25, 20:25])
    # Diffuse white wall must NOT be masked
    assert not np.any(mask_no_dilate[0:15, 0:15])
    # Vibrant red object must NOT be masked (chroma difference is high)
    assert not np.any(mask_no_dilate[40:45, 40:45])

    # Test dilation
    mask_dilated = compute_overexposed_mask(img, min_threshold=250, dilation_kernel_size=5)
    assert np.sum(mask_dilated) > np.sum(mask_no_dilate)
    # Surrounding halo of the bulb is masked by dilation
    assert mask_dilated[19, 20]

    # Test dim image (no overexposure)
    dim_img = np.full((h, w, 3), 150, dtype=np.uint8)
    dim_mask = compute_overexposed_mask(dim_img, min_threshold=250)
    assert not np.any(dim_mask)


def test_filter_multiview_consistency_freespace(mock_intrinsics: CameraIntrinsics):
    h, w = mock_intrinsics.h, mock_intrinsics.w

    # Camera 0 at origin (0, 0, 0), OpenGL convention (facing -Z)
    c2w_0 = np.eye(4, dtype=np.float64)
    # Camera 1 translated along X by 0.35m (side view with clear baseline)
    c2w_1 = np.eye(4, dtype=np.float64)
    c2w_1[0, 3] = 0.35

    kf0 = Keyframe("c0.jpg", 0, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, c2w_0, lambda: None)
    kf1 = Keyframe("c1.jpg", 1, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, c2w_1, lambda: None)

    # Depth maps: flat wall at distance 4.0 meters
    depth0 = np.full((h, w), 4.0, dtype=np.float32)
    depth1 = np.full((h, w), 4.0, dtype=np.float32)

    # 3 points defined in world space:
    # Point 0: True surface point on the wall at (0.0, 0.0, -4.0)
    # Point 1: Floating phantom point in empty space at (0.0, 0.0, -1.5)
    # Point 2: Point behind the wall at (0.0, 0.0, -6.0) (occluded from camera 1)
    pts_world = np.array([
        [0.0, 0.0, -4.0],
        [0.0, 0.0, -1.5],
        [0.0, 0.0, -6.0],
    ], dtype=np.float32)

    # Run filter with free-space carving enabled and min_consensus=1
    mask = filter_multiview_consistency(
        pts_world=pts_world,
        current_idx=0,
        keyframes=[kf0, kf1],
        depth_maps=[depth0, depth1],
        intrinsics=mock_intrinsics,
        min_consensus=1,
        enable_freespace_filter=True,
        max_freespace_violations=0,
    )

    # Point 0 is on the surface (agrees in both views) -> kept
    assert mask[0] == True
    # Point 1 is a floating point in empty air (camera 1 sees 4.0m wall through it) -> CULLED!
    assert mask[1] == False

    # Point 2 is behind the wall; with min_consensus=0 (only free-space check active):
    mask_freespace_only = filter_multiview_consistency(
        pts_world=pts_world,
        current_idx=0,
        keyframes=[kf0, kf1],
        depth_maps=[depth0, depth1],
        intrinsics=mock_intrinsics,
        min_consensus=0,
        enable_freespace_filter=True,
        max_freespace_violations=0,
    )
    # Point 0: valid (not in empty space)
    assert mask_freespace_only[0] == True
    # Point 1: culled (empty space violation)
    assert mask_freespace_only[1] == False
    # Point 2: occluded behind wall (proj_z = 6.0 > obs_depth = 4.0), NOT an empty space violation
    assert mask_freespace_only[2] == True


def test_surfel_initialization_with_filters(mock_intrinsics: CameraIntrinsics):
    h, w = mock_intrinsics.h, mock_intrinsics.w

    # Image with saturated patch
    img_arr = np.full((h, w, 3), 180, dtype=np.uint8)
    img_arr[100:150, 100:150] = [255, 255, 255]  # Blown-out optical bloom
    img = Image.fromarray(img_arr)

    c2w_0 = np.eye(4, dtype=np.float64)
    c2w_1 = np.eye(4, dtype=np.float64)
    c2w_1[0, 3] = 0.3

    kf0 = Keyframe("f0.jpg", 0, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, c2w_0, lambda: img)
    kf1 = Keyframe("f1.jpg", 1, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, c2w_1, lambda: img)

    depth0 = np.full((h, w), 2.5, dtype=np.float32)
    depth1 = np.full((h, w), 2.5, dtype=np.float32)

    init_filtered = SurfelCloudInitializer(
        target_surfels=2000,
        voxel_downsample_m=0.05,
        enable_saturation_mask=True,
        saturation_min_threshold=250,
        enable_freespace_filter=True,
    )
    cloud_filtered = init_filtered.initialize_from_keyframes(
        [kf0, kf1],
        [depth0, depth1],
        mock_intrinsics,
    )

    init_unfiltered = SurfelCloudInitializer(
        target_surfels=2000,
        voxel_downsample_m=0.05,
        enable_saturation_mask=False,
        enable_freespace_filter=False,
    )
    cloud_unfiltered = init_unfiltered.initialize_from_keyframes(
        [kf0, kf1],
        [depth0, depth1],
        mock_intrinsics,
    )

    # Unfiltered cloud contains saturated white points (1.0)
    assert np.max(cloud_unfiltered.colors_rgb) > 0.95
    # Filtered cloud pruned saturated bloom, so maximum color is bounded by diffuse wall intensity
    assert np.max(cloud_filtered.colors_rgb) <= (185.0 / 255.0)

    # For a single keyframe containing a bloom patch, point count is strictly reduced
    single_filt = init_filtered.initialize_from_keyframes([kf0], [depth0], mock_intrinsics)
    single_unfilt = init_unfiltered.initialize_from_keyframes([kf0], [depth0], mock_intrinsics)
    assert len(single_filt) < len(single_unfilt)



def test_voxel_grid_coarsens_instead_of_random_thinning(mock_intrinsics):
    """Over budget, the initializer re-voxelises at a coarser grid rather than dropping points."""
    h, w = mock_intrinsics.h, mock_intrinsics.w
    img = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8))
    mat = np.eye(4, dtype=np.float32)
    kf = Keyframe("img0.jpg", 100, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                  mock_intrinsics.cx, mock_intrinsics.cy, mat, lambda: img)
    depth = np.full((h, w), 2.0, dtype=np.float32)

    initializer = SurfelCloudInitializer(
        max_surfels=1_000,
        voxel_downsample_m=0.005,
    )
    cloud = initializer.initialize_from_keyframes([kf], [depth], mock_intrinsics)

    assert len(cloud) <= 1_000
    assert initializer.applied_voxel_size_m > 0.005
    # Surfels must grow with the coarsened cell, else the surface develops holes.
    assert cloud.scales_2d.max() <= 0.8 * initializer.applied_voxel_size_m + 1e-6
    assert cloud.scales_2d.max() > 0.016


def test_global_cross_view_freespace_carving(mock_intrinsics: CameraIntrinsics):
    """Test global cross-view carving removes floating phantom points."""
    h, w = mock_intrinsics.h, mock_intrinsics.w

    c2w_0 = np.eye(4, dtype=np.float64)
    c2w_1 = np.eye(4, dtype=np.float64)
    c2w_1[0, 3] = 0.5  # 50cm lateral baseline

    kf0 = Keyframe("f0.jpg", 0, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, c2w_0, lambda: None)
    kf1 = Keyframe("f1.jpg", 1, mock_intrinsics.fl_x, mock_intrinsics.fl_y,
                   mock_intrinsics.cx, mock_intrinsics.cy, c2w_1, lambda: None)

    # Both views observe a wall at 3.0m
    depth0 = np.full((h, w), 3.0, dtype=np.float32)
    depth1 = np.full((h, w), 3.0, dtype=np.float32)

    # Point 0: True surface point at z = -3.0
    # Point 1: Phantom floating point at z = -1.5 (violates 3.0m depth in both views)
    pts = np.array([
        [0.0, 0.0, -3.0],
        [0.0, 0.0, -1.5],
    ], dtype=np.float32)
    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    colors = np.array([
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)

    carved_pts, carved_norms, carved_cols = global_cross_view_freespace_carving(
        pts_world=pts,
        normals_world=normals,
        colors_rgb=colors,
        keyframes=[kf0, kf1],
        depth_maps=[depth0, depth1],
        intrinsics=mock_intrinsics,
        max_violations=1,
        margin_m=0.04,
        subsample_kfs=1,
    )

    assert len(carved_pts) == 1
    assert np.allclose(carved_pts[0], [0.0, 0.0, -3.0])

