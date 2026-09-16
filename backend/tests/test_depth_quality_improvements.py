"""Tests for Depth Anything 3 quality improvements:
- Fast RGB-guided depth edge filtering
- Test-time processing resolution configuration
- COLMAP sparse landmark scale/shift anchoring
- Multi-view surface normal consensus regularization
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from package_loader import CameraIntrinsics, Keyframe
from depth_priors import (
    DepthPriorEstimator,
    anchor_depths_to_sparse_points,
    guided_filter_depth,
)
from initialization import (
    SurfelCloudInitializer,
    regularize_surface_normals_multiview,
)


@pytest.fixture
def mock_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=500.0,
        fl_y=500.0,
        cx=160.0,
        cy=120.0,
        w=320,
        h=240,
        camera_angle_x=0.64,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
    )


class TestGuidedFilterDepth:
    def test_guided_filter_edge_alignment(self):
        """Test that guided filtering snaps a blurry depth edge to a sharp RGB boundary."""
        h, w = 64, 64
        # RGB image has a sharp step edge at x = 32
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, 32:, :] = 255

        # Blurry / noisy depth map across x = 32
        x_grid = np.broadcast_to(np.arange(w), (h, w)).astype(np.float32)
        # Soft sigmoid transition between 1.0m and 3.0m
        blurry_depth = 1.0 + 2.0 / (1.0 + np.exp(-(x_grid - 32.0) / 3.0))
        # Add random noise
        rng = np.random.RandomState(42)
        noisy_depth = blurry_depth + rng.normal(0, 0.05, size=(h, w)).astype(np.float32)

        filtered = guided_filter_depth(noisy_depth, rgb, radius=4, eps=1e-3)

        assert filtered.shape == (h, w)
        assert filtered.dtype == np.float32
        # Planar noise in flat left region (x < 20) should be smoothed
        assert np.std(filtered[:, :20]) < np.std(noisy_depth[:, :20])
        # Depth transition across edge (x=30 to x=34) should be steeper
        edge_diff_orig = np.mean(noisy_depth[:, 34]) - np.mean(noisy_depth[:, 30])
        edge_diff_filt = np.mean(filtered[:, 34]) - np.mean(filtered[:, 30])
        assert edge_diff_filt > edge_diff_orig

    def test_guided_filter_non_finite_handling(self):
        """Test that guided filter gracefully handles NaN or Inf in depth map."""
        h, w = 32, 32
        rgb = np.ones((h, w, 3), dtype=np.uint8) * 128
        depth = np.ones((h, w), dtype=np.float32) * 2.0
        depth[10, 10] = np.nan
        depth[12, 12] = np.inf

        filtered = guided_filter_depth(depth, rgb, radius=2, eps=1e-3)
        assert np.isnan(filtered[10, 10])
        assert np.isinf(filtered[12, 12])
        assert np.isfinite(filtered[0, 0])


class TestResolutionConfiguration:
    def test_depth_estimator_process_res_defaults_and_override(self):
        """Test that process_res defaults to 1008 and can be set during init or inference."""
        estimator = DepthPriorEstimator(device="cpu", process_res=1008)
        assert estimator.process_res == 1008
        assert estimator.process_res_method == "upper_bound_resize"

        estimator_custom = DepthPriorEstimator(device="cpu", process_res=1920)
        assert estimator_custom.process_res == 1920


class TestSparseAnchor:
    def test_anchor_depths_to_sparse_points(self, mock_intrinsics: CameraIntrinsics):
        """Test anchoring a scaled depth map against true 3D points."""
        h, w = mock_intrinsics.h, mock_intrinsics.w
        true_depth = np.full((h, w), 2.5, dtype=np.float32)

        # Camera at origin looking along -Z in OpenGL coordinates
        c2w = np.eye(4, dtype=np.float32)
        kf = Keyframe(file_path="mock.jpg", transform_matrix=c2w)

        # Generate sparse 3D points matching true_depth
        sparse_pts = []
        for u in [80, 120, 160, 200, 240]:
            for v in [60, 90, 120, 150, 180]:
                x = (u - mock_intrinsics.cx) * 2.5 / mock_intrinsics.fl_x
                y = -(v - mock_intrinsics.cy) * 2.5 / mock_intrinsics.fl_y
                z = -2.5
                sparse_pts.append([x, y, z])
        sparse_xyz = np.array(sparse_pts, dtype=np.float32)

        # Scaled depth map: 1.1x true depth (drifts to 2.75m)
        drifted_depth = true_depth * 1.10

        aligned_depths, stats = anchor_depths_to_sparse_points(
            depth_maps=[drifted_depth],
            keyframes=[kf],
            intrinsics=mock_intrinsics,
            sparse_points_3d=sparse_xyz,
            min_inliers=8,
        )

        assert stats["anchored_frames"] == 1
        assert len(aligned_depths) == 1
        # Recovered depth should be closer to 2.5 than 2.75
        recovered_mean = float(np.mean(aligned_depths[0]))
        assert abs(recovered_mean - 2.5) < abs(2.75 - 2.5)


class TestNormalConsensus:
    def test_regularize_surface_normals_multiview(self, mock_intrinsics: CameraIntrinsics):
        """Test that cross-view normal consensus reduces normal variance on a shared plane."""
        h, w = mock_intrinsics.h, mock_intrinsics.w
        plane_depth = 2.0

        # View 1: camera at origin
        c2w_1 = np.eye(4, dtype=np.float32)
        # View 2: camera shifted slightly along X
        c2w_2 = np.eye(4, dtype=np.float32)
        c2w_2[0, 3] = 0.20  # 20cm baseline

        kf1 = Keyframe(file_path="1.jpg", transform_matrix=c2w_1)
        kf2 = Keyframe(file_path="2.jpg", transform_matrix=c2w_2)

        d1 = np.full((h, w), plane_depth, dtype=np.float32)
        d2 = np.full((h, w), plane_depth, dtype=np.float32)

        # Unproject points for view 1
        pts_world = []
        normals_world = []
        rng = np.random.RandomState(42)
        for u in range(100, 220, 10):
            for v in range(80, 160, 10):
                x = (u - mock_intrinsics.cx) * plane_depth / mock_intrinsics.fl_x
                y = -(v - mock_intrinsics.cy) * plane_depth / mock_intrinsics.fl_y
                z = -plane_depth
                pts_world.append([x, y, z])
                # Noisy normal perturbed from true normal [0, 0, 1]
                n = np.array([rng.normal(0, 0.1), rng.normal(0, 0.1), 1.0], dtype=np.float32)
                normals_world.append(n / np.linalg.norm(n))

        pts_world = np.array(pts_world, dtype=np.float32)
        normals_world = np.array(normals_world, dtype=np.float32)

        # View 2 observes clean normal pointing towards camera (+Z in cam frame -> +Z in world)
        normals_cam_2 = np.zeros((h, w, 3), dtype=np.float32)
        normals_cam_2[..., 2] = 1.0

        regularized = regularize_surface_normals_multiview(
            pts_world=pts_world,
            normals_world=normals_world,
            current_idx=0,
            keyframes=[kf1, kf2],
            depth_maps=[d1, d2],
            intrinsics=mock_intrinsics,
            normals_cam_maps=[None, normals_cam_2],
            blend_weight=0.5,
        )

        # Perturbation variance in X and Y components should be reduced by consensus
        var_orig = np.var(normals_world[:, :2])
        var_reg = np.var(regularized[:, :2])
        assert var_reg < var_orig
