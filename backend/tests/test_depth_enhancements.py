"""Unit tests for Volumetric TSDF Fusion & Multi-Scale Surfel Extraction (tsdf_fusion.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from package_loader import CameraIntrinsics
from tsdf_fusion import TSDFVolume


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


class TestTSDFFusion:
    def test_tsdf_volume_allocation(self):
        """Verify TSDF grid dimensions and tensor initialization."""
        bbox_min = np.array([-2.0, -2.0, -1.0], dtype=np.float32)
        bbox_max = np.array([2.0, 2.0, 1.0], dtype=np.float32)
        volume = TSDFVolume(bbox_min=bbox_min, bbox_max=bbox_max, voxel_size=0.1, device="cpu")

        assert volume.nx > 0 and volume.ny > 0 and volume.nz > 0
        assert volume.tsdf.shape == (volume.nx, volume.ny, volume.nz)
        assert volume.weight.shape == (volume.nx, volume.ny, volume.nz)
        assert torch.all(volume.tsdf == 1.0)  # Initialized to free space
        assert torch.all(volume.weight == 0.0)

    def test_tsdf_plane_integration_and_multiscale_extraction(self, mock_intrinsics):
        """Integrate a synthetic planar wall at 2.0m and extract single-manifold surfels."""
        # In OpenGL coordinates, camera looks towards -Z
        bbox_min = np.array([-1.5, -1.5, -3.5], dtype=np.float32)
        bbox_max = np.array([1.5, 1.5, -0.5], dtype=np.float32)
        volume = TSDFVolume(bbox_min=bbox_min, bbox_max=bbox_max, voxel_size=0.05, device="cpu")

        # Flat wall at depth = 2.0m
        h, w = mock_intrinsics.h, mock_intrinsics.w
        depth_map = np.full((h, w), 2.0, dtype=np.float32)
        rgb_img = np.full((h, w, 3), 180, dtype=np.uint8)

        # Identity camera looking towards -Z in OpenGL (world Z = -cam Z)
        c2w = np.eye(4, dtype=np.float32)

        # Integrate multiple views (simulating multi-view coverage)
        for _ in range(3):
            volume.integrate_frame(
                depth_map=depth_map,
                rgb_img=rgb_img,
                c2w=c2w,
                intrinsics=mock_intrinsics,
                weight_multiplier=1.0,
            )

        # Weights should be accumulated in front of the wall
        assert torch.any(volume.weight > 1.5)

        # Extract surfels
        cloud = volume.extract_multiscale_surfels(min_weight=1.5, max_surfels=10_000)
        assert len(cloud) > 0
        assert cloud.positions.shape[1] == 3
        assert cloud.normals.shape[1] == 3
        assert cloud.tangent_u.shape[1] == 3
        assert cloud.tangent_v.shape[1] == 3
        assert cloud.scales_2d.shape[1] == 2
        assert cloud.colors_rgb.shape[1] == 3

        # Zero crossing positions should sit tightly around Z = -2.0 in OpenGL world
        mean_z = np.mean(cloud.positions[:, 2])
        assert abs(mean_z - (-2.0)) < 0.1
