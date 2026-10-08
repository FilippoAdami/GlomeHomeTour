"""Tests for GPU Multi-View Epipolar Depth Alignment and Floater Removal.

Tests:
1. Two-view epipolar ray depth consensus alignment.
2. Cross-view free-space violation carving.
3. Uncorroborated floater pruning in overlapping frustums.
4. clean_all_depth_maps end-to-end integration with alignment stats.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from package_loader import CameraIntrinsics
from clean_depth_maps import (
    clean_all_depth_maps,
    gpu_multiview_depth_refine_and_carve,
    gpu_free_space_carve,
)


@pytest.fixture
def test_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=100.0,
        fl_y=100.0,
        cx=25.0,
        cy=25.0,
        w=50,
        h=50,
        camera_angle_x=0.5,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
    )


def test_gpu_multiview_depth_alignment(test_intrinsics: CameraIntrinsics):
    """Test that two slightly disagreeing depth maps converge to a consensus surface."""
    h, w = test_intrinsics.h, test_intrinsics.w
    
    # Camera 1 at origin, Camera 2 shifted slightly along X
    c2w1 = np.eye(4, dtype=np.float32)
    c2w2 = np.eye(4, dtype=np.float32)
    c2w2[0, 3] = 0.1  # 10 cm baseline

    # Frame 1 predicts 2.05m, Frame 2 predicts 1.95m for the same surface
    d1 = np.full((h, w), 2.05, dtype=np.float32)
    d2 = np.full((h, w), 1.95, dtype=np.float32)

    cleaned_maps, stats = gpu_multiview_depth_refine_and_carve(
        depth_maps=[d1, d2],
        c2w_mats=np.stack([c2w1, c2w2]),
        intrinsics=test_intrinsics,
        enable_depth_alignment=True,
        alignment_margin_m=0.15,
        device="cpu",
    )

    assert len(cleaned_maps) == 2
    assert stats["aligned_pixels"] > 0
    assert stats["mean_adjustment_m"] > 0.0

    # Center pixels should have moved closer to 2.0m (consensus)
    center_d1 = cleaned_maps[0][25, 25]
    center_d2 = cleaned_maps[1][25, 25]
    assert center_d1 < 2.05  # d1 adjusted down towards consensus
    assert center_d2 > 1.95  # d2 adjusted up towards consensus


def test_gpu_free_space_carving_violation(test_intrinsics: CameraIntrinsics):
    """Test that a phantom point in front of another camera's surface is carved to 0."""
    h, w = test_intrinsics.h, test_intrinsics.w
    
    c2w1 = np.eye(4, dtype=np.float32)
    c2w2 = np.eye(4, dtype=np.float32)
    c2w2[0, 3] = 0.1

    # Camera 2 sees a wall at 2.0m
    d2 = np.full((h, w), 2.0, dtype=np.float32)

    # Camera 1 has a floating phantom at 0.8m in the center
    d1 = np.full((h, w), 2.0, dtype=np.float32)
    d1[20:30, 20:30] = 0.8  # Floater in front of wall

    cleaned_maps, stats = gpu_multiview_depth_refine_and_carve(
        depth_maps=[d1, d2],
        c2w_mats=np.stack([c2w1, c2w2]),
        intrinsics=test_intrinsics,
        freespace_margin_m=0.08,
        enable_depth_alignment=False,
        device="cpu",
    )

    assert stats["freespace"] > 0
    # The center floater should be carved to 0.0
    assert cleaned_maps[0][25, 25] == 0.0
    # The valid wall should remain
    assert cleaned_maps[0][5, 5] == 2.0


def test_uncorroborated_floater_pruning(test_intrinsics: CameraIntrinsics):
    """Test that uncorroborated floaters in overlapping views are pruned when min_consensus_views > 0."""
    h, w = test_intrinsics.h, test_intrinsics.w
    
    c2w1 = np.eye(4, dtype=np.float32)
    c2w2 = np.eye(4, dtype=np.float32)
    c2w2[0, 3] = 0.05

    # Frame 2 has depth 2.0m
    d2 = np.full((h, w), 2.0, dtype=np.float32)

    # Frame 1 has a floater at 4.5m in center (behind wall, not free-space violation, but uncorroborated)
    d1 = np.full((h, w), 2.0, dtype=np.float32)
    d1[20:30, 20:30] = 4.5

    cleaned_maps, stats = gpu_multiview_depth_refine_and_carve(
        depth_maps=[d1, d2],
        c2w_mats=np.stack([c2w1, c2w2]),
        intrinsics=test_intrinsics,
        enable_depth_alignment=False,
        min_consensus_views=1,
        min_frustum_views=1,
        device="cpu",
    )

    assert stats["uncorroborated"] > 0
    assert cleaned_maps[0][25, 25] == 0.0


def test_clean_all_depth_maps_with_alignment(test_intrinsics: CameraIntrinsics):
    """Test end-to-end clean_all_depth_maps with alignment and carving enabled."""
    h, w = test_intrinsics.h, test_intrinsics.w
    d1 = np.full((h, w), 2.02, dtype=np.float32)
    d2 = np.full((h, w), 1.98, dtype=np.float32)
    c2w1 = np.eye(4, dtype=np.float32)
    c2w2 = np.eye(4, dtype=np.float32)
    c2w2[0, 3] = 0.08

    cleaned_maps, stats = clean_all_depth_maps(
        depth_maps=[d1, d2],
        images=None,
        c2w_mats=np.stack([c2w1, c2w2]),
        intrinsics=test_intrinsics,
        enable_depth_alignment=True,
        device="cpu",
    )

    assert len(cleaned_maps) == 2
    assert "aligned_pixels" in stats
    assert "freespace" in stats
    assert "uncorroborated" in stats
