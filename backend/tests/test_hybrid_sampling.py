import numpy as np
import pytest
from pathlib import Path

from initialization import compute_hybrid_sampling_coords, SurfelCloudInitializer, SurfelCloud
from package_loader import CameraIntrinsics, Keyframe


def test_compute_hybrid_sampling_coords_textured_vs_flat():
    """Verify hybrid sampling captures edges at fine stride and flat areas at coarse stride."""
    h, w = 120, 160
    # Create an image that is flat gray on left half and high-contrast checkerboard on right half
    img_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    # Right half: checkerboard pattern
    for y in range(0, h, 8):
        for x in range(w // 2, w, 8):
            img_rgb[y:y+4, x:x+4] = 255
            img_rgb[y+4:y+8, x+4:x+8] = 0

    # Depth map: smooth plane on left, step edge on right
    depth_map = np.full((h, w), 2.0, dtype=np.float32)
    depth_map[:, w // 2:] = 3.5

    y_coords, x_coords, scales_rel = compute_hybrid_sampling_coords(
        img_rgb=img_rgb,
        depth_map=depth_map,
        energy_threshold=0.08,
        coarse_stride=4,
        fine_stride=1,
    )

    assert len(y_coords) == len(x_coords) == len(scales_rel)
    assert len(y_coords) > 0

    # Right half (textured + depth step) should have much higher density of points than left half
    right_mask = x_coords >= (w // 2)
    left_interior_mask = x_coords < (w // 2 - 4)

    count_right = np.sum(right_mask)
    count_left = np.sum(left_interior_mask)

    # Right half should be at least 3x denser than flat left half
    assert count_right > count_left * 3

    # On the flat left interior, relative scales should reflect coarse stride 4.0
    scales_left = scales_rel[left_interior_mask]
    assert np.all(scales_left == 4.0)

    # On the textured right half, fine points should have scale 1.0
    scales_right = scales_rel[right_mask]
    assert np.any(scales_right == 1.0)


def test_surfel_initializer_with_hybrid_sampling():
    """Verify SurfelCloudInitializer end-to-end unprojection with hybrid sampling enabled."""
    h, w = 60, 80
    intrinsics = CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=100.0,
        fl_y=100.0,
        cx=40.0,
        cy=30.0,
        w=w,
        h=h,
        camera_angle_x=1.0,
        k1=0.0,
        k2=0.0,
        p1=0.0,
        p2=0.0,
    )

    # Create synthetic keyframe
    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 0.0

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:50, 20:60] = 255  # bright square

    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[10:50, 20:60] = 1.5

    class MockKeyframe:
        def __init__(self, c2w):
            self.transform_matrix = c2w
            self.file_path = "mock.jpg"
        def load_image_rgb(self):
            return img

    kfs = [MockKeyframe(c2w)]
    depths = [depth]

    initializer = SurfelCloudInitializer(
        target_surfels=5000,
        enable_hybrid_sampling=True,
        hybrid_energy_threshold=0.08,
        hybrid_coarse_stride=3,
        hybrid_fine_stride=1,
        enable_tube_collapse=False,
    )

    cloud = initializer.initialize_from_keyframes(
        keyframes=kfs,
        depth_maps=depths,
        intrinsics=intrinsics,
        min_conf=0.01,
    )

    assert isinstance(cloud, SurfelCloud)
    assert len(cloud) > 0
    assert np.all(np.isfinite(cloud.positions))
    assert np.all(np.isfinite(cloud.normals))
    assert np.all(np.isfinite(cloud.scales_2d))
    assert cloud.scales_2d.shape[1] == 2
    assert np.all(cloud.scales_2d > 0.0)
