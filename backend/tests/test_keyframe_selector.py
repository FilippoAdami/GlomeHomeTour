"""Unit tests for DynamicKeyframeSelector in backend/ingestion/keyframe_selector.py."""

import numpy as np
import pytest

from ingestion.package_loader import CameraIntrinsics, Keyframe
from ingestion.keyframe_selector import DynamicKeyframeSelector


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


def make_keyframe(idx: int, t_xyz: list[float], r_mat: np.ndarray = None) -> Keyframe:
    if r_mat is None:
        r_mat = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = r_mat
    c2w[:3, 3] = t_xyz
    return Keyframe(
        file_path=f"images/frame_{idx:05d}.jpg",
        timestamp_ns=idx * 33_333_333,
        fl_x=500.0,
        fl_y=500.0,
        cx=320.0,
        cy=240.0,
        transform_matrix=c2w,
        image_loader=lambda: None,
    )


def test_stationary_frames_discarded(mock_intrinsics: CameraIntrinsics):
    """Verify that small micro-movements (< 35cm) are pruned."""
    selector = DynamicKeyframeSelector(min_translation_m=0.35, min_rotation_deg=18.0)

    # 10 frames moving 2cm each -> total 18cm (less than 35cm)
    kfs = [make_keyframe(i, [i * 0.02, 0.0, 0.0]) for i in range(10)]
    res = selector.select_keyframes(kfs, mock_intrinsics)

    # Only frame 0 should be selected
    assert len(res.selected_indices) == 1
    assert res.selected_indices[0] == 0
    assert len(res.discarded_indices) == 9


def test_spatial_translation_gating(mock_intrinsics: CameraIntrinsics):
    """Verify that co-visibility and translation jointly prune redundant views."""
    selector = DynamicKeyframeSelector(min_translation_m=0.35, max_covisibility=0.78)

    # Frames moving 40cm each (co-visibility ~80% with immediate neighbor)
    kfs = [make_keyframe(i, [i * 0.40, 0.0, 0.0]) for i in range(5)]
    res = selector.select_keyframes(kfs, mock_intrinsics)

    # Co-visibility correctly prunes alternating redundant frames [0, 2, 4]
    assert res.selected_indices == [0, 2, 4]
    assert len(res.discarded_indices) == 2

    # Frames moving 85cm each (co-visibility < 0.60)
    kfs_wide = [make_keyframe(i, [i * 0.85, 0.0, 0.0]) for i in range(5)]
    res_wide = selector.select_keyframes(kfs_wide, mock_intrinsics)
    assert res_wide.selected_indices == [0, 1, 2, 3, 4]


def test_angular_rotation_gating(mock_intrinsics: CameraIntrinsics):
    """Verify that stationary camera with angular rotation >= 18 deg is accepted."""
    selector = DynamicKeyframeSelector(min_translation_m=0.35, min_rotation_deg=18.0)

    kfs = []
    for i in range(5):
        # Rotate around Y axis by i * 25 degrees
        angle = np.radians(i * 25.0)
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        kfs.append(make_keyframe(i, [0.0, 0.0, 0.0], r_mat=r))

    res = selector.select_keyframes(kfs, mock_intrinsics)
    assert len(res.selected_indices) == 5
    assert res.selected_indices == [0, 1, 2, 3, 4]


def test_covisibility_overlap_calculation(mock_intrinsics: CameraIntrinsics):
    """Verify co-visibility overlap calculation between identical and offset cameras."""
    selector = DynamicKeyframeSelector()
    rays = selector._generate_canonical_frustum_rays(mock_intrinsics, grid_size=6)

    r_ident = np.eye(3, dtype=np.float32)
    t_0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # Identical poses -> 1.0 overlap
    covis_same = selector._compute_frustum_covisibility(
        r_ident, t_0, r_ident, t_0, mock_intrinsics, rays
    )
    assert pytest.approx(covis_same, abs=1e-2) == 1.0

    # Large translation (10 meters away) -> 0.0 overlap
    t_far = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    covis_far = selector._compute_frustum_covisibility(
        r_ident, t_0, r_ident, t_far, mock_intrinsics, rays
    )
    assert covis_far < 0.1
