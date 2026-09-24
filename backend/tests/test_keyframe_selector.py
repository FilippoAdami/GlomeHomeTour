"""Unit tests for DynamicKeyframeSelector in backend/00_ingestion/keyframe_selector.py."""

import numpy as np
import pytest

from package_loader import CameraIntrinsics, Keyframe
from keyframe_selector import DynamicKeyframeSelector


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


def test_consecutive_keyframes_always_overlap(mock_intrinsics: CameraIntrinsics):
    """The invariant: no two consecutive anchors may be visually disjoint.

    Regression for selections that contained neighbouring keyframes sharing no
    view at all. The trajectory below walks forward, then pans hard (20 deg per
    frame, ~1/3 of the FOV) -- fast enough that accepting a frame only *after*
    overlap collapsed, or subsampling the result, leaves a hole.
    """
    kfs = []
    for i in range(20):  # walk forward
        kfs.append(make_keyframe(i, [i * 0.10, 0.0, 0.0]))
    for j in range(20):  # pan in place
        angle = np.radians(j * 20.0)
        c, s = np.cos(angle), np.sin(angle)
        r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        kfs.append(make_keyframe(20 + j, [1.9, 0.0, 0.0], r_mat=r))

    selector = DynamicKeyframeSelector(
        min_translation_m=0.08, min_rotation_deg=4.0,
        min_covisibility=0.50, max_covisibility=0.85,
    )
    # Precondition: consecutive *input* frames all overlap, so any gap in the
    # output is the selector's doing and not the trajectory's.
    for i in range(len(kfs) - 1):
        assert selector._mutual_covisibility(i, i + 1, kfs, mock_intrinsics) >= 0.50

    uncapped = selector.select_keyframes(kfs, mock_intrinsics)
    res = selector.select_keyframes(kfs, mock_intrinsics, max_keyframes=12)

    # The cap is best-effort: it tightens the overlap threshold, but a hard pan
    # has a floor below which frames stop touching, and coverage outranks the
    # budget. It must still push the count down, and never below connectivity.
    assert len(res.selected_indices) <= len(uncapped.selected_indices)
    assert res.selected_indices == sorted(set(res.selected_indices))

    for a, b in zip(res.selected_indices[:-1], res.selected_indices[1:]):
        covis = selector._mutual_covisibility(a, b, kfs, mock_intrinsics)
        assert covis >= 0.50, f"frames {a},{b} share only {covis:.2f} overlap"


def test_overlap_depends_on_scene_depth(mock_intrinsics: CameraIntrinsics):
    """Overlap must be judged against how far away the scene actually is.

    Regression for keyframe pairs that were visibly disjoint yet scored 0.85-0.90:
    overlap was measured against a plane hardcoded at 2 m while the real room was
    ~0.8 m away, so sidesteps that swept the view were rated as near-duplicates.
    The same pose pair has to score far lower when the subject is close.
    """
    selector = DynamicKeyframeSelector()

    # One sidestep, no rotation. At 3 m this barely changes the view; at 0.5 m it
    # replaces it. Same poses, so any difference is purely the depth assumption.
    kfs = [make_keyframe(0, [0.0, 0.0, 0.0]), make_keyframe(1, [0.55, 0.0, 0.0])]

    far = selector._mutual_covisibility(0, 1, kfs, mock_intrinsics, np.array([3.0, 3.0]))
    near = selector._mutual_covisibility(0, 1, kfs, mock_intrinsics, np.array([0.5, 0.5]))
    assert far > 0.70, f"a 55 cm step at 3 m should keep most of the view, got {far:.2f}"
    assert near < 0.20, f"a 55 cm step at 0.5 m should lose the view, got {near:.2f}"

    # The nearer frame governs: it is the close geometry that leaves frame first.
    mixed = selector._mutual_covisibility(0, 1, kfs, mock_intrinsics, np.array([3.0, 0.5]))
    assert mixed == pytest.approx(near, abs=1e-6)


def test_keyframe_selection_aggressiveness_levels(mock_intrinsics: CameraIntrinsics):
    """Test aggressiveness parameter behavior at 0.0, 0.5, and 1.0."""
    # 20 keyframes walking in a line
    kfs = [make_keyframe(i, [i * 0.15, 0.0, 0.0]) for i in range(20)]

    selector = DynamicKeyframeSelector.for_2dgs_training()

    # Aggressiveness 0.0 -> skips selection, keeps 100% of frames
    res_zero = selector.select_keyframes(kfs, mock_intrinsics, aggressiveness=0.0)
    assert len(res_zero.selected_keyframes) == len(kfs)
    assert res_zero.selection_ratio == 1.0

    # Aggressiveness 0.5 -> default behavior
    res_half = selector.select_keyframes(kfs, mock_intrinsics, aggressiveness=0.5)

    # Aggressiveness 1.0 -> max aggressiveness, keeps minimum keyframes
    res_one = selector.select_keyframes(kfs, mock_intrinsics, aggressiveness=1.0)

    assert len(res_one.selected_keyframes) <= len(res_half.selected_keyframes)
    assert len(res_half.selected_keyframes) <= len(res_zero.selected_keyframes)

