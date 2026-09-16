"""Unit tests for backend/00_ingestion/ subsystem."""

import io
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from package_loader import (
    CapturePackage,
    Keyframe,
    PackageLoader,
    PackageValidationError,
    TrajectorySample,
    find_schemas_dir,
)
from pose_aligner import (
    PoseAligner,
    matrix_to_quaternion,
    opencv_to_opengl,
    opengl_to_opencv,
    quaternion_slerp,
    quaternion_to_matrix,
)
from quality_gate import QualityGate, prune_redundant


@pytest.fixture
def schemas_dir() -> Path:
    return find_schemas_dir()


@pytest.fixture
def fixtures_dir(schemas_dir: Path) -> Path:
    return schemas_dir / "fixtures"


@pytest.fixture
def mock_package_dir(fixtures_dir: Path, tmp_path: Path) -> Path:
    """Create a temporary uncompressed capture package directory using shared fixtures."""
    pkg_dir = tmp_path / "scan_package"
    pkg_dir.mkdir()

    # Copy schema fixture files
    shutil.copy(fixtures_dir / "transforms.example.json", pkg_dir / "transforms.json")
    shutil.copy(fixtures_dir / "coverage_summary.example.json", pkg_dir / "coverage_summary.json")
    shutil.copy(fixtures_dir / "trajectory.example.csv", pkg_dir / "trajectory.csv")

    # Create dummy images referenced by transforms.example.json:
    # images/frame_00000.jpg and images/frame_00001.jpg
    images_dir = pkg_dir / "images"
    images_dir.mkdir()

    # Generate synthetic textured test image (640x480)
    arr = (np.random.RandomState(42).rand(480, 640, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(images_dir / "frame_00000.jpg")
    img.save(images_dir / "frame_00001.jpg")

    return pkg_dir


@pytest.fixture
def mock_package_zip(mock_package_dir: Path, tmp_path: Path) -> Path:
    """Create a temporary ZIP capture package."""
    zip_path = tmp_path / "scan_package.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for p in mock_package_dir.rglob("*"):
            if p.is_file():
                arcname = p.relative_to(mock_package_dir)
                z.write(p, arcname)
    return zip_path


# ==============================================================================
# 1. Package Loader & Schema Validation Tests
# ==============================================================================

def test_package_loader_directory(mock_package_dir: Path):
    loader = PackageLoader(min_keyframes=2)
    pkg = loader.load(mock_package_dir)

    assert isinstance(pkg, CapturePackage)
    assert pkg.intrinsics.camera_model == "OPENCV"
    assert pkg.intrinsics.w == 640
    assert pkg.intrinsics.h == 480
    assert len(pkg.keyframes) == 2
    assert len(pkg.trajectory) == 3

    # Test keyframe access and image loading
    kf0 = pkg.keyframes[0]
    assert kf0.file_path == "images/frame_00000.jpg"
    assert kf0.timestamp_ns == 111
    assert kf0.transform_matrix.shape == (4, 4)

    img_rgb = kf0.load_image_rgb()
    assert img_rgb.shape == (480, 640, 3)
    assert img_rgb.dtype == np.uint8


def test_package_loader_zip(mock_package_zip: Path):
    loader = PackageLoader(min_keyframes=2)
    pkg = loader.load(mock_package_zip)

    assert isinstance(pkg, CapturePackage)
    assert len(pkg.keyframes) == 2
    assert len(pkg.trajectory) == 3

    img_rgb = pkg.keyframes[0].load_image_rgb()
    assert img_rgb.shape == (480, 640, 3)


def test_package_loader_enforces_min_keyframes(mock_package_dir: Path):
    # Expect error when min_keyframes is default 30 but package only has 2
    loader = PackageLoader(min_keyframes=30)
    with pytest.raises(PackageValidationError, match="contains only 2 keyframes"):
        loader.load(mock_package_dir)


def test_package_loader_rejects_corrupted_transforms(mock_package_dir: Path):
    loader = PackageLoader(min_keyframes=1)
    transforms_file = mock_package_dir / "transforms.json"

    # Corrupt transforms by removing required field camera_model
    data = json.loads(transforms_file.read_text())
    del data["camera_model"]
    transforms_file.write_text(json.dumps(data))

    with pytest.raises(PackageValidationError, match="Validation failed for transforms.json"):
        loader.load(mock_package_dir)


def test_package_loader_rejects_corrupted_trajectory_header(mock_package_dir: Path):
    loader = PackageLoader(min_keyframes=1)
    traj_file = mock_package_dir / "trajectory.csv"

    # Corrupt trajectory header
    traj_file.write_text("invalid,header\n1,2\n")

    with pytest.raises(PackageValidationError, match="header .* does not match required contract"):
        loader.load(mock_package_dir)


# ==============================================================================
# 2. Pose Aligner & Timestamp Synchronization Tests
# ==============================================================================

def test_quaternion_slerp_analytical():
    # q0: Identity rotation (0 degrees)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    # q1: 90 degree rotation around Z axis
    # q = [0, 0, sin(45 deg), cos(45 deg)] = [0, 0, sqrt(2)/2, sqrt(2)/2]
    s = math.sin(math.radians(45.0))
    c = math.cos(math.radians(45.0))
    q1 = np.array([0.0, 0.0, s, c])

    # Interpolate exactly midway (u = 0.5) -> should be 45 degree rotation around Z
    q_mid = quaternion_slerp(q0, q1, 0.5)

    # Expected: [0, 0, sin(22.5 deg), cos(22.5 deg)]
    exp_s = math.sin(math.radians(22.5))
    exp_c = math.cos(math.radians(22.5))
    expected = np.array([0.0, 0.0, exp_s, exp_c])

    np.testing.assert_allclose(q_mid, expected, atol=1e-6)


def test_quaternion_slerp_antipodal():
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    # Antipodal equivalent to q0: -q0
    q1 = -q0

    # Shortest path should keep the rotation identity
    q_mid = quaternion_slerp(q0, q1, 0.5)
    np.testing.assert_allclose(np.abs(q_mid[3]), 1.0, atol=1e-6)


def test_pose_aligner_interpolation():
    # Trajectory with 3 points at t = 100, 200, 300
    traj = [
        TrajectorySample(100, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, "TRACKING", 1),
        TrajectorySample(200, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, "TRACKING", 1),
        TrajectorySample(300, 2.0, 4.0, 6.0, 0.0, 0.0, 0.0, 1.0, "TRACKING", 1),
    ]
    aligner = PoseAligner(traj)

    # Query at midpoint t = 150
    t, q, mat = aligner.interpolate_pose(150)
    np.testing.assert_allclose(t, [0.5, 1.0, 1.5], atol=1e-5)
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-5)
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat[:3, 3], [0.5, 1.0, 1.5], atol=1e-5)


def test_coordinate_conversions_roundtrip():
    r = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    t = np.array([1.0, 2.0, 3.0])
    mat_gl = np.eye(4)
    mat_gl[:3, :3] = r
    mat_gl[:3, 3] = t

    mat_cv = opengl_to_opencv(mat_gl)
    mat_gl_roundtrip = opencv_to_opengl(mat_cv)

    np.testing.assert_allclose(mat_gl, mat_gl_roundtrip, atol=1e-6)


# ==============================================================================
# 3. Quality Gate Tests
# ==============================================================================

def test_quality_gate_blur_detection():
    gate = QualityGate(blur_threshold=100.0)

    # Sharp checkerboard pattern -> high Laplacian variance
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp_img[::20, :, :] = 255
    sharp_img[:, ::20, :] = 255

    score_sharp = gate.compute_blur_score(sharp_img)
    assert score_sharp > 500.0

    # Flat blurred / constant image -> zero Laplacian variance
    flat_img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    score_flat = gate.compute_blur_score(flat_img)
    assert score_flat < 1.0


def test_quality_gate_redundancy_filter():
    """Stage 2: near-duplicate viewpoints are dropped, real baselines are kept."""
    sharp_arr = (np.random.RandomState(42).rand(100, 100, 3) * 255).astype(np.uint8)
    img = Image.fromarray(sharp_arr)

    # Frame 0: origin
    mat0 = np.eye(4)
    kf0 = Keyframe("img0.jpg", 100, 500.0, 500.0, 50.0, 50.0, mat0, lambda: img)

    # Frame 1: moved only 1 cm (redundant stationary frame)
    mat1 = np.eye(4)
    mat1[0, 3] = 0.01
    kf1 = Keyframe("img1.jpg", 200, 500.0, 500.0, 50.0, 50.0, mat1, lambda: img)

    # Frame 2: moved 10 cm (accepted baseline)
    mat2 = np.eye(4)
    mat2[0, 3] = 0.10
    kf2 = Keyframe("img2.jpg", 300, 500.0, 500.0, 50.0, 50.0, mat2, lambda: img)

    kept, dropped = prune_redundant([kf0, kf1, kf2], min_translation_m=0.03, min_rotation_deg=2.0)

    assert kept == [0, 2]
    assert dropped == [1]


def test_quality_gate_exposure_filter():
    gate = QualityGate(dark_threshold=15.0)

    # Frame 0: normal
    normal_arr = (np.ones((100, 100, 3)) * 128).astype(np.uint8)
    # Add high frequency noise to prevent blur rejection
    normal_arr[::2, ::2] = 200
    img_normal = Image.fromarray(normal_arr)
    kf0 = Keyframe("img0.jpg", 100, 500.0, 500.0, 50.0, 50.0, np.eye(4), lambda: img_normal)

    # Frame 1: pitch black
    dark_arr = (np.ones((100, 100, 3)) * 5).astype(np.uint8)
    img_dark = Image.fromarray(dark_arr)
    mat1 = np.eye(4)
    mat1[0, 3] = 0.5
    kf1 = Keyframe("img1.jpg", 200, 500.0, 500.0, 50.0, 50.0, mat1, lambda: img_dark)

    res = gate.evaluate([kf0, kf1])
    assert res.metrics[1].rejection_reason == "exposure"


def _block_pattern(period=40, lo=90, hi=190):
    """Mid-gray checkerboard: textured, in focus, nothing clipped.

    Thin bright lines on black would be ~90% crushed black and get (correctly)
    rejected on exposure, and blurred white noise loses all structure and gets
    (correctly) rejected as featureless -- neither isolates the blur gate.
    """
    a = np.full((240, 240), lo, np.uint8)
    for y in range(0, 240, period):
        for x in range(0, 240, period):
            if ((y // period) + (x // period)) % 2 == 0:
                a[y:y + period, x:x + period] = hi
    return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)


def test_quality_gate_uneven_lighting_edge_cases():
    """Dim-but-sharp frames survive; blurred, clipped and featureless ones don't."""
    texture = _block_pattern()

    def kf(name, arr, x):
        mat = np.eye(4)
        mat[0, 3] = x
        img = Image.fromarray(arr)
        return Keyframe(name, 100, 500.0, 500.0, 120.0, 120.0, mat, lambda: img)

    frames = [kf(f"bright_{i}.jpg", texture, 0.5 * i) for i in range(6)]
    # Same detail at a quarter of the exposure -- sharp, just in shadow.
    frames.append(kf("dark.jpg", (texture * 0.25).astype(np.uint8), 4.0))
    # Motion blur: detail gone, structure kept.
    frames.append(kf("blurred.jpg", cv2.GaussianBlur(texture, (0, 0), 2.0), 5.0))
    # Sun in the lens: most of the frame clipped to white.
    blown = texture.copy()
    blown[:200] = 255
    frames.append(kf("blown.jpg", blown, 6.0))
    # Blank wall: nothing to track.
    frames.append(kf("flat.jpg", np.full((240, 240, 3), 128, np.uint8), 7.0))

    res = QualityGate().evaluate(frames)
    reason = {m.file_path: m.rejection_reason for m in res.metrics}
    assert reason["dark.jpg"] is None
    assert reason["blurred.jpg"] == "blur"
    assert reason["blown.jpg"] == "exposure"
    assert reason["flat.jpg"] == "texture"
    assert res.summary["accepted"] >= 7


def test_quality_gate_reject_cap():
    """A scan the metric reads as uniformly blurred still yields usable frames."""
    soft = cv2.GaussianBlur(_block_pattern(), (0, 0), 2.0)
    frames = []
    for i in range(10):
        mat = np.eye(4)
        mat[0, 3] = 0.5 * i
        img = Image.fromarray(soft)
        frames.append(Keyframe(f"f{i}.jpg", 100, 500.0, 500.0, 120.0, 120.0, mat, lambda: img))

    res = QualityGate(blur_threshold=10_000.0).evaluate(frames)
    assert res.summary["accepted"] >= 8


def test_full_ingestion_pipeline(mock_package_dir: Path):
    """Integration test chaining package loading -> quality gating -> pose sync."""
    # 1. Package loading
    loader = PackageLoader(min_keyframes=2)
    pkg = loader.load(mock_package_dir)
    assert len(pkg.keyframes) == 2

    # 2. Quality gating
    gate = QualityGate(blur_threshold=10.0)
    gate_res = gate.evaluate(pkg.keyframes)
    assert len(gate_res.accepted_keyframes) == 2

    # 3. Timestamp synchronization with VIO trajectory
    aligner = PoseAligner(pkg.trajectory)
    synced_kfs = aligner.synchronize_keyframes(gate_res.accepted_keyframes)
    assert len(synced_kfs) == 2
    for kf in synced_kfs:
        assert kf.transform_matrix.shape == (4, 4)

