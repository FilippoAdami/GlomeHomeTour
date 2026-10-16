#!/usr/bin/env python3
"""Tests for COLMAP feature count gating and ARCore pose locking."""

import json
import tempfile
from pathlib import Path
import numpy as np

import sys
_backend = Path(__file__).resolve().parents[1]
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from Utilities.pipeline_paths import bootstrap
bootstrap()

from Utilities.pipeline_step import StepContext
from convert_transforms_to_colmap import OPENGL_TO_OPENCV
from scene.colmap_loader import rotmat2qvec
import sys
sys.path.insert(0, str(_backend / "01_poses_refinment"))
from step_colmap import (
    count_registered_features,
    lock_poses_to_arcore,
    MIN_FEATURE_PRUNE,
    MIN_FEATURE_LOCK,
    MAX_FEATURE_REJECT_FRAC,
)


def test_count_registered_features():
    with tempfile.TemporaryDirectory() as tmp:
        sparse_dir = Path(tmp)
        # Create mock images.txt with:
        # frame_00001.jpg: 2 triangulated points (ids 10, 20), 1 untriangulated (-1)
        # frame_00002.jpg: 0 triangulated points, 2 untriangulated (-1, -1)
        # frame_00003.jpg: 4 triangulated points (ids 1, 2, 3, 4)
        content = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_00001.jpg
10.0 20.0 10 30.0 40.0 -1 50.0 60.0 20
2 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_00002.jpg
15.0 25.0 -1 35.0 45.0 -1
3 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_00003.jpg
1.0 2.0 1 3.0 4.0 2 5.0 6.0 3 7.0 8.0 4
"""
        (sparse_dir / "images.txt").write_text(content)
        counts = count_registered_features(sparse_dir)

        assert counts["frame_00001.jpg"] == 2
        assert counts["frame_00002.jpg"] == 0
        assert counts["frame_00003.jpg"] == 4


def test_lock_poses_to_arcore():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest_dir = root / "00_ingestion"
        sparse_dir = root / "sparse" / "0"
        manifest_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)

        # Original ARCore pose for frame_00001: identity camera-to-world
        c2w = np.eye(4)
        transforms = {
            "camera_model": "PINHOLE",
            "fl_x": 1000.0, "fl_y": 1000.0, "cx": 500.0, "cy": 500.0,
            "w": 1000, "h": 1000, "camera_angle_x": 1.0,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            "frames": [
                {
                    "file_path": "images/frame_00001.jpg",
                    "transform_matrix": c2w.tolist(),
                    "timestamp_ns": 1000,
                    "fl_x": 1000.0, "fl_y": 1000.0, "cx": 500.0, "cy": 500.0,
                    "w": 1000, "h": 1000,
                }
            ]
        }
        (manifest_dir / "transforms.json").write_text(json.dumps(transforms, indent=2))

        # In COLMAP, suppose BA distorted the pose: translation was [5.0, 5.0, 5.0]
        distorted_images_txt = """1 1.0 0.0 0.0 0.0 5.0 5.0 5.0 1 frame_00001.jpg
10.0 20.0 10 30.0 40.0 20
"""
        (sparse_dir / "images.txt").write_text(distorted_images_txt)
        (sparse_dir / "cameras.txt").write_text("1 PINHOLE 1000 1000 1000.0 1000.0 500.0 500.0\n")
        (sparse_dir / "points3D.txt").write_text("# empty\n")

        # Expected pose in COLMAP: c2w @ OPENGL_TO_OPENCV, w2c = inv(c2w)
        expected_c2w = c2w @ OPENGL_TO_OPENCV
        expected_w2c = np.linalg.inv(expected_c2w)
        expected_q = rotmat2qvec(expected_w2c[:3, :3])
        expected_t = expected_w2c[:3, 3]

        class DummyCtx:
            def note(self, msg): pass
            def metric(self, k, v): pass

        locked = lock_poses_to_arcore(sparse_dir, manifest_dir, {"frame_00001.jpg"}, DummyCtx())
        assert locked == 1

        updated_txt = (sparse_dir / "images.txt").read_text()
        lines = [l for l in updated_txt.splitlines() if l.strip()]
        parts = lines[0].split()
        q = [float(x) for x in parts[1:5]]
        t = [float(x) for x in parts[5:8]]

        assert np.allclose(q, expected_q, atol=1e-5)
        assert np.allclose(t, expected_t, atol=1e-5)


def test_feature_gating_thresholds():
    # Verify thresholds
    assert MIN_FEATURE_PRUNE == 30
    assert MIN_FEATURE_LOCK == 100
    assert MAX_FEATURE_REJECT_FRAC == 0.10

    # Categorization test
    counts = {
        "catastrophic.jpg": 12,
        "weak1.jpg": 30,
        "weak2.jpg": 75,
        "weak3.jpg": 99,
        "strong.jpg": 100,
        "rich.jpg": 650,
    }

    catastrophic = {n for n, c in counts.items() if c < MIN_FEATURE_PRUNE}
    assert catastrophic == {"catastrophic.jpg"}

    weak = {n for n, c in counts.items() if MIN_FEATURE_PRUNE <= c < MIN_FEATURE_LOCK}
    assert weak == {"weak1.jpg", "weak2.jpg", "weak3.jpg"}

    strong = {n for n, c in counts.items() if c >= MIN_FEATURE_LOCK}
    assert strong == {"strong.jpg", "rich.jpg"}
