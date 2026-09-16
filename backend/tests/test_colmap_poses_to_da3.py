#!/usr/bin/env python3
"""Substep 4a: the poses handed to DA3 are proper world-to-camera transforms.

This is the check that has to fail *before* a depth run, not after. Feeding DA3
a camera-to-world pose (or one converted twice) does not crash -- it returns
plausible-but-wrong depth, and the symptom only shows up as a warped cloud
20 minutes later. See `backend/CLAUDE.md`, "the bowled point cloud".

    .venv/bin/python tests/test_colmap_poses_to_da3.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

_backend = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "colmap_poses_to_da3", _backend / "02_depth_estimation" / "colmap_poses_to_da3.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
validate_poses = _mod.validate_poses

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _rot(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def _synthetic_scene(n=8):
    """A small trajectory: returns (w2c, camera centres)."""
    centres = np.stack([np.linspace(0, 2, n), np.zeros(n), np.linspace(0, -1, n)], axis=1)
    w2c = np.zeros((n, 4, 4))
    for i in range(n):
        r_c2w = _rot([0, 1, 0], 12.0 * i)
        w2c[i] = np.eye(4)
        w2c[i, :3, :3] = r_c2w.T
        w2c[i, :3, 3] = -r_c2w.T @ centres[i]
    return w2c, centres


def test_accepts_valid_w2c():
    w2c, centres = _synthetic_scene()
    stats = validate_poses(w2c, centres)
    assert abs(stats["det_min"] - 1.0) < 1e-6, stats
    assert stats["max_ortho_err"] < 1e-9, stats
    # The recovered centres must be the ones we built the poses from.
    assert stats["centre_offset_max_m"] < 1e-9, stats


def test_rejects_mirrored_rotation():
    w2c, centres = _synthetic_scene()
    w2c[3, :3, :3] *= -1.0  # det becomes -1: a reflection, not a rotation
    try:
        validate_poses(w2c, centres)
    except ValueError as e:
        assert "det" in str(e).lower(), e
    else:
        raise AssertionError("a mirrored rotation was accepted")


def test_rejects_non_orthonormal():
    w2c, centres = _synthetic_scene()
    w2c[2, :3, :3] *= 1.5  # scaled: still det>0, but not a rigid transform
    try:
        validate_poses(w2c, centres)
    except ValueError as e:
        assert "orthonormal" in str(e).lower() or "det" in str(e).lower(), e
    else:
        raise AssertionError("a scaled rotation was accepted")


def test_rejects_double_conversion():
    """The regression this whole module exists to prevent.

    COLMAP already stores OpenCV world-to-camera. Passing it through
    `arcore_c2w_to_da3_w2c()` anyway applies a second inversion and axis flip;
    the result is still a valid rigid transform, so only the camera-centre
    check catches it.
    """
    w2c, centres = _synthetic_scene()
    doubled = np.stack([np.linalg.inv(p @ OPENGL_TO_OPENCV) for p in w2c])

    validate_poses(doubled)  # rigid on its own terms -- passes without centres

    try:
        validate_poses(doubled, centres)
    except ValueError as e:
        assert "centre" in str(e).lower(), e
    else:
        raise AssertionError("a doubly-converted pose was accepted")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"{len(tests)} passed")
