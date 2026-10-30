"""Coordinate conventions and frustum membership for the pose inspector."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_poses_refinment"))
from inspect_pose_alignment import (
    Pose, camera_calibration, camera_wireframe, frustum_mask, load_frames, pose_delta,
    select_scene,
)


def test_frustum_uses_camera_coordinates_and_rejects_behind_camera():
    c2w = np.array([[0., 0., 1., 10.], [0., 1., 0., 2.], [-1., 0., 0., 3.], [0., 0., 0., 1.]])
    pose = camera_calibration("PINHOLE", 100, 100, [100, 100, 50, 50], c2w)
    camera_points = np.array([[0, 0, 2], [.49, 0, 1], [.51, 0, 1],
                              [0, -.51, 1], [0, 0, -1], [0, 0, 0]])
    world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    assert frustum_mask(world_points, pose).tolist() == [True, True, False, False, False, False]


def test_frustum_includes_lens_distortion():
    points = np.array([[.49, 0, 1], [0, 0, 1]])
    pinhole = camera_calibration("PINHOLE", 100, 100, [100, 100, 50, 50], np.eye(4))
    distorted = camera_calibration("OPENCV", 100, 100,
                                   [100, 100, 50, 50, .2, 0, 0, 0], np.eye(4))
    assert frustum_mask(points, pinhole).tolist() == [True, True]
    assert frustum_mask(points, distorted).tolist() == [False, True]


def test_camera_wireframe_points_forward_and_marks_the_top():
    c2w = np.eye(4)
    c2w[:3, 3] = [1, 2, 3]
    pose = camera_calibration("PINHOLE", 100, 100, [100, 100, 50, 50], c2w)
    points, edges = camera_wireframe(pose, .2)
    np.testing.assert_allclose(points[0], [1, 2, 3])
    np.testing.assert_allclose(points[1:5].mean(0), [1, 2, 3.2])
    assert points[7, 1] < points[1:3, 1].min()  # OpenCV camera up is negative Y.
    assert edges.shape == (11, 2)


@pytest.fixture
def scan(tmp_path):
    scene = tmp_path / "scan_0"
    images = scene / "images"
    sparse = scene / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    for name in ("frame_10.jpg", "frame_2.jpg"):
        (images / name).touch()
    (sparse / "cameras.txt").write_text("1 PINHOLE 80 120 100 90 40 60\n")
    (sparse / "images.txt").write_text(
        "1 0 1 0 0 -1 2 3 1 frame_10.jpg\n\n"
        "2 0 1 0 0 -1 2 3 1 frame_2.jpg\n\n")
    c2w_gl = np.eye(4)
    c2w_gl[:3, 3] = [1, 2, 3]
    data = {"w": 80, "h": 120, "fl_x": 110, "fl_y": 120, "cx": 40, "cy": 60,
            "frames": [{"file_path": f"images/{name}", "transform_matrix": c2w_gl.tolist(),
                        **({"fl_x": 111} if name == "frame_2.jpg" else {})}
                       for name in ("frame_10.jpg", "frame_2.jpg")]}
    (scene / "transforms.json").write_text(json.dumps(data))
    return scene


def test_pairs_arcore_gl_with_colmap_cv_and_keeps_individual_intrinsics(scan):
    frames = load_frames(scan)
    assert [f.name for f in frames] == ["frame_2.jpg", "frame_10.jpg"]
    for frame in frames:
        np.testing.assert_allclose(frame.poses["arcore"].c2w, frame.poses["colmap"].c2w)
        np.testing.assert_allclose(pose_delta(frame), [0, 0], atol=1e-8)
    assert frames[0].poses["arcore"].K[0, 0] == 111
    assert frames[1].poses["arcore"].K[0, 0] == 110
    assert frames[0].poses["colmap"].K[0, 0] == 100
    assert select_scene(scan.parent) == scan
    assert select_scene(scan.parent, "scan_0") == scan


def test_direct_colmap_without_arcore_and_duplicate_images(scan):
    (scan / "transforms.json").unlink()
    assert all(set(f.poses) == {"colmap"} for f in load_frames(scan))
    nested = scan / "images" / "another_camera"
    nested.mkdir()
    (nested / "frame_10.jpg").touch()
    with pytest.raises(ValueError, match="Ambiguous image basename"):
        load_frames(scan)


def test_axis_alignment_transforms_cloud_and_both_saved_pose_sets(scan, monkeypatch):
    import step_colmap

    # Exercise the production world transform without launching the COLMAP executable.
    monkeypatch.setattr(step_colmap, "_run_colmap", lambda _command: None)
    before = load_frames(scan)[0].poses["colmap"]
    xyz = np.array([[1., 2., 1.], [1.2, 2.1, 1.], [5., 2., 1.]])
    sparse = scan / "sparse" / "0"
    (sparse / "points3D.txt").write_text("\n".join(
        f"{i + 1} {p[0]} {p[1]} {p[2]} 120 100 80 0.1 1 0 2 0"
        for i, p in enumerate(xyz)) + "\n")
    angle = np.radians(37)
    rotation = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0],
                         [np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([2., -1.5, -.8])
    expected = np.eye(4)
    expected[:3, :3], expected[:3, 3] = rotation, translation
    expected = expected @ before.c2w
    step_colmap.transform_colmap_model(sparse, rotation, translation, xyz,
                                      np.full_like(xyz, 128))
    step_colmap.transform_arcore_metadata(
        scan / "transforms.json", None, {"frame_2.jpg", "frame_10.jpg"},
        rotation, translation, scan)
    after = load_frames(scan)[0]
    np.testing.assert_allclose(after.poses["colmap"].c2w, expected, atol=1e-8)
    np.testing.assert_allclose(after.poses["arcore"].c2w, expected, atol=1e-8)
    transformed_xyz = xyz @ rotation.T + translation
    np.testing.assert_array_equal(frustum_mask(xyz, before),
                                  frustum_mask(transformed_xyz, after.poses["colmap"]))
    records = [line.split() for line in (sparse / "points3D.txt").read_text().splitlines()]
    np.testing.assert_allclose(np.array([row[1:4] for row in records], dtype=float),
                               transformed_xyz, atol=1e-8)


def test_invalid_pose_and_unsupported_camera_fail_explicitly():
    with pytest.raises(ValueError, match="rigid"):
        Pose(np.zeros((4, 4)), np.eye(3), 100, 100, np.zeros(4))
    with pytest.raises(ValueError, match="Unsupported camera model"):
        camera_calibration("FOV", 100, 100, [100, 100, 50, 50, 1], np.eye(4))
