import csv
import hashlib
import json

import numpy as np
import pytest

from keyframe_budget import read_scene_size
from scene_extent import (
    estimate_metric_scale,
    estimate_north_heading_deg,
    load_depth_scene,
    load_metric_depth_scene,
    write_scene_extent_outputs,
)
from scene.dataset_readers import storePly
from Utilities.scene_alignment import rotate_horizontal


def test_compass_heading_estimation_and_alignment(tmp_path):
    frames = []
    for i in range(5):
        mat = np.eye(4)
        mat[:3, 3] = [float(i), 1.5, float(i)]
        frames.append({
            "file_path": f"images/f{i}.jpg",
            "timestamp_ns": 1000 + i,
            "transform_matrix": mat.tolist(),
            "compass_heading_deg": 90.0,
        })
    tf_path = tmp_path / "transforms.json"
    tf_path.write_text(json.dumps({"frames": frames}))

    north_deg = estimate_north_heading_deg(tmp_path, transforms_path=tf_path)
    assert north_deg is not None
    assert abs(north_deg - 270.0) < 1e-4

    align_deg = (90.0 - north_deg) % 360.0
    assert abs(align_deg - 180.0) < 1e-4
    pt_aligned = rotate_horizontal(np.array([[-1.0, 0.0, 0.0]]), align_deg, up_axis=1)
    assert np.allclose(pt_aligned, [[1.0, 0.0, 0.0]], atol=1e-5)


def test_load_depth_scene_reads_depth_ply_and_pose_centres(tmp_path):
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], dtype=float)
    storePly(str(depth_dir / "points3D_depth.ply"), xyz, np.zeros_like(xyz))

    w2c = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    expected_centres = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    w2c[:, :3, 3] = -expected_centres
    np.savez(depth_dir / "poses_da3.npz", w2c=w2c)

    cloud, centres = load_depth_scene(tmp_path)
    assert np.allclose(cloud.points, xyz)
    assert np.allclose(centres, expected_centres)


def test_read_scene_size_rejects_uncalibrated_colmap_units(tmp_path):
    path = tmp_path / "scene_size.txt"
    path.write_text("coordinate_units: unscaled COLMAP model units\nx: 4\ny: 2\nz: 3\narea_m2: 8\n")
    with pytest.raises(ValueError, match="not calibrated metres"):
        read_scene_size(path)


def _write_arcore_scale_fixture(workspace):
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "ingestion_results.json").write_text(
        json.dumps({"source_format": "arcore_app_certified"}), encoding="utf-8")
    route = np.array([
        [0.00, 0.00, 0.00], [0.18, 0.00, 0.00], [0.40, 0.02, 0.00],
        [0.61, 0.04, 0.03], [0.84, 0.05, 0.06], [1.08, 0.08, 0.08],
        [1.34, 0.10, 0.10], [1.60, 0.10, 0.15], [1.88, 0.15, 0.20],
        [2.15, 0.20, 0.22],
    ])
    # ARCore's origin shifts after one bad transition, while its later local
    # steps keep the same metric scale. Cross-drift baselines must be excluded.
    colmap_route = route.copy()
    route[5:] += [0.0, 0.0, 1.5]
    yaw = np.radians(71.0)
    rotation = np.array([[np.cos(yaw), 0.0, np.sin(yaw)],
                         [0.0, 1.0, 0.0],
                         [-np.sin(yaw), 0.0, np.cos(yaw)]])
    translation = np.array([4.0, -2.0, 1.0])
    frames = []
    with (workspace / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["timestamp_ns", "tracking", "tx", "ty", "tz"])
        writer.writeheader()
        for i, position in enumerate(route):
            timestamp = 1_000_000 + i
            writer.writerow({"timestamp_ns": timestamp, "tracking": "TRACKING",
                             "tx": position[0], "ty": position[1], "tz": position[2]})
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = rotation @ (colmap_route[i] / 0.5) + translation
            frames.append({"file_path": f"images/frame_{i:03d}.jpg",
                           "timestamp_ns": timestamp,
                           "transform_matrix": matrix.tolist()})
    (workspace / "transforms.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
def test_metric_scale_is_robust_and_only_scales_in_memory(tmp_path):
    _write_arcore_scale_fixture(tmp_path)
    calibration = estimate_metric_scale(tmp_path)
    (tmp_path / "metric_scale.json").write_text(json.dumps(calibration), encoding="utf-8")
    assert calibration["available"]
    assert calibration["meters_per_colmap_unit"] == pytest.approx(0.5, abs=1e-10)
    assert calibration["usable_pairs"] >= 6
    assert calibration["relative_mad"] < 0.01
    assert len(calibration["steps_rejected"]) == 1
    assert calibration["steps_rejected"][0]["from_frame"] == "frame_004.jpg"
    assert calibration["steps_rejected"][0]["to_frame"] == "frame_005.jpg"
    assert len(calibration["segments_used"]) == 2

    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    xyz = np.repeat(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0],
                              [2.0, 0.0, 1.0], [1.0, 1.0, 2.0]]), 25, axis=0)
    ply_path = depth_dir / "points3D_depth.ply"
    storePly(str(ply_path), xyz, np.zeros_like(xyz))
    w2c = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    w2c[:, :3, 3] = -np.array([[2.0, 0.0, 0.0], [0.0, 4.0, 2.0]])
    poses_path = depth_dir / "poses_da3.npz"
    np.savez(poses_path, w2c=w2c)
    source_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ply_path, poses_path, tmp_path / "transforms.json")
    }

    cloud, centers, applied = load_metric_depth_scene(tmp_path)
    assert applied["available"]
    assert np.allclose(cloud.points, xyz * 0.5, atol=1e-6)
    assert np.allclose(centers, [[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])

    extent = write_scene_extent_outputs(tmp_path)
    size_report = (depth_dir / "scene_size.txt").read_text(encoding="utf-8")
    assert extent["metric_scale_available"] and extent["floors"] == 1
    assert "coordinate_units: m" in size_report
    assert "area_m2:" in size_report
    assert read_scene_size(depth_dir / "scene_size.txt")["floors"] == 1.0
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in source_hashes.items())

    (tmp_path / "ingestion_results.json").write_text(
        json.dumps({"source_format": "colmap"}), encoding="utf-8")
    rejected = estimate_metric_scale(tmp_path)
    assert not rejected["available"]
    assert "ARCore-origin" in rejected["reason"]


def test_depth_extent_does_not_recompute_metric_scale(tmp_path):
    _write_arcore_scale_fixture(tmp_path)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    xyz = np.repeat(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), 25, axis=0)
    storePly(str(depth_dir / "points3D_depth.ply"), xyz, np.zeros_like(xyz))
    np.savez(depth_dir / "poses_da3.npz", w2c=np.repeat(np.eye(4)[None], 2, axis=0))

    _, _, calibration = load_metric_depth_scene(tmp_path)
    assert not calibration["available"]
    assert calibration["reason"] == "missing COLMAP-stage metric_scale.json"
