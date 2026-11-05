"""Unit tests for single_scan_synchronizer.py."""
import csv
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import plyfile
import pytest
from scipy.spatial.transform import Rotation
import trimesh

ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "07_alignment"
if str(ALIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ALIGNMENT_DIR))

import single_scan_synchronizer as synchronizer
import multi_asset_synchronizer
from single_scan_synchronizer import (
    apply_transformation_to_single_scan,
    backup_raw_scan_if_needed,
    transform_mesh_ply,
    transform_gaussian_splat_ply,
    transform_trajectory_csv,
    transform_transforms_json,
)


def test_transform_mesh_ply(tmp_path):
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    src_ply = tmp_path / "box.ply"
    dst_ply = tmp_path / "box_transformed.ply"
    mesh.export(str(src_ply))

    T = np.eye(4)
    T[:3, 3] = [1.5, -2.0, 3.5]
    transform_mesh_ply(src_ply, dst_ply, T)

    m_out = trimesh.load(dst_ply, process=False)
    np.testing.assert_allclose(m_out.centroid, [1.5, -2.0, 3.5], atol=1e-5)


def test_transform_trajectory_csv(tmp_path):
    src_csv = tmp_path / "trajectory.csv"
    dst_csv = tmp_path / "trajectory_aligned.csv"

    header = ["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw", "tracking", "exported"]
    # identity rotation, position at origin
    row = ["1000", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "1.0", "TRACKING", "1"]

    with open(src_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

    # 90 deg rotation around Y, translation [2.0, 0.0, 1.0]
    R_mat = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [2.0, 0.0, 1.0]

    transform_trajectory_csv(src_csv, dst_csv, T)

    with open(dst_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        h = next(reader)
        r = next(reader)

    # Pos should be [2.0, 0.0, 1.0]
    assert float(r[1]) == pytest.approx(2.0, abs=1e-4)
    assert float(r[2]) == pytest.approx(0.0, abs=1e-4)
    assert float(r[3]) == pytest.approx(1.0, abs=1e-4)


def test_transform_transforms_json(tmp_path):
    src_json = tmp_path / "transforms.json"
    dst_json = tmp_path / "transforms_aligned.json"

    data = {
        "frames": [
            {
                "file_path": "images/frame_000.jpg",
                "transform_matrix": np.eye(4).tolist()
            }
        ]
    }
    with open(src_json, "w", encoding="utf-8") as f:
        json.dump(data, f)

    T = np.array([
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 1.0, 0.0, 0.5],
        [-1.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    transform_transforms_json(src_json, dst_json, T)

    with open(dst_json, "r", encoding="utf-8") as f:
        out_data = json.load(f)

    mat_out = np.array(out_data["frames"][0]["transform_matrix"])
    np.testing.assert_allclose(mat_out, T, atol=1e-6)


def test_single_scan_transforms_each_asset_once_and_copies_identical_outputs(tmp_path, monkeypatch):
    scan_dir = tmp_path / "scan_0"
    export_dir = tmp_path / "aligned" / "scan_0"
    mesh_dir = scan_dir / "mesh_elaboration"
    elements_dir = mesh_dir / "elements"
    gaussian_dir = scan_dir / "gaussian_splat"
    elements_dir.mkdir(parents=True)
    gaussian_dir.mkdir()

    trimesh.creation.box().export(scan_dir / "mesh.ply")
    trimesh.creation.icosphere(subdivisions=1).export(mesh_dir / "complete.ply")
    trimesh.creation.box(extents=[2, 1, 1]).export(elements_dir / "walls.ply")

    gaussian_dtype = np.dtype([
        (name, "<f4") for name in (
            "x", "y", "z", "nx", "ny", "nz", "rot_0", "rot_1", "rot_2", "rot_3",
        )
    ])
    gaussian_vertices = np.zeros(2, dtype=gaussian_dtype)
    gaussian_vertices["x"] = [0.0, 1.0]
    gaussian_vertices["nx"] = 1.0
    gaussian_vertices["rot_0"] = 1.0
    plyfile.PlyData([
        plyfile.PlyElement.describe(gaussian_vertices, "vertex")
    ], text=False, byte_order="<").write(gaussian_dir / "gaussian_splat.ply")

    with (scan_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writerow(["1", "0", "0", "0", "0", "0", "0", "1"])
    (scan_dir / "transforms.json").write_text(json.dumps({
        "frames": [{"transform_matrix": np.eye(4).tolist()}],
    }))

    calls = {"mesh": 0, "gaussian": 0, "trajectory": 0, "transforms": 0}
    for name, key in (
        ("transform_mesh_ply", "mesh"),
        ("transform_gaussian_splat_ply", "gaussian"),
        ("transform_trajectory_csv", "trajectory"),
        ("transform_transforms_json", "transforms"),
    ):
        original = getattr(synchronizer, name)

        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(synchronizer, name, counted)

    transform = np.eye(4)
    transform[:3, 3] = [1.5, -2.0, 3.5]
    result = apply_transformation_to_single_scan(
        scan_dir, transform, export_dir=export_dir, gaussian_mesh_scenes={}
    )

    assert calls == {"mesh": 3, "gaussian": 1, "trajectory": 1, "transforms": 1}
    assert result["updated_assets_count"] == 12
    for relative_path in (
        Path("mesh.ply"),
        Path("mesh_elaboration/complete.ply"),
        Path("mesh_elaboration/elements/walls.ply"),
        Path("gaussian_splat/gaussian_splat.ply"),
        Path("trajectory.csv"),
        Path("transforms.json"),
    ):
        assert (scan_dir / relative_path).read_bytes() == (export_dir / relative_path).read_bytes()

    aligned_mesh = trimesh.load(scan_dir / "mesh.ply", process=False)
    np.testing.assert_allclose(aligned_mesh.centroid, transform[:3, 3], atol=1e-5)


def test_colmap_reconstruction_is_transformed_once_for_all_destinations(tmp_path, monkeypatch):
    source = tmp_path / "sparse" / "0"
    source.mkdir(parents=True)
    (source / "images.txt").write_text("source model")
    outputs = [tmp_path / "scan" / "sparse", tmp_path / "aligned" / "sparse"]
    calls = {"load": 0, "transform": 0, "text": 0, "binary": 0}

    class FakeReconstruction:
        def __init__(self, path):
            calls["load"] += 1
            self.images = {}
            self.points3D = {}

        def transform(self, sim3):
            calls["transform"] += 1

        def write_text(self, path):
            calls["text"] += 1
            (Path(path) / "images.txt").write_text("transformed model")

        def write_binary(self, path):
            calls["binary"] += 1
            (Path(path) / "images.bin").write_bytes(b"transformed binary model")

    monkeypatch.setattr(synchronizer, "HAS_PYCOLMAP", True)
    monkeypatch.setattr(synchronizer, "pycolmap", SimpleNamespace(
        Reconstruction=FakeReconstruction,
        Sim3d=lambda matrix: matrix.copy(),
    ))
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    synchronizer.transform_colmap_sparse(source.parent, outputs[0], transform, [outputs[1]])

    assert calls == {"load": 1, "transform": 1, "text": 2, "binary": 2}
    for relative_path in (Path("0/images.txt"), Path("0/images.bin")):
        assert (outputs[0] / relative_path).read_bytes() == (outputs[1] / relative_path).read_bytes()


def test_combined_gaussian_transform_is_reused_for_per_scan_output(tmp_path, monkeypatch):
    scan_dir = tmp_path / "scan_0"
    relative_path = Path("point_cloud/iteration_30000/point_cloud.ply")
    source = scan_dir / "gaussian_splat" / relative_path
    source.parent.mkdir(parents=True)
    dtype = np.dtype([
        (name, "<f4") for name in (
            "x", "y", "z", "nx", "ny", "nz", "rot_0", "rot_1", "rot_2", "rot_3",
        )
    ])
    vertices = np.zeros(2, dtype=dtype)
    vertices["x"] = [0.0, 1.0]
    vertices["nx"] = 1.0
    vertices["rot_0"] = 1.0
    plyfile.PlyData([
        plyfile.PlyElement.describe(vertices, "vertex")
    ], text=False, byte_order="<").write(source)

    scan = SimpleNamespace(info=SimpleNamespace(
        scan_dir=scan_dir,
        gaussian_splat_path=source,
    ))
    scans = {"scan_0": scan}
    transform = np.eye(4)
    transform[:3, 3] = [1.5, -2.0, 3.5]
    output_path = tmp_path / "aligned" / "aligned_sealed_gaussians.ply"
    reuse_paths = multi_asset_synchronizer.MultiAssetSynchronizer._assemble_gaussian_splats(
        scans, {"scan_0": transform}, output_path, {}
    )
    cached_output = reuse_paths["scan_0"][relative_path]

    def unexpected_transform(*args, **kwargs):
        raise AssertionError("Per-scan synchronization repeated the Gaussian transform")

    monkeypatch.setattr(synchronizer, "transform_gaussian_splat_ply", unexpected_transform)
    synchronizer.apply_transformation_to_single_scan(
        scan_dir,
        transform,
        export_dir=output_path.parent / "scans" / "scan_0",
        gaussian_mesh_scenes={},
        pretransformed_gaussian_splats=reuse_paths["scan_0"],
    )

    exported_path = scan_dir / "gaussian_splat" / relative_path
    assert exported_path.read_bytes() == cached_output.read_bytes()
    combined_vertices = plyfile.PlyData.read(str(output_path))["vertex"].data
    cached_vertices = plyfile.PlyData.read(str(cached_output))["vertex"].data
    np.testing.assert_array_equal(combined_vertices, cached_vertices)
