"""Focused regression checks for structural mesh extraction."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d


MODULE_PATH = Path(__file__).resolve().parents[1] / "06_mesh_elaboration" / "extract_structural_meshes.py"
SPEC = importlib.util.spec_from_file_location("extract_structural_meshes", MODULE_PATH)
extract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extract)


def test_component_grouping_matches_boolean_masks_and_keeps_face_order():
    clusters = np.array([2, 0, 2, 1, 0, 3, 1, 3, 2])
    grouped = extract.component_face_indices(clusters, 4)
    for component, indices in enumerate(grouped):
        np.testing.assert_array_equal(indices, np.flatnonzero(clusters == component))
    assert len(extract.component_face_indices(np.array([], dtype=int), 3)) == 3


def test_closest_major_instance_keeps_first_instance_on_equal_distance():
    trees = [extract.cKDTree([[-1.0, 0.0, 0.0]]), extract.cKDTree([[1.0, 0.0, 0.0]])]
    index, distance = extract.closest_major_instance(np.array([[0.0, 0.0, 0.0]]), trees)
    assert index == 0
    assert distance == 1.0


def test_instance_filenames_are_role_indexed_without_elevation_rules():
    assert extract.instance_filename("floor", 0) == "floor_inst000.ply"
    assert extract.instance_filename("wall", 12) == "wall_inst012.ply"


def test_extraction_adds_neutral_colors_when_source_has_none(tmp_path, monkeypatch):
    mesh_path = tmp_path / "source.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, [7])
    structural_json = tmp_path / "structural.json"
    structural_json.write_text(json.dumps({"classifications": {"7": {"structural_role": "floor"}}}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "reviewed_derived_output.ply"
    sentinel.write_bytes(b"do not delete")
    stale_raw_mesh = output_dir / "ceiling_main.ply"
    stale_raw_mesh.write_bytes(b"stale raw mesh")
    derived_mesh = output_dir / "wall_regularized.ply"
    derived_mesh.write_bytes(b"derived mesh")
    (output_dir / "structural_instances_manifest.json").write_text(json.dumps({
        "instances": [
            {"role": "ceiling", "instance_name": "ceiling_main", "file_name": "ceiling_main.ply"},
            {"role": "wall", "instance_name": "wall_regularized", "file_name": "wall_regularized.ply"},
        ],
    }))
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=mesh_path, labels_path=labels_path, structural_json=structural_json,
        output_dir=output_dir, min_big_area=1.0, merge_dist_thresh=0.25,
    ))

    assert extract.main() == 0
    exported = o3d.io.read_triangle_mesh(str(output_dir / "floor_inst000.ply"))
    assert len(np.asarray(exported.vertex_colors)) == 3
    assert sentinel.read_bytes() == b"do not delete"
    assert not stale_raw_mesh.exists()
    assert derived_mesh.read_bytes() == b"derived mesh"


def test_extraction_exports_multiple_same_role_instances_with_stable_unique_ids(tmp_path, monkeypatch):
    mesh_path = tmp_path / "source.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [0, 2, 0], [2, 2, 0], [0, 4, 0], [10, 3, 0], [12, 3, 0], [10, 5, 0],
        [20, 4, 0], [22, 4, 0], [20, 6, 0], [30, 0, 0], [32, 0, 0], [30, 2, 0],
        [40, 0, 0], [42, 0, 0], [40, 2, 0],
    ])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, [7, 7, 7, 8, 8])
    structural_json = tmp_path / "structural.json"
    structural_json.write_text(json.dumps({"classifications": {
        "7": {"structural_role": "floor"}, "8": {"structural_role": "wall"},
    }}))
    output_dir = tmp_path / "output"
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=mesh_path, labels_path=labels_path, structural_json=structural_json,
        output_dir=output_dir, min_big_area=1.0, merge_dist_thresh=0.25,
    ))

    assert extract.main() == 0
    assert sorted(path.name for path in output_dir.glob("*.ply")) == [
        "floor_inst000.ply", "floor_inst001.ply", "floor_inst002.ply", "wall_inst000.ply", "wall_inst001.ply",
    ]


def test_late_export_failure_preserves_previous_outputs_and_manifest(tmp_path, monkeypatch):
    mesh_path = tmp_path / "source.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [2, 0, 0], [0, 2, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5]])
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, [7, 8])
    structural_json = tmp_path / "structural.json"
    structural_json.write_text(json.dumps({"classifications": {
        "7": {"structural_role": "floor"}, "8": {"structural_role": "ceiling"},
    }}))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    prior_mesh = output_dir / "floor_inst000.ply"
    prior_mesh.write_bytes(b"previous floor")
    prior_manifest = output_dir / "structural_instances_manifest.json"
    prior_manifest.write_text(json.dumps({"instances": [{
        "role": "floor", "instance_name": "floor_inst000", "file_name": "floor_inst000.ply",
    }]}))
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=mesh_path, labels_path=labels_path, structural_json=structural_json,
        output_dir=output_dir, min_big_area=1.0, merge_dist_thresh=0.25,
    ))
    calls = 0

    def fail_second_export(path, *_):
        nonlocal calls
        calls += 1
        if calls == 1:
            Path(path).write_bytes(b"staged floor")
            return True
        return False

    monkeypatch.setattr(extract, "export_colored_ply", fail_second_export)
    assert extract.main() == 1
    assert prior_mesh.read_bytes() == b"previous floor"
    assert json.loads(prior_manifest.read_text())["instances"][0]["file_name"] == "floor_inst000.ply"


def test_extraction_requires_structural_metadata(tmp_path, monkeypatch, capsys):
    mesh_path = tmp_path / "source.ply"
    mesh_path.touch()
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, [])
    missing_metadata = tmp_path / "missing.json"
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=mesh_path, labels_path=labels_path, structural_json=missing_metadata,
        output_dir=tmp_path / "output", min_big_area=1.0, merge_dist_thresh=0.25,
    ))

    assert extract.main() == 1
    assert "Structural classification metadata not found" in capsys.readouterr().out
