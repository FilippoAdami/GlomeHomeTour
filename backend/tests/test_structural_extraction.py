"""Focused regression checks for structural mesh extraction."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest


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


def test_role_cleanup_removes_bad_wall_fill_and_degenerates_without_orientation_filter():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 0, 1],       # horizontal floor
        [2, 0, 0], [2, 1, 0], [2, 0, 1],       # vertical wall
        [3, 0, 0], [4, 0, 0], [3, 0, 1],       # horizontal face mislabeled as wall
        [5, 0, 0], [5, 1, 0], [5, 0, 1],       # vertical face mislabeled as floor
        [6, 0, 0], [6, 1, 0], [6, 2, 0],       # degenerate
        [7, 0, 0], [8, 0, 0], [7, 0.01, 0],    # elongated inferred fill
    ], dtype=float)
    triangles = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    fill_flags = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
    ids = np.arange(len(triangles), dtype=np.int64)

    wall_source = np.array([0, 1, 2, 4, 5], dtype=np.int64)
    floor_source = np.array([0, 1, 3, 4, 5], dtype=np.int64)
    wall_ids, wall_stats = extract._clean_role_triangles(
        vertices, triangles, wall_source, fill_flags, "wall"
    )
    floor_ids, floor_stats = extract._clean_role_triangles(
        vertices, triangles, floor_source, fill_flags, "floor"
    )

    np.testing.assert_array_equal(wall_ids, [0, 1, 2])
    np.testing.assert_array_equal(floor_ids, [0])
    assert wall_stats["wrong_orientation_triangles_removed"] == 0
    assert floor_stats["wrong_orientation_triangles_removed"] == 2
    assert wall_stats["degenerate_triangles_removed"] == 1
    assert wall_stats["elongated_hole_fill_triangles_removed"] == 1
    assert floor_stats["elongated_hole_fill_triangles_removed"] == 1


@pytest.mark.parametrize("role", ["floor", "ceiling", "wall", "stair", "column", "ramp"])
@pytest.mark.parametrize("floater_y", [0.0, 0.04])
def test_final_instance_cleanup_keeps_coplanar_blobs_and_removes_non_coplanar_ones(tmp_path, monkeypatch, role, floater_y):
    x, z = np.meshgrid(np.linspace(0., 1., 21), np.linspace(0., 1., 21))
    vertices = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    faces = []
    for row in range(20):
        for col in range(20):
            start = row * 21 + col
            faces.extend(([start, start + 1, start + 22], [start, start + 22, start + 21]))
    vertices = np.vstack((vertices, [[1.24, floater_y, .5], [1.242, floater_y, .5], [1.24, floater_y, .502]]))
    faces = np.vstack((faces, [[441, 442, 443]])).astype(np.int32)
    colors = (np.arange(len(vertices) * 3).reshape(-1, 3) % 256).astype(np.uint8)
    flags = (np.arange(len(faces)) % 2).astype(np.uint8)
    path = tmp_path / f"{role}_inst000.ply"
    assert extract.export_colored_ply(path, vertices, faces, colors,
                                      {"is_hole_fill": ("uchar", flags)})
    original = o3d.io.read_triangle_mesh(str(path))
    original_vertices = np.asarray(original.vertices).copy()
    original_colors = np.asarray(original.vertex_colors).copy()
    source_bytes = path.read_bytes()
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, np.full(len(faces), 7))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"classifications": {"7": {"structural_role": role}}}))
    output = tmp_path / "output"
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=path, labels_path=labels_path, structural_json=metadata_path,
        output_dir=output, min_big_area=1.0, merge_dist_thresh=0.25,
    ))

    assert extract.main() == 0

    manifest = json.loads((output / "structural_instances_manifest.json").read_text())
    item = manifest["instances"][0]
    assert len(manifest["instances"]) == 1
    retained_blob = floater_y == 0.0
    assert item["merged_small_triangles"] == int(retained_blob)
    assert path.read_bytes() == source_bytes
    result_path = output / item["file_name"]
    result = o3d.io.read_triangle_mesh(str(result_path))
    clean_vertices, clean_faces = np.asarray(result.vertices), np.asarray(result.triangles)
    assert clean_vertices[:, 0].max() > 1.0 if retained_blob else clean_vertices[:, 0].max() <= 1.0
    assert len(clean_faces) == (len(faces) if retained_blob else len(faces) - 1)
    assert item["num_vertices"] == len(clean_vertices)
    assert item["num_triangles"] == len(clean_faces)
    assert item["elevation"]["y_max"] == 0.0
    assert item["geometric_cleanup"]["statistical_outlier_filter"] == "disabled"
    assert item["geometric_cleanup"]["non_coplanar_blob_triangles_removed"] == int(not retained_blob)
    original_ids = extract.cKDTree(original_vertices).query(clean_vertices)[1]
    np.testing.assert_array_equal(clean_vertices, original_vertices[original_ids])
    np.testing.assert_array_equal(np.asarray(result.vertex_colors), original_colors[original_ids])
    source_faces = {tuple(face): i for i, face in enumerate(faces)}
    expected_flags = [flags[source_faces[tuple(face)]] for face in original_ids[clean_faces]]
    np.testing.assert_array_equal(extract.read_ply_face_flags(result_path, "is_hole_fill"), expected_flags)


@pytest.mark.parametrize("role", ["floor", "ceiling", "wall", "stair", "column", "ramp"])
def test_long_thin_original_triangles_survive_fill_cleanup(role):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., .001]])
    faces = np.array([[0, 1, 2]])
    original_ids, _ = extract._clean_role_triangles(vertices, faces, np.array([0]), np.array([0]), role)
    fill_ids, _ = extract._clean_role_triangles(vertices, faces, np.array([0]), np.array([1]), role)
    np.testing.assert_array_equal(original_ids, [0])
    assert len(fill_ids) == 0


def test_extraction_adds_neutral_colors_when_source_has_none(tmp_path, monkeypatch):
    mesh_path = tmp_path / "source.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
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
        [0, 2, 0], [2, 2, 0], [0, 2, 2], [10, 3, 0], [12, 3, 0], [10, 3, 2],
        [20, 4, 0], [22, 4, 0], [20, 4, 2], [30, 0, 0], [30, 2, 0], [30, 0, 2],
        [40, 0, 0], [40, 2, 0], [40, 0, 2],
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


def test_extraction_merges_associated_doors_and_windows_to_nearest_wall(tmp_path, monkeypatch):
    mesh_path = tmp_path / "source.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([
        [0, -3, 0], [2, -3, 0], [0, -3, 2],  # floor
        [0, -1, 0], [0, 1, 0], [0, -1, 2],   # wall
        [0, 5, 0], [2, 5, 0], [0, 5, 2],     # ceiling
    ])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, [7, 8, 9])
    structural_json = tmp_path / "structural.json"
    structural_json.write_text(json.dumps({"classifications": {
        "7": {"structural_role": "floor"},
        "8": {"structural_role": "wall"},
        "9": {"structural_role": "ceiling"},
    }}))

    segmentation_dir = tmp_path / "segmentation_output"
    metadata = {"classes": {
        "19": {"structural_role": "object"},
        "29": {"structural_role": "object"},
    }}
    (segmentation_dir / "3d_masks" / "class_019_object").mkdir(parents=True)
    (segmentation_dir / "3d_masks" / "class_029_object").mkdir(parents=True)
    (segmentation_dir / "class_metadata.json").write_text(json.dumps(metadata))
    door = o3d.geometry.TriangleMesh()
    door.vertices = o3d.utility.Vector3dVector([[0.05, -0.2, 0.2], [0.05, 0.2, 0.2], [0.05, -0.2, 0.6]])
    door.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    door_path = segmentation_dir / "3d_masks" / "class_019_object" / "instance_001.ply"
    assert o3d.io.write_triangle_mesh(str(door_path), door)
    window = o3d.geometry.TriangleMesh()
    window.vertices = o3d.utility.Vector3dVector([[0.05, 0.45, 0.2], [0.05, 0.8, 0.2], [0.05, 0.45, 0.6]])
    window.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    window_path = segmentation_dir / "3d_masks" / "class_029_object" / "instance_001.ply"
    assert o3d.io.write_triangle_mesh(str(window_path), window)
    original_object_bytes = {p: p.read_bytes() for p in (door_path, window_path)}

    associations_path = segmentation_dir / "non_structural_similarity" / "valid_class_associations.json"
    associations_path.parent.mkdir(parents=True)
    associations_path.write_text(json.dumps({"associations": [[19, 1], [29, 2]]}))
    output_dir = tmp_path / "output"
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=mesh_path, labels_path=labels_path, structural_json=structural_json,
        associations_json=None, segmentation_output=segmentation_dir,
        output_dir=output_dir, min_big_area=1.0, merge_dist_thresh=0.25,
    ))

    assert extract.main() == 0
    manifest = json.loads((output_dir / "structural_instances_manifest.json").read_text())
    merges = {(item["text_class_id"], item["target_role"]) for item in manifest["door_window_merges"]}
    assert merges == {(1, "wall"), (2, "wall")}
    assert len(np.asarray(o3d.io.read_triangle_mesh(str(output_dir / "wall_inst000.ply")).triangles)) == 3
    assert len(np.asarray(o3d.io.read_triangle_mesh(str(output_dir / "ceiling_inst000.ply")).triangles)) == 1
    assert {p: p.read_bytes() for p in original_object_bytes} == original_object_bytes


def test_late_export_failure_preserves_previous_outputs_and_manifest(tmp_path, monkeypatch):
    mesh_path = tmp_path / "source.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [2, 0, 0], [0, 0, 2],
                                                [10, 2, 0], [12, 2, 0], [10, 2, 2]])
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

    def fail_second_export(path, *_, **__):
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
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, [7])
    missing_metadata = tmp_path / "missing.json"
    monkeypatch.setattr(extract, "parse_args", lambda: argparse.Namespace(
        mesh_path=mesh_path, labels_path=labels_path, structural_json=missing_metadata,
        output_dir=tmp_path / "output", min_big_area=1.0, merge_dist_thresh=0.25,
    ))

    assert extract.main() == 1
    assert "Structural classification metadata not found" in capsys.readouterr().out
