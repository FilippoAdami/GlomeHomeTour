"""Focused checks for the Stage 06 wall-region diagnostic step."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
MODULE = Path(__file__).parents[1] / "06_mesh_elaboration" / "grow_wall_regions.py"
SPEC = importlib.util.spec_from_file_location("grow_wall_regions", MODULE)
grow_wall_regions = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(grow_wall_regions)


def test_region_growing_splits_turns_and_keeps_horizontal_triangles():
    # Two vertical quads meet at 90 degrees; one horizontal triangle meets their edge.
    vertices = np.asarray([
        [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1],
        [1, 0, 0], [1, 1, 0], [0, 1, 2],
    ], dtype=float)
    triangles = np.asarray([
        [0, 1, 2], [0, 2, 3],  # x=0 wall
        [0, 4, 5], [0, 5, 1],  # z=0 return
        [2, 6, 5],             # horizontal face
    ])

    labels, _, _, _ = grow_wall_regions._grow_regions(vertices, triangles)

    assert len(labels) == len(triangles)
    assert len(np.unique(labels)) == 3
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert labels[4] not in (labels[0], labels[2])


def test_neighbor_clusters_merge_when_closest_orthogonal_offset_is_within_limit():
    vertices = np.asarray([
        [0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 11, .3],
    ], dtype=float)
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]])

    labels, normals, _, black = grow_wall_regions._grow_regions(vertices, triangles, seed_count=1)

    assert abs(float(normals[0] @ normals[1])) > np.cos(np.deg2rad(35))
    # One directed projection is large because the two fits are angled; the closest is 10 cm.
    assert labels[0] == labels[1]
    assert black.all()


def test_growth_compares_new_faces_to_the_evolving_cluster_normal():
    # The end triangles differ by 38 degrees, but each is within 35 degrees
    # of the cluster average after the middle triangle has joined.
    angles = np.deg2rad([-19, 0, 19])
    vertices = np.asarray([
        [0, 0, 0], [1, 0, 0],
        *[[.5, .5 * np.cos(angle), .5 * np.sin(angle)] for angle in angles],
    ])
    triangles = np.asarray([[0, 1, 2], [0, 1, 3], [0, 1, 4]])

    labels, _, _, _ = grow_wall_regions._grow_regions(vertices, triangles, seed_count=1)

    assert len(np.unique(labels)) == 1


def test_small_regions_merge_to_the_most_similar_neighbor():
    vertices = np.asarray([
        [0, 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, .12],
    ], dtype=float)
    triangles = np.asarray([[0, 1, 2], [1, 3, 2]])

    labels, normals, _, black = grow_wall_regions._grow_regions(vertices, triangles, seed_count=1)

    assert np.rad2deg(np.arccos(abs(float(normals[0] @ normals[1])))) > 35
    assert labels[0] == labels[1]
    assert black.all()


def test_disconnected_wall_patches_start_in_separate_regions():
    vertices = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 0, .5], [1, 0, .5], [0, 1, .5],
    ], dtype=float)
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]])

    labels, _, _, black = grow_wall_regions._grow_regions(vertices, triangles)

    assert labels[0] != labels[1]
    assert black.all()


def test_large_detached_regions_do_not_merge_when_their_planes_align(monkeypatch):
    monkeypatch.setattr(grow_wall_regions, "MIN_REGION_TRIANGLES", 1)
    angle = np.deg2rad(13.0)
    center_a = np.zeros(3)
    center_b = np.asarray([
        (0.5 - 0.05 * np.cos(angle)) / np.sin(angle), 0.0, 0.05,
    ])
    tangent = np.asarray([0.0, 1.0, 0.0])
    tangent_a = np.asarray([-1.0, 0.0, 0.0])
    tangent_b = np.asarray([-np.cos(angle), 0.0, np.sin(angle)])

    def triangle(center, other_tangent):
        return [center + tangent, center + other_tangent, center - tangent - other_tangent]

    vertices = np.asarray(triangle(center_a, tangent_a) + triangle(center_b, tangent_b))
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]])

    labels, normals, _, black = grow_wall_regions._grow_regions(vertices, triangles, seed_count=2)

    delta = center_b - center_a
    plane_distance = min(abs(float(normals[0] @ delta)), abs(float(normals[1] @ delta)))
    assert np.rad2deg(np.arccos(abs(float(normals[0] @ normals[1])))) == pytest.approx(13.0)
    assert plane_distance < grow_wall_regions.MAX_PLANE_DISTANCE
    assert np.linalg.norm(delta) > 1.0
    assert labels[0] != labels[1]
    assert not black.any()


def test_cli_writes_geometry_preserving_colored_copy_and_manifest(tmp_path: Path, monkeypatch):
    source = tmp_path / "wall_inst000.ply"
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0],
    ], dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2], [0, 3, 1]]))
    assert o3d.io.write_triangle_mesh(str(source), mesh)
    (tmp_path / "structural_instances_manifest.json").write_text(json.dumps({
        "instances": [{"role": "wall", "file_name": source.name}],
    }))
    monkeypatch.setattr("sys.argv", ["grow_wall_regions.py", "--instances-dir", str(tmp_path)])

    assert grow_wall_regions.main() == 0
    colored = o3d.io.read_triangle_mesh(str(tmp_path / "wall_regions_colored.ply"))
    manifest = json.loads((tmp_path / "wall_regions_manifest.json").read_text())
    labels = np.load(tmp_path / "wall_regions_triangle_labels.npy")

    assert len(colored.triangles) == 2
    assert len(colored.vertex_colors) == 6
    assert np.all(np.asarray(colored.vertex_colors) == 0)
    assert manifest["triangle_count"] == 2
    assert manifest["region_count"] == 2
    assert manifest["triangle_labels_file"] == "wall_regions_triangle_labels.npy"
    assert labels.shape == (2,)
    assert len(np.unique(labels)) == 2
    assert manifest["normal_angle_threshold_deg"] == 35
    assert manifest["small_region_merge_min_triangles"] == 10000
    assert all(region["painted_black"] for region in manifest["regions"])
