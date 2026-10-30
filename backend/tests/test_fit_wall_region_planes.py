"""Checks trimmed region plane fitting and translucent GLB output."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

o3d = pytest.importorskip("open3d")
MODULE = Path(__file__).parents[1] / "06_mesh_elaboration" / "fit_wall_region_planes.py"
SPEC = importlib.util.spec_from_file_location("fit_wall_region_planes", MODULE)
fit_wall_region_planes = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fit_wall_region_planes)


def test_fits_each_cluster_and_trims_at_most_one_percent_globally(tmp_path: Path):
    region_dir = tmp_path / "wall_regions"
    region_dir.mkdir()
    vertices, triangles, labels, colors = [], [], [], []

    def add_triangle(x: float, y: float, region_id: int, color: tuple[float, float, float]):
        start = len(vertices)
        vertices.extend([[x, y, 0], [x, y + 0.05, 0], [x, y, 0.05]])
        triangles.append([start, start + 1, start + 2])
        labels.append(region_id)
        colors.extend([color] * 3)

    for i in range(50):
        add_triangle(0.0, i * 0.1, 0, (1.0, 0.0, 0.0))
    add_triangle(0.3, 5.0, 0, (1.0, 0.0, 0.0))
    for i in range(50):
        add_triangle(2.0, i * 0.1, 1, (0.0, 1.0, 0.0))
    add_triangle(4.0, 0.0, 2, (0.0, 0.0, 0.0))
    start = len(vertices)
    vertices.extend([[6.0, 0.0, 0.0], [6.0, 0.1, 0.0], [6.0, 0.2, 0.0]])
    triangles.append([start, start + 1, start + 2])
    labels.append(3)
    colors.extend([(0.0, 0.0, 0.0)] * 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors))
    mesh_path = region_dir / "wall_regions_colored.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_vertex_colors=True)
    np.save(region_dir / "wall_regions_triangle_labels.npy", np.asarray(labels, dtype=np.int32))
    (region_dir / "wall_regions_manifest.json").write_text(json.dumps({
        "triangle_count": len(triangles),
        "regions": [
            {"region_id": 0, "color_rgb": [255, 0, 0], "painted_black": False},
            {"region_id": 1, "color_rgb": [0, 255, 0], "painted_black": False},
            {"region_id": 2, "color_rgb": [0, 0, 0], "painted_black": True},
            {"region_id": 3, "color_rgb": [0, 0, 0], "painted_black": True},
        ],
    }))

    output_glb, output_manifest, result = fit_wall_region_planes._process(region_dir)

    assert output_glb.is_file() and output_manifest.is_file()
    assert Path(result["combined_ply"]).is_file()
    assert result["plane_count"] == 2
    assert result["outlier_triangle_count"] == 1
    assert result["outlier_fraction"] <= 0.01
    assert result["enclosure_outside_triangle_count"] <= result["enclosure_outside_triangle_budget"]
    assert sum(plane["triangle_count"] for plane in result["planes"]) == 101
    assert sum(plane["outside_triangle_count"] for plane in result["planes"]) == 0
    assert result["black_region_triangle_count"] == 2
    assert result["unassigned_triangle_count"] == 3
    assert [region["region_id"] for region in result["excluded_black_regions"]] == [2, 3]
    first_plane = next(plane for plane in result["planes"] if plane["region_id"] == 0)
    assert abs(first_plane["normal"][0]) == pytest.approx(1.0, abs=1e-5)
    assert abs(first_plane["offset"]) == pytest.approx(0.0, abs=1e-4)
    assert {plane["region_id"] for plane in result["planes"]} == {0, 1}

    scene = trimesh.load(output_glb, force="scene")
    assert len(scene.geometry) == 2
    expected_colors = {
        "wall_region_000": [255, 0, 0],
        "wall_region_001": [0, 255, 0],
    }
    for name, plane in scene.geometry.items():
        material = plane.visual.material
        assert material.alphaMode == "BLEND"
        assert list(map(int, material.baseColorFactor[:3])) == expected_colors[name]
        assert material.baseColorFactor[3] == 128

    combined = trimesh.load(result["combined_ply"], force="mesh", process=False)
    assert len(combined.faces) == len(triangles) + 2 * result["plane_count"]
    assert combined.visual.vertex_colors.shape[1] == 4
    assert np.count_nonzero(combined.visual.vertex_colors[:, 3] == 128) == 4 * result["plane_count"]


def test_snaps_wall_directions_and_merges_only_close_parallel_planes():
    vertices, triangles, labels, fitted = [], [], [], []

    def add_square(region_id: int, axis: str, coordinate: float, normal: list[float]):
        start = len(vertices)
        if axis == "x":
            vertices.extend([[coordinate, 0, 0], [coordinate, 1, 0],
                             [coordinate, 1, 1], [coordinate, 0, 1]])
        elif axis == "z":
            vertices.extend([[0, 0, coordinate], [1, 0, coordinate],
                             [1, 1, coordinate], [0, 1, coordinate]])
        else:
            vertices.extend([[0, coordinate, 0], [1, coordinate, 0],
                             [1, coordinate, 1], [0, coordinate, 1]])
        triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
        labels.extend([region_id, region_id])
        fitted.append({
            "region_id": region_id,
            "normal": normal,
            "fit_area_scene_units_squared": 1.0,
            "color_rgb": [region_id * 20, 100, 200],
        })

    def add_diagonal_square(region_id: int, normal: np.ndarray):
        start = len(vertices)
        center = normal * 3.0
        axis_u = np.asarray([1.0, 0.0, -1.0]) / np.sqrt(2.0)
        axis_v = np.asarray([0.0, 1.0, 0.0])
        vertices.extend([
            (center - axis_u - axis_v).tolist(),
            (center + axis_u - axis_v).tolist(),
            (center + axis_u + axis_v).tolist(),
            (center - axis_u + axis_v).tolist(),
        ])
        triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
        labels.extend([region_id, region_id])
        fitted.append({
            "region_id": region_id,
            "normal": normal.tolist(),
            "fit_area_scene_units_squared": 1.0,
            "color_rgb": [region_id * 20, 100, 200],
        })

    add_square(10, "x", 0.0, [0.999, 0.0, 0.04])
    add_square(11, "x", 0.10, [0.998, 0.0, -0.063])
    add_square(12, "z", 2.0, [0.02, 0.0, 0.9998])
    add_square(13, "y", 2.0, [0.02, 0.999, 0.01])
    add_diagonal_square(14, np.asarray([1.0, 0.0, 1.0]) / np.sqrt(2.0))
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    snapped, planes, info = fit_wall_region_planes._snap_and_merge_planes(
        vertices, triangles, labels, np.ones(len(triangles), dtype=bool), fitted
    )

    assert len(snapped) == 4
    assert len(planes) == 3
    assert info["max_snap_rotation_degrees"] == 30.0
    assert [item["region_id"] for item in info["excluded_before_snap"]] == [14]
    assert info["excluded_before_snap"][0]["rotation_degrees"] > 30.0
    merged = next(plane for plane in planes if plane["source_region_ids"] == [10, 11])
    orthogonal = next(plane for plane in planes if plane["source_region_ids"] == [12])
    horizontal = next(plane for plane in planes if plane["source_region_ids"] == [13])
    assert abs(np.dot(merged["normal"], orthogonal["normal"])) < 1e-8
    assert abs(np.dot(merged["normal"], horizontal["normal"])) < 1e-8
    assert abs(np.dot(orthogonal["normal"], horizontal["normal"])) < 1e-8
    assert info["parallel_merge_distance_scene_units"] == 0.15


def test_merge_is_blocked_when_it_would_collapse_a_previously_separate_gap():
    vertices, triangles, labels, fitted = [], [], [], []

    def add_x_square(region_id: int, x: float, width: float):
        start = len(vertices)
        vertices.extend([[x, 0, 0], [x, width, 0],
                         [x, width, width], [x, 0, width]])
        triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
        labels.extend([region_id, region_id])
        fitted.append({
            "region_id": region_id,
            "normal": [1.0, 0.0, 0.0],
            "fit_area_scene_units_squared": width * width,
            "color_rgb": [region_id * 20, 100, 200],
        })

    add_x_square(1, 0.0, 1.0)
    add_x_square(2, 0.14, 1.5)
    add_x_square(3, 0.28, 1.0)
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    _, planes, info = fit_wall_region_planes._snap_and_merge_planes(
        vertices, triangles, labels, np.ones(len(triangles), dtype=bool), fitted
    )

    assert {tuple(plane["source_region_ids"]) for plane in planes} == {(1,), (2,), (3,)}
    assert info["merge_rejections"] == [{
        "source_region_ids": [2, 3],
        "nearby_region_id": 1,
        "proposed_offset": pytest.approx(-0.14),
        "proposed_distance_scene_units": pytest.approx(0.14),
        "previous_distances_scene_units": [
            {"region_id": 2, "distance": pytest.approx(0.14)},
            {"region_id": 3, "distance": pytest.approx(0.28)},
        ],
    }]


def test_enclosing_translation_limits_outside_triangles_to_one_percent():
    vertices, triangles = [], []
    for i, x in enumerate([-0.5, *[i / 1000 for i in range(100)], 0.5]):
        start = len(vertices)
        y = i * 0.02
        vertices.extend([[x, y, 0], [x, y + 0.05, 0], [x, y, 0.05]])
        triangles.append([start, start + 1, start + 2])
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    v0 = vertices[triangles[:, 0]]
    areas = np.linalg.norm(np.cross(vertices[triangles[:, 1]] - v0,
                                    vertices[triangles[:, 2]] - v0), axis=1) * 0.5

    offset, inside, outside, side, _, mesh_inside_fraction = fit_wall_region_planes._enclose_plane(
        vertices, triangles, np.arange(len(triangles)), areas,
        np.asarray([1.0, 0.0, 0.0]), -0.05,
        np.asarray([[10.0, 0.0, 0.0]]), np.asarray([1.0]),
    )

    assert len(outside) <= int(len(triangles) * 0.01)
    assert side == "negative_normal"
    assert mesh_inside_fraction == pytest.approx(1.0)
    projections = vertices[triangles[inside]] @ np.asarray([1.0, 0.0, 0.0])
    boundary = -offset
    if side == "positive_normal":
        assert np.all(projections.max(axis=1) <= boundary)
    else:
        assert np.all(projections.min(axis=1) >= boundary)
