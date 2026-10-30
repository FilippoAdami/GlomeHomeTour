from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

SCRIPT = Path(__file__).parents[1] / "06_mesh_elaboration" / "fit_enclosing_walls_to_regularized_contours.py"
SPEC = importlib.util.spec_from_file_location("fit_enclosing_walls_to_regularized_contours", SCRIPT)
fit_contours = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(fit_contours)


def _write_source_shell(path: Path) -> None:
    vertices = []
    triangles = []

    def add_quad(points):
        start = len(vertices)
        vertices.extend(points)
        triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])

    add_quad([[0, 0, 0], [1, 0, 0], [1, 1.5, 0], [0, 1.5, 0]])
    add_quad([[0, 1.5, 0], [1, 1.5, 0], [1, 3, 0], [0, 3, 0]])
    add_quad([[0, 1.5, 0], [1, 1.5, 0], [1, 1.5, 1], [0, 1.5, 1]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.paint_uniform_color([0.4, 0.5, 0.6])
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_vertex_colors=True)


def test_fits_per_floor_segment_counts_at_source_horizontal_transition(tmp_path: Path):
    workspace = tmp_path / "scan_0"
    elaboration = workspace / "mesh_elaboration"
    levels_dir = elaboration / "intermediate_floor_levels"
    levels_dir.mkdir(parents=True)
    source = elaboration / "wall_enclosing_shell.ply"
    _write_source_shell(source)
    original_bytes = source.read_bytes()
    (elaboration / "wall_planes_manifest.json").write_text(json.dumps({"wall_planes": [{"id": 1}]}))
    (levels_dir / "floor_level_intermediate_manifest.json").write_text(json.dumps({
        "levels": [
            {"level": 0, "elevation_y_median": 0.5},
            {"level": 1, "elevation_y_median": 2.0},
        ],
    }))
    lower = [[0, 0], [2, 0], [2, 2], [0, 2]]
    upper = [[0, 0], [2, 0], [2, 1], [1.5, 1], [1.5, 2], [0, 2]]
    for level, contour in enumerate([lower, upper]):
        (levels_dir / f"floor_level_{level:03d}_polygons.json").write_text(json.dumps({
            "level": level,
            "areas": [{"area_id": 0, "regularized_wall_contour_world_xz_m": contour}],
        }))

    result = fit_contours.fit_enclosing_walls_to_regularized_contours(workspace)

    output = Path(result["output_ply"])
    fitted = o3d.io.read_triangle_mesh(str(output))
    assert output != source and output.is_file()
    assert source.read_bytes() == original_bytes
    assert result["transition_heights_y_m"] == [1.5]
    assert [floor["wall_plane_count"] for floor in result["floors"]] == [4, 6]
    assert result["plane_count"] == 10
    vertices = np.asarray(fitted.vertices)
    assert vertices[:, 1].min() == 0.0 and vertices[:, 1].max() == 3.0
    assert np.count_nonzero(np.isclose(vertices[:, 1], 1.5)) >= 18


def test_fitted_wall_reapplies_detected_relief_toward_contour_interior():
    x_axis = np.linspace(0.01, 1.99, 60)
    y_axis = np.linspace(0.01, 1.99, 20)
    vertices = np.array([
        [x, y, 0.025 * np.sin(2 * np.pi * x / 0.25)]
        for y in y_axis for x in x_axis
    ])
    triangles = []
    for row in range(len(y_axis) - 1):
        for column in range(len(x_axis) - 1):
            lower = row * len(x_axis) + column
            upper = lower + len(x_axis)
            triangles.extend([[lower, lower + 1, upper + 1], [lower, upper + 1, upper]])
    source = o3d.geometry.TriangleMesh()
    source.vertices = o3d.utility.Vector3dVector(vertices)
    source.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    source.paint_uniform_color([0.5, 0.5, 0.5])
    floors = [{
        "level": 0,
        "floor_y_m": 0.0,
        "areas": [{"area_id": 0, "contour": np.array([[0, 0], [2, 0], [2, 2], [0, 2]])}],
    }]

    fitted, _ = fit_contours._fit_panels(source, floors, [(0.0, 2.0)])
    wall = np.asarray(fitted.vertices)
    front_panel = wall[(wall[:, 0] > 0.1) & (wall[:, 0] < 1.9)
                       & (wall[:, 1] > 0.1) & (wall[:, 1] < 1.9)]

    assert np.max(front_panel[:, 2]) > 0.015
    assert np.min(front_panel[:, 2]) >= -1e-9


def test_transition_mesh_splits_t_junction_edges_without_opening_the_seam():
    vertices = np.array([
        [0, 0, 0], [0, 0, 2], [0, 1, 2], [0, 1, 0],
        [0, 1, 0], [0, 1, 1], [0, 1, 2],
        [0, 2, 0], [0, 2, 1], [0, 2, 2],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 8], [4, 8, 7], [5, 6, 9], [5, 9, 8],
    ], dtype=np.int32)
    colors = np.vstack((np.tile([1., 0., 0.], (4, 1)),
                        np.tile([0., 0., 1.], (6, 1))))

    vertices, faces, colors = fit_contours._conform_transition_edges(
        vertices, faces, colors, [1.0]
    )
    welded = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    welded.merge_vertices(digits_vertex=6)
    edge_counts = np.bincount(welded.edges_unique_inverse)
    boundary = welded.vertices[welded.edges_unique[edge_counts == 1]]

    assert not np.any(np.all(np.isclose(boundary[:, :, 1], 1.0, atol=1e-6), axis=1))
    assert not np.any(edge_counts > 2)
    assert len(faces) > 6
