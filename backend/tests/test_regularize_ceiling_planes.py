import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest
import trimesh
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parents[1] / "06_mesh_elaboration"))
import regularize_ceiling_planes as ceiling_fitting
from regularize_ceiling_planes import (
    _clearance_adjustment,
    _enforce_wall_top_constraint,
    _extracted_wall_meshes,
    _fit_plane_above_wall_top,
    _fit_group_patches,
    _is_upper_wall_ceiling,
    _plane_mesh,
    _projected_footprint,
    _source_support,
    _vertical_junctions,
    _wall_contours_at,
    _wall_top_plane,
)
from fill_mesh_holes import export_ply_with_face_attributes
from structural_geometry import polygon_triangles


def test_associated_ceiling_keeps_natural_gap_and_only_lifts_to_ten_cm_when_needed():
    footprint = Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    floor = np.array([0.0, 1.0, 0.0, -3.0])

    unchanged, translation, gap, final_gap = _clearance_adjustment(
        footprint, np.array([0.0, 1.0, 0.0, -2.7]), floor
    )
    np.testing.assert_allclose(unchanged, [0, 1, 0, -2.7])
    np.testing.assert_allclose(translation, [0, 0, 0])
    assert gap == pytest.approx(0.30)
    assert final_gap == pytest.approx(0.30)

    moved, translation, gap, final_gap = _clearance_adjustment(
        footprint, np.array([0.0, 1.0, 0.0, -2.95]), floor
    )
    assert gap == pytest.approx(0.05)
    assert final_gap == pytest.approx(0.10)
    np.testing.assert_allclose(translation, [0, -0.05, 0], atol=1e-10)
    np.testing.assert_allclose(moved, [0, 1, 0, -2.90], atol=1e-10)

    sloped_floor = np.array([0.1, np.sqrt(0.99), 0.0, -3.0])
    moved, translation, gap, final_gap = _clearance_adjustment(
        Polygon([(0, 0), (2, 0), (2, 1), (0, 1)]),
        np.array([0.0, 1.0, 0.0, -2.95]), sloped_floor,
    )
    assert gap == 0.0
    assert final_gap == pytest.approx(0.10)
    points = np.array([[0, 2.95, 0], [2, 2.95, 0], [2, 2.95, 1], [0, 2.95, 1]])
    assert np.min((points + translation) @ sloped_floor[:3] + sloped_floor[3]) >= 0.10 - 1e-9


def test_wall_slice_recovers_closed_contour_from_vertical_panels():
    vertices, triangles = [], []
    corners = [(0, 0), (2, 0), (2, 1), (0, 1)]
    for first, second in zip(corners, corners[1:] + corners[:1]):
        start = len(vertices)
        vertices.extend([
            [first[0], 0, first[1]], [second[0], 0, second[1]],
            [second[0], 2, second[1]], [first[0], 2, first[1]],
        ])
        triangles.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
    walls = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )

    contour = _wall_contours_at(walls, 1.0)

    assert contour.area == pytest.approx(2.0)
    assert contour.bounds == pytest.approx((0, 0, 2, 1))


def test_wall_top_plane_follows_a_sloped_upper_boundary():
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    vertices = np.vstack((
        np.column_stack((corners[:, 0], np.zeros(4), corners[:, 1])),
        np.column_stack((corners[:, 0], 3.0 + 0.2 * corners[:, 0] - 0.1 * corners[:, 1], corners[:, 1])),
    ))
    faces = []
    for edge in range(4):
        following = (edge + 1) % 4
        a, b = edge, following
        c, d = following + 4, edge + 4
        faces.extend(([a, b, c], [a, c, d]))
    wall = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )

    equation, info = _wall_top_plane(wall)

    predicted = -(equation[0] * vertices[4:, 0]
                  + equation[2] * vertices[4:, 2] + equation[3]) / equation[1]
    np.testing.assert_allclose(predicted, vertices[4:, 1] + 0.02, atol=1e-8)
    assert info["boundary_vertex_count"] >= 4
    assert info["fit_sample_count"] == 4
    assert info["fit_rmse_m"] == pytest.approx(0.02, abs=1e-8)


def test_wall_top_plane_uses_all_wall_sides_instead_of_the_highest_loop_strip():
    corners = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    meshes = []
    top_points = []
    for first, second in zip(corners, np.roll(corners, -1, axis=0)):
        first_y = 3.0 + 0.15 * first[0] - 0.25 * first[1]
        second_y = 3.0 + 0.15 * second[0] - 0.25 * second[1]
        points = np.array([
            [first[0], 0., first[1]], [second[0], 0., second[1]],
            [second[0], second_y, second[1]], [first[0], first_y, first[1]],
        ])
        top_points.extend(points[[2, 3]])
        meshes.append(o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)),
        ))

    equation, info = _wall_top_plane(meshes)

    top_points = np.unique(np.asarray(top_points), axis=0)
    predicted = -(equation[0] * top_points[:, 0]
                  + equation[2] * top_points[:, 2] + equation[3]) / equation[1]
    np.testing.assert_allclose(predicted, top_points[:, 1] + 0.02, atol=1e-7)
    assert info["fit_sample_count"] == 4
    assert len(info["boundary_loops"]) == 4


def test_wall_top_plane_uses_extracted_pli_contour_and_ignores_small_high_outlier(tmp_path):
    records, top_points = [], []
    for index, (first, second) in enumerate((((-1., -1.), (1., -1.)),
                                             ((-1., 1.), (1., 1.)),
                                             ((-1., -1.), (-1., 1.)))):
        vertices = []
        for x, z in (first, second):
            vertices.append([x, 0., z])
        for x, z in (first, second):
            y = 3.0 + 0.2 * x - 0.1 * z
            vertices.append([x, y, z])
            top_points.append([x, y, z])
        path = tmp_path / f"wall_inst{index:03d}.ply"
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices)),
            o3d.utility.Vector3iVector(np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32)),
        )
        assert o3d.io.write_triangle_mesh(str(path), mesh)
        records.append({"role": "wall", "file_name": path.name})

    outlier_path = tmp_path / "wall_inst003.ply"
    outlier = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[10., 5., 10.], [10.01, 5., 10.], [10., 5., 10.01]]),
        o3d.utility.Vector3iVector([[0, 1, 2]]),
    )
    assert o3d.io.write_triangle_mesh(str(outlier_path), outlier)
    records.append({"role": "wall", "file_name": outlier_path.name})
    (tmp_path / "structural_instances_manifest.json").write_text(
        json.dumps({"instances": records}),
    )

    meshes, paths = _extracted_wall_meshes(tmp_path)
    equation, info = _wall_top_plane(meshes)

    top_points = np.asarray(top_points)
    predicted = -(equation[0] * top_points[:, 0]
                  + equation[2] * top_points[:, 2] + equation[3]) / equation[1]
    np.testing.assert_allclose(predicted, top_points[:, 1] + 0.02, atol=1e-6)
    assert paths == [tmp_path / record["file_name"] for record in records]
    assert info["fit_sample_count"] >= 3
    assert len(info["boundary_loops"]) == 3


def test_ceiling_is_raised_only_enough_to_clear_wall_top_plane():
    polygon = Polygon([(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)])
    ceiling_equation = np.array([0., 1., 0., -2.])
    baseline = np.array([-0.2, 1., 0., -3.])
    mesh = _plane_mesh(polygon, ceiling_equation)
    patch = {
        "polygon": polygon,
        "mesh": mesh,
        "plane_equation": ceiling_equation.tolist(),
        "translation": np.zeros(3),
    }

    _enforce_wall_top_constraint(patch, baseline)

    assert patch["wall_top_vertical_translation_m"] == pytest.approx(1.2)
    np.testing.assert_allclose(patch["plane_equation"], [0., 1., 0., -3.2])
    assert patch["wall_top_clearance_after_m"] == pytest.approx(0., abs=1e-10)
    assert patch["wall_top_constraint_applied"] is True
    assert np.min(mesh.vertices[:, 1] - (3.0 + 0.2 * mesh.vertices[:, 0])) >= -1e-10


def test_constrained_ceiling_fit_uses_the_wall_top_plane_without_raising_paired_floors():
    polygon = Polygon([(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)])
    wall_top = np.array([-0.2, 1., 0., -3.])

    fitted, stats = _fit_plane_above_wall_top(
        np.array([0., 1., 0., -2.]), polygon, wall_top,
    )

    boundary = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    assert np.min(
        -(fitted[0] * boundary[:, 0] + fitted[2] * boundary[:, 1] + fitted[3]) / fitted[1]
        - (3.0 + 0.2 * boundary[:, 0])
    ) >= -1e-9
    np.testing.assert_allclose(fitted, wall_top / np.linalg.norm(wall_top[:3]), atol=1e-8)
    assert stats["applied"] is True
    assert stats["maximum_boundary_adjustment_m"] == pytest.approx(1.2)


def test_constrained_ceiling_fit_keeps_a_feasible_solver_result_even_if_status_is_false(monkeypatch):
    polygon = Polygon([(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)])
    wall_top = np.array([-.2, 1., 0., -3.])
    monkeypatch.setattr(ceiling_fitting, "minimize", lambda *args, **kwargs: SimpleNamespace(
        success=False, x=np.array([0., 0., 3.2]),
    ))

    fitted, stats = _fit_plane_above_wall_top(np.array([0., 1., 0., -3.]), polygon, wall_top)

    np.testing.assert_allclose(fitted, [0., 1., 0., -3.2])
    assert stats["optimizer_success"] is False
    assert stats["optimizer_feasible"] is True
    assert stats["fallback_to_wall_top_plane"] is False


def test_wall_top_constraint_does_not_raise_intermediate_paired_ceilings():
    assert not _is_upper_wall_ceiling({"wall_top_max_clearance_before_m": -2.16})
    assert _is_upper_wall_ceiling({"wall_top_max_clearance_before_m": -0.002})


def test_adjacent_ceiling_planes_get_a_vertical_junction():
    first_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    second_polygon = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    planes = ([0, 1, 0, -2.0], [0, 1, 0, -1.8])
    patches = []
    for group_id, polygon, equation in zip((0, 1), (first_polygon, second_polygon), planes):
        mesh = _plane_mesh(polygon, np.asarray(equation, dtype=float))
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh, face_colors=np.tile([180, 180, 180, 255], (len(mesh.faces), 1))
        )
        patches.append({
            "group_id": group_id, "floor_level": 0, "level_key": "floor:0",
            "polygon": polygon, "plane_equation": equation, "mesh": mesh,
        })

    junctions = _vertical_junctions(patches)

    assert len(junctions) == 1
    assert junctions[0]["mesh"].area == pytest.approx(0.2)
    assert junctions[0]["is_junction"]


def test_downprojection_union_preserves_source_holes():
    source = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)],
                     holes=[[(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]])
    xz, faces = polygon_triangles(source)
    vertices = np.column_stack((xz[:, 0], np.full(len(xz), 2.0), xz[:, 1]))

    footprint = _projected_footprint(vertices, faces)

    assert footprint.area == pytest.approx(3.0)
    assert len(footprint.interiors) == 1


def test_pattern_support_uses_only_original_samples_inside_the_final_contour():
    record = {
        "group": {"group_id": 3, "associated_floor_level": None},
        "equation": np.array([0.0, 1.0, 0.0, -2.0]),
        "level_key": "orphan:0",
        "source": {
            "support_centers": np.array([[0.5, 2.0, 0.5], [1.5, 2.0, 0.5]]),
            "support_areas": np.array([1.0, 2.0]),
            "support_colors": np.array([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]]),
            "source_triangle_count": 2,
        },
    }
    support = {
        "polygon": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        "area_id": None, "floor_equation": None, "support_source": "wall_slice",
        "support_height_y_m": 1.7,
    }

    patch = _fit_group_patches(record, support)

    assert patch["source_triangle_count"] == 1
    np.testing.assert_allclose(patch["support_centers"], [[0.5, 2.0, 0.5]])
    np.testing.assert_allclose(patch["support_areas"], [1.0])


def test_associated_ceiling_support_uses_cleaned_ceiling_role_inside_group_footprint(tmp_path):
    vertices = np.array([
        [0, 2, 0], [1, 2, 0], [0, 2, 1],
        [0.1, 2, 0.1], [0.9, 2, 0.1], [0.1, 2, 0.9],
        [2, 2, 2], [3, 2, 2], [2, 2, 3],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
    colors = np.array([
        [255, 0, 0], [255, 0, 0], [255, 0, 0],
        [0, 255, 0], [0, 255, 0], [0, 255, 0],
        [0, 0, 255], [0, 0, 255], [0, 0, 255],
    ], dtype=np.uint8)
    source = tmp_path / "cleaned_floor_level.ply"
    assert export_ply_with_face_attributes(
        source, vertices, faces, colors,
        face_attributes={
            "surface_role": ("uchar", np.array([2, 0, 2], dtype=np.uint8)),
            "is_fitted_plane": ("uchar", np.zeros(3, dtype=np.uint8)),
            "is_hole_fill": ("uchar", np.zeros(3, dtype=np.uint8)),
        },
    )

    support = _source_support(
        source, np.array([0.0, 1.0, 0.0, -2.0]), surface_role=2,
        footprint=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    )

    assert support["source_triangle_count"] == 1
    np.testing.assert_allclose(support["support_colors"], [[1.0, 0.0, 0.0]])
