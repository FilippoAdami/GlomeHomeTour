"""Checks for parallel ceiling and floor cut planes in floor projections."""
import importlib.util
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

pytest.importorskip("open3d")
MODULE = Path(__file__).parents[1] / "08_artifacts_generation" / "extract_floor_ceiling_residual.py"
SPEC = importlib.util.spec_from_file_location("extract_floor_ceiling_residual", MODULE)
projection = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(projection)


def test_tilted_ceiling_clearance_is_perpendicular_and_follows_tilt():
    angle = np.deg2rad(20.0)
    normal = np.array([np.sin(angle), -np.cos(angle), 0.0])
    ceiling = np.r_[normal, -normal[1]]  # passes through (0, 1, 0)

    cut = projection._offset_plane(ceiling, projection.PROJECTION_CEILING_CLEARANCE_M)
    point_on_ceiling = np.array([0.0, 1.0, 0.0, 1.0])

    assert np.isclose(abs(cut @ point_on_ceiling), 0.15)
    cut_y = projection._plane_y(cut, np.array([[0.0, 0.0, 0.0]]))[0]
    assert np.isclose(cut_y, 1.0 - 0.15 / np.cos(angle))


def test_floor_fallback_offset_goes_downward():
    floor = np.array([0.0, 1.0, 0.0, 0.0])

    cut = projection._offset_plane(floor, -0.10)

    assert np.isclose(projection._plane_y(cut, np.array([[0.0, 0.0, 0.0]]))[0], -0.10)


def test_uses_nearest_ceiling_beneath_floor_only_within_40_cm():
    level = {"mean_y_m": 1.0}
    ceilings = [
        {"mean_y_m": 0.2, "component": 1},
        {"mean_y_m": 0.7, "component": 2},
        {"mean_y_m": 0.59, "component": 3},
        {"mean_y_m": 1.1, "component": 4},
    ]

    assert projection._nearest_lower_ceiling(level, ceilings)["component"] == 2
    assert projection._nearest_lower_ceiling(level, [ceilings[0]]) is None


def test_projection_source_excludes_removed_faces_and_every_wall_class():
    labels = np.array([2, 5, 8, 5, 3])
    retained = np.array([True, True, False, True, True])

    selected = projection._projection_source_mask(retained, labels, [5])

    np.testing.assert_array_equal(selected, [True, False, False, False, True])


def test_pre_collapse_mesh_preserves_original_triangle_heights(tmp_path: Path):
    vertices = np.array([[0.0, -1.0, 0.0], [1.0, 0.5, 0.0], [0.0, 2.0, 1.0]])
    path = tmp_path / "floor_pre_collapse.ply"

    projection._write_pre_collapse_mesh(path, vertices, np.array([[0, 1, 2]]))

    saved = o3d.io.read_triangle_mesh(str(path))
    assert len(saved.triangles) == 1
    np.testing.assert_allclose(np.asarray(saved.vertices)[:, 1], vertices[:, 1])


def test_projection_includes_full_floor_and_lower_ceiling_but_removes_upper_ceiling():
    selected = projection._select_projection_faces(
        below_lower_limit=np.array([True, True, False, False]),
        above_upper_limit=np.array([False, False, False, True]),
        upper_floor_faces=np.zeros(4, dtype=bool),
        floor_instance_faces=np.array([True, False, False, False]),
        lower_ceiling_instance_faces=np.array([False, True, False, False]),
        upper_ceiling_instance_faces=np.array([False, False, True, False]),
    )

    np.testing.assert_array_equal(selected, [True, True, False, False])


def test_hole_fill_patch_inherits_its_only_surrounding_class():
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.5, 0.0, -1.0], [2.0, 0.0, 0.5], [0.5, 0.0, 2.0], [-1.0, 0.0, 0.5],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [1, 0, 4], [2, 1, 5], [3, 2, 6], [0, 3, 7],
    ])
    labels, stats = projection._propagate_hole_fill_classes(
        vertices, faces, np.array([-1, -1, 7, 7, 7, 7]), np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    )

    np.testing.assert_array_equal(labels[:2], [7, 7])
    assert stats["patch_count"] == 1
    assert stats["patches"][0]["assignment_method"] == "single_neighbor_class"


def test_hole_fill_patch_with_multiple_neighbors_chooses_coplanar_class():
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.5], [1.0, 1.0, 0.5],
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3], [3, 0, 4], [1, 2, 5]])
    labels, stats = projection._propagate_hole_fill_classes(
        vertices, faces, np.array([-1, -1, 2, 5]), np.array([1, 1, 0, 0], dtype=bool)
    )

    np.testing.assert_array_equal(labels[:2], [2, 2])
    assert stats["patches"][0]["assignment_method"] == "most_confining_coplanar_neighbor"


def test_height_cut_rejects_triangles_that_cross_either_cut_plane():
    vertices = np.array([
        [0.0, 0.9, 0.0], [1.0, 0.9, 0.0], [0.0, 0.9, 1.0],
        [2.0, 0.9, 0.0], [3.0, 1.1, 0.0], [2.0, 0.9, 1.0],
        [4.0, 0.8, 0.0], [5.0, 0.9, 0.0], [4.0, 0.9, 1.0],
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    above = projection._triangles_outside_plane(
        vertices, faces, np.array([0.0, 1.0, 0.0, -1.0]), above=True
    )
    below = projection._triangles_outside_plane(
        vertices, faces, np.array([0.0, 1.0, 0.0, -0.85]), above=False
    )

    np.testing.assert_array_equal(above, [False, True, False])
    np.testing.assert_array_equal(below, [False, False, True])


def test_floor_level_carries_all_source_faces_from_its_plane_components():
    planes = [
        {"kind": "surface", "mean_y_m": 0.0, "plane_area_m2": 2.0,
         "component": 1, "component_area_m2": 2.0, "component_face_ids": np.array([1, 3, 5])},
        {"kind": "surface", "mean_y_m": 0.01, "plane_area_m2": 1.0,
         "component": 2, "component_area_m2": 1.0, "component_face_ids": np.array([7, 9])},
    ]

    levels = projection._floor_levels(planes)

    np.testing.assert_array_equal(levels[0]["component_face_ids"], [1, 3, 5, 7, 9])
