from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '06_mesh_elaboration'))

from recognize_and_regularize_patterns import (  # noqa: E402
    _aligned_plane_to_source,
    _plan_footprint_in_source_frame,
    clip_mesh_y_bounds,
    planar_surface,
    wall_mesh,
)


def test_wall_regions_keep_their_shape_and_only_clip_global_height():
    wall = {
        'axis': 'Dir A',
        'offset_m': 2.0,
        'span_w_m': [0.0, 4.0],
        'bounds_y': [-1.0, 4.0],
        'normal_unit': [1.0, 0.0, 0.0],
        'trim_regions': [
            {'span_m': [0.0, 1.0], 'bounds_y': [-1.0, 4.0]},
            {'span_m': [1.0, 4.0], 'bounds_y': [0.5, 4.0]},
        ],
    }

    vertices, triangles = wall_mesh(
        wall, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), (0.0, 3.0)
    )

    assert len(triangles) == 4
    assert np.min(vertices[:, 1]) == 0.0
    assert np.max(vertices[:, 1]) == 3.0
    assert np.allclose(vertices[:, 0], 2.0)
    assert np.allclose(np.sort(np.unique(vertices[:, 2])), [0.0, 1.0, 4.0])
    assert np.any(np.isclose(vertices[:, 1], 0.5))


def test_wall_unmodeled_triangles_are_clipped_at_global_floor_and_ceiling():
    vertices = np.array([[0.0, -1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    clipped_v, clipped_t, clipped_c = clip_mesh_y_bounds(
        vertices, np.array([[0, 1, 2]]), colors, 0.0, 0.75
    )

    assert len(clipped_t) == 2
    assert np.min(clipped_v[:, 1]) == 0.0
    assert np.max(clipped_v[:, 1]) == 0.75
    assert np.all((clipped_c >= 0.0) & (clipped_c <= 1.0))
    assert np.all(np.linalg.norm(np.cross(
        clipped_v[clipped_t[:, 1]] - clipped_v[clipped_t[:, 0]],
        clipped_v[clipped_t[:, 2]] - clipped_v[clipped_t[:, 0]],
    ), axis=1) > 0.0)


def test_fitted_surface_mesh_preserves_cutouts():
    footprint = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    vertices, triangles = planar_surface(footprint, np.array([0.0, 1.0, 0.0, -2.0]), True)
    tri = vertices[triangles]
    area = np.sum(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)) / 2

    assert np.isclose(area, footprint.area)
    assert np.allclose(vertices[:, 1], 2.0)
    assert not any(footprint.interiors and Point(face.mean(axis=0)[[0, 2]]).within(Polygon(footprint.interiors[0]))
                   for face in tri)


def test_aligned_ceiling_plan_returns_to_source_frame_with_its_hole():
    angle = np.deg2rad(-11.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    translation = np.zeros(3)
    perimeter = [[0, 0], [4, 0], [4, 4], [0, 4]]
    cutout = [[[1, 1], [3, 1], [3, 3], [1, 3]]]

    footprint = _plan_footprint_in_source_frame(
        perimeter, cutout, rotation, translation
    )

    assert np.isclose(footprint.area, 12.0)
    source_plane = _aligned_plane_to_source(
        [0.2, -1.0, 0.0, 2.0], rotation, translation
    )
    aligned_point = np.array([1.0, 2.2, 0.0])
    source_point = rotation.T @ aligned_point
    assert np.isclose(source_plane @ np.r_[source_point, 1.0], 0.0)
