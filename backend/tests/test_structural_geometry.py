"""Regression checks for finite structural surfaces and their junctions."""
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '06_mesh_elaboration'))
from recognize_and_regularize_patterns import ear_clip_polygon
from structural_geometry import (
    beam_surface, close_slab, cut_small_overhangs, pattern_frame, perimeter_pattern_axes, snap_small_excursions,
    supported_horizontal_extensions, surface_wall_sections, wall_profile_mesh,
)


@pytest.mark.parametrize('clockwise', [False, True])
def test_concave_triangulation_preserves_original_indices(clockwise):
    points = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    if clockwise:
        points = points[::-1]
    triangles = ear_clip_polygon(points)
    polygon = Polygon(points)
    pieces = [Polygon(points[t]) for t in triangles]
    assert all(polygon.covers(piece) for piece in pieces)
    assert sum(piece.area for piece in pieces) == pytest.approx(polygon.area)
    mesh = trimesh.Trimesh(vertices=np.column_stack([points, np.zeros(len(points))]), faces=triangles)
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    delta = points[edges[counts == 1, 1]] - points[edges[counts == 1, 0]]
    assert np.all(np.min(np.abs(delta), axis=1) == 0)


def make_beams(slope=0.):
    normal = np.array([slope, -1., 0.])
    normal /= np.linalg.norm(normal)
    footprint = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)
    vertices, faces = beam_surface(normal, 2., [0, -1, 0], np.array([0., 0., 1.]),
                                  .5, .12, .2, .23, footprint)
    return vertices, faces, normal, footprint


@pytest.mark.parametrize('slope', [0., .4])
def test_beams_follow_pitched_plane_and_exact_concave_footprint(slope):
    vertices, faces, normal, footprint = make_beams(slope)
    polygon = Polygon(footprint)
    assert all(polygon.buffer(1e-8).covers(Polygon(vertices[t][:, [0, 2]])) for t in faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    horizontal = np.abs(mesh.face_normals @ normal) > .9
    # Each relief face is exactly parallel to its supporting plane.
    assert np.allclose(np.abs(mesh.face_normals[horizontal] @ normal), 1.)
    residuals = vertices @ normal + 2.
    assert len(np.unique(np.round(residuals, 8))) == 2
    assert np.ptp(residuals) == pytest.approx(.2)


@pytest.mark.parametrize('floor_y', [1.95, 2.8])
def test_slab_is_closed_and_floor_never_cuts_beams(floor_y):
    vertices, faces, _, _ = make_beams()
    solid, equation, lift = close_slab(vertices, faces, [0, 1, 0, -floor_y])
    assert solid.is_watertight and solid.is_winding_consistent
    assert solid.volume > 0
    assert -equation[3] >= vertices[:, 1].max()
    assert lift == pytest.approx(max(0, vertices[:, 1].max()-floor_y), abs=2e-5)
    # The underside was retained exactly, including beam flanks.
    assert np.all(trimesh.proximity.closest_point_naive(solid, vertices)[1] < 1e-8)


def test_slab_clearance_accounts_for_slope_at_lowest_floor_corner():
    vertices, faces, _, _ = make_beams(.4)
    solid, equation, _ = close_slab(vertices, faces, [.1, 1., 0., -2.])
    y_top = -(vertices[:, 0]*equation[0]+equation[3])/equation[1]
    assert y_top.min() >= vertices[:, 1].max()-1e-9
    assert solid.is_watertight


def test_small_misalignments_snap_both_ways_large_extensions_remain():
    points = np.array([[-.1, .1], [3.1, 2.9], [-1., 4.], [4., -1.]])
    result = snap_small_excursions(points, {'West Wall':0, 'East Wall':3, 'South Wall':0, 'North Wall':3})
    np.testing.assert_allclose(result, [[0, 0], [3, 3], [-1, 4], [4, -1]])
    np.testing.assert_allclose(points[0], [-.1, .1])


def test_broad_supported_patio_survives_but_crumbs_do_not():
    footprint = [[0, 0], [2, 0], [2, 2], [0, 2]]
    a, b = np.meshgrid(np.arange(1.95, 3., .01), np.arange(.5, 1.5, .01))
    patio = np.column_stack([a.ravel(), b.ravel()])
    points = np.vstack([patio, [[4., 4.]]*20])
    result = Polygon(supported_horizontal_extensions(footprint, points, {}))
    assert result.contains(Point(2.5, 1.))
    assert not result.covers(Point(4., 4.))
    assert result.is_valid
    points = np.asarray(result.exterior.coords[:-1])
    triangles = ear_clip_polygon(points)
    assert sum(Polygon(points[t]).area for t in triangles) == pytest.approx(result.area)


def test_wall_follows_roof_relief_without_gaps_or_vertical_extras():
    vertices, faces, _, _ = make_beams(.4)
    normal, along = np.array([1., 0., 0.]), np.array([0., 0., 1.])
    sections = surface_wall_sections(vertices, faces, normal, 0., along)
    wall_v, wall_t = wall_profile_mesh(normal, 0., along, (0., 3.),
                                      np.array([0., 1., 0., 0.]), np.array([.4, -1., 0., 2.]), sections)
    assert len(wall_t) > 2
    for i in range(0, len(wall_v), 4):
        a, b, top_b, top_a = wall_v[i:i+4]
        mid = (a[2]+b[2])/2
        active = [s for s in sections if s[0]-1e-8 <= mid <= s[1]+1e-8]
        assert active
        for vertex in (top_a, top_b):
            expected = min(s[2]+(vertex[2]-s[0])*(s[3]-s[2])/(s[1]-s[0]) for s in active)
            assert vertex[1] == pytest.approx(expected)
        assert a[1] == b[1] == 0.


def test_pattern_frame_removes_axis_rounding_drift():
    n, origin, relief, u, v = pattern_frame(np.array([.4, -1., .06]), 2., [0, -1, 0], np.array([.05, .06, 1.]))
    assert abs(u @ n) < 1e-15
    assert abs(v @ n) < 1e-15
    assert origin @ n == pytest.approx(-2/np.linalg.norm([.4, -1., .06]))


def test_nonunit_plane_equation_has_same_beam_geometry():
    footprint = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    def make(scale):
        return beam_surface(np.array([.4, -1., 0.])*scale, 2.*scale, [0, -1, 0],
                            np.array([0., 0., 1.]), .5, .12, .2, .23, footprint)
    a, ta = make(1.)
    b, tb = make(3.)
    np.testing.assert_allclose(a, b, atol=1e-8)
    np.testing.assert_array_equal(ta, tb)


def test_cut_small_overhang_at_recess_before_beams_and_slab():
    walls = [[0, 0], [3, 0], [3, 2], [2.6, 2], [2.6, 3], [0, 3]]
    surface = [[0, 0], [3, 0], [3, 2.08], [2, 2.08], [2, 3], [0, 3]]
    trimmed = cut_small_overhangs(surface, walls)
    assert Polygon(trimmed).difference(Polygon(walls)).area < 1e-12
    vertices, faces = beam_surface(np.array([0., -1., 0.]), 2., [0, -1, 0],
                                  np.array([0., 0., 1.]), .5, .12, .2, .23, np.asarray(trimmed))
    slab, _, _ = close_slab(vertices, faces, [0, 1, 0, -2.5])
    assert slab.is_watertight
    assert all(Polygon(walls).buffer(1e-8).covers(Point(v[[0, 2]])) for v in slab.vertices)


def test_cut_keeps_broad_supported_horizontal_extension():
    walls = [[0, 0], [2, 0], [2, 2], [0, 2]]
    patio = [[0, 0], [3, 0], [3, 2], [0, 2]]
    assert Polygon(cut_small_overhangs(patio, walls)).equals(Polygon(patio))


@pytest.mark.parametrize('normal', [[0., -1., 0.], [.4, -1., .06]])
@pytest.mark.parametrize('axis_index, sign', [(0, 1), (0, -1), (1, 1), (1, -1)])
def test_beams_follow_lifted_perimeter_edge_on_flat_and_pitched_roofs(normal, axis_index, sign):
    normal = np.array(normal)
    normal /= np.linalg.norm(normal)
    angle = np.deg2rad(10.5)
    axes = np.array([[np.cos(angle), 0., np.sin(angle)], [-np.sin(angle), 0., np.cos(angle)]])
    edge = axes[axis_index].copy()
    edge[1] = -(edge @ normal)/normal[1]
    edge /= np.linalg.norm(edge)
    noisy = sign*edge + .03*np.cross(normal, edge)
    noisy /= np.linalg.norm(noisy)
    transverse, longitudinal = perimeter_pattern_axes(normal, noisy, axes)
    np.testing.assert_allclose(longitudinal, sign*edge, atol=1e-14)
    assert transverse @ normal == pytest.approx(0., abs=1e-14)
    assert transverse @ longitudinal == pytest.approx(0., abs=1e-14)
    np.testing.assert_allclose(np.cross(normal, transverse), longitudinal, atol=1e-14)
