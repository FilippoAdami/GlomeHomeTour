"""Physical footprint regressions independent of triangle density and scene origin."""
import sys
from pathlib import Path

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point, Polygon


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_mesh_elaboration"))
from surface_footprints import constrain_footprint, observed_footprint, polygons, wall_regions


A = np.array([1.0, 0.0, 0.0])
B = np.array([0.0, 0.0, 1.0])


def triangulated_polygon(polygon, cell):
    """Fill a polygon with fixed-size quad triangles, preserving holes and components."""
    vertices, faces = [], []
    min_x, min_z, max_x, max_z = polygon.bounds
    for x in np.arange(min_x, max_x, cell):
        for z in np.arange(min_z, max_z, cell):
            if not polygon.covers(Point(x + cell / 2, z + cell / 2)):
                continue
            start = len(vertices)
            vertices.extend([[x, 0, z], [x + cell, 0, z], [x + cell, 0, z + cell], [x, 0, z + cell]])
            faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
    return np.asarray(vertices), np.asarray(faces)


def footprint(polygon, cell=.05, translation=(0, 0, 0)):
    vertices, faces = triangulated_polygon(polygon, cell)
    return observed_footprint(vertices + np.asarray(translation), faces, A, B, cell_size=.02, min_feature=.10)


def x_wall(x, z0, z1):
    return {"normal_unit": [1, 0, 0], "offset_m": x, "span_w_m": [z0, z1]}


def z_wall(z, x0, x1):
    return {"normal_unit": [0, 0, 1], "offset_m": z, "span_u_m": [x0, x1]}


def test_observed_footprint_is_stable_across_remeshing_and_translation():
    rectangle = Polygon([(0, 0), (4, 0), (4, 3), (0, 3)])
    coarse = footprint(rectangle, .20)
    fine = footprint(rectangle, .05)
    translated = footprint(rectangle, .05, (13, 2, -7))
    assert coarse.symmetric_difference(fine).area < .20
    assert translated.symmetric_difference(translate(fine, xoff=13, yoff=-7)).area < .20


def test_observed_footprint_keeps_a_large_orthogonal_notch():
    notched = Polygon([(0, 0), (4, 0), (4, 4), (2.5, 4), (2.5, 2), (0, 2), (0, 0)])
    result = footprint(notched)
    assert result.covers(Point(3, 3))
    assert not result.covers(Point(1, 3))
    assert result.symmetric_difference(notched).area < .25


def test_observed_footprint_preserves_disconnected_patches_and_holes():
    ring = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]])
    patch = Polygon([(6, 0), (8, 0), (8, 2), (6, 2)])
    result = footprint(ring.union(patch))
    assert len(polygons(result)) == 2
    assert not result.covers(Point(2, 2))
    assert result.covers(Point(7, 1))


def test_wall_regions_close_only_small_endpoint_gaps_and_keep_recesses_and_rooms():
    nearly_closed = [x_wall(0, .10, 4), x_wall(4, 0, 3.90), z_wall(0, .10, 3.90), z_wall(4, .10, 3.90)]
    region = wall_regions(nearly_closed, A, B, tolerance=.15)
    assert len(region) == 1 and region[0].area == 16

    too_wide = [x_wall(0, .20, 4), x_wall(4, 0, 3.80), z_wall(0, .20, 3.80), z_wall(4, .20, 3.80)]
    assert wall_regions(too_wide, A, B, tolerance=.15) == []

    recessed_and_two_rooms = [
        x_wall(0, 0, 4), z_wall(0, 0, 4), x_wall(4, 0, 1), z_wall(1, 3, 4), x_wall(3, 1, 4), z_wall(4, 0, 3),
        x_wall(6, 0, 2), x_wall(8, 0, 2), z_wall(0, 6, 8), z_wall(2, 6, 8),
    ]
    regions = wall_regions(recessed_and_two_rooms, A, B)
    assert len(regions) == 2
    assert not any(room.covers(Point(3.5, 2)) for room in regions)


def test_constrain_footprint_trims_thin_slivers_but_keeps_broad_supported_patio():
    room = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    sliver = Polygon([(3, 1), (3.08, 1), (3.08, 2), (3, 2)])
    trimmed = constrain_footprint(room.union(sliver), [room])
    assert not trimmed.covers(Point(3.04, 1.5))

    patio = Polygon([(3, .5), (4, .5), (4, 2.5), (3, 2.5)])
    retained = constrain_footprint(room.union(patio), [room])
    assert retained.covers(Point(3.5, 1.5))


def test_cleanup_removes_short_jogs_without_diagonals_and_preserves_patio():
    from surface_footprints import clean_footprint
    poly = Polygon([(0,0),(3,0),(3,1),(3.06,1),(3.06,1.8),(3,1.8),(3,3),(0,3)])
    clean = clean_footprint(poly)
    assert clean.equals(Polygon([(0,0),(3,0),(3,3),(0,3)]))
    edges = np.diff(np.array(clean.exterior.coords), axis=0)
    assert np.min(np.linalg.norm(edges, axis=1)) >= .10-1e-9
    assert np.max(np.min(np.abs(edges), axis=1)) < 1e-9
    patio = Polygon([(3,1),(4,1),(4,2),(3,2)])
    assert clean_footprint(clean.union(patio)).equals(clean.union(patio))


def test_cleanup_fills_only_holes_smaller_than_40cm_in_both_dimensions():
    from surface_footprints import clean_footprint
    poly = Polygon([(0,0),(4,0),(4,4),(0,4)], holes=[
        [(1,1),(1.39,1),(1.39,1.39),(1,1.39)],
        [(2,1),(2.40,1),(2.40,1.40),(2,1.40)],
        [(1,2),(1.2,2),(1.2,3),(1,3)]])
    clean = clean_footprint(poly)
    assert len(clean.interiors) == 2
    assert clean.covers(Point(1.2,1.2))
    assert not clean.covers(Point(2.2,1.2))
    assert not clean.covers(Point(1.1,2.5))
    shifted = clean_footprint(translate(poly,xoff=123,yoff=-57))
    assert shifted.symmetric_difference(translate(clean,xoff=123,yoff=-57)).area < 1e-9


def test_cleanup_short_asymmetric_steps_moves_whole_adjacent_edge():
    from surface_footprints import clean_footprint
    poly = Polygon([(0,0),(3,0),(3,1),(2.94,1),(2.94,3),(0,3)])
    clean = clean_footprint(poly)
    edges = np.diff(np.array(clean.exterior.coords),axis=0)
    assert np.max(np.min(np.abs(edges),axis=1)) < 1e-9
    assert np.min(np.linalg.norm(edges,axis=1)) >= .10-1e-9
    assert clean.difference(poly).area < 1e-9
