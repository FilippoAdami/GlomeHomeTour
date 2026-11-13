from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_path_planning import LocalNavigator, local_refined_route, rounded_path, segment_clear, validate_door_crossing


def scene_for(mesh):
    result = o3d.t.geometry.RaycastingScene()
    result.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return result


def test_local_path_routes_around_wall_and_checks_final_segments():
    floor = o3d.geometry.TriangleMesh.create_box(width=4, height=.02, depth=4)
    floor.translate([-2, -.02, -2])
    wall = o3d.geometry.TriangleMesh.create_box(width=.1, height=2.2, depth=1.2)
    wall.translate([-.05, 0, -.6])
    surface = scene_for(floor + wall)
    support = scene_for(floor)
    start, end = np.array([-1, 1, 0]), np.array([1, 1, 0])
    assert not segment_clear(surface, support, start, end)
    navigator = LocalNavigator(surface, support, np.asarray(floor.vertices), [start, end], spacing=.15)
    path = navigator.route(start, end)
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], end)
    assert len(path) > 2
    assert all(segment_clear(surface, support, a, b) for a, b in zip(path, path[1:]))
    rounded = rounded_path(surface, support, path)
    np.testing.assert_allclose(rounded[[0, -1]], [start, end])
    assert all(segment_clear(surface, support, a, b) for a, b in zip(rounded, rounded[1:]))


def test_segment_outside_floor_is_rejected():
    floor = o3d.geometry.TriangleMesh.create_box(width=2, height=.02, depth=2)
    floor.translate([-1, -.02, -1])
    scene = scene_for(floor)
    assert not segment_clear(scene, scene, [0, 1, 0], [1.5, 1, 0])


def test_grid_resource_limit_fails_before_allocation():
    floor = o3d.geometry.TriangleMesh.create_box(width=50, height=.02, depth=50)
    floor.translate([-25, -.02, -25])
    scene = scene_for(floor)
    with pytest.raises(ValueError, match='Grid resource limit exceeded'):
        LocalNavigator(scene, scene, np.asarray(floor.vertices), [[-1, 1, 0], [1, 1, 0]])


@pytest.mark.parametrize('size', [8., 50.])
def test_local_fine_grid_finds_a_narrow_offset_opening(size):
    half = size / 2
    floor = o3d.geometry.TriangleMesh.create_box(width=size, height=.02, depth=size)
    floor.translate([-half, -.02, -half])
    left = o3d.geometry.TriangleMesh.create_box(width=.12, height=3, depth=half + 1.065)
    left.translate([-.06, 0, -half])
    right = o3d.geometry.TriangleMesh.create_box(width=.12, height=3, depth=half - 1.335)
    right.translate([-.06, 0, 1.335])
    geometry, support = scene_for(floor + left + right), scene_for(floor)
    points = np.asarray(floor.vertices)
    start, end = [-1., 1., 0.], [1., 1., 0.]
    with pytest.raises(ValueError, match='grid path|resource limit'):
        LocalNavigator(geometry, support, points, [start, end], spacing=.12).route(start, end)
    if size == 50:
        with pytest.raises(ValueError, match='resource limit'):
            LocalNavigator(geometry, support, points, [start, end], spacing=.06)
    path, diagnostic = local_refined_route(geometry, support, points, start, end)
    assert diagnostic['method'] == 'local_refined_grid' and diagnostic['grid_cells'] < 3_000_000
    np.testing.assert_allclose(path[[0, -1]], [start, end])
    assert any(p[2] > 1.0 for p in path)
    assert all(segment_clear(geometry, support, a, b) for a, b in zip(path, path[1:]))
    rounded = rounded_path(geometry, support, path)
    assert all(segment_clear(geometry, support, a, b) for a, b in zip(rounded, rounded[1:]))


def test_local_refinement_reports_exhaustion_when_wall_has_no_opening():
    floor = o3d.geometry.TriangleMesh.create_box(width=4, height=.02, depth=4)
    floor.translate([-2, -.02, -2])
    wall = o3d.geometry.TriangleMesh.create_box(width=.12, height=3, depth=4)
    wall.translate([-.06, 0, -2])
    geometry, support = scene_for(floor + wall), scene_for(floor)
    with pytest.raises(ValueError, match='Local fine-grid search exhausted margins'):
        local_refined_route(geometry, support, np.asarray(floor.vertices), [-1, 1, 0], [1, 1, 0])


def test_straight_route_gets_a_clear_curved_arc():
    floor = o3d.geometry.TriangleMesh.create_box(width=4, height=.02, depth=4)
    floor.translate([-2, -.02, -2])
    scene = scene_for(floor)
    route = rounded_path(scene, scene, [[-1, 1, 0], [1, 1, 0]])
    np.testing.assert_array_equal(route[[0, -1]], [[-1, 1, 0], [1, 1, 0]])
    assert np.max(np.abs(route[:, 2])) > .2
    assert all(segment_clear(scene, scene, a, b) for a, b in zip(route, route[1:]))


def test_door_exception_is_limited_to_designated_surface():
    door = o3d.geometry.TriangleMesh.create_box(width=.1, height=2, depth=1)
    door.translate([-.05, 0, -.5])
    panel = {'panel_index': 0, 'panel_center_xyz_m': [0, 1, 0], 'panel_plane_normal_xyz': [1, 0, 0],
             'panel_dimensions_m': {'depth': .1, 'width': 1., 'height': 2.}}
    crossing = validate_door_crossing(scene_for(door), np.asarray(door.vertices), np.asarray(door.triangles),
                                     panel, [-.4, 1, 0], [.4, 1, 0])
    assert len(crossing['hit_points']) == 2
    obstacle = o3d.geometry.TriangleMesh.create_box(width=.05, height=.5, depth=.5)
    obstacle.translate([.25, .75, -.25])
    obstructed = door + obstacle
    with pytest.raises(ValueError, match='outside'):
        validate_door_crossing(scene_for(obstructed), np.asarray(obstructed.vertices), np.asarray(obstructed.triangles),
                               panel, [-.4, 1, 0], [.4, 1, 0])
    with pytest.raises(ValueError, match='center'):
        validate_door_crossing(scene_for(door), np.asarray(door.vertices), np.asarray(door.triangles),
                               panel, [-.4, 1, .1], [.4, 1, .1])
    shifted = validate_door_crossing(scene_for(door), np.asarray(door.vertices), np.asarray(door.triangles),
                                     panel, [-.4, 1.4, -.2], [.4, 1.4, -.2], [0, 1.4, -.2])
    assert shifted['center'] == [0, 1.4, -.2]
    with pytest.raises(ValueError, match='outside the designated panel'):
        validate_door_crossing(scene_for(door), np.asarray(door.vertices), np.asarray(door.triangles),
                               panel, [-.4, 1.9, 0], [.4, 1.9, 0], [0, 1.9, 0])
