from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_drone_path import _cut, _view_yaw, refine_panorama_framing, retime_path, sample_drone_trajectory, smooth_motion, soften_corners
from cinematic_trajectory import look_yaw
from cinematic_trajectory_validation import validate_trajectory


@pytest.mark.parametrize('interpolation', ['pchip', 'linear'])
def test_panorama_turns_at_least_half_a_circle_without_stopping(interpolation):
    floor = o3d.geometry.TriangleMesh.create_box(width=16, height=.02, depth=8)
    floor.translate([-8, -.02, -4])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    def photo(number, position):
        pose = np.eye(4)
        pose[:3, :3] = look_yaw(0).as_matrix()
        pose[:3, 3] = position
        return {'number': number, 'kind': 'photo', 'scan': 'scan_0', 'floor': 'BASE',
                'position': position, 'camera_to_world': pose.tolist()}
    panorama = {'number': 2, 'kind': 'panorama', 'scan': 'scan_0', 'floor': 'BASE',
                'position': [0, 1, 0], 'minimum_turn_degrees': 240}
    last = photo(3, [5, 1, 0])
    last['camera_to_world'] = np.block([[look_yaw(.6).as_matrix(), np.array([[5], [1], [0]])],
                                        [np.array([[0, 0, 0, 1]])]]).tolist()
    plan = {'checkpoints': [photo(1, [-5, 1, 0]), panorama, last], 'edges': [
        {'from_checkpoint': 1, 'to_checkpoint': 2, 'pieces': [
            {'kind': 'local', 'scan': 'scan_0', 'points': [[-5, 1, 0], [0, 1, 0]]}]},
        {'from_checkpoint': 2, 'to_checkpoint': 3, 'pieces': [
            {'kind': 'local', 'scan': 'scan_0', 'points': [[0, 1, 0], [5, 1, 0]]}]},
    ]}
    floors = {('scan_0', 'BASE'): scene}
    flight = sample_drone_trajectory(plan, scene, floors, interpolation=interpolation)
    result = validate_trajectory(flight, plan, scene, np.asarray(floor.vertices), np.asarray(floor.triangles), floors)
    assert set(result['floor']) == {'BASE'}
    positions = flight['camera_to_world'][:, :3, 3]
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) * flight['fps']
    checkpoint = flight['checkpoints'][1]
    frame = checkpoint['frame']
    assert speed[frame-1] > .2 and speed[frame] > .2
    for camera, event in zip(plan['checkpoints'], flight['checkpoints']):
        if camera['kind'] == 'photo':
            np.testing.assert_allclose(flight['camera_to_world'][event['frame']], camera['camera_to_world'], atol=1e-6)
    forward = flight['camera_to_world'][:, :3, 2]
    yaw = np.unwrap(np.arctan2(forward[:, 2], forward[:, 0]))
    assert abs(yaw[checkpoint['turn_end_frame']] - yaw[checkpoint['turn_start_frame']]) >= np.radians(240)


def test_corner_is_smoothed_without_stopping_or_crossing_geometry():
    floor = o3d.geometry.TriangleMesh.create_box(width=6, height=.02, depth=6)
    floor.translate([-3, -.02, -3])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    t = np.linspace(-1, 0, 44)
    points = np.vstack((np.column_stack((t, np.ones(len(t)), np.zeros(len(t)))),
                        np.column_stack((np.zeros(len(t)-1), np.ones(len(t)-1), -t[-2::-1]))))
    result = soften_corners(points, ['scan_0'] * len(points), [], [], scene, {('scan_0', 'BASE'): scene})
    vectors = np.diff(result, axis=0)
    norm = np.linalg.norm(vectors, axis=1)
    angle = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', vectors[:-1], vectors[1:]) /
                                        (norm[:-1] * norm[1:]), -1, 1)))
    assert angle.max() < 25
    assert norm.min() > .012


def test_cut_at_existing_vertex_does_not_duplicate_points():
    before, after = _cut([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 1.)
    np.testing.assert_array_equal(before, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(after, [[1, 0, 0], [2, 0, 0]])


def test_retiming_bounds_acceleration_through_a_sharp_corner():
    t = np.linspace(-1, 0, 44)
    points = np.vstack((np.column_stack((t, np.ones(len(t)), np.zeros(len(t)))),
                        np.column_stack((np.zeros(len(t)-1), np.ones(len(t)-1), -t[-2::-1]))))
    floor = o3d.geometry.TriangleMesh.create_box(width=6, height=.02, depth=6)
    floor.translate([-3, -.02, -3])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    points = soften_corners(points, ['scan_0'] * len(points), [], [], scene, {('scan_0', 'BASE'): scene})
    sampled, scans, events, doors = retime_path(points, ['scan_0'] * len(points),
                                                [{'number': 1, 'frame': 43}], [], 30, .7)
    velocity = np.diff(sampled, axis=0) * 30
    acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=1) * 30
    assert acceleration.max() <= 1.2 + 1e-6
    assert np.linalg.norm(velocity, axis=1).min() > .1
    assert scans[events[0]['frame']] == 'scan_0' and not doors


def test_shape_preserving_retime_reduces_jerk_without_moving_checkpoint():
    t = np.linspace(0, 1, 81) ** 1.3
    points = np.column_stack((5 * t, np.ones(len(t)), .4 * np.sin(2 * np.pi * t)))
    event = [{'number': 1, 'frame': len(points) // 2}]
    linear, _, _, _ = retime_path(points, ['scan_0'] * len(points), event, [], 30, .8, interpolation='linear')
    smooth, _, events, _ = retime_path(points, ['scan_0'] * len(points), event, [], 30, .8)
    jerk = lambda xyz: np.linalg.norm(np.diff(xyz, n=3, axis=0), axis=1) * 30**3
    assert len(smooth) == len(linear)
    assert np.quantile(jerk(smooth), .95) < .5 * np.quantile(jerk(linear), .95)
    assert jerk(smooth).max() < jerk(linear).max()
    np.testing.assert_allclose(smooth[events[0]['frame']], points[event[0]['frame']], atol=1e-8, rtol=0)


def test_timed_motion_smoothing_reduces_jerk_and_keeps_endpoints():
    floor = o3d.geometry.TriangleMesh.create_box(width=8, height=.02, depth=8)
    floor.translate([-4, -.02, -4])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    t = np.linspace(-1, 1, 121)
    points = np.column_stack((t, np.ones(len(t)), np.maximum(t, 0)))
    events = [{'frame': 0}, {'frame': 120}]
    cameras = [{'position': points[0]}, {'position': points[-1]}]
    result = smooth_motion(points, ['scan_0'] * len(points), events, cameras, [], scene,
                           {('scan_0', 'BASE'): scene}, 30)
    np.testing.assert_allclose(result[[0, -1]], points[[0, -1]])
    assert np.linalg.norm(np.diff(result, n=3, axis=0), axis=1).max() < .1 * np.linalg.norm(np.diff(points, n=3, axis=0), axis=1).max()


def test_view_turns_toward_the_open_side_after_a_door():
    door = o3d.geometry.TriangleMesh.create_box(width=.02, height=3, depth=8)
    door.translate([-.01, 0, -4])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(door))
    points = np.column_stack((np.linspace(-2, 2, 241), np.ones(241), np.zeros(241)))
    yaw = _view_yaw(points, scene, 30, [(90, 150)])
    assert np.max(np.abs(np.diff(yaw))) * 30 <= np.deg2rad(30)
    for lo, hi in ((10, 80), (160, 230)):
        direction = np.column_stack((np.cos(yaw[lo:hi]), np.zeros(hi-lo), np.sin(yaw[lo:hi])))
        rays = np.column_stack((points[lo:hi], direction)).astype(np.float32)
        depth = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        assert np.mean(depth < .5) < .05
    assert np.mean(np.cos(yaw[180:230])) > .2


def test_open_space_view_prefers_the_direction_of_travel():
    floor = o3d.geometry.TriangleMesh.create_box(width=12, height=.02, depth=8)
    floor.translate([-6, -.02, -4])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    points = np.column_stack((np.linspace(-4, 4, 301), np.ones(301), np.zeros(301)))
    yaw = _view_yaw(points, scene, 30, [])
    assert np.mean(np.cos(yaw)) > .95


def test_panorama_framing_avoids_near_wall_without_extra_spin_or_moving_photos():
    floor = o3d.geometry.TriangleMesh.create_box(width=6, height=.02, depth=6)
    floor.translate([-3, -.02, -3])
    wall = o3d.geometry.TriangleMesh.create_box(width=.02, height=2, depth=.5)
    wall.translate([.7, 0, -.25])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor + wall))
    poses = np.tile(np.eye(4), (301, 1, 1))
    poses[:, :3, 3] = [0, 1, 0]
    poses[:, :3, :3] = np.stack([look_yaw(angle).as_matrix() for angle in np.linspace(-np.pi / 2, np.pi / 2, 301)])
    flight = {'camera_to_world': poses, 'fps': 30, 'scans': ['s'] * len(poses), 'door_crossings': [],
              'checkpoints': [{'number': 1, 'frame': 0},
                              {'number': 2, 'frame': 150, 'turn_start_frame': 0, 'turn_end_frame': 300},
                              {'number': 3, 'frame': 300}]}
    plan = {'checkpoints': [{'number': 1, 'kind': 'photo'}, {'number': 2, 'kind': 'panorama'},
                            {'number': 3, 'kind': 'photo'}]}
    result, changes = refine_panorama_framing(flight, plan, scene, ['BASE'] * len(poses))
    assert len(changes) == 1 and changes[0]['near_surface_penalty_after'] < changes[0]['near_surface_penalty_before']
    assert changes[0]['window_penalty_after'] <= changes[0]['window_penalty_before']
    rays = np.stack([np.r_[pose[150, :3, 3], pose[150, :3, 2]] for pose in (poses, result)]).astype(np.float32)
    depths = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    assert depths[0] < 1. and depths[1] > depths[0] + .1
    np.testing.assert_allclose(result[[0, -1]], poses[[0, -1]], atol=1e-12, rtol=0)
    before = np.unwrap(np.arctan2(poses[:, 2, 2], poses[:, 0, 2]))
    after = np.unwrap(np.arctan2(result[:, 2, 2], result[:, 0, 2]))
    assert abs(after[-1] - after[0]) >= np.pi - 1e-12
    assert np.abs(np.diff(after)).sum() <= np.abs(np.diff(before)).sum() + np.radians(1)
    open_scene = o3d.t.geometry.RaycastingScene()
    open_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    untouched, open_changes = refine_panorama_framing(flight, plan, open_scene, ['BASE'] * len(poses))
    assert not open_changes
    np.testing.assert_array_equal(untouched, poses)


def test_panorama_framing_skips_a_one_sided_window_at_low_fps():
    floor = o3d.geometry.TriangleMesh.create_box(width=2, height=.02, depth=2)
    floor.translate([-1, -.02, -1])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    poses = np.tile(np.eye(4), (3, 1, 1))
    flight = {'camera_to_world': poses, 'fps': 1, 'scans': ['s'] * 3, 'door_crossings': [],
              'checkpoints': [{'number': 1, 'frame': 1, 'turn_start_frame': 1, 'turn_end_frame': 2}]}
    with np.errstate(divide='raise', invalid='raise'):
        result, changes = refine_panorama_framing(flight, {'checkpoints': [{'number': 1, 'kind': 'panorama'}]},
                                                  scene, ['BASE'] * 3)
    assert not changes
    np.testing.assert_array_equal(result, poses)


def test_nearby_panorama_turns_are_distinct_and_include_the_checkpoint():
    floor = o3d.geometry.TriangleMesh.create_box(width=20, height=.02, depth=8)
    floor.translate([-10, -.02, -4])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    points = np.column_stack((np.linspace(-5, 5, 900), np.ones(900), np.zeros(900)))
    events = [{'number': 1, 'frame': 200}, {'number': 2, 'frame': 550}, {'number': 3, 'frame': 850}]
    yaw = _view_yaw(points, scene, 30, [], events)
    for event in events:
        a, b = event['turn_start_frame'], event['turn_end_frame']
        assert a <= event['frame'] <= b
        assert abs(yaw[b] - yaw[a]) >= np.pi
    assert all(a['turn_end_frame'] < b['turn_start_frame'] for a, b in zip(events, events[1:]))
    assert np.max(np.abs(np.diff(yaw))) * 30 < np.deg2rad(65.1)


def test_upper_floor_panorama_favors_the_published_view_direction():
    floor = o3d.geometry.TriangleMesh.create_box(width=12, height=.02, depth=8)
    floor.translate([-6, -.02, -4])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    points = np.column_stack((np.linspace(-4, 4, 601), np.ones(601), np.zeros(601)))
    event = {'number': 1, 'frame': 300, 'turn_floor': 'UP', 'turn_yaw': np.pi / 2}
    yaw = _view_yaw(points, scene, 30, [], [event])
    error = (yaw[event['frame']] - event['turn_yaw'] + np.pi) % (2*np.pi) - np.pi
    assert abs(error) < np.deg2rad(50)
