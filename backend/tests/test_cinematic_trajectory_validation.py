from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_trajectory import look_yaw, sample_trajectory
from cinematic_trajectory_validation import validate_trajectory


def test_validation_detects_a_mesh_inserted_into_sampled_flight():
    floor = o3d.geometry.TriangleMesh.create_box(width=4, height=.02, depth=4)
    floor.translate([-2, -.02, -2])
    floor_scene = o3d.t.geometry.RaycastingScene()
    floor_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    def photo(number, x):
        pose = np.eye(4)
        pose[:3, :3] = look_yaw(0).as_matrix()
        pose[:3, 3] = [x, 1, 0]
        return {'number': number, 'kind': 'photo', 'scan': 'scan_0', 'floor': 'BASE',
                'position': [x, 1, 0], 'camera_to_world': pose.tolist()}
    plan = {'checkpoints': [photo(1, -1), photo(2, 1)], 'edges': [
        {'from_checkpoint': 1, 'to_checkpoint': 2,
         'pieces': [{'kind': 'local', 'scan': 'scan_0', 'points': [[-1, 1, 0], [1, 1, 0]]}]}]}
    flight = sample_trajectory(plan)
    v, f = np.asarray(floor.vertices), np.asarray(floor.triangles)
    result = validate_trajectory(flight, plan, floor_scene, v, f, {('scan_0', 'BASE'): floor_scene})
    assert set(result['floor']) == {'BASE'}
    plan['checkpoints'][0]['camera_to_world'][1][1] += .000005
    with pytest.raises(ValueError, match='photo pose mismatch'):
        validate_trajectory(flight, plan, floor_scene, v, f, {('scan_0', 'BASE'): floor_scene})
    plan['checkpoints'][0]['camera_to_world'][1][1] -= .000005
    wall = o3d.geometry.TriangleMesh.create_box(width=.1, height=2, depth=1)
    wall.translate([-.05, 0, -.5])
    combined = floor + wall
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(combined))
    with pytest.raises(ValueError, match='clearance|crosses sealed mesh'):
        validate_trajectory(flight, plan, scene, np.asarray(combined.vertices), np.asarray(combined.triangles),
                            {('scan_0', 'BASE'): floor_scene})


def test_upper_floor_turn_must_happen_on_the_upper_floor():
    base = o3d.geometry.TriangleMesh.create_box(width=6, height=.02, depth=4)
    base.translate([-3, -.02, -2])
    upper = o3d.geometry.TriangleMesh.create_box(width=.4, height=.02, depth=4)
    upper.translate([-.2, .3, -2])
    def ray_scene(mesh):
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return scene
    mesh = base + upper
    scene = ray_scene(mesh)
    frames = 301
    poses = np.tile(np.eye(4), (frames, 1, 1))
    for i, angle in enumerate(np.linspace(0, np.pi, frames)):
        poses[i, :3, :3] = look_yaw(angle).as_matrix()
    poses[:, :3, 3] = np.column_stack((np.linspace(-2, 2, frames), np.ones(frames), np.zeros(frames)))
    checkpoints = [
        {'number': 1, 'kind': 'photo', 'scan': 'scan_0', 'floor': 'BASE', 'position': poses[0, :3, 3].tolist()},
        {'number': 2, 'kind': 'panorama', 'scan': 'scan_0', 'floor': 'UP',
         'position': poses[150, :3, 3].tolist(), 'minimum_turn_degrees': 180},
        {'number': 3, 'kind': 'photo', 'scan': 'scan_0', 'floor': 'BASE', 'position': poses[-1, :3, 3].tolist()},
    ]
    for checkpoint, frame in zip(checkpoints, (0, 150, 300)):
        if checkpoint['kind'] == 'photo':
            checkpoint['camera_to_world'] = poses[frame].tolist()
    events = [{'number': 1, 'frame': 0},
              {'number': 2, 'frame': 150, 'turn_start_frame': 0, 'turn_end_frame': 300},
              {'number': 3, 'frame': 300}]
    flight = {'camera_to_world': poses, 'scans': ['scan_0'] * frames, 'fps': 30,
              'checkpoints': events, 'door_crossings': [], 'camera_mode': 'drone_smooth'}
    args = flight, {'checkpoints': checkpoints, 'edges': []}, scene, np.asarray(mesh.vertices), np.asarray(mesh.triangles), {
        ('scan_0', 'BASE'): ray_scene(base), ('scan_0', 'UP'): ray_scene(upper)}
    result = validate_trajectory(*args)
    assert result['panorama_turns'][0]['required_degrees'] == 180
    assert result['panorama_turns'][0]['actual_degrees'] == pytest.approx(180)
    checkpoints[1]['minimum_turn_degrees'] = 360
    with pytest.raises(ValueError, match='panorama turn is too short'):
        validate_trajectory(*args)
    checkpoints[1]['minimum_turn_degrees'] = 180
    events[1]['turn_floor'] = 'UP'
    with pytest.raises(ValueError, match='does not turn 180 degrees on its floor'):
        validate_trajectory(*args)


def test_floor_support_is_checked_between_rendered_frames():
    left = o3d.geometry.TriangleMesh.create_box(width=.9, height=.02, depth=2)
    left.translate([-1, -.02, -1])
    right = o3d.geometry.TriangleMesh.create_box(width=.9, height=.02, depth=2)
    right.translate([.1, -.02, -1])
    floor = left + right
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(floor))
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = [[-.5, 1, 0], [.5, 1, 0]]
    checkpoints = [{'number': i + 1, 'kind': 'photo', 'scan': 's', 'floor': 'f',
                    'position': pose[:3, 3].tolist(), 'camera_to_world': pose.tolist()}
                   for i, pose in enumerate(poses)]
    flight = {'camera_to_world': poses, 'scans': ['s', 's'], 'fps': 1, 'door_crossings': [],
              'checkpoints': [{'number': 1, 'frame': 0}, {'number': 2, 'frame': 1}]}
    with pytest.raises(ValueError, match='floor support between frames'):
        validate_trajectory(flight, {'checkpoints': checkpoints, 'edges': []}, scene,
                            np.asarray(floor.vertices), np.asarray(floor.triangles), {('s', 'f'): scene})
