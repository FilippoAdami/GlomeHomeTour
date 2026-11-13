from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '08_artifacts_generation'))
from cinematic_trajectory import SPEED_FACTOR, append_overhead, look_yaw, sample_trajectory


def photo(number, position, rotation):
    pose = np.eye(4)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = position
    return {'number': number, 'kind': 'photo', 'scan': 'scan_0', 'position': position,
            'camera_to_world': pose.tolist()}


def test_exact_photo_poses_and_full_turn_at_published_checkpoint():
    first = photo(1, [0, 1, 0], look_yaw(0))
    panorama = {'number': 2, 'kind': 'panorama', 'scan': 'scan_0', 'position': [1, 1, 0],
                'minimum_turn_degrees': 360}
    last = photo(3, [2, 1, 0], look_yaw(.7))
    plan = {'checkpoints': [first, panorama, last], 'edges': [
        {'from_checkpoint': 1, 'to_checkpoint': 2,
         'pieces': [{'kind': 'local', 'scan': 'scan_0', 'points': [[0, 1, 0], [1, 1, 0]]}]},
        {'from_checkpoint': 2, 'to_checkpoint': 3,
         'pieces': [{'kind': 'local', 'scan': 'scan_0', 'points': [[1, 1, 0], [2, 1, 0]]}]},
    ]}
    trajectory = sample_trajectory(plan)
    poses = trajectory['camera_to_world']
    for checkpoint, event in zip(plan['checkpoints'], trajectory['checkpoints']):
        pose = poses[event['frame']]
        np.testing.assert_allclose(pose[:3, 3], checkpoint['position'], atol=1e-8)
        if checkpoint['kind'] == 'photo':
            np.testing.assert_allclose(pose, checkpoint['camera_to_world'], atol=1e-8)
    turn = trajectory['checkpoints'][1]
    rotations = poses[turn['turn_start_frame']:turn['turn_end_frame'] + 1, :3, :3]
    yaw = np.unwrap(np.arctan2(rotations[:, 2, 2], rotations[:, 0, 2]))
    assert abs(yaw[-1] - yaw[0]) >= 2 * np.pi - 1e-7
    angular = Rotation.from_matrix(poses[:, :3, :3])
    max_angular = np.max((angular[:-1].inv() * angular[1:]).magnitude()) * trajectory['fps']
    assert max_angular <= np.deg2rad(25.1 * SPEED_FACTOR)


def test_door_crossing_accelerates_and_then_slows():
    first = photo(1, [0, 1, 0], look_yaw(0))
    last = photo(2, [3, 1, 0], look_yaw(0))
    last['scan'] = 'scan_1'
    plan = {'checkpoints': [first, last], 'edges': [{'from_checkpoint': 1, 'to_checkpoint': 2, 'pieces': [
        {'kind': 'local', 'scan': 'scan_0', 'points': [[0, 1, 0], [1.25, 1, 0]]},
        {'kind': 'door', 'scan_from': 'scan_0', 'scan_to': 'scan_1', 'points': [[1.25, 1, 0], [1.75, 1, 0]],
         'crossing': {'panel_index': 5}},
        {'kind': 'local', 'scan': 'scan_1', 'points': [[1.75, 1, 0], [3, 1, 0]]},
    ]}]}
    trajectory = sample_trajectory(plan)
    speed = np.linalg.norm(np.diff(trajectory['camera_to_world'][:, :3, 3], axis=0), axis=1) * trajectory['fps']
    event = trajectory['door_crossings'][0]
    a, b = event['start_frame'], event['end_frame']
    assert speed[(a+b)//2] > speed[a] * 1.02
    assert speed[(a+b)//2] > speed[b] * 1.02


def test_overhead_camera_points_down_and_records_the_ceiling():
    import open3d as o3d
    roof = o3d.geometry.TriangleMesh.create_box(width=4, height=.1, depth=4)
    roof.translate([-2, 3, -2])
    vertices, faces = np.asarray(roof.vertices), np.asarray(roof.triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(roof))
    pose = np.eye(4)
    pose[:3, 3] = [0, 1, 0]
    flight = {'camera_to_world': pose[None], 'scans': ['scan_0'], 'fps': 30}
    append_overhead(flight, vertices, faces, scene)
    assert len(flight['overhead']['ceiling_hits']) == 2
    np.testing.assert_allclose(flight['camera_to_world'][-1, :3, 2], [0, -1, 0], atol=1e-7)
    assert flight['camera_to_world'][-1, 1, 3] > vertices[:, 1].max()
    assert flight['camera_to_world'][120, 1, 2] < -.99
    final = flight['camera_to_world'][-1]
    camera_points = (vertices - final[:3, 3]) @ final[:3, :3]
    assert np.all(np.abs(camera_points[:, 0]) < camera_points[:, 2] * np.tan(np.deg2rad(35)))
    assert np.all(np.abs(camera_points[:, 1]) < camera_points[:, 2] * np.tan(np.deg2rad(35)) * 16 / 9)


def test_overhead_ascent_preserves_incoming_motion():
    import open3d as o3d
    roof = o3d.geometry.TriangleMesh.create_box(width=4, height=.1, depth=4)
    roof.translate([-2, 3, -2])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(roof))
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = [[-.013, 1, 0], [0, 1, 0]]
    flight = {'camera_to_world': poses, 'scans': ['scan_0', 'scan_0'], 'fps': 30, 'camera_mode': 'drone_smooth'}
    append_overhead(flight, np.asarray(roof.vertices), np.asarray(roof.triangles), scene)
    velocity = np.diff(flight['camera_to_world'][:3, :3, 3], axis=0) * flight['fps']
    assert np.linalg.norm(velocity[1] - velocity[0]) * flight['fps'] < 1.5


def test_overhead_continues_final_panorama_before_looking_down():
    import open3d as o3d
    roof = o3d.geometry.TriangleMesh.create_box(width=4, height=.1, depth=4)
    roof.translate([-2, 3, -2])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(roof))
    poses = np.tile(np.eye(4), (31, 1, 1))
    for i, angle in enumerate(np.linspace(0, np.pi, len(poses))):
        poses[i, :3, :3] = look_yaw(angle).as_matrix()
    poses[:, :3, 3] = [0, 1, 0]
    flight = {'camera_to_world': poses, 'scans': ['scan_0'] * len(poses), 'fps': 30,
              'camera_mode': 'drone_smooth',
              'checkpoints': [{'turn_start_frame': 0, 'turn_end_frame': 30}]}
    append_overhead(flight, np.asarray(roof.vertices), np.asarray(roof.triangles), scene)
    forward = flight['camera_to_world'][:, :3, 2]
    yaw = np.unwrap(np.arctan2(forward[:75, 2], forward[:75, 0]))
    assert yaw[65] - yaw[30] > np.deg2rad(35)
    assert forward[45, 1] > -.01
    np.testing.assert_allclose(forward[-1], [0, -1, 0], atol=1e-7)
    rotations = Rotation.from_matrix(flight['camera_to_world'][:, :3, :3])
    rates = (rotations[:-1].inv() * rotations[1:]).as_rotvec() * flight['fps']
    assert np.linalg.norm(rates[30] - rates[29]) < .3


def test_overhead_retries_a_non_ceiling_intersection():
    import open3d as o3d
    roof = o3d.geometry.TriangleMesh.create_box(width=4, height=.1, depth=4)
    roof.translate([-2, 3, -2])
    wall = o3d.geometry.TriangleMesh.create_box(width=.1, height=2, depth=2)
    mesh = roof + wall
    actual = o3d.t.geometry.RaycastingScene()
    actual.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(roof))
    class Scene:
        calls = 0
        def list_intersections(self, rays):
            self.calls += 1
            if self.calls == 1:
                # The first candidate is reported to hit a vertical wall.
                return {'t_hit': o3d.core.Tensor([.5], o3d.core.Dtype.Float32),
                        'primitive_ids': o3d.core.Tensor([len(roof.triangles)+2], o3d.core.Dtype.UInt32),
                        'ray_ids': o3d.core.Tensor([30], o3d.core.Dtype.UInt32)}
            return actual.list_intersections(rays)
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = [[-.013, 1, 0], [0, 1, 0]]
    flight = {'camera_to_world': poses, 'scans': ['scan_0'] * 2, 'fps': 30, 'camera_mode': 'drone_smooth'}
    scene = Scene()
    append_overhead(flight, np.asarray(mesh.vertices), np.asarray(mesh.triangles), scene)
    assert scene.calls == 2
    velocity = np.diff(flight['camera_to_world'][:3, :3, 3], axis=0) * flight['fps']
    assert np.linalg.norm(velocity[1] - velocity[0]) * flight['fps'] < 1.5
