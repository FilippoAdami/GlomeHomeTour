"""Sample one shared camera trajectory from validated spatial tour legs."""
from __future__ import annotations

import math
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp

SPEED_FACTOR = 2.9

def look_yaw(yaw):
    right = [-math.sin(yaw), 0, math.cos(yaw)]
    down = [0, -1, 0]
    forward = [math.cos(yaw), 0, math.sin(yaw)]
    return Rotation.from_matrix(np.column_stack((right, down, forward)))


def heading(vector):
    vector = np.asarray(vector)
    if np.linalg.norm(vector[[0, 2]]) < 1e-7:
        return None
    return math.atan2(vector[2], vector[0])


def _flatten(edge):
    points, bounds, running = [], [], 0.0
    for piece in edge['pieces']:
        row = np.asarray(piece['points'], dtype=float)
        if len(row) < 2:
            raise ValueError('Every trajectory piece needs at least two points')
        if points and not np.allclose(points[-1], row[0], atol=1e-6, rtol=0):
            raise ValueError('Trajectory pieces are disconnected')
        distance = float(np.linalg.norm(np.diff(row, axis=0), axis=1).sum())
        if distance < 1e-8:
            raise ValueError('Trajectory piece has no length')
        bounds.append((running, running + distance, piece))
        running += distance
        points.extend(row if not points else row[1:])
    points = np.asarray(points)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if np.any(np.diff(distances) <= 1e-9):
        raise ValueError('Repeated spatial points make trajectory sampling ambiguous')
    return points, distances, bounds


def _ease(value):
    return value * value * (3 - 2 * value)


def sample_trajectory(plan, fps=30):
    cameras = plan['checkpoints']
    edges = plan['edges']
    if fps <= 0 or len(edges) != len(cameras) - 1:
        raise ValueError('Invalid frame rate or checkpoint edge count')
    flattened = [_flatten(edge) for edge in edges]
    starts, ends, turn_records = {}, {}, {}
    for i, camera in enumerate(cameras):
        n = camera['number']
        if camera['kind'] == 'photo':
            rotation = Rotation.from_matrix(np.asarray(camera['camera_to_world'])[:3, :3])
            starts[n] = ends[n] = rotation
        else:
            previous = heading(flattened[i - 1][0][-1] - flattened[i - 1][0][-2]) if i else None
            following = heading(flattened[i][0][1] - flattened[i][0][0]) if i < len(edges) else None
            yaw = previous if previous is not None else following if following is not None else 0.0
            delta = 0 if following is None else (following - yaw + math.pi) % (2 * math.pi) - math.pi
            turn = math.radians(camera['minimum_turn_degrees']) * (1 if delta >= 0 else -1)
            if following is not None and abs(turn) > math.pi * 1.5:
                yaw = following - turn
            starts[n], ends[n] = look_yaw(yaw), look_yaw(yaw + turn)
            turn_records[n] = (yaw, turn)
    positions, rotations, scans, events, door_events = [], [], [], [], []
    def append(position, rotation, scan):
        positions.append(np.asarray(position, dtype=float))
        rotations.append(rotation.as_matrix())
        scans.append(scan)
    append(cameras[0]['position'], starts[cameras[0]['number']], cameras[0]['scan'])
    events.append({'number': cameras[0]['number'], 'frame': 0})
    for i, camera in enumerate(cameras):
        number, point, scan = camera['number'], np.asarray(camera['position']), camera['scan']
        if camera['kind'] == 'panorama':
            yaw, turn = turn_records[number]
            count = max(1, math.ceil(abs(turn) / math.radians(16 * SPEED_FACTOR) * fps))
            start_frame = len(positions) - 1
            for u in np.arange(1, count + 1) / count:
                append(point, look_yaw(yaw + turn * _ease(u)), scan)
            events[-1].update(turn_start_frame=start_frame, turn_end_frame=len(positions) - 1,
                              signed_turn_degrees=math.degrees(turn))
        if i == len(edges):
            break
        points, distances, bounds = flattened[i]
        edge = edges[i]
        if edge['from_checkpoint'] != number or edge['to_checkpoint'] != cameras[i + 1]['number']:
            raise ValueError('Edge order differs from published checkpoints')
        door_bounds = [(a, b, p) for a, b, p in bounds if p['kind'] == 'door']
        speed_knots = {0.0: 0.0, float(distances[-1]): 0.0}
        for j, (a, b, _) in enumerate(door_bounds):
            prior = 0.0 if j == 0 else door_bounds[j - 1][1]
            following = distances[-1] if j + 1 == len(door_bounds) else door_bounds[j + 1][0]
            speed = min(.7, .8 * (a - prior), .8 * (following - b))
            if speed <= 1e-4:
                raise ValueError('Insufficient room to approach or exit a door smoothly')
            speed_knots[a] = speed_knots[b] = speed
            speed_knots[(a + b) / 2] = min(.8, 1.6 * speed)
        local_bounds = [(a, b) for a, b, piece in bounds if piece['kind'] == 'local']
        for a, b in local_bounds:
            if b - a > 1.4:
                speed_knots[a + .5] = speed_knots[b - .5] = .55
            elif b - a > .35 or not door_bounds:
                speed_knots[(a + b) / 2] = .55
        knots = sorted(speed_knots)
        speeds = [speed_knots[k] for k in knots]
        times = [0.0]
        for a, b, v0, v1 in zip(knots, knots[1:], speeds, speeds[1:]):
            if b <= a or v0 + v1 <= 0:
                raise ValueError('Invalid timing knots')
            times.append(times[-1] + 2 * (b - a) / (v0 + v1))
        times = np.asarray(times)
        # Limit rotation over the whole segment, including turns toward door crossings.
        targets = [ends[number]]
        rotation_distances = [0.0]
        for a, b, piece in door_bounds:
            direction = np.asarray(piece['points'][-1]) - np.asarray(piece['points'][0])
            yaw = heading(direction)
            if yaw is None:
                raise ValueError('Door normal has no horizontal direction')
            targets.append(look_yaw(yaw))
            rotation_distances.append(a)
        targets.append(starts[cameras[i + 1]['number']])
        rotation_distances.append(float(distances[-1]))
        rotation_angle = sum((a.inv() * b).magnitude() for a, b in zip(targets, targets[1:]))
        scale = max(1.0, rotation_angle / math.radians(15) / times[-1])
        times *= scale
        speed_values = np.asarray(speeds) / scale
        times /= SPEED_FACTOR
        speed_values *= SPEED_FACTOR
        for _ in range(4):
            interpolator = CubicHermiteSpline(times, knots, speed_values)
            count = max(1, math.ceil(times[-1] * fps))
            sample_times = np.linspace(0, times[-1], count + 1)[1:]
            travel = np.clip(interpolator(sample_times), 0, distances[-1])
            xyz = np.column_stack([np.interp(travel, distances, points[:, axis]) for axis in range(3)])
            sampled_rotations = np.empty((count, 3, 3))
            for j, (lo, hi) in enumerate(zip(rotation_distances, rotation_distances[1:])):
                mask = (travel >= lo) & ((travel < hi) if j + 2 < len(rotation_distances) else (travel <= hi))
                if mask.any():
                    u = _ease(np.clip((travel[mask] - lo) / (hi - lo), 0, 1))
                    sampled_rotations[mask] = Slerp([0, 1], Rotation.concatenate(targets[j:j+2]))(u).as_matrix()
            frames = Rotation.from_matrix(np.concatenate((targets[0].as_matrix()[None], sampled_rotations)))
            max_turn = float(np.max((frames[:-1].inv() * frames[1:]).magnitude())) * fps
            if max_turn <= math.radians(25.01 * SPEED_FACTOR):
                break
            factor = max_turn / math.radians(25 * SPEED_FACTOR) * 1.01
            times *= factor
            speed_values /= factor
        else:
            raise ValueError('Could not satisfy the camera angular-speed limit')
        piece_bounds = np.array([b for _, b, _ in bounds])
        piece_indices = np.minimum(np.searchsorted(piece_bounds, travel, side='right'), len(bounds) - 1)
        base_frame = len(positions) - 1
        for k, (position, matrix, piece_index) in enumerate(zip(xyz, sampled_rotations, piece_indices)):
            piece = bounds[piece_index][2]
            owner = piece['scan'] if piece['kind'] == 'local' else (
                piece['scan_from'] if travel[k] < sum(bounds[piece_index][:2]) / 2 else piece['scan_to'])
            positions.append(position)
            rotations.append(matrix)
            scans.append(owner)
        for a, b, piece in door_bounds:
            door_events.append({'panel_index': piece['crossing']['panel_index'],
                                'from_checkpoint': number, 'to_checkpoint': cameras[i + 1]['number'],
                                'start_frame': base_frame + int(np.searchsorted(travel, a)) + 1,
                                'end_frame': base_frame + int(np.searchsorted(travel, b)) + 1})
        events.append({'number': cameras[i + 1]['number'], 'frame': len(positions) - 1})
    poses = np.tile(np.eye(4), (len(positions), 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = positions
    return {'fps': fps, 'camera_to_world': poses, 'scans': scans, 'checkpoints': events,
            'door_crossings': door_events}


def append_overhead(flight, vertices, faces, scene):
    """Rise through horizontal ceiling surfaces, then frame the scene from above."""
    import open3d as o3d
    poses = flight['camera_to_world']
    start = poses[-1, :3, 3]
    low, high = np.min(vertices, axis=0), np.max(vertices, axis=0)
    center = (low + high) / 2
    last_event = flight.get('checkpoints', [{}])[-1]
    continuing_turn = flight.get('camera_mode') == 'drone_smooth' and 'turn_start_frame' in last_event
    if continuing_turn:
        forward = poses[:, :3, 2]
        yaw = np.unwrap(np.arctan2(forward[:, 2], forward[:, 0]))
        direction = np.sign(yaw[last_event['turn_end_frame']] - yaw[last_event['turn_start_frame']])
        # shortcut: fixed extra sweep; use scene-coverage optimization if rooms remain unseen.
        end_yaw = yaw[-1] + direction * math.radians(60)
        right = look_yaw(end_yaw).as_matrix()[:, 0]
    else:
        right = poses[-1, :3, 0].copy()
    right[1] = 0
    right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-8 else np.array([1., 0., 0.])
    forward = np.array([0., -1., 0.])
    down_axis = np.cross(forward, right)
    half = (high - low) / 2
    horizontal_extent = np.abs(right) @ half
    vertical_extent = np.abs(down_axis) @ half
    height = high[1] + max(horizontal_extent / math.tan(math.radians(35)),
                           vertical_extent / (math.tan(math.radians(35)) * (16 / 9))) * 1.2
    end = np.array([center[0], height, center[2]])
    u = np.arange(1, 7 * flight['fps'] + 1) / (7 * flight['fps'])
    t = _ease(u)
    xyz = start + t[:, None] * (end - start)
    # Delay lateral travel until the camera is above the room's ceiling.
    lateral = _ease(np.clip((u - .38) / .62, 0, 1))
    xyz[:, [0, 2]] = start[[0, 2]] + lateral[:, None] * (end - start)[[0, 2]]
    velocity = np.zeros(3)
    if flight.get('camera_mode') in ('drone', 'drone_smooth') and len(poses) > 1:
        velocity = (poses[-1, :3, 3] - poses[-2, :3, 3]) * flight['fps']
    if continuing_turn:
        seconds = u * 7
        yaw_rate = (yaw[-1] - yaw[-2]) * flight['fps']
        T = 2.0
        tau = np.clip(seconds / T, 0, 1)
        D = end_yaw - yaw[-1]
        v0 = yaw_rate * T
        c3 = 10 * D - 6 * v0
        c4 = -15 * D + 8 * v0
        c5 = 6 * D - 3 * v0
        headings = np.where(seconds < T, yaw[-1] + v0 * tau + c3 * tau**3 + c4 * tau**4 + c5 * tau**5, end_yaw)
        look_down = _ease(np.clip((seconds - .6) / 3.04, 0, 1))
        level = Rotation.concatenate([look_yaw(float(value)) for value in headings])
        pitch = Rotation.from_rotvec(np.column_stack((-math.pi / 2 * look_down, np.zeros((len(u), 2)))))
        rotation = level * pitch
    else:
        down = Rotation.from_matrix(np.column_stack((right, down_axis, forward)))
        look_down = _ease(np.clip(u / .52, 0, 1))
        rotation = Slerp([0, 1], Rotation.concatenate([Rotation.from_matrix(poses[-1, :3, :3]), down]))(look_down)
    seconds = u * 7
    continuation = velocity * (seconds * (1 - np.clip(seconds / 3.5, 0, 1)) ** 3)[:, None]
    for scale in (1., 1.2, 1.4, 1.6, 2., .7, 0.):
        candidate = xyz + continuation * (1 + (scale-1) * _ease(np.clip(seconds / .8, 0, 1)))[:, None]
        points = np.vstack((start, candidate))
        rays = np.column_stack((points[:-1], np.diff(points, axis=0))).astype(np.float32)
        hits = scene.list_intersections(o3d.core.Tensor(rays))
        ts, ids, ray_ids = (hits[k].numpy() for k in ('t_hit', 'primitive_ids', 'ray_ids'))
        valid = (ts > 0) & (ts < 1)
        ts, ids, ray_ids = ts[valid], ids[valid], ray_ids[valid]
        triangles = np.asarray(vertices)[np.asarray(faces)[ids]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
        crossings = points[ray_ids] + ts[:, None] * (points[ray_ids + 1] - points[ray_ids])
        if not len(crossings) or (np.all(np.abs(normals[:, 1]) >= .7) and np.all(crossings[:, 1] >= start[1] + .4)):
            break
    else:
        raise ValueError('Overhead ascent crosses a surface that is not a ceiling')
    added = np.tile(np.eye(4), (len(u), 1, 1))
    added[:, :3, :3], added[:, :3, 3] = rotation.as_matrix(), candidate
    flight['overhead'] = {'start_frame': len(poses) - 1, 'ceiling_hits': crossings.tolist(),
                          'ceiling_frames': (len(poses) - 1 + ray_ids).tolist(), 'end_position': end.tolist()}
    flight['camera_to_world'] = np.concatenate((poses, added))
    flight['scans'].extend([flight['scans'][-1]] * len(added))
    if continuing_turn:
        last_event['turn_end_frame'] = min(len(flight['camera_to_world']) - 1, len(poses) - 1 + int(T * flight['fps']))
