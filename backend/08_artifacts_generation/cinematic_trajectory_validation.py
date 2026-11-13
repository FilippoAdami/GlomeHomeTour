"""Validate rendered camera samples and assign the viewer's floor plans."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from cinematic_trajectory import SPEED_FACTOR


def validate_trajectory(flight, plan, scene, vertices, faces, floor_scenes, attributor=None):
    poses = np.asarray(flight['camera_to_world'], dtype=float)
    scans = np.asarray(flight['scans'])
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(scans) != len(poses) or not np.isfinite(poses).all():
        raise ValueError('Invalid sampled camera poses or scan assignments')
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)):
        raise ValueError('Sampled camera poses are not rigid transforms')
    events = flight['checkpoints']
    if (len(events) != len(plan['checkpoints']) or not events or events[0]['frame'] != 0
            or any(not 0 <= event['frame'] < len(poses) for event in events)
            or any(a['frame'] >= b['frame'] for a, b in zip(events, events[1:]))):
        raise ValueError('Invalid checkpoint event sequence')
    points = poses[:, :3, 3]
    crossings = [piece['crossing'] for edge in plan['edges'] for piece in edge['pieces'] if piece['kind'] == 'door']
    planned_crossings = [crossing['panel_index'] for crossing in crossings]
    if [event['panel_index'] for event in flight['door_crossings']] != planned_crossings:
        raise ValueError('Sampled door crossings differ from the planned order')
    door_index = np.full(len(points), -1, dtype=int)
    for crossing_id, event in enumerate(flight['door_crossings']):
        first = max(0, event['start_frame'] - 1)
        last = min(len(points), event['end_frame'] + 2)
        if np.any(door_index[first:last] >= 0):
            raise ValueError('Overlapping door crossings')
        door_index[first:last] = crossing_id
    def within_door(samples, indices):
        allowed = np.zeros(len(samples), dtype=bool)
        for crossing_id, crossing in enumerate(crossings):
            mask = indices == crossing_id
            if not mask.any():
                continue
            center = np.asarray(crossing['center'])
            normal = np.asarray(crossing['normal'])
            relative = samples[mask] - center
            along = relative @ normal
            off_axis = np.linalg.norm(relative - along[:, None] * normal, axis=1)
            allowed[mask] = (np.abs(along) <= crossing['surface_slab_half_depth_m']) & (off_axis < .03)
        return allowed
    vectors = points[1:] - points[:-1]
    counts = np.maximum(1, np.ceil(np.linalg.norm(vectors, axis=1) / .02).astype(int))
    segment_ids = np.repeat(np.arange(len(vectors)), counts)
    offsets = np.concatenate([((np.arange(n) + .5) / n) for n in counts]) if len(counts) else np.array([])
    between = points[segment_ids] + offsets[:, None] * vectors[segment_ids]
    all_points = np.vstack((points, between)).astype(np.float32)
    indices = np.r_[door_index, door_index[segment_ids]]
    allowed = within_door(all_points, indices)
    overhead = flight.get('overhead')
    def within_ceiling(samples, frame_indices):
        result = np.zeros(len(samples), dtype=bool)
        if overhead and overhead['ceiling_hits']:
            active = frame_indices >= overhead['start_frame']
            hits = np.asarray(overhead['ceiling_hits'])
            result[active] = (np.linalg.norm(samples[active, None] - hits[None, :], axis=2).min(axis=1) < .28)
        return result
    sample_frames = np.r_[np.arange(len(points)), segment_ids]
    allowed |= within_ceiling(all_points, sample_frames)
    distance = scene.compute_distance(o3d.core.Tensor(all_points)).numpy()
    margins = np.r_[np.zeros(len(points)), np.linalg.norm(vectors[segment_ids], axis=1) / (2 * counts[segment_ids])]
    unsafe = np.flatnonzero((distance < .08 + margins) & ~allowed)
    if len(unsafe):
        raise ValueError(f'Insufficient sealed-mesh clearance at sampled point {unsafe[0]}')
    moving = np.linalg.norm(vectors, axis=1) > 1e-9
    moving_indices = np.flatnonzero(moving)
    rays = np.column_stack((points[:-1][moving], vectors[moving])).astype(np.float32)
    if len(rays):
        hits = scene.list_intersections(o3d.core.Tensor(rays))
        distance_along = hits['t_hit'].numpy()
        ids = hits['primitive_ids'].numpy()
        rays_hit = hits['ray_ids'].numpy()
        valid = (distance_along > 0) & (distance_along < 1)
        distance_along, ids, rays_hit = distance_along[valid], ids[valid], rays_hit[valid]
        ray_frames = moving_indices[rays_hit]
        hit_points = points[ray_frames] + distance_along[:, None] * vectors[ray_frames]
        ceiling_hits = within_ceiling(hit_points, ray_frames)
        allowed_hits = within_door(hit_points, door_index[ray_frames]) | ceiling_hits
        if not allowed_hits.all():
            raise ValueError(f'Sampled flight crosses sealed mesh at frame {ray_frames[np.flatnonzero(~allowed_hits)[0]]}')
        if len(ids):
            triangles = np.asarray(vertices)[np.asarray(faces)[ids]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
            if np.any(np.abs(normals[ceiling_hits, 1]) < .7):
                raise ValueError('Overhead ascent crosses a non-ceiling surface')
            for crossing_id, crossing in enumerate(crossings):
                select = (door_index[ray_frames] == crossing_id) & ~ceiling_hits
                if np.any(np.abs(normals[select] @ np.asarray(crossing['normal'])) < .5):
                    raise ValueError(f"Non-door-shaped surface hit at panel {crossing['panel_index']}")
                if select.any():
                    if attributor is None:
                        raise ValueError(f"Cannot verify source door faces at panel {crossing['panel_index']}")
                    attributor.attribute(crossing, hit_points[select])
    levels = np.empty(len(points), dtype=object)
    levels[:] = None
    for scan in sorted(set(scans)):
        indices_for_scan = np.flatnonzero(scans == scan)
        hits_by_floor = []
        names = []
        for (owner, name), floor in floor_scenes.items():
            if owner != scan:
                continue
            selected = points[indices_for_scan].astype(np.float32)
            rays = np.column_stack((selected, np.broadcast_to(np.array([0, -1, 0], np.float32), selected.shape)))
            hits_by_floor.append(floor.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy())
            names.append(name)
        if not names:
            raise ValueError(f'No floor plans for {scan}')
        hits_by_floor = np.asarray(hits_by_floor)
        nearest = np.argmin(hits_by_floor, axis=0)
        distance_to_floor = hits_by_floor[nearest, np.arange(len(indices_for_scan))]
        supported = np.isfinite(distance_to_floor) & (distance_to_floor > 0) & (distance_to_floor <= 2.8)
        levels[indices_for_scan[supported]] = np.asarray(names)[nearest[supported]]
    if overhead:
        levels[overhead['start_frame'] + 1:] = None
    unsupported_allowed = within_door(points, door_index)
    if overhead:
        unsupported_allowed[overhead['start_frame']:] = True
    invalid = np.flatnonzero((levels == None) & ~unsupported_allowed)  # noqa: E711
    if len(invalid):
        raise ValueError(f'No floor support at frame {invalid[0]}')
    # Check the rendered chords too, including gaps between widely spaced frames.
    for scan in sorted(set(scans[segment_ids])):
        select = (scans[segment_ids] == scan) & ~allowed[len(points):]
        if overhead:
            select &= segment_ids < overhead['start_frame']
        if not select.any():
            continue
        samples = between[select].astype(np.float32)
        rays = np.column_stack((samples, np.broadcast_to(np.array([0, -1, 0], np.float32), samples.shape)))
        supports = [floor.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
                    for (owner, _), floor in floor_scenes.items() if owner == scan]
        if not supports:
            raise ValueError(f'No floor plans for {scan}')
        hits = np.asarray(supports)
        if not np.all(np.any((hits > .08) & (hits <= 2.8), axis=0)):
            raise ValueError(f'No floor support between frames in {scan}')
    for i in range(len(levels)):
        if levels[i] is None:
            if i == 0:
                raise ValueError('No starting floor plan')
            levels[i] = levels[i - 1]
    panorama_turns = []
    for checkpoint, event in zip(plan['checkpoints'], flight['checkpoints']):
        frame = event['frame']
        passage = checkpoint.get('passage_position', checkpoint['position'])
        is_drone = flight.get('camera_mode') in ('drone', 'drone_smooth')
        tolerance = .03 if is_drone else 1e-6
        if (event['number'] != checkpoint['number'] or np.linalg.norm(points[frame] - passage) > tolerance
                or np.linalg.norm(points[frame] - checkpoint['position']) > (.31 if is_drone else 1e-6)):
            raise ValueError(f'Checkpoint {checkpoint["number"]} position or order mismatch')
        if checkpoint['kind'] == 'photo' and not np.allclose(poses[frame], checkpoint['camera_to_world'], atol=1e-6, rtol=0):
            raise ValueError(f'Checkpoint {checkpoint["number"]} photo pose mismatch')
        if checkpoint['kind'] == 'panorama':
            a, b = event['turn_start_frame'], event['turn_end_frame']
            if not 0 <= a <= frame <= b < len(poses):
                raise ValueError(f'Invalid panorama turn window for checkpoint {checkpoint["number"]}')
            forward = poses[a:b+1, :3, 2]
            yaw = np.unwrap(np.arctan2(forward[:, 2], forward[:, 0]))
            minimum = checkpoint['minimum_turn_degrees']
            if abs(yaw[-1] - yaw[0]) + 1e-6 < np.radians(minimum):
                raise ValueError(f'Checkpoint {checkpoint["number"]} panorama turn is too short')
            floor_turn = None
            if 'turn_floor' in event:
                if event['turn_floor'] != checkpoint['floor']:
                    raise ValueError(f'Checkpoint {checkpoint["number"]} turn floor does not match its viewpoint')
                first = last = frame
                while first > a and levels[first-1] == event['turn_floor'] and scans[first-1] == scans[frame]:
                    first -= 1
                while last < b and levels[last+1] == event['turn_floor'] and scans[last+1] == scans[frame]:
                    last += 1
                floor_forward = poses[first:last+1, :3, 2]
                floor_yaw = np.unwrap(np.arctan2(floor_forward[:, 2], floor_forward[:, 0]))
                if len(floor_yaw) < 2 or abs(floor_yaw[-1] - floor_yaw[0]) + 1e-6 < np.pi:
                    raise ValueError(f'Checkpoint {checkpoint["number"]} does not turn 180 degrees on its floor')
                floor_turn = float(np.degrees(abs(floor_yaw[-1] - floor_yaw[0])))
            panorama_turns.append({'checkpoint': checkpoint['number'], 'required_degrees': minimum,
                                   'actual_degrees': float(np.degrees(abs(yaw[-1] - yaw[0]))),
                                   'floor_turn_degrees': floor_turn})
        if levels[frame] != checkpoint['floor']:
            raise ValueError(f'Checkpoint {checkpoint["number"]} is assigned to {levels[frame]} rather than {checkpoint["floor"]}')
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    angular_speed = (rotations[:-1].inv() * rotations[1:]).magnitude() * flight['fps']
    angular_limit = 120 if flight.get('camera_mode') in ('drone', 'drone_smooth') else 25.1 * SPEED_FACTOR
    if np.max(angular_speed, initial=0) > np.radians(angular_limit):
        raise ValueError('Camera angular speed exceeds the tour limit')
    last_inside = overhead['start_frame'] if overhead else len(points)-1
    inside_angular = angular_speed[:last_inside]
    velocity = np.diff(points[:last_inside+1], axis=0) * flight['fps']
    acceleration = np.linalg.norm(np.diff(velocity, axis=0), axis=1) * flight['fps']
    all_velocity = np.diff(points, axis=0) * flight['fps']
    all_acceleration = np.linalg.norm(np.diff(all_velocity, axis=0), axis=1) * flight['fps']
    speed = np.linalg.norm(all_velocity, axis=1)
    jerk = np.linalg.norm(np.diff(all_velocity, n=2, axis=0), axis=1) * flight['fps'] ** 2
    if flight.get('camera_mode') == 'drone_smooth':
        turning = np.zeros(len(inside_angular), dtype=bool)
        for event in events:
            if 'turn_start_frame' in event:
                turning[max(0, event['turn_start_frame']-flight['fps']*2):
                        min(len(turning), event['turn_end_frame']+flight['fps']*2)] = True
        if (np.max(inside_angular[turning], initial=0) > np.radians(65.1)
                or np.max(inside_angular[~turning], initial=0) > np.radians(37.6)):
            raise ValueError('Interior camera angular speed exceeds the smooth-turn limit')
    if flight.get('camera_mode') == 'drone_smooth' and np.max(acceleration, initial=0) > 1.21:
        raise ValueError('Interior camera acceleration exceeds 1.2 metres per second squared')
    doors = []
    for event in flight['door_crossings']:
        first, last = event['start_frame'], event['end_frame']
        doors.append({'panel_index': event['panel_index'], 'duration_seconds': (last - first) / flight['fps'],
                      'minimum_speed_m_s': float(np.min(speed[first:last], initial=np.inf)) if last > first else 0.,
                      'maximum_speed_m_s': float(np.max(speed[first:last], initial=0))})
    return {'floor': levels.tolist(), 'minimum_non_door_clearance_m': float(distance[~allowed].min()),
            'panorama_turns': panorama_turns,
            'max_speed_m_s': float(np.max(speed, initial=0)),
            'max_jerk_m_s3': float(np.max(jerk, initial=0)),
            'door_motion': doors,
            'max_angular_speed_deg_s': float(np.degrees(np.max(angular_speed, initial=0))),
            'max_inside_angular_speed_deg_s': float(np.degrees(np.max(inside_angular, initial=0))),
            'inside_total_camera_rotation_degrees': float(np.degrees(inside_angular.sum()) / flight['fps']),
            'max_inside_acceleration_m_s2': float(np.max(acceleration, initial=0)),
            'max_total_acceleration_m_s2': float(np.max(all_acceleration, initial=0)),
            'door_mesh_hits': int(len(ids)) if len(rays) else 0}
