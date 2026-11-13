"""Continuous floor-supported drone paths and geometry-aware viewing directions."""
from __future__ import annotations

import math
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from cinematic_path_planning import clearance_and_support
from cinematic_trajectory import look_yaw


def _length(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _cut(points, distance):
    points = np.asarray(points, dtype=float)
    lengths = _length(points)
    exact = np.searchsorted(lengths, distance)
    if exact < len(lengths) and abs(lengths[exact] - distance) < 1e-9:
        return points[:exact + 1].copy(), points[exact:].copy()
    index = min(np.searchsorted(lengths, distance, side='right') - 1, len(points) - 2)
    index = max(0, index)
    t = (distance - lengths[index]) / (lengths[index + 1] - lengths[index])
    point = points[index] + t * (points[index + 1] - points[index])
    before = np.vstack((points[:index + 1], point))
    after = np.vstack((point, points[index + 1:]))
    return before, after


def _clear(scene, floors, points, clearance=.105):
    points = np.asarray(points, dtype=np.float32)
    supported = np.zeros(len(points), dtype=bool)
    for floor in floors:
        supported |= clearance_and_support(scene, floor, points, clearance)
    if not supported.all():
        return False
    rays = np.column_stack((points[:-1], np.diff(points, axis=0))).astype(np.float32)
    return bool(np.all(scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy() > 1))


def soften_corners(xyz, scans, events, door_ranges, scene, floor_scenes):
    """Round remaining sharp joins only when their complete replacement is clear."""
    xyz = xyz.copy()
    rejected = set()
    for _ in range(24):
        vectors = np.diff(xyz, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        cosine = np.einsum('ij,ij->i', vectors[:-1], vectors[1:]) / np.maximum(norms[:-1] * norms[1:], 1e-12)
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        candidates = [i + 1 for i in np.argsort(angles)[::-1] if angles[i] > 10 and i + 1 not in rejected]
        if not candidates:
            break
        center = candidates[0]
        lo, hi = max(0, center - 24), min(len(xyz) - 1, center + 24)
        while lo < center and scans[lo] != scans[center]:
            lo += 1
        while hi > center and scans[hi] != scans[center]:
            hi -= 1
        for d in door_ranges:
            start, end = d[0], d[1]
            if start <= center <= end:
                lo = hi = center
                break
            if lo < end < center:
                lo = end
            if center < start < hi:
                hi = start
        if hi - lo < 8:
            rejected.add(center)
            continue
        original = xyz[lo:hi+1]
        weights = np.sin(np.linspace(0, math.pi, len(original)))**2
        for e in events:
            if lo <= e['frame'] <= hi and e.get('is_photo', False):
                dist = np.abs(np.arange(len(original)) - (e['frame'] - lo))
                weights *= np.clip(dist / 4.0, 0, 1)
        smooth = gaussian_filter1d(original, sigma=7, axis=0, mode='nearest')
        candidate = original + weights[:, None] * (smooth - original)
        floors = [floor for (scan, _), floor in floor_scenes.items() if scan == scans[center]]
        candidate_steps = np.linalg.norm(np.diff(candidate, axis=0), axis=1)
        if (min(candidate_steps, default=0) < .012 or max(candidate_steps, default=0) > .03
                or any(np.linalg.norm(candidate[e['frame'] - lo] - e['position']) > .3
                       for e in events if lo <= e['frame'] <= hi)
                or not _clear(scene, floors, candidate, .093)):
            rejected.add(center)
            continue
        xyz[lo:hi+1] = candidate
    return xyz


def round_passages(plan, scene, floor_scenes):
    """Replace checkpoint corners with clearance-checked arcs near the marker."""
    cameras, edges = plan['checkpoints'], plan['edges']
    for i in range(len(cameras) - 2, 0, -1):
        camera = cameras[i]
        if camera.get('kind') == 'photo':
            continue
        left, right = edges[i - 1]['pieces'][-1], edges[i]['pieces'][0]
        if left['kind'] != 'local' or right['kind'] != 'local' or left['scan'] != right['scan']:
            continue
        a, b = np.asarray(left['points'], float), np.asarray(right['points'], float)
        da, db = _length(a)[-1], _length(b)[-1]
        floors = [floor for (scan, _), floor in floor_scenes.items() if scan == camera['scan']]
        for radius in (.75, .55, .4, .25, .15):
            r = min(radius, da * .85, db * .7)
            if r < .04:
                break
            before, _ = _cut(a, da - r)
            _, after = _cut(b, r)
            u = np.linspace(0, 1, max(9, math.ceil(2 * r / .025) + 1))[:, None]
            curve = (1-u)**2 * before[-1] + 2 * (1-u) * u * a[-1] + u**2 * after[0]
            mid = len(curve) // 2
            if np.linalg.norm(curve[mid] - np.asarray(camera['position'])) > .3:
                continue
            if _clear(scene, floors, curve):
                left['points'] = np.vstack((before, curve[1:mid+1])).tolist()
                right['points'] = np.vstack((curve[mid:-1], after)).tolist()
                camera['passage_position'] = curve[mid].tolist()
                break


def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def refine_panorama_framing(flight, plan, scene, floors):
    """Reduce nearby obstructions around panoramas without adding camera spin."""
    poses = np.asarray(flight['camera_to_world'], dtype=float).copy()
    fps = flight['fps']
    scans = flight['scans']
    events = flight['checkpoints']
    inside_end = flight['overhead']['start_frame'] if flight.get('overhead') else len(poses) - 1
    turning = np.zeros(len(poses) - 1, dtype=bool)
    anchors = {0, inside_end}
    for event, camera in zip(events, plan['checkpoints']):
        if camera['kind'] == 'photo':
            anchors.add(event['frame'])
        if 'turn_start_frame' in event:
            a, b = event['turn_start_frame'], event['turn_end_frame']
            anchors.update((a, b))
            turning[max(0, a - 2 * fps):min(len(turning), b + 2 * fps)] = True
    for door in flight['door_crossings']:
        anchors.update((door['start_frame'], door['end_frame']))
    changes = []
    for event, camera in zip(events, plan['checkpoints']):
        if camera['kind'] != 'panorama':
            continue
        frame = event['frame']
        if any(door['start_frame'] <= frame <= door['end_frame'] for door in flight['door_crossings']):
            continue
        lo = max(0, event['turn_start_frame'], frame - 3 * fps)
        hi = min(inside_end, event['turn_end_frame'], frame + 3 * fps)
        for anchor in anchors:
            if lo <= anchor < frame:
                lo = max(lo, anchor)
            elif frame < anchor <= hi:
                hi = min(hi, anchor)
        if 'turn_floor' in event:
            first = last = frame
            while first > lo and floors[first - 1] == floors[frame] and scans[first - 1] == scans[frame]:
                first -= 1
            while last < hi and floors[last + 1] == floors[frame] and scans[last + 1] == scans[frame]:
                last += 1
            lo, hi = first, last
        if min(frame - lo, hi - frame) < max(1, fps // 2):
            continue
        indices = np.arange(lo, hi + 1)
        u = np.where(indices <= frame, (indices - lo) / (frame - lo), (hi - indices) / (hi - frame))
        weight = 6 * u**5 - 15 * u**4 + 10 * u**3
        score_mask = np.abs(indices - frame) <= fps
        origins = poses[indices, :3, 3]
        forward = poses[indices, :3, 2]

        def near_penalty(degrees):
            angle = np.radians(degrees) * weight[:, None] + np.radians([-24., 0., 24.])[None, :]
            c, s = np.cos(angle), np.sin(angle)
            directions = np.stack((forward[:, 0, None] * c - forward[:, 2, None] * s,
                                   np.broadcast_to(forward[:, 1, None], c.shape),
                                   forward[:, 0, None] * s + forward[:, 2, None] * c), axis=-1)
            rays = np.concatenate((np.broadcast_to(origins[:, None, :], directions.shape), directions), axis=-1)
            hits = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6).astype(np.float32)))['t_hit'].numpy()
            penalty = np.maximum(1. - np.minimum(hits.reshape(-1, 3), 1.), 0.) ** 2
            return np.average(penalty, axis=1, weights=[.5, 1., .5])

        original_penalty = near_penalty(0.)
        before = float(original_penalty[score_mask].mean())
        before_window = float(original_penalty.mean())
        if before < .02:
            continue
        start, stop = max(0, lo - 1), min(len(poses), hi + 2)
        old = Rotation.from_matrix(poses[start:stop, :3, :3])
        old_rates = (old[:-1].inv() * old[1:]).magnitude() * fps
        limits = np.where(turning[start:stop-1], math.radians(65.), math.radians(37.5))
        best = None
        for degrees in (-30., -20., -10., 10., 20., 30.):
            penalty = near_penalty(degrees)
            score, window_score = float(penalty[score_mask].mean()), float(penalty.mean())
            if before - score < max(.005, .1 * before) or window_score > before_window + 1e-7:
                continue
            correction = Rotation.from_rotvec(np.column_stack((np.zeros(len(indices)),
                                                                  -np.radians(degrees) * weight,
                                                                  np.zeros(len(indices)))))
            rotations = poses[start:stop, :3, :3].copy()
            rotations[indices - start] = (correction * Rotation.from_matrix(poses[indices, :3, :3])).as_matrix()
            candidate = Rotation.from_matrix(rotations)
            rates = (candidate[:-1].inv() * candidate[1:]).magnitude() * fps
            if (np.any(rates > limits) or rates.sum() > old_rates.sum() + math.radians(1) * fps):
                continue
            if best is None or score < best[0]:
                best = score, degrees, rotations, window_score
        if best is not None:
            poses[start:stop, :3, :3] = best[2]
            changes.append({'checkpoint': camera['number'], 'start_frame': lo, 'end_frame': hi,
                            'yaw_adjustment_degrees': best[1], 'near_surface_penalty_before': before,
                            'near_surface_penalty_after': best[0], 'window_penalty_before': before_window,
                            'window_penalty_after': best[3]})
    return poses, changes


def _view_yaw(points, scene, fps, door_ranges, panoramas=(), photos=()):
    """Prefer open views while penalizing unnecessary camera rotation."""
    step = max(1, math.ceil(fps / 6))
    frames = np.unique(np.r_[np.arange(0, len(points), step), len(points)-1]).astype(int)
    angles = np.arange(72) * (2 * math.pi / 72)
    sample = points[frames]
    offsets = np.radians([-24., 0., 24.])
    direction = np.stack((np.cos(angles[:, None] + offsets),
                          np.zeros((len(angles), len(offsets))),
                          np.sin(angles[:, None] + offsets)), axis=-1)
    origins = np.broadcast_to(sample[:, None, None, :], (len(frames), len(angles), len(offsets), 3))
    dirs = np.broadcast_to(direction[None], origins.shape)
    rays = np.concatenate((origins, dirs), axis=-1).reshape(-1, 6).astype(np.float32)
    hits = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy().reshape(len(frames), len(angles), len(offsets))
    hits = np.minimum(hits, 3.)
    visibility = .3 * hits[:, :, 0] + 1.4 * hits[:, :, 1] + .3 * hits[:, :, 2]
    lookahead = np.minimum(frames + fps, len(points) - 1)
    velocity = points[lookahead] - sample
    travel = np.arctan2(velocity[:, 2], velocity[:, 0])
    preference = np.full(len(frames), 3.0)
    costs = (-visibility + preference[:, None] * (1 - np.cos(angles[None, :] - travel[:, None]))
             + 48. * np.maximum(1. - hits[:, :, 1], 0) ** 2)

    for d in door_ranges:
        start, end = d[0], d[1]
        if len(d) > 2 and d[2] is not None:
            door_yaw = d[2]
        else:
            diff = points[min(len(points) - 1, end)] - points[start]
            door_yaw = math.atan2(diff[2], diff[0]) if np.linalg.norm(diff[[0, 2]]) > 1e-6 else travel[min(len(travel) - 1, np.searchsorted(frames, start))]
        upcoming_yaw = d[3] if len(d) > 3 and d[3] is not None else door_yaw

        app_mask = (frames >= start - int(1.2 * fps)) & (frames < start)
        if app_mask.any():
            costs[app_mask] -= preference[app_mask, None] * (1 - np.cos(angles[None, :] - travel[app_mask, None]))
            costs[app_mask] += 120. * (1 - np.cos(angles[None, :] - door_yaw))

        trav_mask = (frames >= start) & (frames <= end)
        if trav_mask.any():
            f_trav = frames[trav_mask]
            u = np.clip((f_trav - start) / max(end - start, 1), 0, 1)
            delta = _wrap(upcoming_yaw - door_yaw)
            guide = door_yaw + u * delta
            costs[trav_mask] -= preference[trav_mask, None] * (1 - np.cos(angles[None, :] - travel[trav_mask, None]))
            costs[trav_mask] += 40. * (1 - np.cos(angles[None, :] - guide[:, None]))

        dep_mask = (frames > end) & (frames <= end + int(0.8 * fps))
        if dep_mask.any():
            u = np.clip((frames[dep_mask] - end) / max(int(0.8 * fps), 1), 0, 1)
            costs[dep_mask] += (25. * (1 - u[:, None])) * (1 - np.cos(angles[None, :] - upcoming_yaw))

    for event in panoramas:
        if 'turn_yaw' in event:
            envelope = np.exp(-.5 * ((frames - event['frame']) / fps) ** 2)
            costs += 30. * envelope[:, None] * (1 - np.cos(angles[None, :] - event['turn_yaw']))
    for frame, yaw in photos:
        envelope = np.exp(-.5 * ((frames - frame) / (.7 * fps)) ** 2)
        costs += 500. * envelope[:, None] * (1 - np.cos(angles[None, :] - yaw))
    difference = _wrap(angles[None, :] - angles[:, None])
    transition = 120. * difference**2 + .35 * (np.abs(difference) > 1e-6)
    transition[np.abs(difference) > np.radians(5.01)] = 1e6
    windows = []
    for i, event in enumerate(panoramas):
        center = event['frame']
        lo = max(0, center - 6 * fps, (panoramas[i-1]['frame'] + center) // 2 if i else 0)
        hi = min(len(points)-1, center + 5 * fps,
                 (center + panoramas[i+1]['frame']) // 2 - 1 if i+1 < len(panoramas) else len(points)-1)
        if 'turn_floor_window' in event:
            fl_start, fl_end = event['turn_floor_window']
            lo = max(lo, fl_start)
            hi = min(hi, fl_end)
        first, last = np.searchsorted(frames, [lo, hi], side='left')
        last = min(last, len(frames)-1)
        if windows:
            first = max(first, windows[-1][1] + 1)
        windows.append((int(first), int(last)))
    for i in range(len(windows)-1, -1, -1):
        first, last = windows[i]
        if last - first < 29 and i:
            previous_first, previous_last = windows[i-1]
            previous_center = int(np.searchsorted(frames, panoramas[i-1]['frame']))
            earliest = max(previous_first + 30, previous_center + 1)
            first = max(earliest, last - 29)
            windows[i] = (first, last)
            windows[i-1] = (previous_first, min(previous_last, first - 1))
        if last - first < 29:
            raise ValueError(f'Not enough continuous flight time for panorama {panoramas[i]["number"]}')
    for (first, last), event in zip(windows, panoramas):
        event['turn_start_frame'], event['turn_end_frame'] = int(frames[first]), int(frames[last])
    dp = costs[0].copy()
    back = [None] * len(frames)
    pan_start = {a for a, _ in windows}
    progress_at_start = {a: 60 for a, _ in windows}
    progress_at_end = {b: progress_at_start[a] for a, b in windows}

    progress = max(progress_at_start.values(), default=0)
    state_ids = np.arange(2 * (progress + 1) * len(angles)).reshape(2, progress + 1, len(angles))
    focus = np.zeros(len(frames))
    for (first, last), event in zip(windows, panoramas):
        if 'turn_floor' in event:
            distance = np.linalg.norm(sample[first:last+1] - points[event['frame']], axis=1)
            focus[first:last+1] = np.maximum(distance - .25, 0) ** 2
    active = False
    active_progress = 0
    if 0 in pan_start:
        expanded = np.full((2, progress + 1, len(angles)), np.inf)
        expanded[:, 0] = dp
        dp, active, active_progress = expanded, True, progress_at_start[0]
    for i in range(1, len(frames)):
        end_states = None
        if i-1 in progress_at_end:
            active_progress = progress_at_end[i-1]
            choices = dp[:, active_progress, :]
            if not np.isfinite(choices).any():
                raise ValueError('Not enough time for a continuous panorama turn')
            sign = np.argmin(choices, axis=0)
            end_states = state_ids[sign, active_progress, np.arange(len(angles))]
            dp, active = choices[sign, np.arange(len(angles))], False
        if active:
            new = np.full_like(dp, np.inf)
            pointer = np.full(dp.shape, -1, np.int16)
            for sign in range(2):
                for advance in (0, 1, 2, 3):
                    source = np.roll(dp[sign], advance if sign == 0 else -advance, axis=1)
                    source_ids = np.roll(state_ids[sign], advance if sign == 0 else -advance, axis=1)
                    penalty = (300. * focus[i] * advance + 120. * (advance * 2 * math.pi / 72) ** 2
                               + .35 * (advance > 0))
                    candidate = source[:active_progress + 1-advance] + penalty
                    target = new[sign, advance:active_progress + 1]
                    mask = candidate < target
                    target[:] = np.where(mask, candidate, target)
                    pointer_target = pointer[sign, advance:active_progress + 1]
                    pointer_target[:] = np.where(mask, source_ids[:active_progress + 1-advance], pointer_target)
            dp = new + costs[i][None, None, :]
            back[i] = pointer
        else:
            choices = dp[:, None] + transition
            pointer = np.argmin(choices, axis=0)
            dp = costs[i] + choices[pointer, np.arange(len(angles))]
            back[i] = (end_states[pointer] if end_states is not None else pointer).astype(np.int16)
        if i in pan_start:
            expanded = np.full((2, progress + 1, len(angles)), np.inf)
            expanded[:, 0] = dp
            pointer = np.full(expanded.shape, -1, np.int16)
            pointer[:, 0] = back[i]
            dp, back[i], active, active_progress = expanded, pointer, True, progress_at_start[i]
    sequence = np.empty(len(frames), int)
    state = int(np.argmin(dp))
    if active:
        if not np.isfinite(dp[:, active_progress, :]).any():
            raise ValueError('Not enough time for a continuous panorama turn')
        sign, angle = np.unravel_index(np.argmin(dp[:, active_progress, :]), (2, len(angles)))
        state = int(state_ids[sign, active_progress, angle])
    sequence[-1] = state % len(angles)
    for i in range(len(frames)-1, 0, -1):
        state = int(back[i].reshape(-1)[state])
        sequence[i-1] = state % len(angles)
    unwrapped = np.unwrap(angles[sequence])
    interpolated = np.interp(np.arange(len(points)), frames, unwrapped)
    result = gaussian_filter1d(interpolated, sigma=max(1, (1.15 if panoramas else 2.) * fps), mode='nearest')
    if any('turn_floor' in event for event in panoramas):
        focused = gaussian_filter1d(interpolated, sigma=max(1, .85 * fps), mode='nearest')
        for (first, last), event in zip(windows, panoramas):
            if 'turn_floor' in event:
                t = np.arange(len(points))
                blend = np.clip(np.minimum(t - frames[first], frames[last] - t) / fps, 0, 1)
                blend = blend * blend * (3 - 2 * blend)
                result += blend * (focused - result)
    return result


def smooth_motion(points, scans, events, cameras, door_ranges, scene, floor_scenes, fps):
    """Smooth the timed flight while preserving door passages and safety checks."""
    door_bounds = [(d[0], d[1]) for d in door_ranges]
    weights = np.ones(len(points))
    for start, end in [(0, 0), (len(points)-1, len(points)-1)]:
        distance = np.maximum(np.maximum(start - np.arange(len(points)), np.arange(len(points)) - end), 0)
        u = np.clip(distance / (1.5 * fps), 0, 1)
        weights = np.minimum(weights, u*u*(3-2*u))
    for start, end in door_bounds:
        distance = np.maximum(np.maximum(start - np.arange(len(points)), np.arange(len(points)) - end), 0)
        u = np.clip(distance / (0.4 * fps), 0, 1)
        weights = np.minimum(weights, u*u*(3-2*u))
    for seconds in (.7, .5, .3):
        filtered = gaussian_filter1d(points, sigma=seconds * fps, axis=0, mode='nearest')
        adjusted = weights.copy()
        for event, camera in zip(events, cameras):
            frame = event['frame']
            shift = filtered[frame] - points[frame]
            max_drift = 0.0 if camera.get('kind') == 'photo' else 0.3
            for factor in np.linspace(1, 0, 101):
                if np.linalg.norm(points[frame] + factor * adjusted[frame] * shift - camera['position']) <= max_drift:
                    break
            distance = (np.arange(len(points)) - frame) / (1.5 * fps)
            adjusted *= 1 - (1-factor) * np.exp(-.5 * distance**2)
        candidate = points + adjusted[:, None] * (filtered - points)
        safe = True
        for scan in set(scans):
            selected = np.array([owner == scan for owner in scans])
            for start, end in door_bounds:
                selected[max(0, start-2):min(len(points), end+3)] = False
            floors = [floor for (owner, _), floor in floor_scenes.items() if owner == scan]
            for block in np.split(np.flatnonzero(selected), np.flatnonzero(np.diff(np.flatnonzero(selected)) > 1) + 1):
                if len(block) and not _clear(scene, floors, candidate[block], .085):
                    safe = False
                    break
            if not safe:
                break
        if safe:
            return candidate
    return points


def retime_path(points, scans, events, door_ranges, fps, max_speed, cameras=None, interpolation='pchip'):
    """Slow smoothly for curvature and limit acceleration along the route."""
    if interpolation not in ('pchip', 'linear'):
        raise ValueError(f'Unsupported trajectory interpolation: {interpolation}')
    distance = _length(points)
    ds = np.diff(distance)
    if np.any(ds <= 1e-8):
        raise ValueError('Repeated sampled points prevent smooth timing')
    directions = np.diff(points, axis=0) / ds[:, None]
    curvature = np.zeros(len(points))
    curvature[1:-1] = np.linalg.norm(np.diff(directions, axis=0), axis=1) / ((ds[:-1] + ds[1:]) / 2)
    curvature = maximum_filter1d(curvature, size=5, mode='nearest')
    for d in door_ranges:
        s, e = d[0], d[1]
        curvature[max(0, s - 6):min(len(curvature), e + 7)] = 0.0
    velocity = np.minimum(max_speed, np.sqrt(1.05 / np.maximum(curvature, 1e-8)))
    for d in door_ranges:
        s, e = d[0], d[1]
        lo = max(0, s - 6)
        hi = min(len(velocity), e + 7)
        velocity[lo:hi] = np.maximum(velocity[lo:hi], min(max_speed, 1.25))
    for event in events:
        if 'turn_floor' in event:
            f = event['frame']
            lo = max(0, f - 45)
            hi = min(len(velocity), f + 45)
            velocity[lo:hi] = np.minimum(velocity[lo:hi], 0.22)
        else:
            cam = next((c for c in (cameras or []) if c['number'] == event['number']), None)
            if cam and cam.get('kind') == 'panorama':
                f = event['frame']
                lo = max(0, f - int(1.2 * fps))
                hi = min(len(velocity), f + int(1.2 * fps))
                velocity[lo:hi] = np.minimum(velocity[lo:hi], 0.35)
    if cameras is not None and cameras[-1].get('kind') == 'panorama':
        f = events[-1]['frame']
        lo = max(0, f - int(1.2 * fps))
        velocity[lo:] = np.minimum(velocity[lo:], 0.35)
    a_accel = 0.95
    a_brake = 1.10
    for _ in range(15):
        for i in range(1, len(velocity)):
            velocity[i] = min(velocity[i], math.sqrt(velocity[i-1] ** 2 + 2 * a_accel * ds[i-1]))
        for i in range(len(velocity) - 2, -1, -1):
            velocity[i] = min(velocity[i], math.sqrt(velocity[i+1] ** 2 + 2 * a_brake * ds[i]))
        times = np.r_[0., np.cumsum(2 * ds / (velocity[:-1] + velocity[1:]))]
        count = max(1, math.ceil(times[-1] * fps))
        knots = np.unique(np.r_[0, [event['frame'] for event in events], len(points)-1]).astype(int)
        knot_frames = np.rint(times[knots] / times[-1] * count).astype(int)
        if np.any(np.diff(knot_frames) <= 0):
            raise ValueError('Checkpoints are too close together for the requested frame rate')
        sampled_time = PchipInterpolator(knot_frames, times[knots])(np.arange(count + 1))
        sampled_distance = np.interp(sampled_time, times, distance)
        sampled_points = (PchipInterpolator(distance, points, axis=0)(sampled_distance) if interpolation == 'pchip'
                          else np.column_stack([np.interp(sampled_distance, distance, points[:, axis]) for axis in range(3)]))
        sampled_velocity = np.diff(sampled_points, axis=0) * fps
        measured = np.linalg.norm(np.diff(sampled_velocity, axis=0), axis=1) * fps
        violations = np.flatnonzero(measured > 1.20)
        if not len(violations):
            break
        for index in violations:
            nearest = int(np.searchsorted(distance, sampled_distance[index + 1]))
            lo, hi = max(0, nearest - 3), min(len(velocity), nearest + 4)
            velocity[lo:hi] *= .92 * math.sqrt(1.20 / measured[index])
    indices = np.clip(np.searchsorted(distance, sampled_distance, side='right') - 1, 0, len(scans) - 1)
    sampled_scans = [scans[index] for index in indices]
    def frame(index):
        return int(np.argmin(np.abs(sampled_time - times[index])))
    sampled_events = [{**event, 'frame': frame(event['frame'])} for event in events]
    sampled_doors = [(frame(d[0]), frame(d[1]), *d[2:]) for d in door_ranges]
    return sampled_points, sampled_scans, sampled_events, sampled_doors


def sample_drone_trajectory(plan, scene, floor_scenes, fps=30, speed=1.25, interpolation='pchip'):
    """Pass checkpoints with restrained camera turns and curvature-aware speed."""
    if fps <= 0 or speed <= 0 or interpolation not in ('pchip', 'linear') or len(plan['edges']) != len(plan['checkpoints']) - 1:
        raise ValueError('Invalid drone path inputs')
    round_passages(plan, scene, floor_scenes)
    cameras, edges = plan['checkpoints'], plan['edges']
    points, owners, checkpoints, doors = [], [], [], []
    total = 0.
    def add(path, owner):
        nonlocal total
        path = np.asarray(path, float)
        if points and not np.allclose(points[-1], path[0], atol=1e-6):
            raise ValueError('Disconnected drone path')
        for p in path if not points else path[1:]:
            if points:
                total += float(np.linalg.norm(p - points[-1]))
            points.append(p)
            owners.append(owner)
    add([cameras[0]['position']], cameras[0]['scan'])
    checkpoints.append({'number': cameras[0]['number'], 'distance': 0.})
    for i, edge in enumerate(edges):
        if edge['from_checkpoint'] != cameras[i]['number'] or edge['to_checkpoint'] != cameras[i+1]['number']:
            raise ValueError('Edge order differs from checkpoints')
        for piece in edge['pieces']:
            if piece['kind'] == 'door':
                start = total
                segment = np.asarray(piece['points'], float)
                midpoint = (segment[0] + segment[-1]) / 2
                add([segment[0], midpoint], piece['scan_from'])
                owners[-1] = piece['scan_to']
                add([midpoint, segment[-1]], piece['scan_to'])
                v = segment[-1] - segment[0]
                door_heading = math.atan2(v[2], v[0])
                to_cam = next(c for c in cameras if c['number'] == edge['to_checkpoint'])
                if to_cam['kind'] == 'photo':
                    rot = np.asarray(to_cam['camera_to_world'])[:3, :3]
                    upcoming = math.atan2(rot[2, 2], rot[0, 2])
                else:
                    upcoming = to_cam.get('yaw', door_heading)
                doors.append({'panel_index': piece['crossing']['panel_index'], 'from_checkpoint': edge['from_checkpoint'],
                              'to_checkpoint': edge['to_checkpoint'], 'start': start, 'end': total,
                              'door_yaw': door_heading, 'upcoming_yaw': upcoming})
            else:
                add(piece['points'], piece['scan'])
        camera = cameras[i + 1]
        checkpoints.append({'number': camera['number'], 'distance': total})
    points = np.asarray(points)
    distance = _length(points)
    if np.any(np.diff(distance) <= 1e-9):
        raise ValueError('Repeated points in drone path')
    count = max(1, math.ceil(distance[-1] / .7 * fps))
    travel = np.linspace(0, distance[-1], count + 1)
    xyz = np.column_stack([np.interp(travel, distance, points[:, axis]) for axis in range(3)])
    indices = np.clip(np.searchsorted(distance, travel, side='right') - 1, 0, len(owners)-1)
    scans = [owners[i] for i in indices]
    events = [{'number': row['number'], 'frame': int(np.argmin(abs(travel-row['distance'])))} for row in checkpoints]
    door_ranges = [(int(np.searchsorted(travel, d['start'])), int(np.searchsorted(travel, d['end'])),
                    d['door_yaw'], d['upcoming_yaw']) for d in doors]
    event_positions = [{'frame': event['frame'], 'position': np.asarray(camera['position']),
                        'is_photo': camera['kind'] == 'photo'}
                       for event, camera in zip(events, cameras)]
    for i, camera in enumerate(cameras):
        if camera['kind'] == 'panorama':
            neighbors = [cameras[j] for j in (i-1, i+1) if 0 <= j < len(cameras)]
            if any(other['scan'] == camera['scan'] and other['floor'] != camera['floor'] for other in neighbors):
                events[i]['turn_floor'] = camera['floor']
                if 'yaw' in camera:
                    events[i]['turn_yaw'] = camera['yaw']
    xyz = soften_corners(xyz, scans, event_positions, door_ranges, scene, floor_scenes)
    for event, camera in zip(events, cameras):
        if np.linalg.norm(xyz[event['frame']] - camera['position']) > .03:
            camera['passage_position'] = xyz[event['frame']].tolist()
    xyz, scans, events, door_ranges = retime_path(xyz, scans, events, door_ranges, fps, speed, cameras=cameras,
                                                   interpolation=interpolation)
    xyz = smooth_motion(xyz, scans, events, cameras, door_ranges, scene, floor_scenes, fps)
    photos = [(event['frame'], np.asarray(camera['position'], float))
              for event, camera in zip(events, cameras) if camera['kind'] == 'photo']
    for i, (frame, target) in enumerate(photos):
        lo = max(0, frame - fps, (photos[i-1][0] + frame) // 2 + 1 if i else 0)
        hi = min(len(xyz) - 1, frame + fps, (frame + photos[i+1][0]) // 2 if i + 1 < len(photos) else len(xyz) - 1)
        for d in door_ranges:
            start, end = d[0], d[1]
            if end < frame:
                lo = max(lo, start)
            elif start > frame:
                hi = min(hi, end)
        if hi <= lo:
            continue
        t = np.arange(lo, hi + 1)
        u = np.where(t <= frame, (t - lo) / max(frame - lo, 1), (hi - t) / max(hi - frame, 1))
        weight = 6 * u**5 - 15 * u**4 + 10 * u**3
        weight[t == frame] = 1.0
        xyz[lo:hi+1] += weight[:, None] * (target - xyz[frame])
    xyz, scans, events, door_ranges = retime_path(xyz, scans, events, door_ranges, fps, speed, cameras=cameras,
                                                   interpolation=interpolation)
    for event, camera in zip(events, cameras):
        camera['passage_position'] = xyz[event['frame']].tolist()
    points = xyz
    scans_arr = np.asarray(scans)
    levels = np.empty(len(points), dtype=object)
    levels[:] = None
    for scan in sorted(set(scans)):
        indices_for_scan = np.flatnonzero(scans_arr == scan)
        hits_by_floor = []
        names = []
        for (owner, name), floor in floor_scenes.items():
            if owner != scan:
                continue
            selected = points[indices_for_scan].astype(np.float32)
            rays = np.column_stack((selected, np.broadcast_to(np.array([0, -1, 0], np.float32), selected.shape)))
            hits_by_floor.append(floor.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy())
            names.append(name)
        hits_by_floor = np.asarray(hits_by_floor)
        nearest = np.argmin(hits_by_floor, axis=0)
        distance_to_floor = hits_by_floor[nearest, np.arange(len(indices_for_scan))]
        supported = np.isfinite(distance_to_floor) & (distance_to_floor > 0) & (distance_to_floor <= 2.8)
        levels[indices_for_scan[supported]] = np.asarray(names)[nearest[supported]]
    for i in range(len(levels)):
        if levels[i] is None:
            levels[i] = levels[i-1]
    panoramas = []
    for i, (event, camera) in enumerate(zip(events, cameras)):
        if camera['kind'] != 'panorama':
            continue
        event['near_scan_transition'] = any(
            0 <= j < len(cameras) and cameras[j]['scan'] != camera['scan'] for j in (i - 1, i + 1))
        if 'turn_floor' in event:
            f = event['frame']
            first = last = f
            while first > 0 and levels[first-1] == event['turn_floor'] and scans[first-1] == scans[f]:
                first -= 1
            while last < len(levels)-1 and levels[last+1] == event['turn_floor'] and scans[last+1] == scans[f]:
                last += 1
            event['turn_floor_window'] = (first, last)
        panoramas.append(event)
    photo_rotations = [(event['frame'], Rotation.from_matrix(np.asarray(camera['camera_to_world'])[:3, :3]))
                       for event, camera in zip(events, cameras) if camera['kind'] == 'photo']
    photo_yaws = [(frame, math.atan2(rotation.as_matrix()[2, 2], rotation.as_matrix()[0, 2]))
                  for frame, rotation in photo_rotations]
    yaw = _view_yaw(xyz, scene, fps, door_ranges, panoramas, photo_yaws)
    rotations = Rotation.concatenate([look_yaw(float(value)) for value in yaw]).as_matrix()
    for i, (frame, target) in enumerate(photo_rotations):
        half = int(2.0 * fps)
        lo = max(0, frame - half, (photo_rotations[i-1][0] + frame) // 2 + 1 if i else 0)
        hi = min(len(xyz) - 1, frame + half,
                 (frame + photo_rotations[i+1][0]) // 2 if i + 1 < len(photo_rotations) else len(xyz) - 1)
        for d in door_ranges:
            start, end = d[0], d[1]
            if end < frame:
                lo = max(lo, start)
            elif start > frame:
                hi = min(hi, end)
        if hi <= lo or frame < lo or frame > hi:
            continue
        t = np.arange(lo, hi + 1)
        u = np.where(t <= frame, (t - lo) / max(frame - lo, 1), (hi - t) / max(hi - frame, 1))
        weight = 6 * u**5 - 15 * u**4 + 10 * u**3
        weight[t == frame] = 1.0
        correction = target * Rotation.from_matrix(rotations[frame]).inv()
        blend = Slerp([0, 1], Rotation.concatenate([Rotation.identity(), correction]))(weight)
        rotations[lo:hi+1] = (blend * Rotation.from_matrix(rotations[lo:hi+1])).as_matrix()
    for _ in range(3):
        camera_rotations = Rotation.from_matrix(rotations)
        rates = (camera_rotations[:-1].inv() * camera_rotations[1:]).magnitude() * fps
        turning = np.zeros(len(rates), dtype=bool)
        for event in panoramas:
            turning[max(0, event['turn_start_frame']-2*fps):min(len(rates), event['turn_end_frame']+2*fps)] = True
        acceleration = np.linalg.norm(np.diff(xyz, n=2, axis=0), axis=1) * fps**2
        factor = max(1., np.max(rates[turning], initial=0) / math.radians(65),
                     np.max(rates[~turning], initial=0) / math.radians(37.5),
                     math.sqrt(np.max(acceleration, initial=0) / 1.18))
        if factor <= 1:
            break
        count = math.ceil((len(xyz)-1) * factor * 1.04)
        anchors = {0, len(xyz)-1}
        for event in events:
            anchors.add(event['frame'])
            if 'turn_start_frame' in event:
                anchors.update((event['turn_start_frame'], event['turn_end_frame']))
        for bounds in door_ranges:
            anchors.update(bounds[:2])
        source = np.array(sorted(anchors))
        destination = np.rint(source * count / (len(xyz)-1)).astype(int)
        if np.any(np.diff(destination) <= 0):
            raise ValueError('Cannot retime closely spaced tour events')
        sample = np.clip(PchipInterpolator(destination, source)(np.arange(count+1)), 0, len(xyz)-1)
        xyz = PchipInterpolator(np.arange(len(xyz)), xyz, axis=0)(sample)
        rotations = Slerp(np.arange(len(rotations)), camera_rotations)(sample).as_matrix()
        scans = [scans[min(int(index), len(scans)-1)] for index in sample]
        remap = dict(zip(source.tolist(), destination.tolist()))
        for event in events:
            event['frame'] = remap[event['frame']]
            if 'turn_start_frame' in event:
                event['turn_start_frame'] = remap[event['turn_start_frame']]
                event['turn_end_frame'] = remap[event['turn_end_frame']]
        door_ranges = [(remap[d[0]], remap[d[1]], *d[2:]) for d in door_ranges]
    poses = np.tile(np.eye(4), (len(xyz), 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotations, xyz
    door_events = [{'panel_index': d['panel_index'], 'from_checkpoint': d['from_checkpoint'],
                    'to_checkpoint': d['to_checkpoint'], 'start_frame': bounds[0], 'end_frame': bounds[1]}
                   for d, bounds in zip(doors, door_ranges)]
    return {'fps': fps, 'camera_to_world': poses, 'scans': list(scans), 'checkpoints': events,
            'door_crossings': door_events, 'camera_mode': 'drone_smooth'}
