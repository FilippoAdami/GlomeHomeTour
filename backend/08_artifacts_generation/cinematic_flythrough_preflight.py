#!/usr/bin/env python3
"""Inspect published tour inputs and straight-line obstructions; this does not plan a flight."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import open3d as o3d

STAGE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def ordered_checkpoints(tour):
    cameras = tour['cameras']
    if not cameras or [c['number'] for c in cameras] != list(range(1, len(cameras) + 1)):
        raise ValueError('Viewer checkpoint numbers must be consecutive and already ordered')
    scans = []
    for camera in cameras:
        if camera['kind'] not in ('photo', 'panorama'):
            raise ValueError(f"Unknown camera kind at checkpoint {camera['number']}")
        p = np.asarray(camera['position'], dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(f"Invalid position at checkpoint {camera['number']}")
        if not scans or scans[-1] != camera['scan']:
            if camera['scan'] in scans:
                raise ValueError(f"Scan {camera['scan']} is split across checkpoint groups")
            scans.append(camera['scan'])
    return cameras


def paired_panel_route(start, end, panels):
    """Return a candidate sequence of paired panels, including transit scans."""
    graph = {}
    for panel in panels:
        scans = sorted({item['scan_id'] for item in panel['source_instances']})
        if len(scans) != 2:
            continue
        for a, b in (scans, scans[::-1]):
            graph.setdefault(a, []).append((b, panel['panel_index']))
    queue = deque([(start, [], [start])])
    seen = {start}
    while queue:
        scan, crossings, route = queue.popleft()
        if scan == end:
            return {'scans': route, 'panel_indices': crossings}
        for neighbor, panel in graph.get(scan, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, crossings + [panel], route + [neighbor]))
    raise ValueError(f'No paired-panel connection from {start} to {end}')


def full_turns(cameras):
    """Return the cinematic planner's minimum turn for each panorama."""
    return {c['number']: 240 for c in cameras if c['kind'] == 'panorama'}



def photo_poses(cameras, photo_manifest):
    selections = {}
    for scan in photo_manifest['scans']:
        selection = json.loads(Path(scan['selected_images']).read_text())
        selections[scan['scan_id']] = {p['image_id']: p for p in selection['images']}
    poses = {}
    for camera in cameras:
        if camera['kind'] != 'photo':
            continue
        photo = selections[camera['scan']][camera['source_id']]
        matrix = np.asarray(photo['camera_to_world'], dtype=float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[:3, 3], camera['position'], atol=1e-5, rtol=0)
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5, rtol=0)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5, rtol=0)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
            raise ValueError(f"Invalid aligned photo pose at checkpoint {camera['number']}")
        if 'yaw' in camera:
            yaw = math.atan2(matrix[2, 2], matrix[0, 2])
            viewer_yaw = float(camera['yaw'])
            if not math.isfinite(viewer_yaw) or abs((yaw - viewer_yaw + math.pi) % (2 * math.pi) - math.pi) > 1e-5:
                raise ValueError(f"Viewer and aligned photo orientation disagree at checkpoint {camera['number']}")
        poses[camera['number']] = matrix.tolist()
    return poses


def inspect(scene_dir):
    started = time.perf_counter()
    viewer = scene_dir / 'artifacts/Generate Interactive Viewer'
    tour_path = viewer / 'tour.json'
    viewer_manifest_path = viewer / 'generate_interactive_viewer_manifest.json'
    photo_manifest_path = scene_dir / 'artifacts/Extract Best Views/extract_best_views_manifest.json'
    panel_path = scene_dir / 'multi_room_alignment/stats/aligned_door_panels.json'
    alignment_report_path = scene_dir / 'multi_room_alignment/stats/multi_room_alignment_report.json'
    mesh_path = scene_dir / 'multi_room_alignment/aligned_sealed_meshes.ply'
    manifest = json.loads(viewer_manifest_path.read_text())
    if manifest.get('macro_step_status') != 'complete':
        raise ValueError('Interactive viewer macro step is not complete')
    cameras = ordered_checkpoints(json.loads(tour_path.read_text()))
    if manifest.get('camera_count') != len(cameras):
        raise ValueError('Viewer manifest and tour camera counts disagree')
    panels = json.loads(panel_path.read_text())['panels']
    turns = full_turns(cameras)
    poses = photo_poses(cameras, json.loads(photo_manifest_path.read_text()))
    transitions = []
    for a, b in zip(cameras, cameras[1:]):
        if a['scan'] != b['scan']:
            transitions.append({'from_checkpoint': a['number'], 'to_checkpoint': b['number'],
                                **paired_panel_route(a['scan'], b['scan'], panels)})
    input_seconds = time.perf_counter() - started
    mesh_started = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not len(mesh.triangles):
        raise ValueError(f'Empty or unreadable sealed mesh: {mesh_path}')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    points = np.asarray([c['position'] for c in cameras], dtype=np.float32)
    distances = scene.compute_distance(o3d.core.Tensor(points)).numpy()
    rays = np.concatenate((points[:-1], points[1:] - points[:-1]), axis=1)
    if np.any(np.linalg.norm(rays[:, 3:], axis=1) == 0):
        raise ValueError('Adjacent checkpoint positions coincide; direct-ray preflight is undefined')
    hits = scene.list_intersections(o3d.core.Tensor(rays))
    offsets, values = hits['ray_splits'].numpy(), hits['t_hit'].numpy()
    segments = []
    for i, (a, b) in enumerate(zip(cameras, cameras[1:])):
        fractions = sorted(float(x) for x in values[offsets[i]:offsets[i + 1]] if 0 < x < 1)
        segments.append({'from_checkpoint': a['number'], 'to_checkpoint': b['number'],
                         'length_m': float(np.linalg.norm(points[i + 1] - points[i])),
                         'straight_line_mesh_hit_fractions': fractions})
    mesh_seconds = time.perf_counter() - mesh_started
    checkpoints = [{**c, 'nearest_sealed_surface_m': float(distances[i]),
                    **({'camera_to_world': poses[c['number']]} if c['kind'] == 'photo' else
                       {'minimum_turn_degrees': turns[c['number']]})}
                   for i, c in enumerate(cameras)]
    photo_manifest = json.loads(photo_manifest_path.read_text())
    inputs = [tour_path, viewer_manifest_path, photo_manifest_path, panel_path, mesh_path,
              *(Path(scan['selected_images']) for scan in photo_manifest['scans'])]
    if alignment_report_path.is_file():
        inputs.append(alignment_report_path)
    return {'status': 'preflight_only',
            'limitations': ['Straight-line hits are not a planned or validated route; door surfaces are included.',
                            'Panel graph routes are candidates; geometry and Phase 07 relationships still require validation.',
                            'No trajectory, minimap animation, or video has been generated.'],
            'sparse_floor_rule': 'Each panorama requires at least a half-turn during continuous flight.',
            'phase07_alignment_report_present': alignment_report_path.is_file(),
            'input_sha256': {str(path): digest(path) for path in inputs},
            'checkpoints': checkpoints, 'candidate_scan_transitions': transitions, 'straight_lines': segments,
            'timings_seconds': {'input_and_topology': input_seconds, 'mesh_preflight': mesh_seconds,
                                'total': time.perf_counter() - started}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=STAGE.parent / 'current_scene')
    parser.add_argument('--output', type=Path, required=True, help='Write the preflight JSON to this path')
    args = parser.parse_args()
    report = inspect(args.scene_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Preflight only: {len(report['checkpoints'])} checkpoints; report: {args.output}")


if __name__ == '__main__':
    main()
