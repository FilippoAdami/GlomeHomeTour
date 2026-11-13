"""Build and validate local candidate flight legs in published checkpoint order."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import json
import time

import numpy as np
import open3d as o3d
from plyfile import PlyData

from cinematic_flythrough_preflight import ordered_checkpoints, full_turns, photo_poses
from cinematic_path_planning import LocalNavigator, clearance_and_support, local_refined_route, rounded_path, segment_clear, validate_door_crossing
from generate_floor_level_meshes import _read_mesh
from generate_interactive_viewer import floor_geometry


class DoorSurfaceAttributor:
    """Tie crossing hits to their original Phase 05 door-class mesh faces."""
    def __init__(self, scene_dir):
        self.scene_dir = Path(scene_dir)
        self.transforms = json.loads((self.scene_dir / 'multi_room_alignment/stats/aligned_scans_transforms.json').read_text())['scans']
        self.sources = {}

    def source(self, scan):
        if scan not in self.sources:
            record = self.transforms[scan]
            vertices, faces, _ = _read_mesh(Path(record['mesh_path']))
            segmented = PlyData.read(str(self.scene_dir / scan / 'segmentation/mesh_segmented.ply'),
                                     mmap='r', known_list_len={'face': {'vertex_indices': 3}})
            face_data = segmented['face'].data
            if len(face_data) != len(faces) or not {'class_id', 'instance_id'}.issubset(face_data.dtype.names or ()):
                raise ValueError(f'Missing source face labels for {scan}')
            raycaster = o3d.t.geometry.RaycastingScene()
            raycaster.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces.astype(np.uint32)))
            self.sources[scan] = (vertices, faces, segmented, raycaster,
                                  np.linalg.inv(np.asarray(record['transform_matrix'], dtype=float)))
        return self.sources[scan]

    def attribute(self, panel, hit_points):
        points = np.asarray(hit_points, dtype=float)
        matches = [[] for _ in points]
        for scan in sorted({item['scan_id'] for item in panel['source_instances']}):
            classes = {item['class_id'] for item in panel['source_instances'] if item['scan_id'] == scan}
            vertices, faces, segmented, raycaster, inverse = self.source(scan)
            local = points @ inverse[:3, :3].T + inverse[:3, 3]
            closest = raycaster.compute_closest_points(o3d.core.Tensor(local.astype(np.float32)))
            ids = closest['primitive_ids'].numpy()
            distance = np.linalg.norm(local - closest['points'].numpy(), axis=1)
            labels = segmented['face'].data
            segment_vertices = segmented['vertex'].data
            for i in np.flatnonzero(distance <= 1e-5):
                face_id = int(ids[i])
                label = labels[face_id]
                if int(label['class_id']) not in classes:
                    continue
                indices = label['vertex_indices']
                triangle = np.stack([segment_vertices[name][indices] for name in ('x', 'y', 'z')], axis=1)
                if not np.allclose(triangle, vertices[faces[face_id]], atol=1e-5, rtol=0):
                    raise ValueError(f'Phase 05 face geometry differs from the sealed source for {scan}')
                matches[i].append({'scan': scan, 'source_face_id': face_id, 'class_id': int(label['class_id']),
                                   'source_instance_index': int(label['instance_id']), 'distance_m': float(distance[i])})
        if any(not row for row in matches):
            raise ValueError(f"Panel {panel['panel_index']} crossing includes a surface without matching source door-class geometry")
        return matches


def match_pair_panels(pairs, panels):
    result = []
    for pair in pairs:
        ids = {(pair['scan_from'], pair['door_from']['source_class_id'], pair['door_from']['source_instance_id']),
               (pair['scan_to'], pair['door_to']['source_class_id'], pair['door_to']['source_instance_id'])}
        if len(ids) != 2 or any(item[1] is None or item[2] is None for item in ids):
            raise ValueError(f'Missing source instance identities for {pair}')
        matches = [panel for panel in panels if ids <= {(x['scan_id'], x['class_id'], x['instance_id'])
                                                        for x in panel['source_instances']}]
        if len(matches) != 1:
            raise ValueError(f'Expected one aligned panel for {pair["scan_from"]} -> {pair["scan_to"]}; found {len(matches)}')
        result.append((pair['scan_from'], pair['scan_to'], matches[0]))
    return result


def scan_routes(start, end, pairs, limit=64):
    """Enumerate bounded simple scan routes in increasing door count."""
    queue = deque([(start, [], {start})])
    count = 0
    expanded = 0
    while queue:
        expanded += 1
        if expanded > 4096:
            raise ValueError('Scan route search limit reached (4096 expansions)')
        current, path, seen = queue.popleft()
        if current == end:
            yield path
            count += 1
            if count >= limit:
                if queue:
                    raise ValueError(f'Scan route search limit reached ({limit} routes)')
                return
            if path:
                continue
        for a, b, panel in pairs:
            neighbor = b if current == a else a if current == b else None
            if neighbor is not None and (neighbor not in seen or (neighbor == end and path and panel is not path[-1][2])):
                queue.append((neighbor, path + [(current, neighbor, panel)], seen | {neighbor}))


def scan_route(start, end, pairs):
    route = next(scan_routes(start, end, pairs), None)
    if route is not None:
        return route
    raise ValueError(f'No trajectory-paired door route from {start} to {end}')


def choose_scan_route(start, end, pairs, build, score=None, candidates=None):
    errors = []
    best = None
    evaluated = 0
    routes = iter(scan_routes(start, end, pairs))
    while True:
        try:
            crossings = next(routes)
        except StopIteration:
            break
        except ValueError as error:
            if best is None:
                raise
            errors.append(str(error))
            break
        try:
            candidate = build(crossings)
        except ValueError as error:
            errors.append(str(error))
            continue
        except RuntimeError as error:
            if best is None:
                raise
            errors.append(str(error))
            break
        if score is None:
            return candidate, errors
        value = score(candidate)
        if candidates is not None:
            candidates.append((value, candidate))
        evaluated += 1
        if best is None or value < best[0]:
            best = value, candidate
        if evaluated == 3:
            break
    if best is not None:
        errors.append(f'Compared {evaluated} feasible routes (maximum 3); initial best score {best[0]:.3f}')
        return best[1], errors
    detail = errors[-1] if errors else 'no trajectory-paired door route'
    raise ValueError(f'{detail}; bounded scan route search exhausted (at most 64 routes)')


def route_quality(pieces, scene):
    """Prefer short, clear, gently turning routes with fewer door crossings."""
    total = 0.
    for piece in pieces:
        points = np.asarray(piece['points'], dtype=float)
        vectors = np.diff(points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        total += float(lengths.sum())
        if piece['kind'] == 'door':
            total += .25
            continue
        nonzero = lengths > 1e-9
        vectors, lengths = vectors[nonzero], lengths[nonzero]
        if not len(lengths):
            continue
        directions = vectors / lengths[:, None]
        if len(directions) > 1:
            total += .1 * float(np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)).sum())
        midpoints = (points[:-1] + points[1:])[nonzero] / 2
        clearance = scene.compute_distance(o3d.core.Tensor(midpoints.astype(np.float32))).numpy()
        total += .5 * float(np.sum(lengths * np.maximum(.4 - clearance, 0) / .4))
    return total


def panel_axes(panel):
    center = np.asarray(panel['panel_center_xyz_m'], dtype=float)
    normal = np.asarray(panel['panel_plane_normal_xyz'], dtype=float)
    norm = np.linalg.norm(normal)
    if center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"Invalid aligned panel geometry: {panel['panel_index']}")
    return center, normal / norm


def build_plan(scene_dir: Path, pairs):
    started = time.perf_counter()
    scene_dir = Path(scene_dir)
    alignment = scene_dir / 'multi_room_alignment'
    tour = json.loads((scene_dir / 'artifacts/Generate Interactive Viewer/tour.json').read_text())
    cameras = ordered_checkpoints(tour)
    manifest = json.loads((scene_dir / 'artifacts/Extract Best Views/extract_best_views_manifest.json').read_text())
    poses = photo_poses(cameras, manifest)
    turns = full_turns(cameras)
    panels = json.loads((alignment / 'stats/aligned_door_panels.json').read_text())['panels']
    pairs = match_pair_panels(pairs, panels)
    attributor = DoorSurfaceAttributor(scene_dir)
    floor_records = json.loads((scene_dir / 'artifacts/Generate Floor Plans/floor_level_metadata.json').read_text())['floor_planes']
    vertices, faces, colors = _read_mesh(alignment / 'aligned_sealed_meshes.ply')
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces.astype(np.uint32)))
    floors, floor_points = {}, {}
    for scan in {c['scan'] for c in cameras} | {x for a, b, _ in pairs for x in (a, b)}:
        records = [record for record in floor_records if record['scan_id'] == scan]
        if not records:
            raise ValueError(f'No floor geometry for {scan}')
        floors[scan], floor_points[scan] = floor_geometry(records, alignment)
    navs = {}
    all_endpoints = {}
    for scan in floors:
        endpoints = [c['position'] for c in cameras if c['scan'] == scan]
        for a, b, panel in pairs:
            if scan in (a, b):
                center, normal = panel_axes(panel)
                endpoints.extend(center + offset * normal for offset in (-.8, -.25, .25, .8))
        all_endpoints[scan] = np.asarray(endpoints)
    def navigator(scan, spacing=.12):
        key = scan, spacing
        if key not in navs:
            navs[key] = LocalNavigator(scene, floors[scan], floor_points[scan], all_endpoints[scan], spacing=spacing)
        return navs[key]
    def anchor(scan, panel, center, target=None):
        _, normal = panel_axes(panel)
        candidates = np.array([center - .25 * normal, center + .25 * normal])
        supported = clearance_and_support(scene, floors[scan], candidates, .1)
        if supported.sum() != 1:
            raise ValueError(f"Panel {panel['panel_index']} has {supported.sum()} supported sides for {scan}")
        entry = candidates[np.flatnonzero(supported)[0]]
        side = (entry - center) / np.linalg.norm(entry - center)
        distance = .8 if target is None else float(np.clip(np.dot(np.asarray(target) - center, side) - .03, .3, .8))
        guide = center + distance * side
        if not segment_clear(scene, floors[scan], entry, guide):
            raise ValueError(f"Panel {panel['panel_index']} lacks a clear perpendicular approach in {scan}")
        return entry, guide
    def local_piece(scan, start, end, start_guide=None, end_guide=None):
        route_start = start if start_guide is None else start_guide
        route_end = end if end_guide is None else end_guide
        try:
            if segment_clear(scene, floors[scan], route_start, route_end):
                route = np.stack((route_start, route_end))
                routing = {'method': 'straight'}
            else:
                route = navigator(scan).route(route_start, route_end)
                routing = {'method': 'coarse_grid', 'spacing_m': .12}
        except ValueError as coarse_error:
            try:
                route, routing = local_refined_route(scene, floors[scan], floor_points[scan], route_start, route_end)
                routing['coarse_failure'] = str(coarse_error)
            except ValueError as fine_error:
                try:
                    refined = navigator(scan, .06)
                    route = refined.route(route_start, route_end)
                    routing = {'method': 'scan_refined_grid', 'spacing_m': .06, 'grid_cells': len(refined.points),
                               'coarse_failure': str(coarse_error), 'local_failure': str(fine_error)}
                except ValueError as global_error:
                    raise ValueError(f'{scan}: coarse grid failed ({coarse_error}); local refinement failed ({fine_error}); '
                                     f'scan-wide refinement failed ({global_error})') from global_error
        sections = ([np.asarray(start)[None]] if start_guide is not None else []) + [route]
        sections += [np.asarray(end)[None]] if end_guide is not None else []
        points = np.vstack(sections)
        return {'kind': 'local', 'scan': scan, 'points': rounded_path(scene, floors[scan], points).tolist(),
                'routing': routing}
    edges = []
    last_door_exit = None
    last_door_normal = None
    for first, last in zip(cameras, cameras[1:]):
        scan = first['scan']
        current = np.asarray(first['position'])
        exit_guide = None
        if last_door_exit is not None and last_door_normal is not None:
            if np.linalg.norm(current - last_door_exit) < 0.5:
                guide_cand = current + 0.35 * last_door_normal
                if segment_clear(scene, floors[scan], current, guide_cand):
                    exit_guide = guide_cand
            last_door_exit = None
            last_door_normal = None
        attempts = [0]
        errors = []
        def try_crossings(crossings, crossing_index, current, scan, exit_guide):
            if crossing_index == len(crossings):
                return [local_piece(scan, current, last['position'], exit_guide)], None, None
            source, dest, panel = crossings[crossing_index]
            panel_center, normal = panel_axes(panel)
            desired_height = panel_center[1]
            tangent = np.cross([0., 1., 0.], normal)
            tangent /= np.linalg.norm(tangent)
            desired_offset = 0.
            if crossing_index == len(crossings) - 1:
                margin = panel['panel_dimensions_m']['height'] / 2 - .15
                desired_height = np.clip(last['position'][1], panel_center[1] - margin, panel_center[1] + margin)
                margin = panel['panel_dimensions_m']['width'] / 2 - .15
                desired_offset = np.clip(np.dot(np.asarray(last['position']) - panel_center, tangent), -margin, margin)
            candidates = ((offset, height) for offset in dict.fromkeys(np.linspace(desired_offset * 0.5, 0, 6))
                          for height in dict.fromkeys(np.linspace(desired_height, panel_center[1], 11)))

            for offset, height in candidates:
                attempts[0] += 1
                if attempts[0] > 256:
                    raise RuntimeError('Door candidate search limit reached (256 attempts)')
                center = panel_center + offset * tangent
                center[1] = height
                try:
                    entry, entry_guide = anchor(source, panel, center)
                    exit, next_exit_guide = anchor(dest, panel, center, last['position'] if crossing_index == len(crossings) - 1 else None)
                    if np.dot(entry - center, normal) * np.dot(exit - center, normal) >= 0:
                        raise ValueError('Door anchors are not on opposite sides')
                    crossing = validate_door_crossing(scene, vertices, faces, panel, entry, exit, center)
                    crossing['source_instances'] = panel['source_instances']
                    crossing['source_face_attribution'] = attributor.attribute(panel, crossing['hit_points'])
                    approach = local_piece(source, current, entry, exit_guide, entry_guide)
                    following, later_exit, later_normal = try_crossings(
                        crossings, crossing_index + 1, exit, dest, next_exit_guide)
                    pieces = [approach,
                              {'kind': 'door', 'scan_from': source, 'scan_to': dest,
                               'points': [entry.tolist(), exit.tolist()], 'crossing': crossing}]
                    return pieces + following, (later_exit if later_exit is not None else np.asarray(exit)), (
                        later_normal if later_normal is not None else (exit - entry) / np.linalg.norm(exit - entry))
                except ValueError as error:
                    errors.append(f"panel {panel['panel_index']}: {error}")
            raise ValueError(errors[-1] if errors else f"No candidates for panel {panel['panel_index']}")
        candidates = []
        try:
            result, rejected_routes = choose_scan_route(
                scan, last['scan'], pairs, lambda route: try_crossings(route, 0, current, scan, exit_guide),
                lambda candidate: route_quality(candidate[0], scene), candidates)
            pieces, last_door_exit, last_door_normal = result
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Cannot connect checkpoint {first['number']} -> {last['number']}: {error}") from error
        edges.append({'from_checkpoint': first['number'], 'to_checkpoint': last['number'], 'pieces': pieces,
                      'door_candidates_tested': attempts[0], 'rejected_candidates': errors[-10:],
                      'route_search_notes': rejected_routes, 'route_quality_score': route_quality(pieces, scene),
                      'route_search_capped': len(candidates) >= 3 or any('search limit reached' in note.lower()
                                                                          for note in rejected_routes),
                      'route_alternatives': [{'pieces': candidate[0], 'route_quality_score': value}
                                             for value, candidate in sorted(candidates, key=lambda row: row[0])
                                             if candidate is not result and candidate[0] != pieces]})
    return {'status': 'candidate_spatial_legs_only',
            'limitations': ['Door hits match source door classes and panel geometry; source instance identity may be coarser after Phase 07 splits.',
                            'Continuity and clearance at the joins, timing, orientation, and animated video are not yet validated.'],
            'sparse_floor_rule': 'Each panorama requires at least a half-turn during continuous flight.',
            'checkpoints': [{**camera, **({'camera_to_world': poses[camera['number']]} if camera['kind'] == 'photo'
                                          else {'minimum_turn_degrees': turns[camera['number']]})} for camera in cameras],
            'edges': edges, 'planning_seconds': time.perf_counter() - started}
