#!/usr/bin/env python3
"""Plan, validate, and render a paired continuous Phase 08 cinematic tour."""
from __future__ import annotations

import argparse
from collections.abc import Callable
import copy
import hashlib
import heapq
from importlib import metadata
from importlib.util import find_spec
import json
import logging
import os
from pathlib import Path
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import TypeVar

import imageio_ffmpeg
import numpy as np
import open3d as o3d
from PIL import Image, ImageFont

from cinematic_flythrough_plan import DoorSurfaceAttributor, build_plan
from cinematic_flythrough_preflight import digest
from cinematic_minimap import AnimatedMinimap, compose_minimap
from cinematic_rendering import GaussianRenderer, OpticalMeshRenderer
from panorama_rendering import panel_faces
from cinematic_trajectory import append_overhead
from cinematic_drone_path import refine_panorama_framing, sample_drone_trajectory
from cinematic_trajectory_validation import validate_trajectory
from generate_floor_level_meshes import _read_mesh
from generate_interactive_viewer import floor_geometry

STAGE = Path(__file__).resolve().parent
MACRO = 'Generate Cinematic Flythrough'
LOGGER = logging.getLogger('generate_cinematic_flythrough')
T = TypeVar('T')


def write_json(path, value):
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(json.dumps(value, indent=2) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _run_timed_step(name: str, operation: Callable[[], T], timings: list[dict], macro_started: float) -> T:
    started = time.perf_counter()
    LOGGER.info('Starting substep: %s', name)
    try:
        result = operation()
    except Exception:
        elapsed = time.perf_counter() - started
        timings.append({'step': name, 'status': 'failed', 'elapsed_seconds': round(elapsed, 3)})
        LOGGER.exception('Substep %s failed after %.2f s; macro elapsed %.2f s',
                         name, elapsed, time.perf_counter() - macro_started)
        raise
    elapsed = time.perf_counter() - started
    timings.append({'step': name, 'status': 'complete', 'elapsed_seconds': round(elapsed, 3)})
    LOGGER.info('Completed substep: %s in %.2f s', name, elapsed)
    return result


def validated_trajectory(plan, geometry, vertices, faces, floor_geometry_scenes, attributor, max_attempts=16):
    """Try bounded spatial-route combinations, validating each actual sampled flight."""
    options = [[{'pieces': edge['pieces'], 'route_quality_score': edge['route_quality_score']},
                *edge.get('route_alternatives', [])] for edge in plan['edges']]
    initial = (0,) * len(options)
    pending = [(sum(rows[0]['route_quality_score'] for rows in options), 0, initial)]
    seen = {(initial, 0)}
    failures = []
    spatial_cap = any(edge.get('route_search_capped', False) for edge in plan['edges'])
    elapsed = {'sampling': 0., 'overhead': 0., 'validation': 0., 'framing': 0.}
    while pending and len(failures) < max_attempts:
        cost, mode, choices = heapq.heappop(pending)
        interpolation = ('pchip', 'linear')[mode]
        candidate = copy.deepcopy(plan)
        for edge, rows, choice in zip(candidate['edges'], options, choices):
            edge.pop('route_alternatives', None)
            selected = rows[choice]
            edge['pieces'] = copy.deepcopy(selected['pieces'])
            edge['route_quality_score'] = selected['route_quality_score']
            edge['selected_route_alternative'] = choice
        stage = 'sampling'
        tick = time.perf_counter()
        try:
            flight = sample_drone_trajectory(candidate, geometry, floor_geometry_scenes, interpolation=interpolation)
            elapsed[stage] += time.perf_counter() - tick
            stage, tick = 'overhead', time.perf_counter()
            append_overhead(flight, vertices, faces, geometry)
            elapsed[stage] += time.perf_counter() - tick
            stage, tick = 'validation', time.perf_counter()
            validation = validate_trajectory(flight, candidate, geometry, vertices, faces,
                                             floor_geometry_scenes, attributor)
            elapsed[stage] += time.perf_counter() - tick
        except ValueError as error:
            elapsed[stage] += time.perf_counter() - tick
            failures.append({'route_alternatives': list(choices), 'interpolation': interpolation,
                             'stage': stage, 'reason': str(error)})
            LOGGER.warning('Trajectory attempt %d failed during %s: %s', len(failures), stage, error)
            if mode == 0 and (choices, 1) not in seen:
                seen.add((choices, 1))
                heapq.heappush(pending, (cost, 1, choices))
            for i, rows in enumerate(options):
                if choices[i] + 1 >= len(rows):
                    continue
                next_choice = list(choices)
                next_choice[i] += 1
                next_choice = tuple(next_choice)
                if (next_choice, 0) not in seen:
                    seen.add((next_choice, 0))
                    next_cost = sum(rows[index]['route_quality_score'] for rows, index in zip(options, next_choice))
                    heapq.heappush(pending, (next_cost, 0, next_choice))
            continue
        tick = time.perf_counter()
        refined_poses, framing_changes = refine_panorama_framing(
            flight, candidate, geometry, validation['floor'])
        elapsed['framing'] += time.perf_counter() - tick
        framing = {'status': 'unchanged', 'changes': []}
        if framing_changes:
            refined = {**flight, 'camera_to_world': refined_poses}
            tick = time.perf_counter()
            try:
                refined_validation = validate_trajectory(refined, candidate, geometry, vertices, faces,
                                                         floor_geometry_scenes, attributor)
            except ValueError as error:
                framing = {'status': 'rejected', 'reason': str(error), 'changes': framing_changes}
                LOGGER.warning('Framing adjustment failed trajectory validation: %s', error)
            else:
                flight, validation = refined, refined_validation
                framing = {'status': 'validated', 'changes': framing_changes}
            elapsed['validation'] += time.perf_counter() - tick
        return candidate, flight, validation, {
            'attempt_count': len(failures) + 1, 'selected_route_alternatives': list(choices),
            'interpolation': interpolation, 'framing': framing,
            'failed_attempts': failures, 'max_attempts': max_attempts, 'spatial_route_search_capped': spatial_cap,
            'seconds': {key: round(value, 3) for key, value in elapsed.items()}}
    limit = f'; search limit {max_attempts} reached' if pending else ''
    if spatial_cap:
        affected = [(edge.get('from_checkpoint'), edge.get('to_checkpoint')) for edge in plan['edges']
                    if edge.get('route_search_capped', False)]
        limit += f'; spatial route search was capped for checkpoint legs {affected}'
    detail = failures[-1]['reason'] if failures else 'no candidate routes'
    raise ValueError(f'No validated trajectory after {len(failures)} route attempts{limit}: {detail}')


def pairing_data(scene, supplied):
    alignment = scene / 'multi_room_alignment'
    path = supplied or alignment / 'stats/multi_room_alignment_report.json'
    if not path.is_file():
        raise ValueError(f'No preserved Phase 07 pairing report: {path}')
    data = json.loads(path.read_text())
    if supplied:
        if data.get('status') != 'reconstructed_from_current_inputs_not_original_report':
            raise ValueError('Supplied reconstructed pairing provenance is missing')
        for source, expected in data['input_sha256'].items():
            if not Path(source).is_file() or digest(Path(source)) != expected:
                raise ValueError(f'Reconstructed pairing source changed: {source}')
        return path, data['pairs'], 'reconstructed from current inputs; original report unavailable'
    metrics = data.get('alignment_metrics', {})
    if metrics.get('trajectory_door_pairs_status') not in ('computed', 'reused'):
        raise ValueError('Phase 07 did not retain trajectory-derived door identities')
    current = json.loads((alignment / 'stats/aligned_scans_transforms.json').read_text())['scans']
    if set(data.get('transforms', {})) != set(current) or any(
        not np.allclose(data['transforms'][name], current[name]['transform_matrix'], atol=1e-8, rtol=0)
        for name in current
    ):
        raise ValueError('Phase 07 pairing report does not match the saved alignment transforms')
    return path, metrics['trajectory_door_pairs'], 'retained Phase 07 report'


def input_paths(scene, pairing_path, tour):
    alignment = scene / 'multi_room_alignment'
    viewer = scene / 'artifacts/Generate Interactive Viewer'
    photos_path = scene / 'artifacts/Extract Best Views/extract_best_views_manifest.json'
    photos = json.loads(photos_path.read_text())
    paths = [pairing_path, viewer / 'tour.json', viewer / 'generate_interactive_viewer_manifest.json',
             photos_path, alignment / 'aligned_sealed_meshes.ply', alignment / 'aligned_sealed_meshes.glb',
             alignment / 'stats/phase07_region_replacements.json', alignment / 'stats/aligned_door_panels.json',
             alignment / 'stats/aligned_scans_transforms.json',
             scene / 'artifacts/Generate Floor Plans/floor_level_metadata.json']
    paths.extend(Path(row['selected_images']) for row in photos['scans'])
    pairing_report = json.loads(Path(pairing_path).read_text())
    if pairing_report.get('status') == 'reconstructed_from_current_inputs_not_original_report':
        paths.extend(Path(path) for path in pairing_report['input_sha256'])
    paths.extend(viewer / level['styles']['objects']['image'].split('?', 1)[0] for level in tour['levels'])
    transforms = json.loads((alignment / 'stats/aligned_scans_transforms.json').read_text())['scans']
    paths.extend(scene / scan / 'metric_scale.json' for scan in transforms)
    paths.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    paths.append(Path(ImageFont.truetype('DejaVuSans-Bold.ttf', 10).path))
    gsplat = find_spec('gsplat')
    if gsplat and gsplat.origin:
        package = Path(gsplat.origin).parent
        paths.extend(package.rglob('*.py'))
    paths.extend(STAGE / name for name in ('generate_cinematic_flythrough.py', 'cinematic_flythrough_plan.py',
                 'cinematic_flythrough_preflight.py', 'cinematic_path_planning.py', 'cinematic_drone_path.py',
                 'cinematic_trajectory.py', 'cinematic_trajectory_validation.py', 'cinematic_rendering.py',
                 'cinematic_minimap.py', 'panorama_rendering.py', 'generate_interactive_viewer.py',
                 'generate_floor_level_meshes.py'))
    paths.extend(alignment / 'scans' / scan / 'gaussian_splat/gaussian_splat_sealed.ply' for scan in transforms)
    paths.extend(Path(record['mesh_path']) for record in transforms.values())
    paths.extend(scene / scan / 'segmentation/mesh_segmented.ply' for scan in transforms)
    records = json.loads((scene / 'artifacts/Generate Floor Plans/floor_level_metadata.json').read_text())['floor_planes']
    paths.extend(alignment / record['source_mesh'] for record in records)
    unique = sorted(set(Path(p).absolute() for p in paths))
    result = {str(path): digest(path) for path in unique}
    merged = (alignment / 'aligned_sealed_gaussians.ply').absolute()
    result[str(merged)] = digest(merged) if merged.is_file() else None
    if gsplat and gsplat.origin:
        binary = (package / 'csrc.so').absolute()
        result[str(binary)] = digest(binary) if binary.is_file() else None
    return result


def verify_inputs(inputs):
    for name, expected in inputs.items():
        path = Path(name)
        actual = digest(path) if path.is_file() else None
        if actual != expected:
            raise ValueError(f'Input changed after it was recorded: {path}')


def runtime_versions():
    versions = {}
    for package in ('numpy', 'scipy', 'open3d', 'torch', 'plyfile', 'Pillow',
                    'imageio-ffmpeg', 'gsplat', 'opencv-python', 'trimesh'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': sys.version, 'packages': versions}


def metric_scale_status(scene):
    transforms = json.loads((scene / 'multi_room_alignment/stats/aligned_scans_transforms.json').read_text())['scans']
    for scan in transforms:
        path = scene / scan / 'metric_scale.json'
        if not path.is_file():
            raise ValueError(f'{scan}: missing {path}; calibrate and apply metric scale upstream before generating a cinematic tour')
        data = json.loads(path.read_text())
        if not (data.get('available') is True and data.get('scale_applied_to_colmap_geometry') is True
                and data.get('output_coordinate_units') == 'meters'):
            raise ValueError(f'{scan}: uncalibrated geometry in {path}; calibrate and apply metric scale upstream before generating a cinematic tour')
    return {'units': 'meters', 'status': 'calibrated_and_applied', 'scans': sorted(transforms)}


def floor_scenes(scene):
    alignment = scene / 'multi_room_alignment'
    records = json.loads((scene / 'artifacts/Generate Floor Plans/floor_level_metadata.json').read_text())['floor_planes']
    groups = {(row['scan_id'], row['floor_level_name']) for row in records}
    return {key: floor_geometry([row for row in records if (row['scan_id'], row['floor_level_name']) == key],
                                alignment)[0] for key in groups}


def diagnostic(path, plan, flight):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    fig, axis = plt.subplots(figsize=(9, 8))
    scans = sorted({piece['scan'] for edge in plan['edges'] for piece in edge['pieces'] if piece['kind'] == 'local'})
    palette = {scan: plt.cm.tab10(i % 10) for i, scan in enumerate(scans)}
    cameras = {camera['number']: camera for camera in plan['checkpoints']}
    final_inside = flight['overhead']['start_frame'] + 1
    points = flight['camera_to_world'][:final_inside, :3, 3][:, [0, 2]]
    segments = np.stack((points[:-1], points[1:]), axis=1)
    axis.add_collection(LineCollection(segments, colors=[palette[scan] for scan in flight['scans'][:len(segments)]], linewidths=1.4))
    axis.update_datalim(points)
    for edge in plan['edges']:
        endpoint_scans = {cameras[edge['from_checkpoint']]['scan'], cameras[edge['to_checkpoint']]['scan']}
        for piece in edge['pieces']:
            pts = np.asarray(piece['points'])
            if piece['kind'] == 'door':
                axis.plot(pts[:, 0], pts[:, 2], color='black', linewidth=2.4)
                center = np.asarray(piece['crossing']['center'])
                axis.scatter(center[0], center[2], marker='x', color='black', s=85)
                axis.annotate(f"door {piece['crossing']['panel_index']} {piece['scan_from']} → {piece['scan_to']}",
                              (center[0], center[2]), xytext=(5, 5), textcoords='offset points', fontsize=7)
            elif piece['scan'] not in endpoint_scans:
                middle = pts[len(pts) // 2]
                axis.annotate('transit ' + piece['scan'], (middle[0], middle[2]), fontsize=8)
    axis.annotate('up', points[-1], xytext=(10, 14), textcoords='offset points', fontsize=8,
                  arrowprops={'arrowstyle': '->'})
    for camera in plan['checkpoints']:
        x, _, z = camera['position']
        axis.scatter(x, z, s=180, marker='s' if camera['floor'] != 'BASE' else 'o',
                     color='#d37a19' if camera['kind'] == 'panorama' else '#174785', edgecolor='white', zorder=3)
        axis.text(x, z, str(camera['number']), color='white', ha='center', va='center', fontsize=8, weight='bold', zorder=4)
        if camera['kind'] == 'panorama' and flight.get('camera_mode') != 'drone_smooth':
            axis.annotate(str(camera['minimum_turn_degrees']) + '°', (x, z), xytext=(8, 8), textcoords='offset points', fontsize=8)
    axis.set(title='Validated flight: numbered checkpoints and designated door crossings', xlabel='Aligned world X (m)', ylabel='Aligned world Z (m)')
    axis.set_aspect('equal')
    axis.grid(alpha=.25)
    axis.legend(handles=[Line2D([0], [0], color=palette[scan], label=scan) for scan in scans], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def video_writers(output, width, height, fps):
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    processes, logs = [], []
    try:
        for name in ('mesh', 'gaussian'):
            log = (output / f'{name}_ffmpeg.log').open('w')
            logs.append(log)
            command = [exe, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                       '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                       '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                       str(output / f'{name}.mp4')]
            processes.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log, bufsize=0))
    except BaseException:
        for process in processes:
            process.kill()
            process.wait()
            process.stdin.close()
        for log in logs:
            log.close()
        raise
    return processes, logs


def write_frame(stream, data, processes=()):
    remaining = memoryview(data)
    fd = stream.fileno()
    os.set_blocking(fd, False)
    last_progress = time.monotonic()
    while remaining:
        if any(process.poll() is not None for process in processes):
            raise BrokenPipeError('Video encoder exited while rendering')
        if not select.select([], [fd], [], 1)[1]:
            if time.monotonic() - last_progress > 30:
                raise TimeoutError('Video encoder stopped accepting frames for 30 seconds')
            continue
        try:
            written = os.write(fd, remaining)
        except BlockingIOError:
            continue
        if not written:
            raise BrokenPipeError('Video encoder stopped accepting frames')
        remaining = remaining[written:]
        last_progress = time.monotonic()


class AsyncVideoWriter:
    def __init__(self, processes, logs):
        self.processes = processes
        self.logs = logs
        self.queue: queue.Queue[tuple[bytes, bytes] | None] = queue.Queue(maxsize=16)
        self.exception: BaseException | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    self.queue.task_done()
                    break
                mesh_bytes, gaussian_bytes = item
                write_frame(self.processes[0].stdin, mesh_bytes, self.processes)
                write_frame(self.processes[1].stdin, gaussian_bytes, self.processes)
                self.queue.task_done()
        except BaseException as exc:
            self.exception = exc
            self.queue.task_done()

    def write_frames(self, mesh_bytes: bytes, gaussian_bytes: bytes):
        while True:
            if self.exception is not None:
                raise self.exception
            try:
                self.queue.put((mesh_bytes, gaussian_bytes), timeout=0.2)
                break
            except queue.Full:
                continue

    def close(self, mode: str):
        while True:
            if self.exception is not None:
                raise self.exception
            try:
                self.queue.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=60)
        if self._thread.is_alive():
            raise TimeoutError('AsyncVideoWriter worker thread timed out')
        if self.exception is not None:
            raise self.exception
        for process in self.processes:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        for process in self.processes:
            if process.wait(timeout=60) != 0:
                raise RuntimeError(f'{mode} video encoder failed; see ffmpeg logs')

    def cleanup(self):
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=2)
            except Exception:
                pass
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        if self._thread.is_alive():
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
            self._thread.join(timeout=2.0)


def render_pair(scene, output, tour, flight, floors, width, height, mode, save_samples, preloaded_models=None):
    start = time.perf_counter()
    load_start = time.perf_counter()
    if preloaded_models is not None:
        LOGGER.info('Reusing preloaded models for %s rendering (%dx%d)...', mode, width, height)
        mesh_geom, gaussians = preloaded_models
        mesh = OpticalMeshRenderer(mesh_geom[0], mesh_geom[1], mesh_geom[2], mesh_geom[3], width, height)
    else:
        LOGGER.info('Loading models for %s rendering (%dx%d)...', mode, width, height)
        alignment = scene / 'multi_room_alignment'
        import trimesh
        glb = trimesh.load(alignment / 'aligned_sealed_meshes.glb', force='scene', process=False)
        regions = json.loads((alignment / 'stats/phase07_region_replacements.json').read_text())['regions']
        transforms = json.loads((alignment / 'stats/aligned_scans_transforms.json').read_text())['scans']
        vertices, faces, colors, types = [], [], [], []
        seen_scans = set()
        offset = 0
        for node in sorted(glb.graph.nodes_geometry):
            transform, name = glb.graph[node]
            if not name.startswith('sealed_'):
                raise ValueError(f'Unexpected GLB geometry: {name}')
            scan = name[len('sealed_'):]
            if scan in seen_scans:
                raise ValueError(f'Duplicate GLB scan geometry: {scan}')
            seen_scans.add(scan)
            part = glb.geometry[name]
            v, f = trimesh.transform_points(part.vertices, transform), np.asarray(part.faces)
            vertices.append(v)
            faces.append(f + offset)
            colors.append(np.asarray(part.visual.vertex_colors))
            types.append(panel_faces(scan, v, f, regions)[0])
            offset += len(v)
        if seen_scans != set(transforms):
            raise ValueError('GLB scan geometry does not match the alignment')
        mesh_geom = (np.concatenate(vertices), np.concatenate(faces), np.concatenate(colors),
                     np.concatenate(types))
        mesh = OpticalMeshRenderer(mesh_geom[0], mesh_geom[1], mesh_geom[2], mesh_geom[3], width, height)
        del glb, vertices, faces, colors, types
        aligned_gs_path = alignment / 'aligned_sealed_gaussians.ply'
        if aligned_gs_path.is_file():
            gaussians = GaussianRenderer(aligned_gs_path, transforms={scan: row['transform_matrix'] for scan, row in transforms.items()})
        else:
            scan_inputs = [(alignment / 'scans' / scan / 'gaussian_splat/gaussian_splat_sealed.ply',
                            np.asarray(row['transform_matrix'])) for scan, row in sorted(transforms.items())]
            gaussians = GaussianRenderer(scan_inputs)
    minimap = AnimatedMinimap(scene / 'artifacts/Generate Interactive Viewer', tour, flight, floors, width)
    load_seconds = time.perf_counter() - load_start
    LOGGER.info('Models loaded in %.2f s', load_seconds)

    durations = {'mesh_rendering': 0.0, 'gaussian_rendering': 0.0, 'minimap_and_composition': 0.0}
    processes, logs = video_writers(output, width, height, flight['fps'])
    writer = AsyncVideoWriter(processes, logs)
    sample_dir = output / 'representative_frames'
    if save_samples:
        sample_dir.mkdir(exist_ok=True)
    frame_count = len(flight['camera_to_world'])
    encoding_io_time = 0.0
    try:
        for i, pose in enumerate(flight['camera_to_world']):
            t = time.perf_counter()
            mesh_image = mesh.render(pose, width, height)
            durations['mesh_rendering'] += time.perf_counter() - t
            t = time.perf_counter()
            gaussian_image = gaussians.render(pose, width, height)
            durations['gaussian_rendering'] += time.perf_counter() - t
            t = time.perf_counter()
            overlay = minimap.advance(i)
            mesh_image = compose_minimap(mesh_image, overlay)
            gaussian_image = compose_minimap(gaussian_image, overlay)
            durations['minimap_and_composition'] += time.perf_counter() - t
            if i in save_samples:
                for label in save_samples[i]:
                    Image.fromarray(mesh_image).save(sample_dir / f'{label}_mesh.png')
                    Image.fromarray(gaussian_image).save(sample_dir / f'{label}_gaussian.png')
            t_io = time.perf_counter()
            writer.write_frames(mesh_image.tobytes(), gaussian_image.tobytes())
            encoding_io_time += time.perf_counter() - t_io
            if (i + 1) % 300 == 0 or (i + 1) == frame_count:
                LOGGER.info('%s rendering: %d/%d frames (%.1f%%) in %.1f s',
                            mode, i + 1, frame_count, 100 * (i + 1) / frame_count, time.perf_counter() - start)
        t_close = time.perf_counter()
        writer.close(mode)
        encoding_io_time += time.perf_counter() - t_close
    except BaseException:
        writer.cleanup()
        for log in logs:
            try:
                details = Path(log.name).read_text(errors='replace')[-4000:]
            except OSError:
                continue
            if details:
                LOGGER.error('%s encoder diagnostics: %s', mode, details)
        raise
    finally:
        for log in logs:
            log.close()
    total_render_wall = time.perf_counter() - start
    timings_result = {
        'model_loading': round(load_seconds, 3),
        'mesh_rendering': round(durations['mesh_rendering'], 3),
        'gaussian_rendering': round(durations['gaussian_rendering'], 3),
        'minimap_and_composition': round(durations['minimap_and_composition'], 3),
        'video_encoding_and_io': round(encoding_io_time, 3),
        'total_wall': round(total_render_wall, 3),
        'mesh_renderer': 'GLB Open3D OpenGL with one-bounce classified panel optics',
        'mesh_samples_per_pixel': 4
    }
    LOGGER.info('%s render_pair completed in %.2f s (mesh: %.2f s, gaussian: %.2f s, minimap: %.2f s, encoding/io: %.2f s)',
                mode, total_render_wall, durations['mesh_rendering'], durations['gaussian_rendering'],
                durations['minimap_and_composition'], encoding_io_time)
    return timings_result, (mesh_geom, gaussians)


def preview(scene, output, pairing_path, overwrite=False):
    started = time.perf_counter()
    timings = []
    LOGGER.info('Starting %s (preview mode)', MACRO)
    LOGGER.info('Scene: %s', scene)
    LOGGER.info('Output: %s', output)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if not overwrite:
            raise FileExistsError(f'Refusing to replace existing output: {output}')

    staged = Path(tempfile.mkdtemp(prefix='.cinematic-', dir=output.parent))

    file_handler = logging.FileHandler(staged / 'generate_cinematic_flythrough.log', mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    try:
        pairing_path_resolved = None
        pairs = None
        origin = None
        tour = None
        viewer_manifest = None
        inputs = None
        scale_status = None
        runtime = None

        def step_input_validation():
            nonlocal pairing_path_resolved, pairs, origin, tour, viewer_manifest, inputs, scale_status, runtime
            pairing_path_resolved, pairs, origin = pairing_data(scene, pairing_path)
            tour = json.loads((scene / 'artifacts/Generate Interactive Viewer/tour.json').read_text())
            viewer_manifest = json.loads((scene / 'artifacts/Generate Interactive Viewer/generate_interactive_viewer_manifest.json').read_text())
            if viewer_manifest.get('macro_step_status') != 'complete' or viewer_manifest.get('camera_count') != len(tour['cameras']):
                raise ValueError('Viewer manifest is incomplete or disagrees with tour data')
            scale_status = metric_scale_status(scene)
            inputs = input_paths(scene, pairing_path_resolved, tour)
            runtime = runtime_versions()

        _run_timed_step('input_validation_and_setup', step_input_validation, timings, started)

        plan = None
        def step_spatial_planning():
            nonlocal plan
            plan = build_plan(scene, pairs)

        _run_timed_step('spatial_path_planning', step_spatial_planning, timings, started)

        alignment = scene / 'multi_room_alignment'
        vertices = None
        faces = None
        geometry = None
        floor_geometry_scenes = None

        def step_scene_geometry_loading():
            nonlocal vertices, faces, geometry, floor_geometry_scenes
            vertices, faces, _ = _read_mesh(alignment / 'aligned_sealed_meshes.ply')
            geometry = o3d.t.geometry.RaycastingScene()
            geometry.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces.astype(np.uint32)))
            floor_geometry_scenes = floor_scenes(scene)

        _run_timed_step('scene_geometry_loading', step_scene_geometry_loading, timings, started)

        plan, flight, validation, trajectory_search = _run_timed_step(
            'validated_trajectory_search',
            lambda: validated_trajectory(plan, geometry, vertices, faces, floor_geometry_scenes,
                                         DoorSurfaceAttributor(scene)), timings, started)
        floors = validation.pop('floor')
        validation_seconds = trajectory_search['seconds']['validation']

        plan['status'] = 'validated_sampled_trajectory'
        plan['limitations'] = [plan['limitations'][0],
                               'Panorama checkpoints receive at least a half-turn while the camera continues moving.',
                               'GLB panels use idealized one-bounce mirror and window optics; missing reflected or transmitted geometry appears white.',
                               'Closed door and ceiling surfaces are rendered continuously as the camera crosses them; exterior Gaussian coverage depends on the scans.']
        plan['sparse_floor_rule'] = 'Each panorama receives at least a half-turn during continuous flight.'

        samples = {}
        events = flight['checkpoints']

        def step_serialization_and_diagnostics():
            nonlocal samples
            write_json(staged / 'spatial_plan.json', plan)
            np.savez_compressed(staged / 'trajectory.npz', camera_to_world=flight['camera_to_world'],
                                scans=np.asarray(flight['scans']), floors=np.asarray(floors))
            write_json(staged / 'flight_settings.json', {
                'fps': flight['fps'], 'horizontal_fov_degrees': 70, 'checkpoint_events': events,
                'door_crossings': flight['door_crossings'], 'overhead': flight['overhead'],
                'camera_mode': flight['camera_mode']})
            diagnostic(staged / 'trajectory_topdown.png', plan, flight)
            checkpoint = next((e for e, c in zip(events, plan['checkpoints']) if c['kind'] == 'photo'), events[0])
            changes = [i for i in range(1, len(floors)) if floors[i] != floors[i-1]]
            samples = {checkpoint['frame']: ['checkpoint']}
            if flight['door_crossings']:
                door = flight['door_crossings'][0]
                samples.setdefault((door['start_frame'] + door['end_frame']) // 2, []).append('door_crossing')
            if changes:
                samples.setdefault(changes[0], []).append('floor_transition')
            samples.setdefault(len(floors) - 1, []).append('overhead')

        _run_timed_step('artifacts_serialization_and_diagnostics', step_serialization_and_diagnostics, timings, started)

        render_timing = {}
        loaded_models = None
        def step_preview_rendering():
            nonlocal render_timing, loaded_models
            render_dir = staged / 'preview'
            render_dir.mkdir(exist_ok=True)
            render_timing, loaded_models = render_pair(scene, render_dir, tour, flight, floors, 270, 480, 'preview', samples)

        _run_timed_step('preview_video_rendering', step_preview_rendering, timings, started)
        verify_inputs(inputs)
        if runtime_versions() != runtime:
            raise ValueError('Python or rendering package versions changed during preview rendering')

        total_wall_preview = time.perf_counter() - started
        manifest = {'schema_version': 2, 'macro_step': MACRO, 'macro_step_status': 'preview_complete',
                    'input_sha256': inputs, 'pairing_provenance': origin,
                    'scale': scale_status,
                    'gaussian_cache_policy': 'Legacy tensor caches are ignored; Gaussian PLY inputs are read directly.',
                    'runtime_versions': runtime,
                    'trajectory_search': trajectory_search,
                    'route_decisions': [{**{key: edge[key] for key in ('from_checkpoint', 'to_checkpoint',
                                         'door_candidates_tested', 'rejected_candidates', 'route_search_notes',
                                         'route_quality_score', 'route_search_capped', 'selected_route_alternative')},
                                         'local_routing': [piece['routing'] for piece in edge['pieces']
                                                           if piece['kind'] == 'local']} for edge in plan['edges']],
                    'ordered_checkpoints': plan['checkpoints'], 'checkpoint_events': events,
                    'door_crossings': flight['door_crossings'], 'trajectory_file': 'trajectory.npz',
                    'overhead': flight['overhead'],
                    'camera_mode': flight['camera_mode'],
                    'trajectory_sha256': digest(staged / 'trajectory.npz'),
                    'flight_settings_sha256': digest(staged / 'flight_settings.json'),
                    'fps': 30, 'horizontal_fov_degrees': 70,
                    'frame_count': len(flight['camera_to_world']), 'duration_seconds': len(floors)/30,
                    'last_pose_timestamp_seconds': (len(floors)-1)/30,
                    'sparse_floor_rule': plan['sparse_floor_rule'],
                    'floor_display_rule': 'Nearest supporting floor in the current scan; retain the previous floor inside unsupported doors and throughout the overhead ascent.',
                    'validation': validation, 'limitations': plan['limitations'],
                    'preview': {'width': 270, 'height': 480, 'mesh': 'preview/mesh.mp4',
                                'gaussian': 'preview/gaussian.mp4', 'representative_frames': samples,
                                'timings_seconds': render_timing},
                    'timings': timings,
                    'total_duration_seconds': round(total_wall_preview, 3),
                    'planning_seconds': round(plan['planning_seconds'], 3),
                    'trajectory_validation_seconds': round(validation_seconds, 3),
                    'preview_total_wall_seconds': round(total_wall_preview, 3)}
        write_json(staged / 'generate_cinematic_flythrough_manifest.json', manifest)
        LOGGER.info('Preview complete: %s in %.2f s', output, total_wall_preview)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()

    if output.exists() or output.is_symlink():
        backup = Path(tempfile.mkdtemp(prefix='.cinematic-backup-', dir=output.parent))
        backup.rmdir()
        output.rename(backup)
        try:
            staged.rename(output)
        except BaseException:
            backup.rename(output)
            raise
        if backup.is_symlink():
            backup.unlink()
        else:
            shutil.rmtree(backup)
    else:
        staged.rename(output)
    return loaded_models


def final(scene, output, preloaded_models=None):
    started = time.perf_counter()
    LOGGER.info('Starting %s (final mode)', MACRO)
    LOGGER.info('Scene: %s', scene)
    LOGGER.info('Output: %s', output)

    manifest_path = output / 'generate_cinematic_flythrough_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('macro_step') != MACRO or manifest.get('macro_step_status') != 'preview_complete':
        raise ValueError('A completed preview is required before final rendering')
    if manifest.get('schema_version') != 2:
        raise ValueError('Preview predates complete input identity checks; generate a new preview')

    timings = manifest.get('timings', [])

    file_handler = logging.FileHandler(output / 'generate_cinematic_flythrough.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    destination = output / 'final'
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Refusing to replace existing final outputs: {destination}')
    staged = Path(tempfile.mkdtemp(prefix='.final-', dir=output))

    try:
        flight = None
        floors = None
        tour = None

        def step_final_verification():
            nonlocal flight, floors, tour
            if digest(output / 'trajectory.npz') != manifest['trajectory_sha256']:
                raise ValueError('Preview trajectory has changed')
            if runtime_versions() != manifest.get('runtime_versions'):
                raise ValueError('Python or rendering package versions changed after preview')
            settings_path = output / 'flight_settings.json'
            if not settings_path.is_file() or digest(settings_path) != manifest.get('flight_settings_sha256'):
                raise ValueError('Preview flight settings have changed')
            settings = json.loads(settings_path.read_text())
            if any(manifest.get(key) != value for key, value in settings.items()):
                raise ValueError('Preview manifest flight settings have changed')
            verify_inputs(manifest['input_sha256'])
            tour = json.loads((scene / 'artifacts/Generate Interactive Viewer/tour.json').read_text())
            with np.load(output / 'trajectory.npz') as data:
                flight = {'camera_to_world': data['camera_to_world'], 'scans': data['scans'].tolist(),
                          'fps': manifest['fps'], 'checkpoints': manifest['checkpoint_events'],
                          'door_crossings': manifest['door_crossings'], 'overhead': manifest['overhead'],
                          'camera_mode': manifest['camera_mode']}
                floors = data['floors'].tolist()

        _run_timed_step('final_input_verification', step_final_verification, timings, started)

        render_timing = {}
        def step_final_rendering():
            nonlocal render_timing
            render_timing, _ = render_pair(scene, staged, tour, flight, floors, 540, 960, 'final', {},
                                           preloaded_models=preloaded_models)

        _run_timed_step('final_video_rendering', step_final_rendering, timings, started)
        verify_inputs(manifest['input_sha256'])
        if runtime_versions() != manifest['runtime_versions']:
            raise ValueError('Python or rendering package versions changed during final rendering')
        if digest(output / 'trajectory.npz') != manifest['trajectory_sha256']:
            raise ValueError('Preview trajectory changed during final rendering')
        if digest(output / 'flight_settings.json') != manifest['flight_settings_sha256']:
            raise ValueError('Preview flight settings changed during final rendering')

        staged.rename(destination)

        final_total = time.perf_counter() - started
        manifest['final'] = {'width': 540, 'height': 960, 'mesh': 'final/mesh.mp4', 'gaussian': 'final/gaussian.mp4',
                             'timings_seconds': render_timing}
        manifest['macro_step_status'] = 'complete'
        manifest['final_total_wall_seconds'] = round(final_total, 3)
        manifest['total_wall_seconds'] = round(manifest['preview_total_wall_seconds'] + final_total, 3)
        manifest['total_duration_seconds'] = manifest['total_wall_seconds']
        manifest['total_wall_seconds_scope'] = 'Active preview and final runs; excludes time spent inspecting the preview.'
        manifest['timings'] = timings
        try:
            write_json(manifest_path, manifest)
        except BaseException:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        LOGGER.info('Final complete: %s in %.2f s (total active wall: %.2f s)',
                    output, final_total, manifest['total_wall_seconds'])
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    finally:
        root_logger.removeHandler(file_handler)
        file_handler.close()


def both(scene, output, pairing_path, overwrite=False):
    started = time.perf_counter()
    LOGGER.info('Starting %s (combined preview + final mode)', MACRO)
    loaded_models = preview(scene, output, pairing_path, overwrite=overwrite)
    final(scene, output, preloaded_models=loaded_models)
    manifest = json.loads((output / 'generate_cinematic_flythrough_manifest.json').read_text())
    LOGGER.info('Completed combined %s run in %.2f s (active wall: %.2f s)',
                MACRO, time.perf_counter() - started, manifest['total_wall_seconds'])


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, default=STAGE.parent / 'current_scene')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pairings', type=Path, help='Explicit, provenance-checked reconstructed Phase 07 pairings')
    parser.add_argument('--mode', choices=('preview', 'final', 'both'), default='preview')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory if present in preview mode')
    args = parser.parse_args()
    scene = args.scene_dir.resolve()
    output = (args.output_dir or scene / 'artifacts' / MACRO).resolve()
    pairings = args.pairings.resolve() if args.pairings else None
    if args.mode == 'preview':
        preview(scene, output, pairings, overwrite=args.overwrite)
    elif args.mode == 'final':
        final(scene, output)
    elif args.mode == 'both':
        both(scene, output, pairings, overwrite=args.overwrite)


if __name__ == '__main__':
    main()

