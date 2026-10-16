#!/usr/bin/env python3
"""3D-only reconstruction from frozen 2D observations; no learned-model imports/calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import open3d as o3d
from scipy import sparse

from panoptic3d import (assign_faces, balance_votes, boundary_distance, boundary_edges, cached_mask_extent, dual_graph, extent_conflicts,
                       object_metrics, object_source_aliases, observation_components, physical_ids, resolve_tracks, restrict_instance_votes, safe_removal, summarize_metrics,
                       view_groups)

# This existing reader imports no model and performs only COLMAP text parsing.
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / '03_FastGS_DNSplatter'))
from gsplat_train import load_cameras

REVIEW = {
    'frame_00969': 'thin wall decorations; similar adjacent pictures',
    'frame_00856': 'ceiling and wall boundary; disconnected ceiling pieces',
    'frame_00773': 'bed frame and wall-adjacent furniture',
    'frame_00693': 'thin shelves; books and small objects',
    'frame_00602': 'nightstand; small teddy; missing geometry',
    'frame_00455': 'repeated wardrobe doors; object boundaries',
    'frame_01041': 'small fan; thin spokes; floor leakage',
    'frame_01086': 'door handle and thin wall-adjacent geometry',
    'frame_01117': 'chair; occluded disconnected pieces; floor boundary',
    'frame_01305': 'window and curtain; difficult light boundary',
    'frame_01340': 'small wall fixtures and hanging object',
    'frame_00350': 'boxing bag and thin chains; floaters',
    'frame_00259': 'shelves and small books; wall leakage',
    'frame_00221': 'small dinosaur; thin ornament and picture boundaries',
    'frame_00184': 'thin shelves; small objects; baseline cleanup holes',
    'frame_00562': 'thin framed picture; incomplete surface; floaters',
}
DIAGNOSTIC = {'frame_00560', 'frame_00617', 'frame_00983', 'frame_01006', 'frame_01169', 'frame_01193'}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs(out):
    metadata = json.loads((out / 'inputs.json').read_text())
    if any(digest(Path(path)) != expected for path, expected in metadata['files'].items()):
        raise RuntimeError('a frozen input or baseline changed during reconstruction')
    return metadata


def output_directory(stage, name):
    if not name or Path(name).name != name or not name.startswith('panoptic3d_'):
        raise ValueError('run name must be a single new panoptic3d_* directory name')
    out = stage / name
    if out.is_symlink() or out.resolve().parent != stage.resolve():
        raise ValueError('output must remain a direct child of the 06c stage')
    return out


def ray_scene(vertices, triangles, transform):
    raw = vertices @ transform[:3, :3].T + transform[:3, 3]
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(raw.astype('float32')),
                        o3d.core.Tensor(triangles.astype('uint32')))
    return scene


def cast(scene, camera, stride, offset=0):
    """Sample original pixel centres without resampling masks or changing intrinsics."""
    y, x = np.mgrid[offset:camera['height']:stride, offset:camera['width']:stride]
    return cast_pixels(scene, camera, y, x)


def cast_pixels(scene, camera, y, x):
    k, v = camera['K'], camera['viewmat']
    local = np.stack([(x + .5 - k[0, 2]) / k[0, 0],
                      (y + .5 - k[1, 2]) / k[1, 1], np.ones_like(x)], -1)
    directions = local @ v[:3, :3]
    center = -v[:3, :3].T @ v[:3, 3]
    rays = np.concatenate([np.broadcast_to(center, directions.shape), directions], -1).astype('float32')
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    face = hit['primitive_ids'].numpy().astype('int64')
    face[face == 2**32 - 1] = -1
    # Rays have camera-local z=1; t_hit is camera Z, not Euclidean range.
    depth = hit['t_hit'].numpy()
    normal = hit['primitive_normals'].numpy()
    cosine = np.abs(np.sum(normal * directions, axis=-1)) / np.linalg.norm(directions, axis=-1)
    return face.astype('int32'), depth, cosine, y, x


def native_depth_points(camera, y, x, depth, transform):
    k, v = camera['K'], camera['viewmat']
    local = np.stack([(x + .5 - k[0, 2]) / k[0, 0],
                      (y + .5 - k[1, 2]) / k[1, 1], np.ones_like(x)], -1)
    local *= np.where(np.isfinite(depth), depth, 0)[..., None]
    raw = (local - v[:3, 3]) @ v[:3, :3]
    inverse = np.linalg.inv(transform)
    return raw @ inverse[:3, :3].T + inverse[:3, 3]


def load_depth_context(workspace, cameras):
    root = workspace / '02_depth_estimation/depth'
    metadata = root / 'poses_da3.npz'
    if not metadata.exists():
        return None, set(), 'cached camera metadata unavailable; depth disabled'
    with np.load(metadata) as z:
        mapping = {Path(str(n)).stem: i for i, n in enumerate(z['names'])}
        for c in cameras:
            i = mapping.get(c['image_path'].stem)
            if i is None or not np.allclose(c['K'], z['K'][i], atol=.01) or not np.allclose(c['viewmat'], z['w2c'][i], atol=1e-4):
                return None, set(), 'cached cameras differ; depth disabled'
    alignment = json.loads((root / 'depth_alignment.json').read_text())
    unreliable = {Path(n).stem for key in ('unreliable', 'unalignable', 'scale_outliers')
                  for n in alignment.get(key, [])}
    return root, unreliable, 'matching cached COLMAP poses/intrinsics; unreliable frames excluded'


def depth_evidence(depth, cached, valid, interior):
    """Calibrate disagreement to robust in-frame residual; reject poorly aligned caches."""
    usable = valid & np.isfinite(cached) & (cached > 0) & interior
    residual = depth[usable] - cached[usable]
    if len(residual) < 100:
        return np.full(depth.shape, 2, np.int8), None
    median = float(np.median(np.abs(residual)))
    mad = float(np.median(np.abs(residual - np.median(residual))))
    if median > .1:
        return np.full(depth.shape, 2, np.int8), dict(median_error_m=median, enabled=False)
    tolerance = min(.2, max(.05, median + 3 * 1.4826 * mad))
    state = np.full(depth.shape, 2, np.int8)
    state[usable] = 0
    state[usable & (depth - cached > tolerance)] = -1  # Missing foreground/occluded mesh.
    state[usable & (cached - depth > tolerance)] = 1   # Mesh occupies observed free space.
    return state, dict(median_error_m=median, tolerance_m=tolerance, enabled=True)


def pose_support(inputs):
    """Use triangulated tracks, not raw keypoint detections, to gate weak poses."""
    lines = iter(line for line in (inputs / 'sparse/0/images.txt').read_text().splitlines()
                 if not line.startswith('#'))
    result = {}
    for line in lines:
        header = line.split()
        if not header:
            continue
        if len(header) < 10:
            raise ValueError('malformed COLMAP image header')
        points = next(lines, '').split()
        if len(points) % 3:
            raise ValueError('malformed COLMAP observations')
        count = sum(int(points[i]) >= 0 for i in range(2, len(points), 3))
        result[Path(header[9]).stem] = dict(registered_points=count, trusted=count >= 100,
            status='supported' if count >= 100 else 'weak_pose_excluded_from_geometry')
    return result


def read_scene_review(stage, path):
    if path is None:
        return {}
    review = json.loads(path.read_text())
    if review['mesh_sha256'] != digest(stage / 'inputs/mesh.ply'):
        raise ValueError('scene review belongs to a different mesh')
    referenced = {a['frame'] for key in ('annotations', 'distinct_masks', 'part_of', 'unresolved_masks') for a in review.get(key, [])}
    referenced |= {o['frame'] for key in ('same_object', 'disconnected_same_object', 'object_extents')
                   for pair in review.get(key, []) for o in pair['observations']}
    if any(f'{frame}_final.npz' not in review.get('anchor_archive_sha256', {}) for frame in referenced):
        raise ValueError('every reviewed observation needs its frozen archive hash')
    for name, expected in review.get('anchor_archive_sha256', {}).items():
        if digest(stage / 'full362_final_pipeline/npz_data' / name) != expected:
            raise ValueError('scene review anchor masks changed')
    return review


def region_templates(stage, cameras, scene, review, poses=None):
    """Reviewed physical regions are transferred only through shared visible triangles."""
    templates = defaultdict(list)
    by_stem = {c['image_path'].stem: c for c in cameras}
    for spec in review.get('distinct_masks', []):
        with np.load(stage / 'full362_final_pipeline/npz_data' / f"{spec['frame']}_final.npz") as z:
            if int(z['ids'][spec['mask_index']]) != spec['source_id']:
                raise ValueError('review mask source mismatch')
            active = (z['masks'][spec['mask_index']] > 0) & (z['labels'] == spec['source_id'])
            for component, mask in observation_components(active, True):
                y, x = np.where(mask)
                faces = np.empty(0, np.int32)
                if poses is None or poses[spec['frame']]['trusted']:
                    faces, _, _, _, _ = cast_pixels(scene, by_stem[spec['frame']], y, x)
                templates[spec['source_id']].append(dict(frame=spec['frame'], mask_index=spec['mask_index'],
                    component=component, faces=np.unique(faces[faces >= 0])))
    return templates


def anchored_regions(mask, source, frame, mask_index, templates, scene, camera):
    seeds = templates.get(int(source), [])
    if not seeds:
        return [(0, mask)]
    explicit = [s for s in seeds if s['frame'] == frame and s['mask_index'] == mask_index]
    pieces = observation_components(mask, True)
    if explicit:
        return pieces
    groups = defaultdict(lambda: np.zeros(mask.shape, bool))
    for component, region in pieces:
        y, x = np.where(region)
        faces, _, _, _, _ = cast_pixels(scene, camera, y, x)
        faces = np.unique(faces[faces >= 0])
        overlaps = [len(np.intersect1d(faces, s['faces'], assume_unique=True)) for s in seeds]
        best = int(np.argmax(overlaps))
        # No nearest-position guess: fragments without repeated face evidence stay uncertain.
        key = best + 1 if overlaps[best] >= 3 else 0
        groups[key] |= region
    if len(groups) < 2:
        return [(0, mask)]
    return [(k, region) for k, region in sorted(groups.items())]


def collect(workspace, out, stride, review_file=None):
    stage = workspace / '06c_semantic_segmentation'
    inputs, frozen = stage / 'inputs', stage / 'full362_final_pipeline/npz_data'
    review = read_scene_review(stage, review_file)
    write_json(out / 'observation_policy.json', dict(distinct_masks=review.get('distinct_masks', [])))
    cameras = sorted(load_cameras(inputs), key=lambda c: c['image_path'].name)
    poses = pose_support(inputs)
    groups = view_groups(cameras)
    group_counts = np.maximum(1, np.bincount(groups, weights=[poses[c['image_path'].stem]['trusted'] for c in cameras]))
    mesh = o3d.io.read_triangle_mesh(str(inputs / 'mesh.ply'))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    transform = np.load(inputs / 'T_mesh_to_raw.npy')
    if not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError('invalid mesh-to-camera transform')
    scene = ray_scene(vertices, triangles, transform)
    templates = region_templates(stage, cameras, scene, review, poses)
    depth_root, unreliable, depth_reason = load_depth_context(workspace, cameras)
    (out / 'evidence').mkdir()
    fixed_review = {c['image_path'].stem: REVIEW[c['image_path'].stem] for c in cameras if c['image_path'].stem in REVIEW}
    if len(fixed_review) != len(REVIEW):
        raise ValueError('fixed review cameras missing')
    write_json(out / 'review_set.json', dict(frames=fixed_review, stride=stride,
               rule='fixed before optimization; base evidence grid offset=0, evaluation grid offset=1; dense boundary/small-mask evidence may share evaluation pixels',
               protected='all objects, with separate macro averages for small objects'))
    observations, frames, cannot_link = [], [], []
    row_parts, col_parts, value_parts = [], [], []
    protected_paths = [inputs / 'mesh.ply', inputs / 'T_mesh_to_raw.npy',
                       inputs / 'sparse/0/cameras.txt', inputs / 'sparse/0/images.txt']
    for path in (stage / 'mesh_projection_step1/triangle_class_votes.npz',
                 stage / 'mesh_dispute_resolution_step2/mesh_dispute_resolved.ply',
                 stage / 'mesh_objects_step3/mesh_cleaned_no_floaters.ply',
                 stage / 'mesh_objects_step3/objects_manifest.json'):
        if path.exists():
            protected_paths.append(path)
    if depth_root:
        protected_paths += [depth_root / 'poses_da3.npz', depth_root / 'depth_alignment.json']
    protected_paths += sorted(frozen.glob('*_final.npz')) + [c['image_path'] for c in cameras]
    if depth_root:
        protected_paths += [depth_root / 'depth_maps' / f"{c['image_path'].stem}.npy" for c in cameras
                            if (depth_root / 'depth_maps' / f"{c['image_path'].stem}.npy").exists()]
    baseline_exports = stage / 'mesh_objects_step3/extracted_instances'
    if baseline_exports.exists():
        protected_paths += sorted(baseline_exports.glob('*.ply'))
    input_hashes = {str(p.resolve()): digest(p) for p in protected_paths}
    start = time.perf_counter()
    for frame_index, camera in enumerate(cameras):
        stem = camera['image_path'].stem
        trusted_pose = poses[stem]['trusted']
        archive = frozen / f'{stem}_final.npz'
        face, depth, cosine, y, x = cast(scene, camera, stride)
        with np.load(archive) as data:
            labels = data['labels']
            if labels.shape != (camera['height'], camera['width']) or np.any(labels < 0):
                raise ValueError(f'{archive}: label/camera schema mismatch')
            full_distance = boundary_distance(labels)
            distance = full_distance[y, x]
            sampled_labels = labels[y, x]
            masks, ids, embeddings = [], [], []
            original_indices, component_indices = [], []
            for original_index, (mask, source, embedding) in enumerate(zip(data['masks'], data['ids'], data['embeddings'])):
                for component_index, region in anchored_regions((mask > 0) & (labels == source), source,
                        stem, original_index, templates, scene, camera):
                    masks.append(region)
                    ids.append(source)
                    embeddings.append(embedding)
                    original_indices.append(original_index)
                    component_indices.append(component_index)
            obsmap = np.full(face.shape, -1, np.int32)
            full_obsmap = np.full(labels.shape, -1, np.int32)
            frame_obs = []
            for mask_index, (mask, source, embedding) in enumerate(zip(masks, ids, embeddings)):
                pixels = mask[y, x].astype(bool) & (sampled_labels == source) & (obsmap < 0)
                if not pixels.any():
                    # Preserve even a subpixel or fully overwritten detection in the catalog.
                    pixel_count = int(np.count_nonzero(mask & (labels == source)))
                else:
                    pixel_count = int(pixels.sum()) * stride**2
                oid = len(observations)
                vector = embedding.astype('float64')
                vector /= max(1e-12, np.linalg.norm(vector))
                observations.append(dict(observation_id=oid, frame=stem, frame_index=frame_index,
                    mask_index=original_indices[mask_index], component_index=component_indices[mask_index],
                    source_id=int(source), pixels=pixel_count,
                    group=int(groups[frame_index]), embedding=vector.tolist()))
                obsmap[pixels] = oid
                full_obsmap[mask & (full_obsmap < 0)] = oid
                frame_obs.append(oid)
            if np.any((sampled_labels > 0) & (obsmap < 0)):
                raise ValueError(f'{stem}: label pixels lack a source mask')
        # Distinct archived masks in the same frame are explicit separation evidence.
        for i, a in enumerate(frame_obs):
            for b in frame_obs[i + 1:]:
                if observations[a]['pixels'] and observations[b]['pixels']:
                    cannot_link.append((a, b))
        valid = face >= 0
        state = np.full(face.shape, 2, np.int8)
        calibration = None
        original = None
        cached = np.zeros(face.shape, np.float32)
        depth_path = depth_root / 'depth_maps' / f'{stem}.npy' if depth_root else None
        if trusted_pose and depth_path and depth_path.exists() and stem not in unreliable:
            original = np.load(depth_path, mmap_mode='r')
            if original.shape == labels.shape:
                cached = original[y, x].astype('float32')
                state, calibration = depth_evidence(depth, cached, valid, distance >= 6)
        weights = cosine.astype('float32')**1.5 * np.clip(distance / 12, .2, 1)
        weights[(~valid) | (obsmap < 0) | (cosine < .17) | (state == 1) | (state == -1)] = 0
        extents = {}
        if original is not None and original.shape == labels.shape:
            for oid, mask in zip(frame_obs, masks):
                # Registered depth may justify abstention even when missing mesh defeats calibration.
                extent = cached_mask_extent(mask, original, .2)
                if extent:
                    extents[oid] = extent
                    observations[oid]['depth_extent'] = extent
        withheld = valid & extent_conflicts(depth, state, obsmap, extents)
        weights[withheld] = 0
        # Spend additional rays at boundaries and small masks, never use size to discard objects.
        dense = (labels > 0) & (full_distance <= 12)
        for mask, source in zip(masks, ids):
            active = (mask > 0) & (labels == source)
            if np.count_nonzero(active) <= .02 * labels.size:
                dense |= active
        dense[y, x] = False
        ey, ex = np.where(dense)
        ef, ed, ec, _, _ = cast_pixels(scene, camera, ey, ex)
        eo = np.full(len(ey), -1, np.int32)
        for oid, mask, source in zip(frame_obs, masks, ids):
            pixels = (mask[ey, ex] > 0) & (labels[ey, ex] == source) & (eo < 0)
            eo[pixels] = oid
        es = np.full(len(ey), 2, np.int8)
        if calibration and calibration['enabled']:
            reference = original[ey, ex].astype('float32')
            usable = (ef >= 0) & np.isfinite(reference) & (reference > 0) & (full_distance[ey, ex] >= 6)
            es[usable] = 0
            es[usable & (ed - reference > calibration['tolerance_m'])] = -1
            es[usable & (reference - ed > calibration['tolerance_m'])] = 1
        ew = ec.astype('float32')**1.5 * np.clip(full_distance[ey, ex] / 12, .2, 1)
        ew[(ef < 0) | (eo < 0) | (ec < .17) | (es == 1) | (es == -1)] = 0
        extra_withheld = (ef >= 0) & extent_conflicts(ed, es, eo, extents)
        ew[extra_withheld] = 0
        if not trusted_pose:
            weights[:] = 0
            ew[:] = 0
        all_weights = np.r_[weights.ravel(), ew]
        all_observations = np.r_[obsmap.ravel(), eo]
        all_faces = np.r_[face.ravel(), ef]
        valid_vote = all_weights > 0
        # One score per face and observation per frame, independent of pixel footprint.
        coo = sparse.coo_matrix((all_weights[valid_vote], (all_observations[valid_vote], all_faces[valid_vote])),
                                 shape=(len(observations), len(triangles))).tocsr()
        counts = sparse.coo_matrix((np.ones(valid_vote.sum()), (all_observations[valid_vote], all_faces[valid_vote])),
                                    shape=coo.shape).tocsr()
        reduced = coo.copy()
        reduced.data /= counts.data
        reduced.data /= group_counts[groups[frame_index]]
        rr, cc = reduced.nonzero()
        row_parts.append(rr.astype('int32')); col_parts.append(cc.astype('int32'))
        value_parts.append(reduced.data.astype('float32'))
        np.savez_compressed(out / 'evidence' / f'{stem}.npz', face=face, depth=depth,
            cached_depth=cached, state=state, labels=sampled_labels, observation=obsmap,
            distance=np.minimum(distance, 65504).astype('float16'), cosine=cosine.astype('float16'),
            extra_face=ef, extra_observation=eo, extra_state=es,
            extent_withheld=withheld, extra_extent_withheld=extra_withheld)
        np.savez_compressed(out / 'evidence' / f'{stem}_regions.npz', observation=full_obsmap)
        frames.append(dict(stem=stem, group=int(groups[frame_index]), depth=calibration, pose=poses[stem],
                           hit_pixels=int(valid.sum()), unknown_pixels=int((sampled_labels == 0).sum()),
                           no_hit_pixels=int((~valid & (sampled_labels > 0)).sum())))
        if (frame_index + 1) % 30 == 0:
            print(f'evidence {frame_index + 1}/{len(cameras)} ({time.perf_counter()-start:.1f}s)', flush=True)
    support = sparse.coo_matrix((np.concatenate(value_parts),
                                 (np.concatenate(row_parts), np.concatenate(col_parts))),
                                 shape=(len(observations), len(triangles))).tocsr()
    sparse.save_npz(out / 'observation_support.npz', support)
    write_json(out / 'observations.json', observations)
    write_json(out / 'cannot_link.json', cannot_link)
    write_json(out / 'frames.json', frames)
    write_json(out / 'inputs.json', dict(workspace=str(workspace.resolve()), stride=stride,
        depth_policy=depth_reason, diverse_views=int(len(group_counts)), triangles=len(triangles),
        pose_policy='fewer than 100 triangulated COLMAP tracks: catalog/diagnostics only; no surface or deletion evidence',
        files=input_hashes))
    verify_inputs(out)
    print(f'evidence complete: {len(observations)} masks, {len(group_counts)} diverse views, {support.nnz} correspondences', flush=True)


def palette(ids):
    ids = np.asarray(ids)
    colors = np.stack([(197 * (ids - 1) + 31) % 255,
                       (109 * (ids - 1) + 71) % 255,
                       (37 * (ids - 1) + 93) % 255], -1).astype('uint8')
    colors[ids <= 0] = [128, 128, 128]
    return colors


def export_mesh(path, vertices, triangles, labels, parts=None):
    """Face properties preserve exact boundaries; RGB on vertices is only a preview."""
    from plyfile import PlyData, PlyElement
    colors = palette(labels)
    vertex_colors = np.zeros((len(vertices), 3), np.float64)
    count = np.zeros(len(vertices), np.int32)
    for corner in range(3):
        np.add.at(vertex_colors, triangles[:, corner], colors)
        np.add.at(count, triangles[:, corner], 1)
    vertex_colors = (vertex_colors / np.maximum(count, 1)[:, None]).astype('uint8')
    vertex = np.empty(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for j, key in enumerate(('x', 'y', 'z')):
        vertex[key] = vertices[:, j]
    for j, key in enumerate(('red', 'green', 'blue')):
        vertex[key] = vertex_colors[:, j]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,)), ('instance_id', 'i4'), ('part_id', 'i4'),
                                          ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    face['vertex_indices'] = triangles
    face['instance_id'] = labels
    face['part_id'] = labels if parts is None else parts
    for j, key in enumerate(('red', 'green', 'blue')):
        face[key] = colors[:, j]
    PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(face, 'face')], text=False).write(str(path))


def reviewed_object_bounds(workspace, review, observations, observation_instances, transform):
    specs = review.get('object_extents', [])
    if not specs:
        return []
    stage = workspace / '06c_semantic_segmentation'
    cameras = load_cameras(stage / 'inputs')
    by_frame = {c['image_path'].stem: c for c in cameras}
    depth_root, unreliable, _ = load_depth_context(workspace, cameras)
    poses = pose_support(stage / 'inputs')
    result = []
    for spec in specs:
        if not spec.get('complete_object_coverage') or not spec.get('reason'):
            raise ValueError('object extent needs reviewed complete-object coverage and a reason')
        lows, highs, instances = [], [], []
        for anchor in spec['observations']:
            matches = [o for o in observations if all(o.get(k, 0) == v for k, v in anchor.items())]
            if len(matches) != 1:
                raise ValueError('object extent anchor must identify exactly one observation')
            o = matches[0]
            if not depth_root or o['frame'] in unreliable or not poses[o['frame']]['trusted'] or 'depth_extent' not in o:
                raise ValueError('object extent requires reliable registered cached depth')
            with np.load(stage / 'full362_final_pipeline/npz_data' / f"{o['frame']}_final.npz") as archive:
                mask = (archive['masks'][o['mask_index']] > 0) & (archive['labels'] == o['source_id'])
                if o.get('component_index', 0):
                    mask = dict(observation_components(mask, True))[o['component_index']]
            cached = np.load(depth_root / 'depth_maps' / f"{o['frame']}.npy", mmap_mode='r')
            if not cached_mask_extent(mask, cached, .2):
                raise ValueError('object extent lacks positive cached depth on half its mask')
            y, x = np.where(mask & np.isfinite(cached) & (cached > 0))
            step = max(1, len(y) // 20000)
            y, x = y[::step], x[::step]
            cloud = native_depth_points(by_frame[o['frame']], y, x, cached[y, x].astype('float32'), transform)
            lo, hi = np.percentile(cloud, [.5, 99.5], axis=0)
            lows.append(lo); highs.append(hi); instances.append(int(observation_instances[o['observation_id']]))
        if not instances or len(set(instances)) != 1:
            raise ValueError('object extent anchors must already have one established identity')
        result.append(dict(instance_id=instances[0], native_lower_m=(np.min(lows, axis=0)-.2).tolist(),
            native_upper_m=(np.max(highs, axis=0)+.2).tolist(), padding_m=.2, quantiles=[.5, 99.5], evidence=spec))
    return result


def silhouette_conflicts(out, physical_labels, observation_instances, observations, scoped_instances):
    """Count unobscured contradictions only for reviewed complete, opaque objects."""
    result = np.zeros(len(physical_labels), np.uint16)
    if not scoped_instances:
        return result
    visible = defaultdict(set)
    for o in observations:
        iid = int(observation_instances[o['observation_id']])
        if o['pixels'] and iid in scoped_instances:
            visible[o['frame']].add(iid)
    groups = defaultdict(list)
    for frame in json.loads((out / 'frames.json').read_text()):
        if (visible[frame['stem']] and frame.get('pose', {}).get('trusted', False)
                and frame['depth'] and frame['depth']['enabled']):
            groups[frame['group']].append(frame)
    for frames in groups.values():
        parts = []
        for frame in frames:
            with np.load(out / 'evidence' / f"{frame['stem']}.npz") as z:
                valid = ((z['face'] >= 0) & (z['observation'] >= 0) & (z['labels'] > 0)
                         & (z['state'] == 1) & (z['distance'] >= 12))
                faces = z['face'][valid]
                observed = observation_instances[z['observation'][valid]]
                predicted = physical_labels[faces]
                contradicted = np.isin(predicted, list(visible[frame['stem']])) & (predicted != observed)
                parts.append(faces[contradicted])
        result[np.unique(np.concatenate(parts))] += 1
    return result


def component_depth_evidence(workspace, out, vertices, triangles, labels, graph):
    """Check every detached centroid; pixel labels alone can also land on floaters."""
    from scipy.sparse.csgraph import connected_components
    _, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    candidates = np.flatnonzero(components != sizes.argmax())
    free, agree = np.zeros(len(triangles), np.uint16), np.zeros(len(triangles), np.uint16)
    if not len(candidates):
        return free, agree
    stage = workspace / '06c_semantic_segmentation'
    inputs = stage / 'inputs'
    cameras = sorted(load_cameras(inputs), key=lambda c: c['image_path'].name)
    root, unreliable, _ = load_depth_context(workspace, cameras)
    if root is None:
        return free, agree
    transform = np.load(inputs / 'T_mesh_to_raw.npy')
    centers = vertices[triangles[candidates]].mean(axis=1)
    raw = centers @ transform[:3, :3].T + transform[:3, 3]
    scene = ray_scene(vertices, triangles, transform)
    frames = json.loads((out / 'frames.json').read_text())
    group_free, group_agree = {}, {}
    stride = json.loads((out / 'inputs.json').read_text())['stride']
    for camera, frame in zip(cameras, frames):
        calibration = frame['depth']
        stem = camera['image_path'].stem
        if not frame.get('pose', {}).get('trusted', True) or stem in unreliable or not calibration or not calibration['enabled']:
            continue
        cam = raw @ camera['viewmat'][:3, :3].T + camera['viewmat'][:3, 3]
        projected = cam @ camera['K'].T
        uv = np.floor(projected[:, :2] / np.maximum(cam[:, 2:3], 1e-12)).astype('int64')
        inside = (cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < camera['width']) & (uv[:, 1] >= 0) & (uv[:, 1] < camera['height'])
        ids = np.flatnonzero(inside)
        if not len(ids):
            continue
        center = -camera['viewmat'][:3, :3].T @ camera['viewmat'][:3, 3]
        directions = (raw[ids] - center) / cam[ids, 2:3]
        rays = np.c_[np.broadcast_to(center, directions.shape), directions].astype('float32')
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        visible = np.abs(hit['t_hit'].numpy() - cam[ids, 2]) < .005
        cache = np.load(root / 'depth_maps' / f'{stem}.npy', mmap_mode='r')
        depth = cache[uv[ids, 1], uv[ids, 0]].astype('float32')
        valid = visible & np.isfinite(depth) & (depth > 0)
        with np.load(out / 'evidence' / f'{stem}.npz') as z:
            interior = z['distance'][np.minimum(uv[ids, 1] // stride, z['distance'].shape[0]-1),
                                     np.minimum(uv[ids, 0] // stride, z['distance'].shape[1]-1)] >= 6
        residual = depth - cam[ids, 2]
        gf = group_free.setdefault(frame['group'], np.zeros(len(candidates), bool))
        ga = group_agree.setdefault(frame['group'], np.zeros(len(candidates), bool))
        gf[ids[valid & interior & (residual > calibration['tolerance_m'])]] = True
        ga[ids[valid & (np.abs(residual) <= calibration['tolerance_m'])]] = True
    for values in group_free.values():
        free[candidates] += values
    for values in group_agree.values():
        agree[candidates] += values
    write_json(out / 'unsupported_components.json', [dict(component_id=int(c),
        triangles=int(np.count_nonzero(components[candidates] == c)),
        min_free_views=int(free[candidates[components[candidates] == c]].min()),
        contradicted_faces=int(np.count_nonzero(agree[candidates[components[candidates] == c]])))
        for c in np.unique(components[candidates])])
    return free, agree


def solve(workspace, out, strength, review_file=None, area_balance=0):
    verify_inputs(out)
    stage = workspace / '06c_semantic_segmentation'
    observations = json.loads((out / 'observations.json').read_text())
    support = sparse.load_npz(out / 'observation_support.npz')
    cannot = json.loads((out / 'cannot_link.json').read_text())
    review = read_scene_review(stage, review_file)
    if review_file:
        write_json(out / 'scene_review.json', review)
    policy_path = out / 'observation_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    if review.get('distinct_masks', []) != policy.get('distinct_masks', []):
        raise ValueError('changed observation splitting policy requires new evidence collection')
    reviewed_pairs = review.get('same_object', [])
    roots, decisions = resolve_tracks(support, observations, cannot, reviewed_pairs=reviewed_pairs,
        rejected_pairs=review.get('rejected_merges', []), disconnected_pairs=review.get('disconnected_same_object', []))
    write_json(out / 'configuration.json', dict(smoothness=strength, iterations=4, area_balance_exponent=area_balance,
        view_translation_m=.15, view_angle_degrees=12, prototype_similarity=.75,
        overlap_fraction=.35, reviewed_alias_similarity=.65, reviewed_alias_overlap=.5,
        silhouette_abstention=dict(minimum_pose_groups=2, minimum_mask_interior_px=12,
            scope='reviewed complete opaque physical objects', depth_state='free_space',
            exemption='same resolved physical identity, including aliases and parts', geometry_action='none'),
        source_files={str(p.name): digest(p) for p in (Path(__file__), Path(__file__).with_name('panoptic3d.py'),
                                                     Path(__file__).with_name('panoptic3d_baseline.py'))}))
    unique = sorted(set(roots), key=lambda r: (observations[r]['frame'], observations[r]['mask_index'],
                                             observations[r].get('component_index', 0)))
    root_to_id = {r: i+1 for i, r in enumerate(unique)}
    obs_to_instance = np.array([root_to_id[r] for r in roots], np.int32)
    mapping = sparse.csr_matrix((np.ones(len(roots)), (np.arange(len(roots)), obs_to_instance)),
                               shape=(len(roots), len(unique)+1))
    votes = (support.T @ mapping).tocsr()
    sparse.save_npz(out / 'instance_votes.npz', votes)
    mesh = o3d.io.read_triangle_mesh(str(stage / 'inputs/mesh.ply'))
    mesh.compute_triangle_normals()
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    constraints = reviewed_object_bounds(workspace, review, observations, obs_to_instance,
                                         np.load(stage / 'inputs/T_mesh_to_raw.npy'))
    for constraint in constraints:
        votes, withheld = restrict_instance_votes(votes, vertices[triangles].mean(axis=1),
            constraint['instance_id'], constraint['native_lower_m'], constraint['native_upper_m'])
        constraint.update(withheld_face_ids=withheld.tolist(), withheld_faces=len(withheld))
    write_json(out / 'object_extent_constraints.json', constraints)
    graph = dual_graph(triangles)
    normals = np.asarray(mesh.triangle_normals).astype('float32')
    rgb = np.asarray(mesh.vertex_colors)[triangles].mean(axis=1).astype('float32')
    face_areas = np.linalg.norm(np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                                         vertices[triangles[:, 2]] - vertices[triangles[:, 0]]), axis=1) / 2
    balanced, core_areas = balance_votes(votes, face_areas, area_balance)
    np.savez_compressed(out / 'area_balance.npz', uncontested_core_area_m2=core_areas, exponent=area_balance)
    labels, confidence = assign_faces(balanced, graph, normals, rgb, strength)
    free = np.zeros(len(triangles), np.uint16)
    conflict = np.zeros(len(triangles), np.uint16)
    # Count independent groups, rather than repeated near-identical images.
    frame_info = json.loads((out / 'frames.json').read_text())
    for group in sorted({f['group'] for f in frame_info}):
        free_ids, conflict_ids = [], []
        for frame in (f for f in frame_info if f['group'] == group):
            with np.load(out / 'evidence' / f"{frame['stem']}.npz") as z:
                for prefix in ('', 'extra_') if 'extra_face' in z.files else ('',):
                    face, state = z[f'{prefix}face'], z[f'{prefix}state']
                    free_ids.append(face[(state == 1) & (face >= 0)])
                    conflict_ids.append(face[(state == 0) & (face >= 0)])
        free[np.unique(np.concatenate(free_ids))] += 1
        conflict[np.unique(np.concatenate(conflict_ids))] += 1
    centroid_free, centroid_agree = component_depth_evidence(workspace, out, vertices, triangles, labels, graph)
    # Use centroid evidence as a complete per-face check, without double-counting pixel observations.
    pixel_free = free.copy()
    free = centroid_free
    conflict = np.maximum(conflict, centroid_agree)
    contradicted = (centroid_free >= 3) & (conflict == 0)
    supported = (labels > 0) & ~contradicted
    removed, components = safe_removal(triangles, graph, supported, free, conflict)
    before_edges, after_edges = boundary_edges(triangles), boundary_edges(triangles[~removed])
    before_set = {tuple(e) for e in before_edges}
    new_edges = [e for e in after_edges if tuple(e) not in before_set]
    if new_edges:
        raise RuntimeError('cleanup opened new mesh boundaries')
    arrays = dict(instance_ids=labels, confidence=confidence,
        removed=removed, original_face_ids=np.arange(len(triangles)), observation_instances=obs_to_instance,
        free_space_groups=free, pixel_free_space_groups=pixel_free,
        agreeing_depth_groups=conflict, geometric_components=components)
    write_json(out / 'identity_decisions.json', decisions)
    write_json(out / 'cleanup.json', dict(removed_triangles=int(removed.sum()),
        removed_components=int(len(np.unique(components[removed]))), new_boundary_edges=len(new_edges),
        supported_triangles_removed=int(np.count_nonzero(removed & supported)),
        mask_voted_but_depth_contradicted_triangles_removed=int(np.count_nonzero(removed & (labels > 0))),
        component_splits=0, repairs=[],
        rule='whole vertex-isolated components; >=3 diverse centroid free-space views per face; no agreeing depth; mask votes contradicted by every calibrated observation are not surface support',
        uncertain_preserved=int(np.count_nonzero(labels == 0))))
    (out / 'instances').mkdir(exist_ok=True)
    from scipy.sparse.csgraph import connected_components
    a, b = graph.nonzero()
    same = (labels[a] == labels[b]) & (labels[a] > 0)
    _, labeled_components = connected_components(sparse.csr_matrix(
        (np.ones(same.sum(), bool), (a[same], b[same])), shape=graph.shape), directed=False)
    manifest = []
    trusted_frames = {f['stem'] for f in frame_info if f.get('pose', {}).get('trusted', True)}
    for root in unique:
        iid = root_to_id[root]
        obs_ids = np.flatnonzero(obs_to_instance == iid)
        source_ids = sorted({observations[i]['source_id'] for i in obs_ids})
        face_ids = np.flatnonzero((labels == iid) & ~removed)
        groups = {observations[i]['group'] for i in obs_ids if observations[i]['pixels']}
        manifest.append(dict(instance_id=iid,
            stable_key=f"{observations[root]['frame']}_mask_{observations[root]['mask_index']:02d}" +
                       (f"_part_{observations[root]['component_index']:02d}" if observations[root].get('component_index', 0) else ''),
            source_ids=source_ids, observation_ids=obs_ids.tolist(),
            frames=sorted({observations[i]['frame'] for i in obs_ids}), independent_views=len(groups),
            pose_reliability='supported' if any(observations[i]['frame'] in trusted_frames for i in obs_ids) else 'weak_pose_only',
            category=None, category_status='uncertain: class-agnostic source tracks provide no semantic names',
            source_relationship='same physical surface supported across views' if len(source_ids)>1 else 'source track hypothesis',
            parent_instance_id=None, part_relationship_status='unestablished; adjacency alone is insufficient',
            geometry_triangles=int(len(face_ids)), geometry_components=int(len(set(labeled_components[face_ids]))),
            confidence=float(np.mean(confidence[face_ids])) if len(face_ids) else 0,
            completeness_status='missing_geometry' if not len(face_ids) else 'pending_reprojection',
            identity_status='multi_view_supported' if len(groups)>1 else 'single_view_uncertain'))
    for annotation in review.get('annotations', []):
        for entry in manifest:
            matched = [i for i in entry['observation_ids']
                       if observations[i]['frame'] == annotation['frame'] and
                       observations[i]['source_id'] == annotation['source_id'] and
                       ('mask_index' not in annotation or observations[i]['mask_index'] == annotation['mask_index'])]
            if matched:
                entry.update(category=annotation['category'], category_status='visually_supported',
                             category_evidence=annotation)
    for relation in review.get('part_of', []):
        def anchored(source, mask_index):
            return [entry for entry in manifest if any(
                observations[i]['frame'] == relation['frame'] and observations[i]['source_id'] == source
                and (mask_index is None or observations[i]['mask_index'] == mask_index)
                for i in entry['observation_ids'])]
        parents = anchored(relation['parent_source'], relation.get('parent_mask_index'))
        children = anchored(relation['child_source'], relation.get('child_mask_index'))
        if len(parents) == 1 and len(children) == 1 and parents[0] != children[0]:
            children[0].update(parent_instance_id=parents[0]['instance_id'],
                               part_relationship_status='visually_supported', part_role=relation.get('role'),
                               part_evidence=relation)
        else:
            raise ValueError(f'part review is ambiguous or self-referential: {relation}')
    for unresolved in review.get('unresolved_masks', []):
        for entry in manifest:
            if any(all(observations[i].get(k) == unresolved[k] for k in ('frame', 'mask_index', 'source_id'))
                   for i in entry['observation_ids']):
                entry.update(identity_status='unresolved_within_mask_group', identity_uncertainty=unresolved['reason'])
    physical = physical_ids(manifest)
    physical_labels = physical[labels]
    opaque = {c['instance_id'] for c in constraints if c['evidence'].get('opaque_surface', False)}
    if any(physical[iid] != iid for iid in opaque):
        raise ValueError('opaque silhouette constraint must cover a complete physical object')
    negative = silhouette_conflicts(out, physical_labels, physical[obs_to_instance], observations, opaque)
    abstained = np.flatnonzero((negative >= 2) & (labels > 0) & ~removed)
    previous_tracks, previous_confidence = labels[abstained].copy(), confidence[abstained].copy()
    # Silhouette contradictions change label certainty, never cleanup eligibility or geometry.
    labels[abstained] = 0
    physical_labels[abstained] = 0
    confidence[abstained] = 0
    for entry in manifest:
        count = int(np.count_nonzero(previous_tracks == entry['instance_id']))
        if count:
            face_ids = np.flatnonzero((labels == entry['instance_id']) & ~removed)
            entry.update(geometry_triangles=len(face_ids),
                geometry_components=int(connected_components(graph[face_ids][:, face_ids], directed=False,
                                                              return_labels=False)) if len(face_ids) else 0,
                confidence=float(np.mean(confidence[face_ids])) if len(face_ids) else 0,
                silhouette_abstained_triangles=count,
                completeness_status='pending_reprojection' if len(face_ids) else 'missing_geometry')
    cleanup = json.loads((out / 'cleanup.json').read_text())
    cleanup.update(uncertain_preserved=int(np.count_nonzero((labels == 0) & ~removed)),
                   silhouette_label_abstentions=len(abstained))
    write_json(out / 'cleanup.json', cleanup)
    # Keep direct track/part labels as well as the enclosing physical identity.
    arrays.update(instance_ids=physical_labels, track_ids=labels,
                  observation_instances=physical[obs_to_instance], observation_tracks=obs_to_instance,
                  silhouette_conflict_groups=negative)
    np.savez_compressed(out / 'face_assignments.npz', **arrays)
    # Exact original face indices make every deletion and assignment reversible.
    np.savez_compressed(out / 'geometry_audit.npz', removed_face_ids=np.flatnonzero(removed),
        uncertain_face_ids=np.flatnonzero((labels == 0) | (confidence < .6)),
        assigned_face_ids=np.flatnonzero(labels > 0), assigned_instance_ids=physical_labels[labels > 0],
        assigned_track_ids=labels[labels > 0], silhouette_abstained_face_ids=abstained,
        silhouette_previous_track_ids=previous_tracks, silhouette_previous_confidence=previous_confidence)
    export_mesh(out / 'segmented_mesh.ply', vertices, triangles[~removed], physical_labels[~removed], labels[~removed])
    removed_vertices, inverse = np.unique(triangles[removed], return_inverse=True)
    export_mesh(out / 'removed_geometry.ply', vertices[removed_vertices], inverse.reshape(-1, 3),
                physical_labels[removed], labels[removed])
    (out / 'parts').mkdir(exist_ok=True)
    for entry in manifest:
        iid = entry['instance_id']
        pid = int(physical[iid])
        entry.update(physical_instance_id=pid, entry_kind='object' if pid == iid else 'part')
        face_ids = np.flatnonzero(((physical_labels == iid) if pid == iid else (labels == iid)) & ~removed)
        entry['physical_geometry_triangles'] = int(np.count_nonzero((physical_labels == pid) & ~removed))
        if len(face_ids):
            verts, inverse = np.unique(triangles[face_ids], return_inverse=True)
            directory, prefix = ('instances', 'instance') if pid == iid else ('parts', 'part')
            export_mesh(out / directory / f'{prefix}_{iid:04d}.ply', vertices[verts], inverse.reshape(-1, 3),
                        physical_labels[face_ids], labels[face_ids])
    write_json(out / 'instances_manifest.json', dict(instances=manifest,
        policy='mask tracks are hypotheses, not semantic classes; disconnected geometry retains identity',
        separation_constraints=len(cannot), identity_merges=sum(d['decision']=='merge' for d in decisions)))
    print(f'solve complete: {len(manifest)} hypotheses, {np.count_nonzero(labels):,} assigned faces; {removed.sum()} removed', flush=True)


def dense_completeness(camera, regions, original_scene, revised_scene, tracks, removed,
                       observation_tracks, cached, calibration, extents=None):
    """Every visible frozen-mask pixel contributes, including one-pixel detections."""
    y, x = np.where(regions >= 0)
    face, depth, _, _, _ = cast_pixels(original_scene, camera, y, x)
    valid = face >= 0
    predicted = np.zeros(len(face), np.int32)
    predicted[valid] = tracks[face[valid]]
    deleted = valid.copy()
    deleted[valid] = removed[face[valid]]
    if deleted.any():
        revised_face, revised_depth, _, _, _ = cast_pixels(revised_scene, camera, y[deleted], x[deleted])
        keep_tracks = tracks[~removed]
        p = np.zeros(len(revised_face), np.int32)
        p[revised_face >= 0] = keep_tracks[revised_face[revised_face >= 0]]
        predicted[deleted] = p
        depth[deleted] = revised_depth
        valid[deleted] = revised_face >= 0
    target = observation_tracks[regions[y, x]]
    missing_foreground, incompatible = np.zeros(len(face), bool), np.zeros(len(face), bool)
    state = np.full(len(face), 2, np.int8)
    if cached is not None and calibration and calibration['enabled']:
        reference = cached[y, x].astype('float32')
        usable = (reference > 0) & np.isfinite(reference) & valid
        missing_foreground = usable & (depth - reference > calibration['tolerance_m'])
        incompatible = usable & (np.abs(depth - reference) > calibration['tolerance_m'])
        state[usable] = 0
    possible_extent_conflict = valid & extent_conflicts(depth, state, regions[y, x], extents or {})
    masks = dict(pixels=np.ones(len(face), bool), no_hit=~valid, correct=valid & (predicted == target),
        wrong_label=valid & (predicted != target), missing_foreground_depth=missing_foreground,
        missing_geometry=~valid | missing_foreground,
        depth_conflict=incompatible, uncertain_depth_extent=possible_extent_conflict,
        geometrically_explained=valid & (predicted == target) & ~incompatible & ~possible_extent_conflict)
    result = defaultdict(dict)
    for name, mask in masks.items():
        for iid, value in enumerate(np.bincount(target[mask])):
            if value:
                result[iid][name] = int(value)
    return result


def evaluate(workspace, out, stride):
    """Independent pixel lattice, frozen cameras/masks; report source and hypothesis fidelity."""
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
    from panoptic3d_baseline import recover_baseline_labels

    stage = workspace / '06c_semantic_segmentation'
    inputs = stage / 'inputs'
    cameras = sorted(load_cameras(inputs), key=lambda c: c['image_path'].name)
    observations = json.loads((out / 'observations.json').read_text())
    manifest = json.loads((out / 'instances_manifest.json').read_text())
    metadata = verify_inputs(out)
    configuration = json.loads((out / 'configuration.json').read_text())
    for name, expected in configuration['source_files'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise RuntimeError('implementation changed after solve; rerun solve before evaluating')
    with np.load(out / 'face_assignments.npz') as z:
        labels, removed = z['instance_ids'], z['removed']
        obs_to_instance = z['observation_instances']
        tracks = z['track_ids'] if 'track_ids' in z.files else labels
        obs_to_track = z['observation_tracks'] if 'observation_tracks' in z.files else obs_to_instance
    sources = sorted({o['source_id'] for o in observations})
    mesh = o3d.io.read_triangle_mesh(str(inputs / 'mesh.ply'))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    transform = np.load(inputs / 'T_mesh_to_raw.npy')
    revised_scene = ray_scene(vertices, triangles[~removed], transform)
    original_scene = ray_scene(vertices, triangles, transform)
    kept_labels = labels[~removed]
    kept_tracks = tracks[~removed]
    source_for_instance = np.zeros(len(manifest['instances'])+1, np.int32)
    for entry in manifest['instances']:
        source_for_instance[entry['instance_id']] = entry['source_ids'][0]
    canonical = object_source_aliases(manifest['instances'], max(sources))
    baseline = o3d.io.read_triangle_mesh(str(stage / 'mesh_objects_step3/mesh_cleaned_no_floaters.ply'))
    baseline_vertices, baseline_triangles = np.asarray(baseline.vertices), np.asarray(baseline.triangles)
    baseline_scene = ray_scene(baseline_vertices, baseline_triangles, transform)
    recovered = recover_baseline_labels(stage / 'mesh_objects_step3/mesh_cleaned_no_floaters.ply',
        stage / 'mesh_objects_step3/objects_manifest.json', stage / 'mesh_objects_step3/extracted_instances', sources)
    baseline_sources = recovered['sources']
    write_json(out / 'baseline_recovery.json', recovered['provenance'])
    graph = dual_graph(baseline_triangles)
    a, b = graph.nonzero()
    keep = baseline_sources[a] == baseline_sources[b]
    same_class = sparse.csr_matrix((np.ones(keep.sum(), bool), (a[keep], b[keep])), shape=graph.shape)
    _, fallback_components = connected_components(same_class, directed=False)
    saved = recovered['instance_ids']
    baseline_components = np.where(saved > 0, saved, saved.max(initial=0) + 1 + fallback_components)
    # Match original triangles numerically; old cleanup retained their exact positions.
    tree = cKDTree(vertices[triangles].mean(axis=1))
    distance, original_ids = tree.query(baseline_vertices[baseline_triangles].mean(axis=1), workers=8)
    baseline_match = labels[original_ids]
    baseline_match[distance > 1e-5] = 0
    component_to_instance = np.zeros(baseline_components.max()+1, np.int32)
    pairs = sparse.coo_matrix((np.ones(len(baseline_match)), (baseline_components, baseline_match))).tocsr()
    # Unknown has no object identity and must not dominate an otherwise matched component.
    pairs.data[pairs.indices == 0] = 0
    component_to_instance[:] = np.asarray(pairs.argmax(axis=1)).ravel()
    baseline_physical = component_to_instance[baseline_components]
    baseline_original_sources = np.full(len(triangles), -1, np.int32)
    matched = distance <= 1e-5
    baseline_original_sources[original_ids[matched]] = baseline_sources[matched]
    revised_original_sources = source_for_instance[tracks]
    reassigned = (baseline_original_sources >= 0) & (baseline_original_sources != revised_original_sources) & ~removed
    np.savez_compressed(out / 'baseline_comparison.npz', baseline_source_ids=baseline_original_sources,
        revised_source_ids=revised_original_sources, reassigned_face_ids=np.flatnonzero(reassigned),
        recovered_geometry_face_ids=np.flatnonzero((baseline_original_sources < 0) & ~removed))
    split_counts = np.bincount(component_to_instance, minlength=len(source_for_instance))
    split_counts[0] = 0
    review = json.loads((out / 'review_set.json').read_text())['frames']
    (out / 'review').mkdir(exist_ok=True)
    (out / 'diagnostics').mkdir(exist_ok=True)
    totals = {scope: {version: defaultdict(lambda: defaultdict(int))
                     for version in ('baseline', 'revised')}
              for scope in ('all_source', 'review_source', 'all_instance', 'review_instance',
                            'all_literal_source', 'review_literal_source',
                            'trusted_source', 'trusted_literal_source', 'trusted_instance')}
    frame_results, completeness = [], defaultdict(lambda: defaultdict(int))
    by_frame = defaultdict(list)
    for observation in observations:
        by_frame[observation['frame']].append(observation)
    frame_info = {f['stem']: f for f in json.loads((out / 'frames.json').read_text())}
    depth_root, unreliable, _ = load_depth_context(workspace, cameras)
    new_holes, recovered_rays = 0, 0
    diagnostic_holes, diagnostic_recovered = 0, 0
    floater_pixels = 0
    start = time.perf_counter()
    for index, camera in enumerate(cameras):
        stem = camera['image_path'].stem
        trusted_pose = frame_info[stem].get('pose', {}).get('trusted', True)
        rf, rd, incidence, y, x = cast(revised_scene, camera, stride, offset=1)
        bf, _, _, _, _ = cast(baseline_scene, camera, stride, offset=1)
        of, od, original_incidence, _, _ = cast(original_scene, camera, stride, offset=1)
        revised_visible, baseline_visible = rf >= 0, bf >= 0
        holes = int(np.count_nonzero((of >= 0) & ~revised_visible))
        recovered = int(np.count_nonzero(revised_visible & ~baseline_visible))
        diagnostic_holes += holes
        diagnostic_recovered += recovered
        if trusted_pose:
            new_holes += holes
            recovered_rays += recovered
        rp, bp = np.zeros(rf.shape, np.int32), np.zeros(bf.shape, np.int32)
        ri, bi = np.zeros(rf.shape, np.int32), np.zeros(bf.shape, np.int32)
        ri[revised_visible] = kept_labels[rf[revised_visible]]
        rp[revised_visible] = source_for_instance[kept_tracks[rf[revised_visible]]]
        bp[baseline_visible] = baseline_sources[bf[baseline_visible]]
        bi[baseline_visible] = baseline_physical[bf[baseline_visible]]
        with np.load(stage / 'full362_final_pipeline/npz_data' / f'{stem}_final.npz') as z:
            truth = z['labels'][y, x]
            ti = np.zeros(truth.shape, np.int32)
            regions_path = out / 'evidence' / f'{stem}_regions.npz'
            full_regions = np.load(regions_path)['observation'] if regions_path.exists() else None
            for observation in by_frame[stem]:
                oid = observation['observation_id']
                original = z['masks'][observation['mask_index']]
                component = observation.get('component_index', 0)
                if full_regions is not None:
                    mask = full_regions[y, x] == oid
                elif component:
                    active = (original > 0) & (z['labels'] == observation['source_id'])
                    _, regions = cv2.connectedComponents(active.astype('uint8'), connectivity=8)
                    mask = regions[y, x] == component
                else:
                    mask = original[y, x]
                hit = (mask > 0) & (truth == observation['source_id']) & (ti == 0)
                ti[hit] = obs_to_instance[oid]
        missing_foreground = np.zeros(truth.shape, bool)
        incompatible_depth = np.zeros(truth.shape, bool)
        calibration = frame_info[stem]['depth']
        if depth_root and stem not in unreliable and calibration and calibration['enabled']:
            cache = np.load(depth_root / 'depth_maps' / f'{stem}.npy', mmap_mode='r')[y, x].astype('float32')
            usable = (cache > 0) & np.isfinite(cache) & (of >= 0)
            missing_foreground = usable & (od - cache > calibration['tolerance_m'])
            incompatible_depth = usable & (np.abs(od - cache) > calibration['tolerance_m'])
        removed_pixels = np.zeros(of.shape, bool)
        removed_pixels[of >= 0] = removed[of[of >= 0]]
        if trusted_pose and removed_pixels.sum() > floater_pixels:
            floater_pixels = int(removed_pixels.sum())
            yy, xx = np.where(removed_pixels)
            crop = (slice(max(0, yy.min()-15), min(len(y), yy.max()+16)),
                    slice(max(0, xx.min()-15), min(len(x[0]), xx.max()+16)))
            original_ids = np.zeros(of.shape, np.int32)
            original_ids[of >= 0] = labels[of[of >= 0]]
            before = (palette(original_ids)[..., ::-1] * (.25+.75*original_incidence[..., None])).astype('uint8')
            after = (palette(ri)[..., ::-1] * (.25+.75*incidence[..., None])).astype('uint8')
            rgb = cv2.imread(str(camera['image_path']))[y, x]
            residual = np.full((*of.shape, 3), 128, np.uint8)
            residual[incompatible_depth & ~missing_foreground] = [0, 0, 255]
            residual[missing_foreground] = [255, 0, 0]
            residual[removed_pixels] = [0, 255, 255]
            cv2.imwrite(str(out / 'floater_review.jpg'), np.hstack([a[crop] for a in (rgb, before, after, residual)]))
            write_json(out / 'floater_review.json', dict(frame=stem, removed_hit_pixels=floater_pixels,
                panels=['RGB', 'original shaded geometry', 'revised shaded geometry',
                        'depth: free-space red, missing-foreground blue, removed yellow, unknown gray'],
                sampling_stride=stride, pixel_offset=1, depth_calibration=calibration))
        literal_rp = rp.copy()
        # Apply exactly the same explicit alias map to the reference and both candidates.
        canonical_truth, canonical_bp, canonical_rp = canonical[ti, truth], canonical[bi, bp], canonical[ri, rp]
        per_frame = dict(stem=stem, review=stem in review, pose=frame_info[stem].get('pose'),
                         baseline_missing_rays=int(np.count_nonzero((truth > 0) & ~baseline_visible)),
                         revised_missing_rays=int(np.count_nonzero((truth > 0) & ~revised_visible)))
        for kind, target, predictions, validities in (
                ('source', canonical_truth, (canonical_bp, canonical_rp), (baseline_visible, revised_visible)),
                ('literal_source', truth, (bp, literal_rp), (baseline_visible, revised_visible)),
                ('instance', ti, (bi, ri), (baseline_visible, revised_visible))):
            ids = [int(i) for i in np.unique(np.r_[target.ravel(), predictions[0].ravel(), predictions[1].ravel()]) if i > 0]
            for version, prediction, visible in zip(('baseline', 'revised'), predictions, validities):
                counts = object_metrics(target, prediction, visible, ids)
                scopes = [f'all_{kind}']
                if stem in review:
                    scopes.append(f'review_{kind}')
                if frame_info[stem].get('pose', {}).get('trusted', True):
                    scopes.append(f'trusted_{kind}')
                for scope in scopes:
                    for oid, record in counts.items():
                        for key, value in record.items():
                            totals[scope][version][oid][key] += value
                per_frame[f'{version}_{kind}'] = summarize_metrics(counts)
        if full_regions is None:
            raise ValueError('dense completeness requires current evidence collection')
        cached = None
        if depth_root and stem not in unreliable and calibration and calibration['enabled']:
            cached = np.load(depth_root / 'depth_maps' / f'{stem}.npy', mmap_mode='r')
        extents = {o['observation_id']: o['depth_extent'] for o in observations
                   if o['frame'] == stem and 'depth_extent' in o}
        counts = dense_completeness(camera, full_regions, original_scene, revised_scene, tracks,
                                    removed, obs_to_track, cached, calibration, extents) if trusted_pose else {}
        for iid, record in counts.items():
            for key, value in record.items():
                completeness[iid][key] += value
        if stem in review or stem in DIAGNOSTIC:
            rgb = cv2.imread(str(camera['image_path']))[y, x]
            tiles = [rgb, palette(canonical_truth)[..., ::-1], palette(canonical_bp)[..., ::-1], palette(canonical_rp)[..., ::-1]]
            disagreement = np.zeros((*truth.shape, 3), np.uint8)
            disagreement[(truth > 0) & (canonical_rp != canonical_truth)] = [0, 0, 255]
            disagreement[(truth > 0) & ~revised_visible] = [255, 0, 255]
            disagreement[(truth > 0) & (canonical_rp == canonical_truth)] = [0, 100, 0]
            tiles += [disagreement, palette(ri)[..., ::-1]]
            shaded = (palette(ri)[..., ::-1].astype('float32') * (.25 + .75 * incidence[..., None])).astype('uint8')
            shaded[~revised_visible] = 0
            tiles.append(shaded)
            tiles = [np.ascontiguousarray(tile) for tile in tiles]
            titles = ['RGB', '2D canonical IDs', 'baseline 3D', 'revised 3D', 'disagreement', '3D instances', 'shaded mesh']
            for tile, title in zip(tiles, titles):
                cv2.putText(tile, title, (8, 22), 0, .55, (255, 255, 255), 1, cv2.LINE_AA)
            directory = out / ('review' if stem in review else 'diagnostics')
            cv2.imwrite(str(directory / f'{stem}.jpg'), np.hstack(tiles))
            np.savez_compressed(directory / f'{stem}.npz', frozen_2d=truth, canonical_2d=canonical_truth,
                baseline_source=canonical_bp, revised_source=canonical_rp,
                literal_baseline_source=bp, literal_revised_source=literal_rp, baseline_instance=bi,
                revised_instance=ri, baseline_visible=baseline_visible, revised_visible=revised_visible)
        frame_results.append(per_frame)
        if (index+1) % 60 == 0:
            print(f'evaluate {index+1}/{len(cameras)} ({time.perf_counter()-start:.1f}s)', flush=True)
    metrics = {scope: {version: summarize_metrics(counts) for version, counts in versions.items()}
               for scope, versions in totals.items()}
    aggregates = {}
    for scope, versions in metrics.items():
        aggregates[scope] = {}
        for version, counts in versions.items():
            values = [v for v in counts.values() if v['pixels'] > 0]
            pixel_sizes = [v['pixels'] for v in values]
            cutoff = float(np.percentile(pixel_sizes, 25)) if pixel_sizes else 0
            small = [v for v in values if v['pixels'] <= cutoff]
            aggregates[scope][version] = dict(objects=len(values),
                false_positive_pixels_without_reference_object=sum(v['fp'] for v in counts.values() if v['pixels']==0),
                **{f'macro_{key}': float(np.mean([v[key] for v in values])) if values else 0
                   for key in ('recall', 'iou', 'leakage', 'boundary_f1', 'missing_geometry_fraction')},
                small_object_cutoff_pixels=cutoff,
                **{f'small_macro_{key}': float(np.mean([v[key] for v in small])) if small else 0
                   for key in ('recall', 'iou', 'boundary_f1')})
    for entry in manifest['instances']:
        c = dict(completeness[entry['instance_id']])
        c.setdefault('pixels', 0)
        entry['reprojection'] = c
        explained = c.get('correct', 0) / max(1, c['pixels'])
        entry['explained_fraction'] = explained
        entry['geometrically_explained_fraction'] = c.get('geometrically_explained', 0) / max(1, c['pixels'])
        entry['completeness_status'] = ('unreliable_pose' if entry['pose_reliability'] == 'weak_pose_only' else
            'no_unambiguous_label_pixels' if c['pixels']==0 else
            'missing_geometry' if c.get('missing_geometry', 0) == c['pixels'] else
            'partial_geometry' if c.get('missing_geometry', 0) else
            'uncertain_geometry' if c.get('uncertain_depth_extent', 0) or c.get('depth_conflict', 0) else
            'label_conflict' if explained < .9 else 'mostly_explained')
        entry['baseline_identity_components'] = int(split_counts[entry['instance_id']])
    unchanged = {path: digest(Path(path)) == expected for path, expected in metadata['files'].items()}
    if not all(unchanged.values()):
        raise RuntimeError('a frozen input or baseline changed during reconstruction')
    report = dict(aggregates=aggregates, per_object=metrics,
        new_missing_rays_vs_original=new_holes, recovered_rays_vs_baseline=recovered_rays,
        geometry_evaluation_scope='trusted poses only; all_* scores retain weak frames for diagnostics',
        diagnostic_all_pose_new_missing_rays=diagnostic_holes,
        diagnostic_all_pose_recovered_rays=diagnostic_recovered,
        frozen_input_hashes_unchanged=True,
        baseline_identity_splits=int(np.maximum(split_counts[1:]-1, 0).sum()),
        identity_metric_caveat='hypothesis-relative: saved baseline instances and unresolved same-label components matched by retained geometry; no independent physical-object ground truth',
        source_alias_map={e['instance_id']: e['source_ids'] for e in manifest['instances'] if len(e['source_ids']) > 1},
        source_metric_caveat='frozen 2D is a consistency reference, not human semantic ground truth; aliases are scoped to the established physical object for reference and both candidates; literal scores are also reported',
        runtime_evaluation_seconds=time.perf_counter()-start)
    write_json(out / 'instances_manifest.json', manifest)
    write_json(out / 'metrics.json', report)
    write_json(out / 'frame_metrics.json', frame_results)
    lines = ['# Frozen-mask 3D reconstruction comparison', '',
             'Reference: unchanged 2D masks and original cameras, with an independent pixel lattice.',
             'All metrics are consistency measures; physical identity and semantics require visual evidence.', '',
             '| Fixed review set | Baseline | Revised |', '|---|---:|---:|']
    for key in ('macro_recall', 'macro_iou', 'macro_boundary_f1', 'small_macro_recall', 'small_macro_iou', 'small_macro_boundary_f1'):
        values = aggregates['review_source']
        lines.append(f"| {key} | {values['baseline'][key]:.4f} | {values['revised'][key]:.4f} |")
    cleanup = json.loads((out / 'cleanup.json').read_text())
    lines += ['', f"Removed {cleanup['removed_triangles']} triangles in {cleanup['removed_components']} confirmed components.",
              f"New mesh boundary edges: {cleanup['new_boundary_edges']}; new missing rays versus original: {new_holes}.",
              f'Recovered visible rays versus old cleanup: {recovered_rays}.',
              f"Frozen artifacts verified unchanged: {len(unchanged)} SHA-256 checks.", '',
              'See metrics.json for all-camera and per-object results, identity caveats, and protected-object scores.',
              'See review/*.jpg and review/*.npz for identical-camera masks, mesh views, and disagreements.',
              'Unresolved: semantic categories/part relations without established evidence, within-mask conflation,',
              'single-view identities, missing geometry, and unsupported geometry without calibrated free-space confirmation.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregates['review_source'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=BACKEND / 'current_scene')
    parser.add_argument('--run-name', default='panoptic3d_v1')
    parser.add_argument('--phase', choices=['all', 'evidence', 'solve', 'evaluate'], default='all')
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--smoothness', type=float, default=.15)
    parser.add_argument('--area-balance', type=float, default=0, help='measured uncontested-area damping exponent; calibrate on fixed review')
    parser.add_argument('--review-file', type=Path, help='optional scene-bound visual category/relationship evidence')
    args = parser.parse_args()
    if args.stride < 2 or not np.isfinite(args.smoothness) or args.smoothness < 0 or not 0 <= args.area_balance <= 1:
        parser.error('stride must be >=2, smoothness nonnegative, and area balance between 0 and 1')
    workspace = args.workspace.resolve()
    stage = workspace / '06c_semantic_segmentation'
    out = output_directory(stage, args.run_name)
    if args.phase in ('all', 'evidence'):
        out.mkdir(exist_ok=False)
        collect(workspace, out, args.stride, args.review_file)
    else:
        metadata = json.loads((out / 'inputs.json').read_text())
        if metadata['workspace'] != str(workspace):
            raise ValueError('evidence belongs to a different workspace')
        args.stride = metadata['stride']
    if args.phase in ('all', 'solve'):
        solve(workspace, out, args.smoothness, args.review_file, args.area_balance)
    if args.phase in ('all', 'evaluate'):
        evaluate(workspace, out, args.stride)


if __name__ == '__main__':
    main()
