#!/usr/bin/env python3
"""Render and select orthographic cutaway views independently for each scan."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFilter

from generate_floor_level_meshes import DEFAULT_ALIGNMENT_DIR, _plane, _plane_group, _read_mesh
from panorama_rendering import panel_faces

MACRO = 'Generate Possible Views'
FLOOR_ALLOWANCE = 0.05
CUTAWAY_RADIUS = 0.15
PANEL_CONTEXT_RADIUS = 0.30
MIN_PIXELS = 2
# shortcut: cutaway seeds are sampled at <=3072px; use render resolution if thin surfaces need denser first-hit sampling.
CUTAWAY_MAX_RESOLUTION = 3072


def targets(metadata):
    groups = {}
    for floor in metadata['floor_planes']:
        for key in ('scan_id', 'floor_level_name'):
            name = floor[key]
            if not isinstance(name, str) or not name or name in ('.', '..') or '/' in name or chr(92) in name:
                raise ValueError(f'Invalid {key}: {name!r}')
        groups.setdefault((floor['scan_id'], floor['floor_level_name']), []).append(floor)
    result = []
    for (scan, level), floors in sorted(groups.items()):
        names = {f['scan_level_name'] for f in floors}
        heights = {f['height_group_index'] for f in floors}
        if len(names) != 1 or len(heights) != 1:
            raise ValueError(f'Inconsistent floor metadata for {scan}/{level}')
        ceilings = [c for c in metadata['ceiling_planes'] if c['scan_id'] == scan
                    and c.get('floor_level_name') == level
                    and c['mean_elevation_m'] > max(f['mean_elevation_m'] for f in floors)]
        result.append(dict(scan_id=scan, floor_level_name=level, floor_plane_ids=sorted(f['plane_id'] for f in floors),
                           scan_level_name=names.pop(), height_group_index=heights.pop(), floors=floors,
                           ceiling=min(ceilings, key=lambda c: c['mean_elevation_m']) if ceilings else None))
    if not result:
        raise ValueError('No actual floor slices in metadata')
    return result


def clip_band(vertices, faces, colors, lower, upper):
    """Clip triangles to two half spaces; output face IDs refer to the input mesh."""
    verts, cols, fs, ids = vertices, colors, faces, np.arange(len(faces), dtype=np.int64)
    for normal, offset in (lower, upper):
        distance = verts @ normal + offset
        inside = distance[fs] >= -1e-7
        whole = inside.all(axis=1)
        crossing = np.flatnonzero(inside.any(axis=1) & ~whole)
        new_v, new_c, new_f, new_ids = [], [], [], []
        for index in crossing:
            face = fs[index]
            polygon = [(verts[i], cols[i].astype(np.float64), float(distance[i])) for i in face]
            out = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if (a[2] >= 0) != (b[2] >= 0):
                    t = a[2] / (a[2] - b[2])
                    out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0.0))
                if b[2] >= 0:
                    out.append(b)
            base = len(verts) + len(new_v)
            new_v.extend(p[0] for p in out)
            new_c.extend(p[1] for p in out)
            for j in range(1, len(out) - 1):
                new_f.append((base, base + j, base + j + 1))
                new_ids.append(ids[index])
        fs = np.concatenate((fs[whole], np.asarray(new_f, dtype=np.int32).reshape(-1, 3)))
        ids = np.concatenate((ids[whole], np.asarray(new_ids, dtype=np.int64)))
        if new_v:
            verts = np.concatenate((verts, np.asarray(new_v, np.float32)))
            cols = np.concatenate((cols, np.clip(np.rint(new_c), 0, 255).astype(np.uint8)))
    used, inverse = np.unique(fs.reshape(-1), return_inverse=True)
    return verts[used], inverse.reshape(-1, 3).astype(np.int32), cols[used], ids


def panel_context_neighbors(vertices, faces, panel_ids, radius=PANEL_CONTEXT_RADIUS):
    """Find nearby non-panel faces once; cutaway results classify each panel per view."""
    from scipy.spatial import cKDTree

    ordinary = np.flatnonzero(panel_ids < 0)
    if not len(ordinary) or not np.any(panel_ids >= 0):
        return {}
    centers = vertices[faces].mean(axis=1)
    tree = cKDTree(centers[ordinary])
    result = {}
    for panel_id in np.unique(panel_ids[panel_ids >= 0]):
        panel_faces = np.flatnonzero(panel_ids == panel_id)
        nearby = tree.query_ball_point(centers[panel_faces], radius)
        result[int(panel_id)] = np.unique(ordinary[np.concatenate(nearby)]) if any(len(x) for x in nearby) else np.empty(0, np.int64)
    return result


def apply_panel_context(removed_ids, panel_ids, neighbors):
    """Remove a panel only when most nearby non-panel triangles were cut away."""
    result = np.zeros(len(panel_ids), bool)
    result[removed_ids] = True
    for panel_id, surrounding in neighbors.items():
        faces = np.flatnonzero(panel_ids == panel_id)
        result[faces] = (np.count_nonzero(result[surrounding]) * 2 > len(surrounding)) if len(surrounding) else False
    return np.flatnonzero(result)


def floor_faces(vertices, faces, planes, distance, alignment, whole_triangle=False):
    """Find horizontal faces supported by one of the same-scan floor planes."""
    result = np.zeros(len(faces), bool)
    for start in range(0, len(faces), 250_000):
        stop = min(start + 250_000, len(faces))
        tri = vertices[faces[start:stop]]
        cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        length = np.linalg.norm(cross, axis=1)
        for record in planes:
            normal, offset = _plane(record)
            residual = np.abs(tri @ normal + offset)
            residual = residual.max(axis=1) if whole_triangle else residual.mean(axis=1)
            result[start:stop] |= (length > 1e-12) & (np.abs(cross @ normal) >= alignment * length) & (residual <= distance)
    return result


def cameras(slice_bounds, scene_bounds):
    lo, hi = np.asarray(slice_bounds)
    center = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in scene_bounds[:, 0] for y in scene_bounds[:, 1] for z in scene_bounds[:, 2]])
    result = []
    for dx, dz in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        horizontal = np.array([dx * max(hi[0]-lo[0], .01), 0, dz * max(hi[2]-lo[2], .01)], float)
        horizontal /= np.linalg.norm(horizontal)
        direction = horizontal * np.cos(np.deg2rad(35)) + np.array([0, np.sin(np.deg2rad(35)), 0])
        right = np.cross(np.array([0., 1., 0.]), direction); right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        projected = np.column_stack(((corners-center) @ right, (corners-center) @ up))
        extent = float(2 * np.max(np.abs(projected)) * 1.06)
        distance = float(np.linalg.norm(scene_bounds[1]-scene_bounds[0]) * 2 + 1)
        result.append(dict(eye=(center + direction * distance).tolist(), target=center.tolist(),
                           projection='orthographic', extent=extent, elevation_degrees=35,
                           right=right.tolist(), up=up.tolist()))
    return result


def ray_batches(camera, resolution):
    eye = np.asarray(camera['eye']); target = np.asarray(camera['target'])
    right, up = np.asarray(camera['right']), np.asarray(camera['up'])
    xy = ((np.arange(resolution) + .5) / resolution - .5) * camera['extent']
    direction = (target-eye) / np.linalg.norm(target-eye)
    for start in range(0, resolution, 128):
        stop = min(start + 128, resolution)
        origins = eye + xy[None, :, None] * right - xy[start:stop, None, None] * up
        rays = np.empty((stop-start, resolution, 6), np.float32)
        rays[:, :, :3] = origins; rays[:, :, 3:] = direction
        yield start, stop, direction, rays


def cutaway_faces(scene, vertices, faces, camera, resolution, radius=CUTAWAY_RADIUS, max_edges=None, protected=None):
    """Remove first-hit backfaces and faces touching their radius neighborhood."""
    seeds = np.zeros(len(faces), bool)
    for _, _, direction, rays in ray_batches(camera, resolution):
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        ids = hit['primitive_ids'].numpy()
        normals = hit['primitive_normals'].numpy()
        back = (ids < len(faces)) & ((normals @ direction) > 1e-6)
        seeds[ids[back].astype(np.int64)] = True
    if protected is not None:
        seeds[protected] = False
    seed_ids = np.flatnonzero(seeds)
    if not len(seed_ids):
        return seed_ids, seed_ids
    seed_scene = o3d.t.geometry.RaycastingScene()
    seed_scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32, copy=False)),
                             o3d.core.Tensor(faces[seed_ids].astype(np.uint32)))
    distances = np.full(len(vertices), np.inf, np.float32)
    if max_edges is None:
        tri = vertices[faces]
        max_edges = np.sqrt(np.maximum.reduce([np.sum((tri[:,i]-tri[:,(i+1)%3])**2, axis=1) for i in range(3)]))
    seed_vertices = vertices[np.unique(faces[seed_ids])]
    margin = radius + float(max_edges.max())
    low = seed_vertices.min(axis=0) - margin
    high = seed_vertices.max(axis=0) + margin
    candidates = np.flatnonzero(np.all((vertices >= low) & (vertices <= high), axis=1))
    for start in range(0, len(candidates), 100_000):
        ids = candidates[start:start + 100_000]
        distances[ids] = seed_scene.compute_distance(o3d.core.Tensor(vertices[ids].astype(np.float32))).numpy()
    face_distances = distances[faces]
    nearest = face_distances.min(axis=1)
    removed = (nearest <= radius + 1e-6) | seeds
    if protected is not None:
        removed[protected] = False
    uncertain = ~removed & (nearest <= radius + max_edges)
    if protected is not None:
        uncertain[protected] = False
    tolerance = 0.005
    removed[uncertain & (max_edges <= tolerance)] = True
    refine = np.flatnonzero(uncertain & (max_edges > tolerance))
    # Subdivision bounds the extra removed band by 5 mm, even for large triangles.
    for start in range(0, len(refine), 10_000):
        face_ids = refine[start:start + 10_000]
        triangles = vertices[faces[face_ids]]
        d = face_distances[face_ids]
        edges = max_edges[face_ids]
        while len(face_ids):
            mid = (triangles + np.roll(triangles, -1, axis=1)) * 0.5
            mid_d = seed_scene.compute_distance(o3d.core.Tensor(mid.reshape(-1, 3).astype(np.float32))).numpy().reshape(-1, 3)
            hit = (mid_d <= radius + 1e-6).any(axis=1)
            removed[face_ids[hit]] = True
            keep = ~hit & ~removed[face_ids]
            if not np.any(keep):
                break
            triangles, d, mid, mid_d, edges, face_ids = (x[keep] for x in (triangles, d, mid, mid_d, edges, face_ids))
            a, b, c = triangles.transpose(1, 0, 2)
            ab, bc, ca = mid.transpose(1, 0, 2)
            da, db, dc = d.T; dab, dbc, dca = mid_d.T
            triangles = np.stack((np.stack((a,ab,ca),axis=1), np.stack((ab,b,bc),axis=1),
                                  np.stack((ca,bc,c),axis=1), np.stack((ab,bc,ca),axis=1)),axis=1).reshape(-1,3,3)
            d = np.stack((np.stack((da,dab,dca),axis=1), np.stack((dab,db,dbc),axis=1),
                          np.stack((dca,dbc,dc),axis=1), np.stack((dab,dbc,dca),axis=1)),axis=1).reshape(-1,3)
            edges = np.repeat(edges * .5, 4); face_ids = np.repeat(face_ids, 4)
            possible = (d.min(axis=1) <= radius + edges) & ~removed[face_ids]
            removed[face_ids[possible & (edges <= tolerance)]] = True
            possible &= (edges > tolerance) & ~removed[face_ids]
            triangles, d, edges, face_ids = (x[possible] for x in (triangles, d, edges, face_ids))
    return seed_ids, np.flatnonzero(removed)


def render(scene, vertices, faces, colors, camera, resolution, floor_fallback=None, optical_types=None):
    image = np.full((resolution, resolution, 3), 255, np.uint8)
    ids = np.full((resolution, resolution), -1, np.int32)
    optical_scene = None
    if optical_types is not None and np.any(optical_types):
        optical_faces = np.flatnonzero(optical_types == 0)
        if len(optical_faces):
            optical_scene = o3d.t.geometry.RaycastingScene()
            optical_scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32, copy=False)),
                                        o3d.core.Tensor(faces[optical_faces].astype(np.uint32)))
    for start, stop, _, rays in ray_batches(camera, resolution):
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        primitive = hit['primitive_ids'].numpy()
        valid = primitive < len(faces)
        face_ids = primitive[valid].astype(np.int32)
        ids[start:stop][valid] = face_ids
        uv = hit['primitive_uvs'].numpy()[valid]
        weights = np.column_stack((1-uv.sum(axis=1), uv))
        image[start:stop][valid] = np.clip(np.rint(np.einsum('ij,ijk->ik', weights, colors[faces[face_ids], :3])), 0, 255).astype(np.uint8)
        if floor_fallback is not None:
            fallback_scene, original_faces, source_index, owners, protected = floor_fallback
            pixels = np.flatnonzero(valid.reshape(-1))
            pixels = pixels[source_index[face_ids] != 0]
            if len(pixels):
                other = fallback_scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)[pixels]))
                other_ids = other['primitive_ids'].numpy()
                possible = other_ids < len(original_faces)
                local = original_faces[other_ids[possible].astype(np.int64)]
                selected = pixels[possible]
                delta = other['t_hit'].numpy()[possible] - hit['t_hit'].numpy().reshape(-1)[selected]
                # Prefer the scanned floor when it lies close behind its fitted support.
                keep = protected[local] & (owners[local] == owners[primitive.reshape(-1)[selected]]) & (delta <= .20)
                selected, local = selected[keep], local[keep]
                uv = other['primitive_uvs'].numpy()[possible][keep]
                weights = np.column_stack((1-uv.sum(axis=1), uv))
                image[start:stop].reshape(-1, 3)[selected] = np.clip(np.rint(np.einsum('ij,ijk->ik', weights, colors[faces[local], :3])), 0, 255).astype(np.uint8)
                ids[start:stop].reshape(-1)[selected] = local
        if optical_types is not None:
            primary = ids[start:stop].reshape(-1)
            pixels = np.flatnonzero(primary >= 0)
            pixels = pixels[optical_types[primary[pixels]] != 0]
            if len(pixels):
                # shortcut: one optical bounce and a white environment; extend when captured exterior or lighting exists.
                flat_image = image[start:stop].reshape(-1, 3)
                flat_image[pixels] = 255
                if optical_scene is not None:
                    source_rays = rays.reshape(-1, 6)[pixels]
                    distance = hit['t_hit'].numpy().reshape(-1)[pixels]
                    normals = hit['primitive_normals'].numpy().reshape(-1, 3)[pixels]
                    direction = source_rays[:, 3:].copy()
                    mirror = optical_types[primary[pixels]] == 1
                    direction[mirror] -= 2 * np.sum(direction[mirror] * normals[mirror], axis=1)[:, None] * normals[mirror]
                    secondary = np.column_stack((source_rays[:, :3] + distance[:, None] * source_rays[:, 3:] + direction * .001, direction)).astype(np.float32)
                    reflected = optical_scene.cast_rays(o3d.core.Tensor(secondary))
                    primitives = reflected['primitive_ids'].numpy()
                    valid_optical = primitives < len(optical_faces)
                    local = optical_faces[primitives[valid_optical].astype(np.int64)]
                    uv = reflected['primitive_uvs'].numpy()[valid_optical]
                    weights = np.column_stack((1-uv.sum(axis=1), uv))
                    flat_image[pixels[valid_optical]] = np.clip(np.rint(np.einsum('ij,ijk->ik', weights, colors[faces[local], :3])), 0, 255).astype(np.uint8)
                windows = pixels[optical_types[primary[pixels]] == 2]
                transmitted = flat_image[windows].astype(np.float32)
                reflected_color = np.full_like(transmitted, 255)
                if optical_scene is not None and len(windows):
                    window_source = rays.reshape(-1, 6)[windows]
                    window_distance = hit['t_hit'].numpy().reshape(-1)[windows]
                    window_normals = hit['primitive_normals'].numpy().reshape(-1, 3)[windows]
                    reflected_direction = window_source[:, 3:] - 2 * np.sum(window_source[:, 3:] * window_normals, axis=1)[:, None] * window_normals
                    reflection_rays = np.column_stack((window_source[:, :3] + window_distance[:, None] * window_source[:, 3:] + reflected_direction * .001, reflected_direction)).astype(np.float32)
                    reflection = optical_scene.cast_rays(o3d.core.Tensor(reflection_rays))
                    primitives = reflection['primitive_ids'].numpy()
                    valid_reflections = primitives < len(optical_faces)
                    local = optical_faces[primitives[valid_reflections].astype(np.int64)]
                    uv = reflection['primitive_uvs'].numpy()[valid_reflections]
                    weights = np.column_stack((1-uv.sum(axis=1), uv))
                    reflected_color[valid_reflections] = np.einsum('ij,ijk->ik', weights, colors[faces[local], :3])
                flat_image[windows] = np.clip(np.rint(transmitted * .75 + reflected_color * .10 + np.array([150, 160, 170]) * .15), 0, 255).astype(np.uint8)
    if optical_types is not None and np.any(optical_types == 2):
        mask = ids >= 0
        mask[mask] = optical_types[ids[mask]] == 2
        y, x = np.nonzero(mask)
        if len(x):
            y0, y1 = max(y.min()-4, 0), min(y.max()+5, resolution)
            x0, x1 = max(x.min()-4, 0), min(x.max()+5, resolution)
            window = mask[y0:y1, x0:x1]
            inner = Image.fromarray(window.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))
            deeper = inner.filter(ImageFilter.MinFilter(7))
            inner, deeper = np.asarray(inner) > 0, np.asarray(deeper) > 0
            crop = image[y0:y1, x0:x1]
            crop[window & ~inner] = [55, 65, 75]
            crop[inner & ~deeper] = [215, 225, 230]
    return image, ids


def select(visibility, areas, owners, count, threshold, min_gain):
    covered = np.zeros(len(areas), bool)
    totals = np.bincount(owners, weights=areas, minlength=count)
    attainable_mask = np.zeros(len(areas), bool)
    for visible in visibility:
        attainable_mask[visible] = True
    attainable = np.bincount(owners[attainable_mask], weights=areas[attainable_mask], minlength=count)
    chosen = []
    initial_gain = None
    while True:
        current = np.bincount(owners[covered], weights=areas[covered], minlength=count)
        ratios = np.divide(current, totals, out=np.zeros(count), where=totals > 0)
        if np.all(ratios >= threshold):
            reason = 'coverage_target_reached'; break
        needs = ratios < threshold
        gains = [float(areas[ids[~covered[ids] & needs[owners[ids]]]].sum())
                 if i not in chosen else -1. for i, ids in enumerate(visibility)]
        best = int(np.argmax(gains))
        if initial_gain is None:
            initial_gain = gains[best]
        if gains[best] <= min_gain * initial_gain:
            reason = 'insufficient_marginal_coverage'; break
        chosen.append(best)
        covered[visibility[best]] = True
    return chosen, ratios, np.divide(attainable, totals, out=np.zeros(count), where=totals > 0), reason


def sheet(paths, output):
    thumb = 256; rows = (len(paths) + 3) // 4
    canvas = Image.new('RGB', (thumb * 4, max(rows, 1) * (thumb + 25)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, path in enumerate(paths):
        im = Image.open(path); im.thumbnail((thumb, thumb))
        x, y = (i % 4) * thumb, (i // 4) * (thumb + 25)
        canvas.paste(im, (x, y)); draw.text((x+3, y+thumb+3), path.stem, fill='black')
    canvas.save(output)


def run(alignment_dir, metadata_path, output_dir, resolution=4096, threshold=.7, min_gain=.1):
    start = time.perf_counter(); alignment_dir = Path(alignment_dir).resolve(); output_dir = Path(output_dir).resolve()
    metadata_path = Path(metadata_path).resolve(); scene_dir = alignment_dir.parent
    if output_dir == scene_dir or scene_dir.is_relative_to(output_dir) or output_dir.is_relative_to(alignment_dir) or any(output_dir.is_relative_to(p) for p in scene_dir.glob('scan_*')) or metadata_path.is_relative_to(output_dir):
        raise ValueError('Output must be separate from inputs')
    if resolution < 64 or not 0 < threshold <= 1 or not 0 <= min_gain < 1:
        raise ValueError('Invalid resolution or coverage settings')
    metadata = json.loads(metadata_path.read_text()); entries = targets(metadata)
    panel_manifest = alignment_dir / 'stats' / 'phase07_region_replacements.json'
    regions = json.loads(panel_manifest.read_text()).get('regions', []) if panel_manifest.is_file() else []
    if output_dir.exists() and not (output_dir / 'generate_possible_views_manifest.json').is_file():
        raise FileExistsError(f'Refusing to replace an unowned output directory: {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.possible_views_', dir=output_dir.parent) as tmp:
        stage = Path(tmp) / 'result'; stage.mkdir()
        for folder in ('views', 'scan_levels', 'contact_sheets'): (stage / folder).mkdir()
        vertices_list=[]; faces_list=[]; colors_list=[]; owner_list=[]; source_list=[]; source_index_list=[]; protected_list=[]; optical_list=[]; panel_id_list=[]; offset=0
        cache={}; slices=[]
        for index, entry in enumerate(entries):
            scan=entry['scan_id']; source=alignment_dir/'scans'/scan/'mesh.ply'
            if scan not in cache:
                v,f,c = _read_mesh(source)
                panel_kinds, panel_ids = panel_faces(scan, v, f, regions)
                cache[scan] = (v,f,c,panel_kinds,panel_ids)
            v,f,c,panel_kinds,panel_ids=cache[scan]; normal, off, _ = _plane_group(entry['floors'])
            if normal[1] < 0: normal, off = -normal, -off
            lower=(normal, off + FLOOR_ALLOWANCE)
            lower_source = 'floor_allowance'
            earlier = [e for e in entries if e['scan_id'] == scan and e['height_group_index'] < entry['height_group_index'] and e['ceiling'] and e['ceiling']['mean_elevation_m'] < min(x['mean_elevation_m'] for x in entry['floors'])]
            if earlier:
                previous = max(earlier, key=lambda e: e['ceiling']['mean_elevation_m'])['ceiling']
                pn, po = _plane(previous)
                if pn[1] < 0: pn, po = -pn, -po
                lower = (pn, po)
                lower_source = previous['plane_id']
            ceiling=entry['ceiling']
            if ceiling:
                cn, co = _plane(ceiling)
                if cn[1] < 0: cn, co = -cn, -co
                upper=(-cn, -co)
                upper_record={'plane_id':ceiling['plane_id'], 'source_mesh':str(alignment_dir / ceiling['source_mesh']), 'equation':np.r_[cn,co].tolist(), 'fallback':False}
            else:
                upper=(np.array([0.,-1.,0.]), float(v[:,1].max()+1e-5))
                upper_record={'plane_id':None, 'equation':[0,1,0,-float(v[:,1].max()+1e-5)], 'fallback':True}
            sv,sf,sc,ids=clip_band(v,f,c,lower,upper)
            clipped_count = len(sf)
            if not len(sf): raise ValueError(f'Empty slice {scan}/{entry["floor_level_name"]}')
            used, inverse = np.unique(sf.reshape(-1), return_inverse=True)
            sv, sc, sf = sv[used], sc[used], inverse.reshape(-1, 3).astype(np.int32)
            optical = panel_kinds[ids]
            optical_panel_ids = panel_ids[ids]
            protected = floor_faces(sv, sf, entry['floors'], .10, .90)
            mesh_indices = np.zeros(len(sf), np.int32)
            source_meshes = [{'path':str(source), 'face_count':len(f)}]
            floor_groups = {}
            for record in entry['floors']:
                floor_groups.setdefault(record['source_mesh'], []).append(record)
            for floor_path, floor_records in floor_groups.items():
                floor_source = alignment_dir / floor_path
                fv, ff, fc = _read_mesh(floor_source)
                floor_ids = np.flatnonzero(floor_faces(fv, ff, floor_records, .03, .996, whole_triangle=True))
                if not len(floor_ids):
                    raise ValueError(f'No floor-plane triangles in {floor_source}')
                floor_used, floor_inverse = np.unique(ff[floor_ids].reshape(-1), return_inverse=True)
                sf = np.concatenate((sf, floor_inverse.reshape(-1, 3).astype(np.int32) + len(sv)))
                sv = np.concatenate((sv, fv[floor_used])); sc = np.concatenate((sc, fc[floor_used]))
                ids = np.concatenate((ids, floor_ids))
                protected = np.concatenate((protected, np.ones(len(floor_ids), bool)))
                optical = np.concatenate((optical, np.zeros(len(floor_ids), np.uint8)))
                optical_panel_ids = np.concatenate((optical_panel_ids, np.full(len(floor_ids), -1, np.int32)))
                mesh_indices = np.concatenate((mesh_indices, np.full(len(floor_ids), len(source_meshes), np.int32)))
                source_meshes.append({'path':str(floor_source), 'face_count':len(ff)})
            bounds=np.array([sv[sf].reshape(-1,3).min(axis=0),sv[sf].reshape(-1,3).max(axis=0)])
            slice_path = Path('scan_levels') / scan / entry['floor_level_name'] / 'slice.npz'
            (stage / slice_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(stage / slice_path, vertices=sv, faces=sf, colors=sc, source_face_id=ids, source_mesh_index=mesh_indices, floor_protected=protected, optical_type=optical, optical_panel_id=optical_panel_ids)
            face_offset = sum(len(part) for part in owner_list)
            vertices_list.append(sv); faces_list.append(sf+offset); colors_list.append(sc); offset+=len(sv)
            owner_list.append(np.full(len(sf), index, np.int32)); source_list.append(ids)
            source_index_list.append(mesh_indices); protected_list.append(protected); optical_list.append(optical); panel_id_list.append(optical_panel_ids)
            slices.append({k:entry[k] for k in ('scan_id','floor_level_name','floor_plane_ids','scan_level_name','height_group_index')} | {'source_mesh':str(source), 'source_meshes':source_meshes, 'protected_floor_face_count':int(protected.sum()), 'slice_mesh':slice_path.as_posix(), 'composite_face_offset':face_offset, 'floor_plane_sources': [{'plane_id': x['plane_id'], 'source_mesh': str(alignment_dir / x['source_mesh']), 'equation': x['plane_equation']} for x in entry['floors']], 'source_face_count':len(f), 'clipped_face_count':clipped_count, 'lower_cut_source':lower_source, 'lower_cut_equation':np.r_[lower[0],lower[1]].tolist(), 'upper_cut':upper_record, 'vertex_count':len(sv), 'face_count':len(sf), 'bounds':bounds.tolist()})
        v=np.concatenate(vertices_list); f=np.concatenate(faces_list); c=np.concatenate(colors_list); owners=np.concatenate(owner_list); source_ids=np.concatenate(source_list); source_indices=np.concatenate(source_index_list); floor_protected=np.concatenate(protected_list); optical_types=np.concatenate(optical_list); optical_panel_ids=np.concatenate(panel_id_list)
        if len(f) >= np.iinfo(np.int32).max: raise ValueError('Too many composite faces for pixel provenance')
        areas=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*.5
        # Fitted floor fills are visual support; do not count their overlapping area twice.
        areas[source_indices != 0] = 0
        max_edges=np.sqrt(np.maximum.reduce([np.sum((v[f[:,i]]-v[f[:,(i+1)%3]])**2,axis=1) for i in range(3)]))
        build_time=time.perf_counter()-start
        np.savez(stage/'face_provenance.npz', slice_index=owners, source_face_id=source_ids, source_mesh_index=source_indices, floor_protected=floor_protected, optical_type=optical_types, optical_panel_id=optical_panel_ids)
        candidate_records=[]; scan_records=[]; selected_ids=[]; selected_paths=[]
        render_time=0.; selection_time=0.
        totals = np.bincount(owners, weights=areas, minlength=len(slices))
        scans = list(dict.fromkeys(item['scan_id'] for item in slices))

        def process_scan(scan):
            indices = [i for i, item in enumerate(slices) if item['scan_id'] == scan]
            first = slices[indices[0]]['composite_face_offset']
            last = slices[indices[-1]]['composite_face_offset'] + slices[indices[-1]]['face_count']
            scan_f = f[first:last]; scan_owners = owners[first:last] - indices[0]
            scan_bounds = np.array([np.min([slices[i]['bounds'][0] for i in indices], axis=0),
                                    np.max([slices[i]['bounds'][1] for i in indices], axis=0)])
            render_start=time.perf_counter()
            scan_panel_ids = optical_panel_ids[first:last]
            scan_panel_neighbors = panel_context_neighbors(v, scan_f, scan_panel_ids)
            cutaway_protected = floor_protected[first:last] | (optical_types[first:last] != 0)
            scene=o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.core.Tensor(v.astype(np.float32, copy=False)), o3d.core.Tensor(scan_f.astype(np.uint32)))
            visible=[]; paths=[]; records=[]
            for n, camera in enumerate(cameras(scan_bounds, scan_bounds)):
                ident=f'{scan}_{n:02d}'
                seeds, removed = cutaway_faces(scene, v, scan_f, camera, min(resolution, CUTAWAY_MAX_RESOLUTION), max_edges=max_edges[first:last], protected=cutaway_protected)
                removed = apply_panel_context(removed, scan_panel_ids, scan_panel_neighbors)
                keep = np.ones(len(scan_f), bool); keep[removed] = False
                retained = np.flatnonzero(keep)
                if len(retained):
                    cut_scene = o3d.t.geometry.RaycastingScene()
                    cut_scene.add_triangles(o3d.core.Tensor(v.astype(np.float32, copy=False)), o3d.core.Tensor(scan_f[retained].astype(np.uint32)))
                    original_faces = np.flatnonzero(source_indices[first:last][retained] == 0)
                    fallback = None
                    if len(original_faces):
                        original_scene = o3d.t.geometry.RaycastingScene()
                        original_scene.add_triangles(o3d.core.Tensor(v.astype(np.float32, copy=False)), o3d.core.Tensor(scan_f[retained[original_faces]].astype(np.uint32)))
                        fallback = (original_scene, original_faces, source_indices[first:last][retained], owners[first:last][retained], floor_protected[first:last][retained])
                    image, pixels = render(cut_scene, v, scan_f[retained], c, camera, resolution, floor_fallback=fallback, optical_types=optical_types[first:last][retained])
                    valid = pixels >= 0; pixels[valid] = retained[pixels[valid]] + first
                else:
                    image = np.full((resolution, resolution, 3), 255, np.uint8)
                    pixels = np.full((resolution, resolution), -1, np.int32)
                np.savez_compressed(stage/'views'/f'{ident}_cutaway.npz', seed_face_ids=(seeds+first).astype(np.int32), removed_face_ids=(removed+first).astype(np.int32))
                path=stage/'views'/f'{ident}.png'; Image.fromarray(image).save(path)
                np.savez_compressed(stage/'views'/f'{ident}_face_ids.npz', face_ids=pixels.astype(np.int32))
                counts=np.bincount(pixels[pixels>=0]-first,minlength=len(scan_f)); ids=np.flatnonzero(counts>=MIN_PIXELS)
                visible.append(ids)
                record={'id':ident,'scan_id':scan,'slice_indices':indices,'camera':camera,'image':f'views/{ident}.png','pixel_face_ids':f'views/{ident}_face_ids.npz','cutaway_faces':f'views/{ident}_cutaway.npz', 'first_layer_face_count':len(seeds), 'removed_face_count':len(removed), 'optical_pixel_counts':{'mirror':int(counts[optical_types[first:last] == 1].sum()), 'window':int(counts[optical_types[first:last] == 2].sum())}}
                records.append(record); paths.append(path)
            contact_sheet = f'contact_sheets/{scan}.png'
            sheet(paths, stage / contact_sheet)
            scan_render_time = time.perf_counter()-render_start
            selection_start=time.perf_counter()
            for record, ids in zip(records, visible):
                covered_area = np.bincount(owners[first:last][ids], weights=areas[first:last][ids], minlength=len(slices))
                record['per_slice_coverage'] = np.divide(covered_area, totals, out=np.zeros(len(slices)), where=totals > 0).tolist()
            chosen, coverage, attainable, reason=select(visible,areas[first:last],scan_owners,len(indices),threshold,min_gain)
            scan_selected = [records[i]['id'] for i in chosen]
            chosen_paths = [paths[i] for i in chosen]
            selected_sheet = f'contact_sheets/{scan}_selected.png'
            sheet([paths[i] for i in chosen], stage / selected_sheet)
            for local, i in enumerate(indices):
                item=slices[i]
                item['contact_sheet'] = contact_sheet
                item['coverage']=float(coverage[local]); item['attainable_coverage']=float(attainable[local])
                item['surface_area_m2'] = float(totals[i])
                item['covered_area_m2'] = float(totals[i] * coverage[local])
                item['attainable_area_m2'] = float(totals[i] * attainable[local])
                item['selected_pose_ids']=[records[j]['id'] for j in chosen if np.any((scan_owners[visible[j]] == local) & (areas[first:last][visible[j]] > 0))]
                item['stop_reason'] = ('target_reached' if coverage[local] >= threshold else
                                       'target_unattainable' if attainable[local] < threshold else reason)
                folder=stage/'scan_levels'/scan/item['floor_level_name']
                (folder/'slice.json').write_text(json.dumps(item,indent=2)+'\n')
            scan_record={'scan_id':scan,'slice_indices':indices,'bounds':scan_bounds.tolist(),
                         'candidate_pose_ids':[record['id'] for record in records], 'selected_pose_ids':scan_selected,
                         'optical_panels':[{'id':region['id'], 'class':region['class'], 'count':sum(bool(panel.get('retained_face_indices', [0, 1])) for panel in region.get('sealed_panels', []))} for region in regions if region['scan_id'] == scan and region['class'] in ('mirror','window')],
                         'contact_sheet':contact_sheet,'selected_contact_sheet':selected_sheet,'stop_reason':reason}
            return records, scan_record, scan_selected, chosen_paths, scan_render_time, time.perf_counter()-selection_start

        render_wall_start = time.perf_counter()
        # Open3D ray casts release the GIL; each scan owns its scenes and output paths.
        with ThreadPoolExecutor(max_workers=len(scans)) as pool:
            for records, scan_record, scan_selected, chosen_paths, scan_render_time, scan_selection_time in pool.map(process_scan, scans):
                candidate_records.extend(records)
                scan_records.append(scan_record)
                selected_ids.extend(scan_selected)
                selected_paths.extend(chosen_paths)
                render_time += scan_render_time
                selection_time += scan_selection_time
        scan_work_wall_time = time.perf_counter()-render_wall_start
        sheet(selected_paths,stage/'contact_sheets'/'selected.png')
        manifest={'schema_version':3,'status':'complete','inputs':{'metadata':str(metadata_path),'alignment_dir':str(alignment_dir),'panel_manifest':str(panel_manifest) if panel_manifest.is_file() else None},
                  'settings':{'resolution':resolution,'cutaway_sampling_resolution':min(resolution,CUTAWAY_MAX_RESOLUTION),'geometry_archive_compression':'none','scan_workers':len(scans),'raycast_device':'CPU','coverage_threshold':threshold,'minimum_marginal_fraction_of_best_initial_view':min_gain,'minimum_pixels_per_triangle':MIN_PIXELS,'coverage_excludes_supplemental_floor_faces':True,'floor_allowance':FLOOR_ALLOWANCE,'cutaway_radius':CUTAWAY_RADIUS, 'cutaway_proximity_tolerance':0.005,'panel_neighbor_radius':PANEL_CONTEXT_RADIUS,'panel_removal_requires_majority_neighbor_removal':True,'window_blend':{'transmission':0.75,'reflection':0.10,'gray_tint':0.15,'gray_rgb':[150,160,170]}},
                  'scans':scan_records,'slices':slices,'candidates':candidate_records,'selected_pose_ids':selected_ids,
                  'stop_reason':'per_scan','selected_contact_sheet':'contact_sheets/selected.png','face_provenance':'face_provenance.npz','timings_seconds':{'build_slices':build_time,'render':render_time,'scan_work_wall':scan_work_wall_time,'selection':selection_time,'total':time.perf_counter()-start}}
        (stage/'generate_possible_views_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        backup=Path(tmp)/'previous'
        if output_dir.exists(): output_dir.rename(backup)
        try: stage.rename(output_dir)
        except BaseException:
            if backup.exists(): backup.rename(output_dir)
            raise
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alignment-dir',type=Path,default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resolution',type=int,default=4096)
    parser.add_argument('--coverage-threshold',type=float,default=.7)
    parser.add_argument('--minimum-gain',type=float,default=.1)
    args=parser.parse_args(); scene=args.alignment_dir.resolve().parent
    result=run(args.alignment_dir,args.metadata or scene/'artifacts'/'Generate Floor Plans'/'floor_level_metadata.json',args.output_dir or scene/'artifacts'/MACRO,args.resolution,args.coverage_threshold,args.minimum_gain)
    print(f"Selected {len(result['selected_pose_ids'])} views; {result['stop_reason']}; {result['timings_seconds']['total']:.1f}s")

if __name__ == '__main__': main()
