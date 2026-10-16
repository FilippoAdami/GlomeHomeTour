"""Per-level full-scene collapse and constrained orthogonal distance fitting (metres, Y up)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.optimize import LinearConstraint, minimize
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely import contains_xy
from surface_footprints import (clean_footprint, observed_footprint, polygons,
                                frame_footprint, snap_footprint_to_walls, constrain_footprint, wall_regions)


def sample_ring(ring, spacing=.01):
    distances = np.arange(0., ring.length, spacing)
    return np.array([ring.interpolate(d).coords[0] for d in distances])


def squared_distances(points, vertices):
    start = vertices
    delta = np.roll(vertices, -1, axis=0)-start
    length2 = np.sum(delta*delta, axis=1)
    relative = points[:, None, :]-start[None, :, :]
    fraction = np.clip(np.sum(relative*delta, axis=2)/np.maximum(length2, 1e-20), 0., 1.)
    return np.min(np.sum((relative-fraction[:, :, None]*delta)**2, axis=2), axis=1)


def fit_ring(initial, samples, min_edge=.10):
    """Optimize all line offsets jointly; each adjacent intersection is a right angle."""
    origin = np.mean(samples, axis=0)
    samples = samples-origin
    initial = np.array(initial, dtype=float)-origin
    edges = np.roll(initial, -1, axis=0)-initial
    axes = np.argmin(np.abs(edges), axis=1)
    if np.any(axes == np.roll(axes, 1)):
        raise ValueError('Orthogonal fitter needs alternating horizontal/vertical edges')
    count = len(initial)
    offsets = initial[np.arange(count), axes]
    signs = np.sign(edges[np.arange(count), 1-axes])
    matrix = np.zeros((count, count))
    for i in range(count):
        matrix[i, (i+1)%count] = signs[i]
        matrix[i, (i-1)%count] = -signs[i]
    def vertices(values):
        result = np.empty((count, 2))
        result[np.arange(count), axes] = values
        result[np.arange(count), np.roll(axes, 1)] = np.roll(values, 1)
        return result
    def objective(values):
        return float(np.mean(squared_distances(samples, vertices(values))))
    result = minimize(objective, offsets, method='SLSQP',
                      constraints=LinearConstraint(matrix, min_edge, np.inf),
                      options={'maxiter': 80, 'ftol': 1e-11})
    fitted = vertices(result.x)
    if (np.min(matrix@result.x) < min_edge-1e-7 or not Polygon(fitted).is_valid
            or objective(result.x) > objective(offsets)+1e-10):
        fitted = initial
    return fitted+origin


def fit_distance_footprint(initial, contour, min_edge=.10):
    """Fit sampled contour distances, testing simpler right-angle topologies as well."""
    outputs, records = [], []
    for poly in polygons(initial):
        source = max(polygons(contour), key=lambda p: p.intersection(poly).area)
        source_rings = [source.exterior, *source.interiors]
        fitted_rings = []
        for ring in [poly.exterior, *poly.interiors]:
            evidence = min(source_rings, key=lambda r: r.hausdorff_distance(ring))
            samples = sample_ring(evidence)
            current = np.array(ring.coords[:-1])
            before = float(np.mean(squared_distances(samples, current)))
            current = fit_ring(current, samples, min_edge)
            score = float(np.mean(squared_distances(samples, current)))
            while len(current) > 4:
                best = None
                for i in range(len(current)):
                    j = (i+1)%len(current)
                    axis = int(np.argmax(np.abs(current[j]-current[i])))
                    for anchor in (i,j):
                        candidate = current.copy()
                        neighbor = (j+1)%len(current) if anchor == i else (i-1)%len(current)
                        candidate[[i,j,neighbor], axis] = current[anchor, axis]
                        polygon = Polygon(candidate).simplify(0)
                        if not polygon.is_valid or polygon.is_empty or polygon.area < 1e-8:
                            continue
                        candidate = np.array(polygon.exterior.coords[:-1])
                        if len(candidate) >= len(current) or len(candidate) < 4:
                            continue
                        lengths = np.linalg.norm(np.roll(candidate,-1,axis=0)-candidate, axis=1)
                        if lengths.min() < min_edge-1e-8:
                            continue
                        candidate = fit_ring(candidate, samples, min_edge)
                        candidate_score = float(np.mean(squared_distances(samples,candidate)))
                        if candidate_score < score-1e-10 and (best is None or candidate_score < best[0]):
                            best = (candidate_score,candidate)
                if best is None:
                    break
                score,current = best
            fitted_rings.append(current)
            records.append({'samples':len(samples),'initial_mean_squared_distance_m2':before,
                            'fitted_mean_squared_distance_m2':score,'rms_distance_m':float(np.sqrt(score)),
                            'segments':len(current),'min_segment_m':float(np.min(np.linalg.norm(np.roll(current,-1,axis=0)-current,axis=1)))})
        fitted = Polygon(fitted_rings[0], fitted_rings[1:])
        if not fitted.is_valid:
            raise ValueError('Constrained contour fit produced overlapping rings')
        outputs.append(fitted)
    return unary_union(outputs),records


def plane_y(equation, points):
    return -(points[:,0]*equation[0]+points[:,2]*equation[2]+equation[3])/equation[1]


def collapse_level(vertices, faces, normals, centers, floor, ceilings, floors=()):
    """Select a scene height band; semantic surfaces define heights, never its contour."""
    floor_y = plane_y(floor['equation'],centers)
    lower = floor_y-.03
    upper = np.full(len(centers),np.inf)
    for ceiling in ceilings:
        heights = plane_y(ceiling['equation'],centers)
        footprint = ceiling['footprint']
        # Only use ceiling support here to choose the local upper/lower height envelope.
        local = contains_xy(footprint.buffer(.15),centers[:,0],centers[:,2])
        under = local & (heights<=floor_y+.05) & (heights>=floor_y-.4)
        lower[under] = np.minimum(lower[under],heights[under]-.12)
        above = local & (heights>floor_y+.05)
        upper[above] = np.minimum(upper[above],heights[above]-.25)
    for other in floors:
        heights = plane_y(other['equation'],centers)
        above = heights>floor_y+.15
        upper[above] = np.minimum(upper[above],heights[above]-.12)
    keep = ((centers[:,1]>=lower) & (centers[:,1]<=upper) & (np.abs(normals[:,1])>.5))
    if not np.any(keep):
        raise ValueError(f"No source scene support in level band for {floor['name']}")
    return faces[keep]


def restore_level_footprints(units, scene_path, a, b, wall_planes, output_dir):
    scene = o3d.io.read_triangle_mesh(str(scene_path))
    if scene.is_empty() or not len(scene.triangles):
        raise ValueError(f'Cannot read source scene mesh: {scene_path}')
    scene.compute_triangle_normals()
    vertices,faces = np.asarray(scene.vertices),np.asarray(scene.triangles)
    normals = np.asarray(scene.triangle_normals)
    centers = vertices[faces].mean(axis=1)
    floors = [u for u in units if u['role']=='floor']
    ceilings = [u for u in units if u['role']=='ceiling']
    provenance = {'source_scene_mesh':str(scene_path),'contour_source':'full_scene_per_level_height_collapse',
                  'objective':'mean squared Euclidean distance to uniformly sampled cleaned contour',
                  'sample_spacing_m':.01,'minimum_segment_m':.10,'levels':[]}
    for floor in floors:
        selected = collapse_level(vertices,faces,normals,centers,floor,ceilings,floors)
        initial,contour = observed_footprint(vertices,selected,a,b,return_contour=True)
        # Keep the measured connected level region associated with this floor instance.
        local_source = frame_footprint(floor['footprint'],a,b,inverse=True)
        pieces = [p for p in polygons(initial) if p.intersection(local_source).area>.01]
        if not pieces:
            raise ValueError(f"No collapsed component matches {floor['name']}")
        initial = clean_footprint(unary_union(pieces))
        fitted,scores = fit_distance_footprint(initial,contour)
        fitted = snap_footprint_to_walls(fitted,wall_planes,a,b)
        fitted = constrain_footprint(fitted,wall_regions(wall_planes,a,b,floor['elevation']))
        fitted = clean_footprint(fitted)
        final_scores = []
        for poly in polygons(fitted):
            evidence = max(polygons(contour),key=lambda p:p.intersection(poly).area)
            points = sample_ring(evidence.exterior)
            distances = np.sqrt(squared_distances(points,np.array(poly.exterior.coords[:-1])))
            inside = contains_xy(poly,points[:,0],points[:,1])
            final_scores.append({'samples':len(points),'rms_distance_m':float(np.sqrt(np.mean(distances**2))),
                                 'mean_absolute_distance_m':float(distances.mean()),
                                 'signed_mean_distance_m':float(np.mean(np.where(inside,-distances,distances)))})
        floor['footprint'] = frame_footprint(fitted,a,b)
        floor['footprint_source'] = 'full_scene_per_level_height_collapse'
        used = np.unique(selected)
        collapsed = vertices[used].copy()
        collapsed[:,1] = plane_y(floor['equation'],collapsed)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(collapsed))
        path = Path(output_dir)/f"{floor['name']}_rough_collapsed.ply"
        if not o3d.io.write_point_cloud(str(path),cloud):
            raise OSError(f'Failed to write collapsed level: {path}')
        provenance['levels'].append({'floor':floor['name'],'source_triangles':len(selected),
                                    'collapsed_artifact':str(path),'cleaned_contour_ab':mapping(contour),
                                    'fitted_footprint_ab':mapping(fitted),'fit_scores':scores,'final_boundary_scores':final_scores})
    for ceiling in ceilings:
        supported = [floor for floor in floors if floor['footprint'].intersection(ceiling['footprint']).area>.01]
        above = [floor for floor in supported if -.05<=floor['elevation']-ceiling['elevation']<=.4]
        if above:
            ceiling['footprint'] = unary_union([floor['footprint'] for floor in above])
        else:
            below = [floor for floor in supported if floor['elevation']<ceiling['elevation']]
            if below:
                level_region = unary_union([floor['footprint'] for floor in below])
                # A floor patio cannot imply a roof: bound roof coverage using the
                # full scene's ceiling-facing support, rather than semantic labels.
                equation = ceiling['equation']
                depth = ceiling['classification'].parameters.get('beam_depth_m', 0.)
                support = ((normals@equation[:3]>.7)
                           & (np.abs(centers@equation[:3]+equation[3])<=max(.2,depth+.1)))
                if not np.any(support):
                    raise ValueError(f"No full-scene roof support for {ceiling['name']}")
                coverage, contour = observed_footprint(vertices,faces[support],a,b,return_contour=True)
                local = frame_footprint(level_region,a,b,inverse=True).intersection(coverage)
                fitted,_ = fit_distance_footprint(clean_footprint(local),contour)
                fitted = snap_footprint_to_walls(fitted,wall_planes,a,b)
                fitted = constrain_footprint(fitted,wall_regions(wall_planes,a,b,ceiling['elevation']))
                ceiling['footprint'] = frame_footprint(clean_footprint(fitted),a,b)
            else:
                # Ceiling-only partial scan: collapse the source geometry below this plane.
                keep = (centers[:,1]<=plane_y(ceiling['equation'],centers)+.15)&(np.abs(normals[:,1])>.5)
                initial,contour = observed_footprint(vertices,faces[keep],a,b,return_contour=True)
                fitted,_ = fit_distance_footprint(clean_footprint(initial),contour)
                ceiling['footprint'] = frame_footprint(fitted,a,b)
        ceiling['footprint_source'] = 'full_scene_per_level_height_collapse'
    (Path(output_dir)/'level_footprints_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
