"""Local floor-supported navigation on sealed meshes for the cinematic tour."""
from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


def clearance_and_support(scene, floor_scene, points, clearance=0.08, max_floor_distance=2.8):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if not len(points):
        return np.zeros(0, dtype=bool)
    rays = np.column_stack((points, np.broadcast_to(np.array([0, -1, 0], np.float32), points.shape)))
    floor_distance = floor_scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    floor_valid = (floor_distance > clearance) & (floor_distance <= max_floor_distance)
    valid = np.zeros(len(points), dtype=bool)
    if np.any(floor_valid):
        supported_points = points[floor_valid]
        distance = scene.compute_distance(o3d.core.Tensor(supported_points)).numpy()
        valid[floor_valid] = distance >= clearance
    return valid


def segment_clear(scene, floor_scene, a, b, clearance=0.08, spacing=0.025):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    length = float(np.linalg.norm(b - a))
    if not length:
        return bool(clearance_and_support(scene, floor_scene, [a], clearance)[0])
    # Early-exit direct raycast check: reject immediately if the chord hits an obstacle
    ray = np.asarray([np.r_[a, b - a]], dtype=np.float32)
    if not (scene.cast_rays(o3d.core.Tensor(ray))['t_hit'].numpy()[0] > 1):
        return False
    # Sample spacing plus the clearance margin bounds the unsampled distance by Lipschitz continuity.
    steps = max(1, int(np.ceil(length / spacing)))
    points = a + np.linspace(0, 1, steps + 1)[:, None] * (b - a)
    return bool(clearance_and_support(scene, floor_scene, points, clearance + spacing / 2).all())



def rounded_path(scene, floor_scene, polyline, clearance=0.08):
    """Round each interior corner, checking the actual sampled curve and its chords."""
    points = np.asarray(polyline, dtype=float)
    result = [points[0]]
    for i in range(1, len(points) - 1):
        before, corner, after = points[i - 1:i + 2]
        incoming, outgoing = before - corner, after - corner
        a, b = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if min(a, b) < 1e-9:
            raise ValueError('Degenerate corner in path')
        for radius in (0.25, 0.15, 0.08, 0.04):
            trim = min(radius, a * .4, b * .4)
            start = corner + incoming / a * trim
            end = corner + outgoing / b * trim
            t = np.linspace(0, 1, max(5, int(np.ceil(2 * trim / .025)) + 1))[:, None]
            curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * corner + t ** 2 * end
            candidate = np.vstack((result[-1], curve))
            if all(segment_clear(scene, floor_scene, x, y, clearance) for x, y in zip(candidate, candidate[1:])):
                result.extend(curve)
                break
        else:
            raise ValueError(f'Cannot round path corner {i} without losing clearance or floor support')
    if not segment_clear(scene, floor_scene, result[-1], points[-1], clearance):
        raise ValueError('Final smoothed path segment fails clearance or floor support')
    result.append(points[-1])
    curved = [result[0]]
    for start, end in zip(result, result[1:]):
        delta = end - start
        length = np.linalg.norm(delta[[0, 2]])
        if length > .6:
            sideways = np.array([-delta[2], 0, delta[0]]) / length
            u = np.linspace(0, 1, max(3, int(np.ceil(np.linalg.norm(delta) / .035)) + 1))
            for amplitude in (min(.45, length * .2), min(.22, length * .1), -.15, -.07):
                candidate = start + u[:, None] * delta + (amplitude * np.sin(np.pi * u) ** 2)[:, None] * sideways
                candidate[0], candidate[-1] = start, end
                rays = np.column_stack((candidate[:-1], np.diff(candidate, axis=0))).astype(np.float32)
                if (clearance_and_support(scene, floor_scene, candidate, clearance + .02).all()
                        and np.all(scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy() > 1)):
                    curved.extend(candidate[1:])
                    break
            else:
                curved.append(end)
        else:
            curved.append(end)
    return np.asarray(curved)


def validate_door_crossing(scene, vertices, faces, panel, start, end, crossing_center=None):
    """Allow only approximately coplanar hits in the designated aligned door opening."""
    panel_center = np.asarray(panel['panel_center_xyz_m'], dtype=float)
    center = panel_center.copy() if crossing_center is None else np.asarray(crossing_center, dtype=float)
    normal = np.asarray(panel['panel_plane_normal_xyz'], dtype=float)
    norm = np.linalg.norm(normal)
    if center.shape != (3,) or normal.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(norm) or norm < 1e-9:
        raise ValueError('Invalid aligned door panel center or normal')
    normal /= norm
    if crossing_center is not None:
        offset = center - panel_center
        height = float(panel['panel_dimensions_m']['height'])
        width = float(panel['panel_dimensions_m']['width'])
        tangent = np.cross([0., 1., 0.], normal)
        tangent /= np.linalg.norm(tangent)
        if (not np.isfinite(center).all() or abs(np.dot(offset, normal)) > 1e-6
                or abs(np.dot(offset, tangent)) > width / 2 - .15 or abs(offset[1]) > height / 2 - .15):
            raise ValueError('Door crossing lies outside the designated panel')
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    if np.linalg.norm(np.cross(direction, normal)) > 1e-6 or np.linalg.norm(np.cross(start - center, normal)) > 1e-6:
        raise ValueError('Door crossing must pass through the panel center along its normal')
    if np.dot(start - center, normal) * np.dot(end - center, normal) >= 0:
        raise ValueError('Door crossing endpoints must lie on opposite sides')
    rays = np.asarray([np.r_[start, direction]], dtype=np.float32)
    hits = scene.list_intersections(o3d.core.Tensor(rays))
    distances, ids = hits['t_hit'].numpy(), hits['primitive_ids'].numpy()
    selected = (distances > 0) & (distances < 1)
    distances, ids = distances[selected], ids[selected]
    if not len(ids):
        raise ValueError('No sealed surface was found at the designated door crossing')
    hit_points = start + distances[:, None] * direction
    # Paired source surfaces can be offset from the idealized panel center by wall thickness.
    half_depth = float(panel['panel_dimensions_m']['depth']) / 2 + 0.15
    if np.any(np.abs((hit_points - center) @ normal) > half_depth):
        raise ValueError('Crossing hits geometry outside the designated door slab')
    triangles = np.asarray(vertices)[np.asarray(faces)[ids]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12) or np.any(np.abs(normals @ normal) / lengths < .5):
        raise ValueError('Crossing hits a surface that does not follow the door plane')
    samples = start + np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(direction) / .02)) + 1))[:, None] * direction
    outside = samples[np.abs((samples - center) @ normal) > half_depth]
    if len(outside) and np.any(scene.compute_distance(o3d.core.Tensor(outside.astype(np.float32))).numpy() < .09):
        raise ValueError('Crossing has insufficient clearance outside the designated door slab')
    return {'panel_index': panel['panel_index'], 'center': center.tolist(), 'normal': normal.tolist(),
            'start': start.tolist(), 'end': end.tolist(), 'hit_points': hit_points.tolist(),
            'surface_slab_half_depth_m': half_depth}


class LocalNavigator:
    """Cache a scan's floor-supported grid; paths remain candidates until smoothed and revalidated."""
    def __init__(self, scene, floor_scene, floor_points, endpoints, spacing=0.12, clearance=0.08, xz_bounds=None):
        self.scene, self.floor_scene, self.clearance = scene, floor_scene, clearance
        floor_points = np.asarray(floor_points, dtype=float)
        endpoints = np.asarray(endpoints, dtype=float)
        low, high = floor_points.min(axis=0), floor_points.max(axis=0)
        if xz_bounds is not None:
            bounds = np.asarray(xz_bounds, dtype=float)
            if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                raise ValueError('Invalid local refinement bounds')
            low[[0, 2]] = np.maximum(low[[0, 2]], bounds[0])
            high[[0, 2]] = np.minimum(high[[0, 2]], bounds[1])
            if np.any(low[[0, 2]] >= high[[0, 2]]):
                raise ValueError('Local refinement bounds do not intersect the scan floor')
        lower_y = min(low[1], endpoints[:, 1].min()) + spacing
        upper_y = max(high[1] + 2.4, endpoints[:, 1].max() + 0.5)
        axes = [np.arange(low[0] - spacing, high[0] + 2 * spacing, spacing),
                np.arange(lower_y, upper_y, spacing),
                np.arange(low[2] - spacing, high[2] + 2 * spacing, spacing)]
        self.shape = tuple(len(axis) for axis in axes)
        if np.prod(self.shape, dtype=np.int64) > 3_000_000:
            raise ValueError(f'Grid resource limit exceeded: {self.shape} at {spacing:.3f} m spacing (maximum 3000000 cells)')
        points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3).astype(np.float32)
        # Six-neighbor edges have length spacing; this margin guarantees clearance between vertices.
        valid = clearance_and_support(scene, floor_scene, points, clearance + spacing / 2)
        ids = np.arange(len(points)).reshape(self.shape)
        mask = valid.reshape(self.shape)
        left, right = [], []
        for axis in range(3):
            a, b = [slice(None)] * 3, [slice(None)] * 3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            selected = mask[tuple(a)] & mask[tuple(b)]
            left.append(ids[tuple(a)][selected])
            right.append(ids[tuple(b)][selected])
        left, right = np.concatenate(left), np.concatenate(right)
        self.graph = coo_matrix((np.full(2 * len(left), spacing),
                                (np.r_[left, right], np.r_[right, left])),
                               shape=(len(points), len(points))).tocsr()
        self.points = points
        self.valid_ids = np.flatnonzero(valid)
        self.tree = cKDTree(points[self.valid_ids])

    def endpoint_nodes(self, point):
        point = np.asarray(point, dtype=float)
        indices = self.tree.query_ball_point(point, 0.35)
        indices.sort(key=lambda i: np.linalg.norm(self.points[self.valid_ids[i]] - point))
        return [int(self.valid_ids[i]) for i in indices
                if segment_clear(self.scene, self.floor_scene, point, self.points[self.valid_ids[i]], self.clearance)]

    def route(self, start, end):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if segment_clear(self.scene, self.floor_scene, start, end, self.clearance):
            return np.stack((start, end))
        starts, ends = self.endpoint_nodes(start), self.endpoint_nodes(end)
        if not starts or not ends:
            raise ValueError('Endpoint has no collision-free connection to the floor-supported grid')
        # Compute a shortest graph path from all valid start attachments simultaneously.
        dist, previous, sources = dijkstra(self.graph, indices=starts, min_only=True, return_predecessors=True)
        reachable = [node for node in ends if np.isfinite(dist[node])]
        if not reachable:
            raise ValueError('No floor-supported grid path connects the endpoints')
        node = min(reachable, key=lambda i: dist[i] + np.linalg.norm(self.points[i] - end))
        chain = [node]
        while previous[node] != -9999:
            node = int(previous[node])
            chain.append(node)
        chain.reverse()
        points = np.vstack((start, self.points[chain], end))
        # Greedy visibility shortcuts are safe only when the full segment is checked.
        result = [points[0]]
        index = 0
        while index < len(points) - 1:
            for target in range(len(points) - 1, index, -1):
                if segment_clear(self.scene, self.floor_scene, points[index], points[target], self.clearance):
                    break
            else:
                raise ValueError('Grid edge failed final support or clearance validation')
            result.append(points[target])
            index = target
        return np.asarray(result)


def local_refined_route(scene, floor_scene, floor_points, start, end):
    """Try bounded fine grids around a failed local connection."""
    endpoints = np.asarray([start, end], dtype=float)
    low, high = endpoints[:, [0, 2]].min(axis=0), endpoints[:, [0, 2]].max(axis=0)
    errors = []
    for margin in (.6, 1.2, 2.4):
        navigator = None
        try:
            navigator = LocalNavigator(scene, floor_scene, floor_points, endpoints, spacing=.06,
                                       xz_bounds=[low - margin, high + margin])
            path = navigator.route(start, end)
            return path, {'method': 'local_refined_grid', 'spacing_m': .06, 'margin_m': margin,
                          'grid_cells': len(navigator.points), 'failed_regions': errors}
        except ValueError as error:
            errors.append({'margin_m': margin, 'reason': str(error)})
        finally:
            del navigator
    raise ValueError(f'Local fine-grid search exhausted margins 0.6, 1.2, 2.4 m: {errors}')
