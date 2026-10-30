"""Assign inferred/unobserved surface pieces only after observed-region cleanup."""
from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def read_infill_attributes(path: Path, face_count: int):
    """Read locked Stage04 face provenance and coplanar infill RGB in source face order."""
    ply = PlyData.read(path, known_list_len={'face': {'vertex_indices': 3}})
    faces = ply['face'].data
    if len(faces) != face_count:
        raise ValueError('infill provenance must match source mesh face order')
    names = faces.dtype.names
    flags = faces['is_hole_fill'].astype(bool) if 'is_hole_fill' in names else np.zeros(face_count, bool)
    rgb = (np.column_stack([faces[k] for k in ('fill_r', 'fill_g', 'fill_b')]) / 255.
           if all(k in names for k in ('fill_r', 'fill_g', 'fill_b')) else None)
    return flags, rgb


def assign_deferred_patches(labels, centers, normals, areas, rgb, adj_u, adj_v,
                            gravity_up, wall_cids):
    """Grow accepted owners into geometry/RGB pieces; every assignment retains mesh contact.

    Fit one plane per smooth, color-coherent piece. Competing boundary owners are
    scored by plane residual, normal agreement and original RGB, never display colors.
    Multi-source growth can give different pieces of one infill patch different owners.
    Returns labels, metrics and explicit same-surface instance association face pairs.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if (labels.ndim != 1 or labels.dtype.kind not in 'iu' or np.shape(centers) != (n, 3)
            or np.shape(normals) != (n, 3) or np.shape(rgb) != (n, 3) or np.shape(areas) != (n,)):
        raise ValueError('deferred geometry, RGB and labels must share source face order')
    normals = np.asarray(normals, dtype=float)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-15)
    result = labels.astype(np.int32, copy=True)
    deferred = labels <= 0
    faces = np.flatnonzero(deferred)
    if not len(faces):
        return result, dict(deferred_faces=0, assigned_faces=0, remaining_faces=0, pieces=0), np.empty((0, 2), np.int32)
    up = np.array(gravity_up, dtype=float, copy=True)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError('gravity_up must be finite and nonzero')
    up /= np.linalg.norm(up)
    wall_cids = set(wall_cids)
    horizontal = np.abs(normals @ up) >= np.cos(np.radians(20)) - 1e-12
    local = np.full(n, -1, np.int32)
    local[faces] = np.arange(len(faces))
    # Each undirected edge is used once, including callers with directed copies.
    u, v = np.minimum(adj_u, adj_v), np.maximum(adj_u, adj_v)
    edges = np.unique(np.column_stack((u, v)), axis=0)
    u, v = edges.T
    distance = np.linalg.norm(centers[u] - centers[v], axis=1)
    scale = max(float(np.median(distance[distance > 0])) if np.any(distance > 0) else 1., 1e-12)
    internal = deferred[u] & deferred[v]
    smooth = internal & (np.einsum('ij,ij->i', normals[u], normals[v]) >= np.cos(np.radians(25)))
    smooth &= np.linalg.norm(rgb[u] - rgb[v], axis=1) <= .15
    smooth &= horizontal[u] == horizontal[v]
    count, component = connected_components(csr_matrix((np.ones(smooth.sum(), bool),
        (local[u[smooth]], local[v[smooth]])), shape=(len(faces), len(faces))), directed=False)
    piece = np.full(n, -1, np.int32)
    piece[faces] = component
    weight = np.maximum(areas[faces], 1e-15)
    total = np.bincount(component, weights=weight, minlength=count)
    means = lambda values: np.column_stack([np.bincount(component, weights=weight * values[:, k], minlength=count)
                                           for k in range(3)]) / total[:, None]
    midpoint = means(centers[faces])
    color = means(rgb[faces])
    mean_normal = means(normals[faces])
    delta = centers[faces] - midpoint[component]
    covariance = np.zeros((count, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            covariance[:, a, b] = covariance[:, b, a] = np.bincount(
                component, weights=weight * delta[:, a] * delta[:, b], minlength=count) / total
    eigenvalues, axes = np.linalg.eigh(covariance)
    plane_normal = axes[:, :, 0]
    mean_length = np.linalg.norm(mean_normal, axis=1)
    mean_normal /= np.maximum(mean_length[:, None], 1e-12)
    # Degenerate tiny pieces use their actual face normals rather than an arbitrary SVD axis.
    plane_normal[eigenvalues[:, 1] < scale ** 2 * 1e-6] = mean_normal[eigenvalues[:, 1] < scale ** 2 * 1e-6]
    planar = (np.sqrt(np.maximum(eigenvalues[:, 0], 0)) <= 2 * scale) & (mean_length >= .9)
    plane_normal[~planar] = mean_normal[~planar]
    has_horizontal = np.bincount(component, weights=horizontal[faces], minlength=count) > 0

    def fit_cost(p, point, normal, source_rgb):
        angle = 1 - abs(float(plane_normal[p] @ normal))
        residual = abs(float((midpoint[p] - point) @ normal)) / (2 * scale)
        return 3 * angle + min(residual, 10) * planar[p] + np.linalg.norm(color[p] - source_rgb) / .15

    # Boundary prototypes remain tied to a real owner and a real contacting seed face.
    a, b = np.r_[u, v], np.r_[v, u]
    boundary = deferred[a] & (labels[b] > 0)
    boundary_piece, seed = piece[a[boundary]], b[boundary]
    prototypes = []
    heap = []
    if len(seed):
        pairs, inverse = np.unique(np.column_stack((boundary_piece, labels[seed])), axis=0, return_inverse=True)
        order = np.argsort(inverse, kind='stable')
        bounds = np.r_[0, np.cumsum(np.bincount(inverse))]
        for group, (p, owner) in enumerate(pairs):
            if owner in wall_cids and has_horizontal[p]:
                continue
            source = seed[order[bounds[group]:bounds[group + 1]]]
            costs = np.array([fit_cost(p, centers[f], normals[f], rgb[f]) for f in source])
            # Median matching boundary geometry is robust to a single favorable noisy triangle.
            chosen = source[np.argsort(costs, kind='stable')[len(source) // 2]]
            prototype = len(prototypes)
            prototypes.append((int(owner), centers[chosen], normals[chosen], rgb[chosen]))
            heapq.heappush(heap, (float(np.median(costs)), int(p), prototype))
    cross = internal & (piece[u] != piece[v])
    left, right = piece[u[cross]], piece[v[cross]]
    graph = csr_matrix((np.ones(2 * len(left), bool), (np.r_[left, right], np.r_[right, left])), shape=(count, count))
    owners = np.full(count, -1, np.int32)
    def grow():
        while heap:
            cost, p, prototype = heapq.heappop(heap)
            if owners[p] > 0:
                continue
            owner, point, normal, source_rgb = prototypes[prototype]
            if owner in wall_cids and has_horizontal[p]:
                continue
            owners[p] = owner
            for neighbor in graph.indices[graph.indptr[p]:graph.indptr[p + 1]]:
                if owners[neighbor] <= 0 and not (owner in wall_cids and has_horizontal[neighbor]):
                    transition = .1 + np.linalg.norm(color[p] - color[neighbor]) / .15
                    transition += 1 - abs(float(plane_normal[p] @ plane_normal[neighbor]))
                    score = cost + transition + fit_cost(neighbor, point, normal, source_rgb)
                    heapq.heappush(heap, (float(score), int(neighbor), prototype))

    grow()
    result[faces] = owners[component]
    # Sealed surfaces can contain disconnected tessellation islands. Associate only
    # nearby, coplanar, similarly colored pieces with an already accepted instance.
    links = []
    accepted = np.flatnonzero(labels > 0)
    pending = np.flatnonzero(owners <= 0)
    if len(accepted) and len(pending):
        for is_horizontal in (False, True):
            pending_group = pending[has_horizontal[pending] == is_horizontal]
            candidates_pool = accepted[~np.isin(labels[accepted], list(wall_cids))] if is_horizontal else accepted
            if not len(pending_group) or not len(candidates_pool):
                continue
            tree = cKDTree(centers[candidates_pool])
            distances, nearest = tree.query(midpoint[pending_group], k=min(32, len(candidates_pool)),
                                             distance_upper_bound=6 * scale, workers=8)
            distances = np.asarray(distances).reshape(len(pending_group), -1)
            nearest = np.asarray(nearest).reshape(len(pending_group), -1)
            for row, p in enumerate(pending_group):
                valid = np.isfinite(distances[row]) & (nearest[row] < len(candidates_pool))
                candidates = candidates_pool[nearest[row, valid]]
                if not len(candidates):
                    continue
                eligible = np.abs(normals[candidates] @ plane_normal[p]) >= np.cos(np.radians(25))
                residual = np.abs((midpoint[p] - centers[candidates]) @ plane_normal[p])
                eligible &= residual <= 2 * scale
                eligible &= np.linalg.norm(rgb[candidates] - color[p], axis=1) <= .25
                candidates = candidates[eligible]
                if not len(candidates):
                    continue
                scores = [fit_cost(p, centers[f], normals[f], rgb[f])
                          + np.linalg.norm(midpoint[p] - centers[f]) / (6 * scale) for f in candidates]
                target = int(candidates[int(np.argmin(scores))])
                part_faces = faces[component == p]
                source = int(part_faces[np.argmin(np.linalg.norm(centers[part_faces] - centers[target], axis=1))])
                if np.linalg.norm(centers[source] - centers[target]) > 6 * scale:
                    continue
                owners[p] = labels[target]
                # Retain an explicit association, not an invented source-mesh triangle.
                links.append((source, target))
                prototype = len(prototypes)
                prototypes.append((int(labels[target]), centers[target], normals[target], rgb[target]))
                for neighbor in graph.indices[graph.indptr[p]:graph.indptr[p + 1]]:
                    if owners[neighbor] <= 0:
                        heapq.heappush(heap, (fit_cost(neighbor, centers[target], normals[target], rgb[target]) + .1,
                                             int(neighbor), prototype))
        grow()
        result[faces] = owners[component]
    if np.any(np.isin(result, list(wall_cids)) & horizontal):
        raise RuntimeError('deferred assignment created a near-horizontal wall face')
    return result, dict(deferred_faces=int(len(faces)), assigned_faces=int((result[faces] > 0).sum()),
        remaining_faces=int((result[faces] <= 0).sum()), pieces=int(count),
        planar_pieces=int(planar.sum()), assigned_pieces=int((owners > 0).sum()),
        spatially_associated_pieces=len(links), spatial_max_distance=6 * scale,
        spatial_max_plane_residual=2 * scale), np.asarray(links, dtype=np.int32).reshape(-1, 2)
