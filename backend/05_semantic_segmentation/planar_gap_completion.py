"""Complete explicitly selected unknown faces from compatible labeled surfaces."""
import numpy as np
from scipy.sparse import csr_matrix, csgraph


def complete_planar_gaps(labels, vertices, triangles, normals, centers, edge_u, edge_v,
                         candidates, seeds, tolerance):
    """Fill unknowns only, keeping each proposal tied to its original seed plane.

    Callers select eligible unknown faces and trusted seed surfaces. Coplanar
    boundaries compete locally for ownership; existing object labels stay fixed.
    """
    n = len(labels)
    if (labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer)
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or triangles.shape != (n, 3) or normals.shape != (n, 3)
            or centers.shape != (n, 3) or candidates.shape != (n,) or seeds.shape != (n,)
            or len(edge_u) != len(edge_v) or tolerance <= 0 or not np.isfinite(tolerance)):
        raise ValueError('Labels, geometry, masks and positive tolerance must agree')
    if (np.any(~np.isfinite(vertices)) or np.any(~np.isfinite(normals))
            or np.any(~np.isfinite(centers)) or np.any(triangles < 0)
            or np.any(triangles >= len(vertices)) or np.any(edge_u < 0)
            or np.any(edge_v < 0) or np.any(edge_u >= n) or np.any(edge_v >= n)):
        raise ValueError('Invalid mesh geometry or adjacency')
    candidates = np.asarray(candidates, bool) & (labels <= 0)
    seeds = np.asarray(seeds, bool) & (labels > 0)
    result = labels.copy()
    report = {'filled_faces': 0, 'competing_boundary_faces': 0}
    if not candidates.any() or not seeds.any():
        return result, report
    cosine = np.cos(np.deg2rad(20))
    u, v = edge_u, edge_v
    # Both directed and undirected input graphs have identical semantics.
    angle = np.einsum('ij,ij->i', normals[u], normals[v])
    # Generated sealing fans can fold back with opposite normals on the same plane.
    smooth = abs(angle) >= cosine
    unknown = candidates[u] & candidates[v] & smooth
    boundary = (candidates[u] & (labels[v] > 0)) | (candidates[v] & (labels[u] > 0))
    inside = np.where(candidates[u[boundary]], u[boundary], v[boundary])
    outside = np.where(candidates[u[boundary]], v[boundary], u[boundary])
    delta = centers[inside] - centers[outside]
    same_plane = (abs(np.einsum('ij,ij->i', normals[inside], normals[outside])) >= cosine)
    same_plane &= abs(np.einsum('ij,ij->i', delta, normals[outside])) <= tolerance
    inside, outside = inside[same_plane], outside[same_plane]
    if not len(inside):
        return result, report
    # Nearest compatible boundary controls ownership, rather than vetoing a whole hole.
    rows = np.r_[u[unknown], inside]
    columns = np.r_[v[unknown], outside]
    adjacency = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T).tocoo()
    adjacency.data = np.maximum(np.linalg.norm(centers[adjacency.row] - centers[adjacency.col], axis=1), 1e-12)
    _, _, references = csgraph.dijkstra(adjacency.tocsr(), directed=False,
        indices=np.unique(outside), min_only=True, return_predecessors=True)
    reached = candidates & (references >= 0)
    safe_reference = np.maximum(references, 0)
    allowed = reached & seeds[safe_reference]
    report['competing_boundary_faces'] = int((reached & ~allowed).sum())
    faces = np.flatnonzero(allowed)
    ref = references[faces]
    parallel = abs(np.einsum('ij,ij->i', normals[faces], normals[ref])) >= cosine
    distances = np.einsum('fvi,fi->fv', vertices[triangles[faces]] - centers[ref, None, :], normals[ref])
    compatible = parallel & (abs(distances).max(axis=1) <= tolerance)
    filled = np.zeros(n, dtype=bool)
    filled[faces[compatible]] = True
    result[filled] = labels[references[filled]]
    report['filled_faces'] = int(filled.sum())
    if filled.any():
        references_filled = references[filled]
        delta = vertices[triangles[filled]] - centers[references_filled, None, :]
        report['max_plane_error'] = float(abs(np.einsum('fvi,fi->fv', delta, normals[references_filled])).max())
        report['max_normal_angle_deg'] = float(np.rad2deg(np.arccos(np.clip(
            abs(np.einsum('ij,ij->i', normals[filled], normals[references_filled])), 0, 1))).max())
    report['assigned_to'] = {str(cid): int(((result != labels) & (result == cid)).sum())
                             for cid in np.unique(result[filled])}
    assert np.array_equal(result[labels > 0], labels[labels > 0])
    return result, report


def complete_geometric_regions(labels, vertices, triangles, edge_u, edge_v, candidates, tolerance):
    """Assign each connected unknown region to its best supported neighboring surface.

    Fit the whole region using exact surface-area moments, including folded fans.
    Neighboring triangle planes vote by shared boundary length, discounted by
    their whole-region plane error and normal disagreement. Existing labels and
    ineligible unknown faces stay fixed; no class IDs or semantic roles are used.
    """
    n = len(labels)
    if (labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer)
            or vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.shape != (n, 3) or np.any(triangles < 0) or np.any(triangles >= len(vertices))
            or candidates.shape != (n,) or len(edge_u) != len(edge_v)
            or np.any(edge_u < 0) or np.any(edge_v < 0)
            or np.any(edge_u >= n) or np.any(edge_v >= n)
            or tolerance <= 0 or not np.isfinite(tolerance)):
        raise ValueError('Labels, mesh adjacency, masks and positive tolerance must agree')
    unknown = labels <= 0
    candidates = np.asarray(candidates, bool) & unknown
    result = labels.copy()
    report = {'filled_faces': 0, 'filled_regions': 0, 'regions_without_labeled_boundary': 0,
              'regions': []}
    if not candidates.any():
        return result, report
    faces = np.flatnonzero(unknown)
    index = np.full(n, -1, dtype=np.int32)
    index[faces] = np.arange(len(faces))
    u, v = edge_u, edge_v
    same = unknown[u] & unknown[v]
    adjacency = csr_matrix((np.ones(same.sum(), bool), (index[u[same]], index[v[same]])),
                           shape=(len(faces), len(faces)))
    count, groups = csgraph.connected_components(adjacency, directed=False)
    boundary = unknown[u] != unknown[v]
    inside = np.where(unknown[u[boundary]], u[boundary], v[boundary])
    outside = np.where(unknown[u[boundary]], v[boundary], u[boundary])
    # Deduplicate shared edges so directed/undirected graph inputs vote identically.
    edges = csr_matrix((np.ones(len(inside), bool), (inside, outside)), shape=(n, n)).tocoo()
    inside, outside = edges.row, edges.col
    shared = (triangles[inside, :, None] == triangles[outside, None, :]).any(axis=2)
    if np.any(shared.sum(axis=1) != 2):
        raise ValueError('Adjacent triangles must share exactly one edge')
    endpoints = vertices[triangles[inside][shared].reshape(-1, 2)]
    lengths = np.linalg.norm(endpoints[:, 1] - endpoints[:, 0], axis=1)
    neighbors = vertices[triangles[outside]]
    normals = np.cross(neighbors[:, 1] - neighbors[:, 0], neighbors[:, 2] - neighbors[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norm[:, None], np.finfo(float).tiny)
    centers = neighbors.mean(axis=1)
    boundary_groups = groups[index[inside]]
    face_order = np.argsort(groups, kind='stable')
    face_offsets = np.r_[0, np.cumsum(np.bincount(groups, minlength=count))]
    edge_order = np.argsort(boundary_groups, kind='stable')
    edge_offsets = np.r_[0, np.cumsum(np.bincount(boundary_groups, minlength=count))]
    for group in range(count):
        region_faces = faces[face_order[face_offsets[group]:face_offsets[group + 1]]]
        eligible = region_faces[candidates[region_faces]]
        if not len(eligible):
            continue
        border = edge_order[edge_offsets[group]:edge_offsets[group + 1]]
        if not len(border):
            report['regions_without_labeled_boundary'] += 1
            continue
        points = vertices[triangles[region_faces]]
        areas = .5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
        area = areas.sum()
        if area > 0:
            center = np.average(points.mean(axis=1), axis=0, weights=areas)
            delta = points - center
            summed = delta.sum(axis=1)
            covariance = (np.einsum('fvi,fvj,f->ij', delta, delta, areas)
                          + np.einsum('fi,fj,f->ij', summed, summed, areas)) / (12 * area)
            eigenvalues, axes = np.linalg.eigh(covariance)
            plane_normal = axes[:, 0]
            variance = max(float(eigenvalues[0]), 0.)
            normals_border = normals[border]
            offset = np.einsum('fi,fi->f', center - centers[border], normals_border)
            error = np.einsum('fi,ij,fj->f', normals_border, covariance, normals_border) + offset**2
            angle = np.clip(1 - (normals_border @ plane_normal)**2, 0, 1)
            loss = np.maximum(error - variance, 0) / (variance + tolerance**2)
            loss += angle / np.sin(np.deg2rad(20))**2
            loss[norm[border] == 0] = np.inf
            finite = np.isfinite(loss)
            if finite.any():
                # A common shift preserves votes while preventing exponential underflow.
                votes = lengths[border] * np.exp(-.5 * (loss - loss[finite].min()))
            else:
                votes = lengths[border]
        else:
            # Degenerate regions have no plane; use their actual shared contact.
            variance = 0.
            votes = lengths[border]
        owners, owner_index = np.unique(labels[outside[border]], return_inverse=True)
        support = np.bincount(owner_index, weights=votes, minlength=len(owners))
        winners = np.flatnonzero(support == support.max())
        # Exact geometric ties follow first contact, never numeric class order.
        first = [int(outside[border[owner_index == k]].min()) for k in winners]
        owner = owners[winners[np.argmin(first)]]
        result[eligible] = owner
        report['filled_regions'] += 1
        report['regions'].append({'faces': int(len(region_faces)), 'filled_faces': int(len(eligible)),
            'area': float(area), 'plane_rms': float(np.sqrt(variance)), 'owner': int(owner),
            'geometric_support': {str(cid): float(value) for cid, value in zip(owners, support)}})
    report['filled_faces'] = int((result != labels).sum())
    assert np.array_equal(result[~candidates], labels[~candidates])
    return result, report


def complete_bounded_gaps(labels, vertices, triangles, edge_u, edge_v, patches, candidates, seeds):
    """Inherit an enclosed generated patch's owner from its original boundary.

    Generated fans can fold, so their normals are not reliable ownership evidence.
    Require a closed patch, >=95% boundary length from one owner, and trusted seeds
    on >=50% of the boundary. Unknown or other owners count against agreement.
    """
    n = len(labels)
    if (labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer)
            or vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or triangles.shape != (n, 3) or np.any(triangles < 0) or np.any(triangles >= len(vertices))
            or candidates.shape != (n,) or patches.shape != (n,)
            or seeds.shape != (n,) or len(edge_u) != len(edge_v)
            or np.any(edge_u < 0) or np.any(edge_v < 0)
            or np.any(edge_u >= n) or np.any(edge_v >= n)):
        raise ValueError('Labels, mesh adjacency and masks must agree')
    patches = np.asarray(patches, bool)
    candidates = np.asarray(candidates, bool) & patches & (labels <= 0)
    seeds = np.asarray(seeds, bool) & (labels > 0)
    result = labels.copy()
    report = {'filled_faces': 0, 'filled_regions': 0}
    faces = np.flatnonzero(patches)
    owners = np.unique(labels[seeds])
    if not len(faces) or not len(owners):
        return result, report
    adjacency = csr_matrix((np.ones(len(edge_u)), (edge_u, edge_v)), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T)
    # A patch ending on an open mesh edge has no enclosing semantic boundary.
    degree = np.diff(adjacency.indptr)
    _, groups = csgraph.connected_components(adjacency[faces][:, faces], directed=False)
    count = groups.max() + 1
    closed = np.bincount(groups, weights=degree[faces] != 3, minlength=count) == 0
    region = np.full(n, -1, dtype=np.int32)
    region[faces] = groups
    u, v = edge_u, edge_v
    boundary = patches[u] != patches[v]
    inside = np.where(patches[u[boundary]], u[boundary], v[boundary])
    outside = np.where(patches[u[boundary]], v[boundary], u[boundary])
    shared = (triangles[inside, :, None] == triangles[outside, None, :]).any(axis=2)
    if np.any(shared.sum(axis=1) != 2):
        raise ValueError('Adjacent triangles must share exactly one edge')
    edge = vertices[triangles[inside][shared].reshape(-1, 2)]
    lengths = np.linalg.norm(edge[:, 0] - edge[:, 1], axis=1)
    boundary_region = region[inside]
    total = np.bincount(boundary_region, weights=lengths, minlength=count)
    for owner in owners:
        matching = labels[outside] == owner
        weight = np.bincount(boundary_region, weights=lengths * matching, minlength=count)
        trusted = np.bincount(boundary_region, weights=lengths * matching * seeds[outside], minlength=count)
        conflict = np.bincount(groups, weights=(labels[faces] > 0) & (labels[faces] != owner), minlength=count) > 0
        accepted = closed & ~conflict & (total > 0) & (weight >= .95 * total) & (trusted >= .5 * total)
        fill = accepted[groups] & candidates[faces]
        result[faces[fill]] = owner
        report['filled_regions'] += int(accepted.sum())
    report['filled_faces'] = int((result != labels).sum())
    assert np.array_equal(result[labels > 0], labels[labels > 0])
    return result, report
