"""Reclaim weak labels where trusted surface geometry supports another owner."""
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial import cKDTree


def reclaim_compatible_surfaces(labels, vertices, triangles, normals, centers, colors,
                                 areas, edge_u, edge_v, candidates, seeds, tolerance):
    """Fit coherent candidate patches and match original reference surfaces.

    Callers select mutable faces and protect corroborated objects. Fitted planes
    tolerate local roughness; existing reference triangles must support that plane.
    """
    n = len(labels)
    if (labels.shape != (n,) or triangles.shape != (n, 3) or normals.shape != (n, 3)
            or centers.shape != (n, 3) or colors.shape != (n, 3) or areas.shape != (n,)
            or candidates.shape != (n,) or seeds.shape != (n,) or len(edge_u) != len(edge_v)
            or not np.issubdtype(labels.dtype, np.integer)
            or tolerance <= 0 or not np.isfinite(tolerance)):
        raise ValueError('Labels, geometry, masks and positive tolerance must agree')
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or not np.isfinite(normals).all() or not np.isfinite(centers).all()
            or not np.isfinite(colors).all() or not np.isfinite(areas).all() or np.any(areas < 0)
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))
            or np.any(edge_u < 0) or np.any(edge_v < 0)
            or np.any(edge_u >= n) or np.any(edge_v >= n)):
        raise ValueError('Invalid surface geometry or adjacency')
    result = labels.copy()
    mutable = np.asarray(candidates, bool) & (labels > 0) & (areas > 0)
    faces = np.flatnonzero(mutable)
    source = np.flatnonzero(np.asarray(seeds, bool) & (labels > 0) & (areas > 0))
    report = {'changed_faces': 0, 'supported_patches': 0}
    if not len(faces) or not len(source):
        return result, report
    cosine = np.cos(np.deg2rad(20))
    u, v = edge_u, edge_v
    same = mutable[u] & mutable[v] & (labels[u] == labels[v])
    same &= np.einsum('ij,ij->i', normals[u], normals[v]) >= cosine
    same &= np.linalg.norm(colors[u] - colors[v], axis=1) <= .035
    index = np.full(n, -1, dtype=np.int32)
    index[faces] = np.arange(len(faces))
    _, groups = csgraph.connected_components(csr_matrix((np.ones(same.sum()),
        (index[u[same]], index[v[same]])), shape=(len(faces), len(faces))), directed=False)
    order = np.argsort(groups, kind='stable')
    patches = np.split(faces[order], np.flatnonzero(np.diff(groups[order])) + 1)
    tree = cKDTree(centers[source])
    max_residual = 0.
    for patch in patches:
        points = vertices[triangles[patch]].reshape(-1, 3)
        weights = np.repeat(areas[patch] / 3, 3)
        center = np.average(points, axis=0, weights=weights)
        offset = points - center
        values, axes = np.linalg.eigh((offset * weights[:, None]).T @ offset / weights.sum())
        residual = float(np.sqrt(max(values[0], 0.)))
        if residual > tolerance or values[1] <= max(values[2], 1e-30) * 1e-12:
            continue
        normal = axes[:, 0]
        if normal @ np.average(normals[patch], axis=0, weights=areas[patch]) < 0:
            normal = -normal
        distance, nearest = tree.query(center, k=min(128, len(source)), workers=1)
        distance, reference = np.atleast_1d(distance), source[np.atleast_1d(nearest)]
        source_radius = np.linalg.norm(vertices[triangles[reference]] - centers[reference, None, :], axis=2).max(axis=1)
        near = distance <= np.linalg.norm(offset, axis=1).max() + source_radius + 4 * tolerance
        reference, distance = reference[near], distance[near]
        if not len(reference):
            continue
        compatible = normals[reference] @ normal >= cosine
        compatible &= abs((vertices[triangles[reference]] - center) @ normal).max(axis=1) <= tolerance
        # Compare local colors, so illumination gradients do not become object boundaries.
        _, closest = cKDTree(centers[patch]).query(centers[reference], workers=1)
        compatible &= np.linalg.norm(colors[reference] - colors[patch[closest]], axis=1) <= .035
        if not compatible.any():
            continue
        best = np.argmin(np.where(compatible, distance, np.inf))
        roughness = abs(offset @ normal).reshape(-1, 3).max(axis=1) <= 2 * tolerance
        result[patch[roughness]] = labels[reference[best]]
        report['supported_patches'] += 1
        max_residual = max(max_residual, residual)
    changed = result != labels
    report.update(changed_faces=int(changed.sum()), max_fitted_plane_rms=max_residual,
                  changes_from={str(cid): int((changed & (labels == cid)).sum())
                                for cid in np.unique(labels[changed])})
    assert np.array_equal(result[~np.asarray(candidates, bool)], labels[~np.asarray(candidates, bool)])
    return result, report
