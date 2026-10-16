"""Exclude horizontal triangles from caller-supplied wall classes.

Wall identities and gravity are caller supplied; geometry and label IDs are opaque.
Only horizontal wall faces can change. Mesh vertices and topology are never edited.
"""
import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial import cKDTree

from detect_structural_classes import build_mesh_dual_graph
from refine_class_boundaries import compute_mesh_derivatives


def _components(mask, u, v):
    faces = np.flatnonzero(mask)
    index = np.full(len(mask), -1, dtype=np.int32)
    index[faces] = np.arange(len(faces))
    keep = mask[u] & mask[v]
    graph = csr_matrix((np.ones(keep.sum()), (index[u[keep]], index[v[keep]])),
                       shape=(len(faces), len(faces)))
    _, groups = csgraph.connected_components(graph, directed=False)
    return faces, groups


def _nearest_nonwall(labels, eligible, centers, u, v, query):
    boundary = eligible[u] != eligible[v]
    seeds = np.unique(np.where(eligible[u[boundary]], u[boundary], v[boundary]))
    lengths = np.maximum(np.linalg.norm(centers[u] - centers[v], axis=1), 1e-12)
    graph = csr_matrix((np.r_[lengths, lengths], (np.r_[u, v], np.r_[v, u])),
                       shape=(len(labels), len(labels)))
    distance, _, source = csgraph.dijkstra(graph, directed=False, indices=seeds,
                                          min_only=True, return_predecessors=True)
    distance, source = distance[query], source[query]
    disconnected = source < 0
    if disconnected.any():
        candidates = np.flatnonzero(eligible)
        d, nearest = cKDTree(centers[candidates]).query(centers[query[disconnected]], workers=1)
        distance[disconnected], source[disconnected] = d, candidates[nearest]
    return labels[source], distance, int(disconnected.sum())


def refine_wall_orientations(mesh, labels, priors, source_ids, wall_ids, up_axis, tolerance):
    """Return a separate labeling and audit report; tolerance uses mesh length units."""
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    labels, source_ids = np.asarray(labels), np.asarray(source_ids)
    up = np.array(up_axis, dtype=float, copy=True)
    if (labels.shape != (len(triangles),) or priors.shape != (len(labels), len(source_ids))
            or up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Mesh, labels, priors, gravity and positive tolerance must agree')
    up /= np.linalg.norm(up)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.zeros_like(vertices)
    areas, normals, centers, rgb = compute_mesh_derivatives(vertices, triangles, colors)
    u, v = build_mesh_dual_graph(triangles)
    unique = u < v
    u, v = u[unique], v[unique]
    wall = np.isin(labels, wall_ids)
    horizontal = abs(normals @ up) >= np.sqrt(.5)
    rejected = wall & horizontal & (areas > 0)
    result = labels.copy()
    report = {'horizontal_wall_faces': int(rejected.sum()), 'changed_faces': 0, 'unknown_faces': 0}
    if not rejected.any():
        return result, report
    eligible = (labels > 0) & ~wall
    if not eligible.any():
        raise ValueError('Horizontal walls require at least one existing non-wall owner')
    # Unobserved leftover specks must not grow into new visible object instances.
    own_quality = np.zeros(len(labels))
    for column, owner in enumerate(source_ids):
        rows = np.flatnonzero(eligible & (labels == owner))
        own_quality[rows] = priors[rows, column].toarray().ravel()
    total_quality = np.asarray(priors.sum(axis=1)).ravel()
    anchor = eligible & (own_quality >= 3) & (own_quality >= .75 * total_quality)
    same = labels[u] == labels[v]
    owner_faces, owner_groups = _components(eligible, u[same], v[same])
    corroborated = np.bincount(owner_groups, weights=anchor[owner_faces]) > 0
    if corroborated.any():
        eligible[owner_faces] = corroborated[owner_groups]
    changed, components = _components(rejected, u, v)
    # Freeze every other owner; local reassignment cannot erode accepted objects.
    boundary = rejected[u] != rejected[v]
    inside = np.where(rejected[u[boundary]], u[boundary], v[boundary])
    outside = np.where(rejected[u[boundary]], v[boundary], u[boundary])
    valid = eligible[outside]
    inside, outside = inside[valid], outside[valid]
    common = (triangles[inside, :, None] == triangles[outside, None, :]).any(axis=2)
    edge = vertices[triangles[inside][common].reshape(-1, 2)]
    lengths = np.linalg.norm(edge[:, 1] - edge[:, 0], axis=1)
    angle = np.arccos(np.clip(np.einsum('ij,ij->i', normals[inside], normals[outside]), -1, 1))
    affinity = np.exp(-.5 * (angle / np.deg2rad(25)) ** 2
                      - .5 * (np.linalg.norm(rgb[inside] - rgb[outside], axis=1) / .035) ** 2)
    contact_reward = tolerance * lengths * affinity
    index = np.full(len(labels), -1, dtype=np.int32)
    index[changed] = components
    routed = None
    fallback_route_faces = disconnected_faces = 0
    for component in range(components.max() + 1 if len(components) else 0):
        patch = changed[components == component]
        neighbors = outside[index[inside] == component]
        mass = np.asarray(priors[patch].sum(axis=1)).ravel()
        # First face order, rather than numeric identity, resolves exact ties.
        owners = list(dict.fromkeys(labels[np.sort(np.unique(neighbors))].tolist()))
        if not owners:
            if routed is None:
                routed, route_distance, disconnected_faces = _nearest_nonwall(
                    labels, eligible, centers, u, v, changed)
            selection = components == component
            votes = {}
            for owner, weight in zip(routed[selection], areas[patch] / (route_distance[selection] + tolerance)):
                votes[owner] = votes.get(owner, 0.) + weight
            result[patch] = max(votes, key=votes.get)
            fallback_route_faces += len(patch)
            continue
        scores = []
        for owner in owners:
            column = np.flatnonzero(source_ids == owner)
            own = np.asarray(priors[patch, column[0]].toarray()).ravel() if len(column) else np.zeros(len(patch))
            unary = -np.log((own + 1 / max(len(source_ids), 1)) / (mass + 1))
            contact = (index[inside] == component) & (labels[outside] == owner)
            scores.append(float(areas[patch] @ unary - contact_reward[contact].sum()))
        result[patch] = owners[int(np.argmin(scores))]
    report.update(changed_faces=int((result != labels).sum()),
                  unknown_faces=int(((result != labels) & (result <= 0)).sum()),
                  assigned_to={str(i): int(((result != labels) & (result == i)).sum())
                               for i in np.unique(result[result != labels])},
                  fallback_route_faces=fallback_route_faces, disconnected_route_faces=disconnected_faces,
                  corroborated_owner_faces=int(eligible.sum()) if corroborated.any() else 0)
    assert np.all(result[rejected] > 0) and not np.isin(result[rejected], wall_ids).any()
    return result, report
