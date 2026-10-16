"""Numerical building blocks for frozen-mask 3D reconstruction (no model inference)."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def boundary_distance(labels):
    import cv2
    boundary = np.zeros(labels.shape, bool)
    boundary[1:] |= labels[1:] != labels[:-1]
    boundary[:-1] |= labels[1:] != labels[:-1]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    return cv2.distanceTransform((~boundary).astype('uint8'), cv2.DIST_L2, 5)


def observation_components(mask, split=False):
    """Interpret reviewed multi-object masks; never split unreviewed disconnected objects."""
    if not split:
        return [(0, mask.astype(bool))]
    import cv2
    count, components = cv2.connectedComponents(mask.astype('uint8'), connectivity=8)
    if count <= 1:
        return [(0, mask.astype(bool))]
    return [(i, components == i) for i in range(1, count)]


def cached_mask_extent(mask, cached, tolerance):
    """Conservative depth extent for abstention, never geometry creation or deletion."""
    valid = mask.astype(bool) & np.isfinite(cached) & (cached > 0)
    fraction = np.count_nonzero(valid) / max(1, np.count_nonzero(mask))
    if fraction < .5:
        return None
    low, high = np.percentile(cached[valid].astype('float32'), [.5, 99.5])
    return dict(lower_m=max(0., float(low)-tolerance), upper_m=float(high)+tolerance,
                valid_fraction=fraction, quantiles=[.5, 99.5], padding_m=tolerance)


def extent_conflicts(depth, state, observation, extents):
    lower = np.full(max(extents, default=0)+1, -np.inf)
    upper = np.full(len(lower), np.inf)
    for oid, extent in extents.items():
        lower[oid], upper[oid] = extent['lower_m'], extent['upper_m']
    valid = (observation >= 0) & (observation < len(lower)) & (state == 2)
    result = np.zeros(depth.shape, bool)
    result[valid] = ((depth[valid] < lower[observation[valid]]) |
                     (depth[valid] > upper[observation[valid]]))
    return result


def view_groups(cameras, translation=.15, angle_degrees=12):
    """Group redundant camera poses; each group contributes one effective view."""
    representatives, groups = [], []
    for camera in cameras:
        v = camera['viewmat']
        center, rotation = -v[:3, :3].T @ v[:3, 3], v[:3, :3]
        group = len(representatives)
        for i, (c, r) in enumerate(representatives):
            angle = np.arccos(np.clip((np.trace(rotation @ r.T) - 1) / 2, -1, 1))
            if np.linalg.norm(center - c) < translation and angle < np.radians(angle_degrees):
                group = i
                break
        if group == len(representatives):
            representatives.append((center, rotation))
        groups.append(group)
    return np.asarray(groups, np.int32)


def resolve_tracks(support, observations, cannot_link, min_overlap=.35, min_similarity=.75,
                   reviewed_pairs=(), rejected_pairs=(), disconnected_pairs=(), feature_support=None):
    """Constrained union from shared visible faces/appearance or cached local feature tracks."""
    support = support.tocsr()
    binary = support.copy()
    binary.data[:] = 1
    intersections = sparse.triu(binary @ binary.T, k=1).tocoo()
    sizes = np.asarray(binary.sum(axis=1)).ravel()
    parents = np.arange(len(observations))
    members = {i: {i} for i in parents}
    blocked = {i: set() for i in parents}
    for a, b in cannot_link:
        blocked[a].add(b)
        blocked[b].add(a)
    rejected = {tuple(sorted(pair['source_ids'])) for pair in rejected_pairs}
    for a, oa in enumerate(observations):
        for b, ob in enumerate(observations[:a]):
            if tuple(sorted((oa['source_id'], ob['source_id']))) in rejected:
                blocked[a].add(b)
                blocked[b].add(a)

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return int(i)

    candidates = []
    def anchored(spec):
        matches = [i for i, o in enumerate(observations) if all(o.get(k, 0) == v for k, v in spec.items())]
        if len(matches) != 1:
            raise ValueError(f'review observation must identify one mask: {spec}')
        return matches[0]

    reviewed = {}
    for pair in reviewed_pairs:
        a, b = (anchored(spec) for spec in pair['observations'])
        reviewed[tuple(sorted((a, b)))] = pair
    for a, b, count in zip(intersections.row, intersections.col, intersections.data):
        if observations[a]['frame'] == observations[b]['frame']:
            continue
        overlap = float(count / max(1, min(sizes[a], sizes[b])))
        similarity = float(np.dot(observations[a]['embedding'], observations[b]['embedding']))
        pair = (int(a), int(b))
        thresholds = reviewed.get(pair, {})
        # Frozen exports contain one prototype per source ID; visually reviewed aliases still
        # require strong geometric overlap and cached appearance rather than review alone.
        accepted = ((overlap >= min_overlap and similarity >= min_similarity) or
                    (pair in reviewed and overlap >= thresholds.get('min_overlap', .5) and
                     similarity >= thresholds.get('min_similarity', .65)))
        if count >= 3 and accepted:
            candidates.append((overlap, similarity, int(a), int(b), int(count)))
    # Explicit RGB/shape review can establish one object across disjoint mesh pieces.
    # Unlike ordinary aliases, these records must state why shared faces are unavailable.
    for pair in disconnected_pairs:
        if not pair.get('reason'):
            raise ValueError('disconnected identity review needs evidence')
        a, b = (anchored(spec) for spec in pair['observations'])
        similarity = float(np.dot(observations[a]['embedding'], observations[b]['embedding']))
        if similarity >= pair.get('min_similarity', .75):
            candidates.append((1., similarity, a, b, 0))
    passes = [('mesh', sorted(candidates, reverse=True))]
    if feature_support is not None:
        features = feature_support.tocsr(copy=True)
        if (features.shape[0] != len(observations) or not np.isfinite(features.data).all()
                or np.any(features.data < 0)):
            raise ValueError('invalid cached feature-track support')
        features.eliminate_zeros()
        features.sum_duplicates()
        features.data = np.ones(features.nnz, np.int32)
        counts = sparse.triu(features @ features.T, k=1).tocoo()
        sizes = np.asarray(features.sum(axis=1)).ravel()
        candidates = []
        for a, b, count in zip(counts.row, counts.col, counts.data):
            overlap = float(count / max(1, min(sizes[a], sizes[b])))
            if count >= 3 and overlap >= min_overlap and observations[a]['frame'] != observations[b]['frame']:
                similarity = float(np.dot(observations[a]['embedding'], observations[b]['embedding']))
                candidates.append((overlap, similarity, int(a), int(b), int(count)))
        # Cached local feature matches supply appearance evidence independently of global source prototypes.
        # Apply them after mesh grouping so established co-visible objects constrain mixed masks.
        passes.append(('feature_tracks', sorted(candidates, key=lambda p: (p[4], p[0]), reverse=True)))
    decisions = []
    for evidence, candidates in passes:
        for overlap, similarity, a, b, count in candidates:
            ra, rb = root(a), root(b)
            if ra == rb:
                continue
            conflict = any(blocked[i] & members[rb] for i in members[ra])
            decisions.append(dict(a=a, b=b, overlap=overlap, similarity=similarity, evidence=evidence,
                shared_faces=count if evidence == 'mesh' else 0,
                shared_feature_tracks=count if evidence == 'feature_tracks' else 0,
                decision='separate' if conflict else 'merge'))
            if conflict:
                continue
            keep, drop = min(ra, rb), max(ra, rb)
            parents[drop] = keep
            members[keep] |= members.pop(drop)
    roots = np.array([root(i) for i in range(len(observations))])
    return roots, decisions


def physical_ids(instances):
    """One object identity with separately addressable reviewed parts; reject cyclic reviews."""
    parents = {e['instance_id']: e.get('parent_instance_id') for e in instances}
    result = np.zeros(max(parents, default=0) + 1, np.int32)
    for iid in parents:
        root, seen = iid, set()
        while parents[root] is not None:
            if root in seen or parents[root] not in parents:
                raise ValueError('invalid physical-object part hierarchy')
            seen.add(root)
            root = parents[root]
        result[iid] = root
    return result


def object_source_aliases(instances, maximum_source):
    """Aliases belong to an established object, never to every use of a source ID."""
    result = np.tile(np.arange(maximum_source + 1, dtype=np.int32), (len(instances) + 1, 1))
    for entry in instances:
        aliases = entry['source_ids']
        if len(aliases) > 1:
            result[entry.get('physical_instance_id', entry['instance_id']), aliases] = min(aliases)
    return result


def dual_graph(triangles):
    """Manifold shared-edge adjacency; vertex-only contacts do not join surfaces."""
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    faces = np.repeat(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e, f = edges[order], faces[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    a, b = f[:-1][same], f[1:][same]
    return sparse.coo_matrix((np.ones(2 * len(a), bool),
                             (np.r_[a, b], np.r_[b, a])),
                            shape=(len(triangles), len(triangles))).tocsr()


def assign_faces(votes, graph, normals, rgb, strength=.15, iterations=4):
    """Evidence-constrained Potts updates. Never introduce a label with zero local support."""
    votes = votes.tocsr()
    total = np.asarray(votes.sum(axis=1)).ravel()
    probabilities = votes.multiply(1 / np.maximum(total, 1e-12)[:, None]).tocsr()
    labels = np.asarray(votes.argmax(axis=1)).ravel().astype(np.int32)
    labels[total == 0] = 0
    confidence = np.asarray(probabilities.max(axis=1).toarray()).ravel()
    if not strength:
        return labels, confidence
    a, b = graph.nonzero()
    dot = np.sum(normals[a] * normals[b], axis=1).clip(-1, 1)
    crease = np.exp(-np.arccos(dot)**2 / (2 * np.radians(25)**2))
    appearance = np.exp(-np.sum((rgb[a] - rgb[b])**2, axis=1) / (2 * .12**2))
    weights = sparse.csr_matrix((crease * appearance, (a, b)), shape=graph.shape)
    normalization = np.maximum(np.asarray(weights.sum(axis=1)).ravel(), 1e-12)
    for _ in range(iterations):
        onehot = sparse.csr_matrix((np.ones(len(labels)), (np.arange(len(labels)), labels)),
                                   shape=votes.shape)
        agreement = (weights @ onehot).multiply(1 / normalization[:, None])
        # Restrict smoothing to labels actually supported by this face's observations.
        score = probabilities + strength * agreement.multiply(votes.sign())
        proposed = np.asarray(score.argmax(axis=1)).ravel().astype(np.int32)
        proposed[(total == 0) | (confidence >= .9)] = labels[(total == 0) | (confidence >= .9)]
        if np.array_equal(proposed, labels):
            break
        labels = proposed
    confidence = np.asarray(probabilities[np.arange(len(labels)), labels]).ravel()
    return labels, confidence


def balance_votes(votes, face_areas, exponent):
    """Temper large-surface priors using measured uncontested area, not semantic ID guesses."""
    votes = votes.tocsr()
    total = np.asarray(votes.sum(axis=1)).ravel()
    maximum = np.asarray(votes.max(axis=1).toarray()).ravel()
    core = (total > 0) & (maximum >= .9 * total)
    winners = np.asarray(votes.argmax(axis=1)).ravel()
    areas = np.bincount(winners[core], weights=face_areas[core], minlength=votes.shape[1])
    # A single poorly sampled face must not give an unlimited rare-label boost.
    floor = np.median(face_areas[face_areas > 0]) * 50
    factors = np.maximum(areas, floor)**(-exponent)
    factors[0] = 1
    return votes.multiply(factors).tocsr(), areas


def restrict_instance_votes(votes, centroids, instance, lower, upper):
    """Withhold one reviewed object's off-object votes; preserve competing labels and geometry."""
    votes = votes.copy().tocsr()
    positions = np.flatnonzero(votes.indices == instance)
    faces = np.searchsorted(votes.indptr, positions, side='right') - 1
    outside = np.any((centroids[faces] < lower) | (centroids[faces] > upper), axis=1)
    votes.data[positions[outside]] = 0
    votes.eliminate_zeros()
    return votes, faces[outside]


def safe_removal(triangles, graph, supported, free_space_views, conflict_views, min_views=3):
    """Remove whole unsupported components only with repeated calibrated free-space evidence."""
    n, components = connected_components(graph, directed=False)
    sizes = np.bincount(components, minlength=n)
    observed = np.bincount(components, weights=supported.astype(bool), minlength=n)
    free = np.bincount(components, weights=free_space_views >= min_views, minlength=n)
    conflict = np.bincount(components, weights=conflict_views > 0, minlength=n)
    eligible = (observed == 0) & (free == sizes) & (conflict == 0)
    removed = eligible[components]
    # Shared vertices can still split a pinched surface; keep those candidates for review.
    kept_vertices = np.unique(triangles[~removed])
    changed = True
    while changed:
        changed = False
        for component in np.flatnonzero(eligible):
            faces = components == component
            if removed[faces].any():
                vertices = np.unique(triangles[faces])
                if np.intersect1d(vertices, kept_vertices).size:
                    removed[faces] = False
                    kept_vertices = np.union1d(kept_vertices, vertices)
                    changed = True
    return removed, components


def boundary_edges(triangles):
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def object_metrics(truth, prediction, visible, ids):
    """Missing rays remain recall failures; unknown reference pixels are excluded."""
    import cv2
    valid = truth > 0
    trusted_boundary = cv2.erode(valid.astype('uint8'), np.ones((5, 5), 'uint8')).astype(bool)
    output = {}
    for oid in ids:
        t, p = (truth == oid) & valid, (prediction == oid) & valid
        tp = int(np.count_nonzero(t & p & visible))
        fn = int(np.count_nonzero(t)) - tp
        fp = int(np.count_nonzero(p & ~t & visible))
        tb = t & trusted_boundary & (boundary_distance(t.astype('int32')) <= 1)
        pb = p & visible & trusted_boundary & (boundary_distance(p.astype('int32')) <= 1)
        near_t = cv2.dilate(tb.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
        near_p = cv2.dilate(pb.astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
        output[int(oid)] = dict(tp=tp, fn=fn, fp=fp, pixels=int(t.sum()),
                               missing=int(np.count_nonzero(t & ~visible)),
                               boundary_truth=int(tb.sum()), boundary_pred=int(pb.sum()),
                               boundary_recalled=int(np.count_nonzero(tb & near_p)),
                               boundary_precise=int(np.count_nonzero(pb & near_t)))
    return output


def summarize_metrics(counts):
    result = {}
    for oid, c in counts.items():
        recall = c['tp'] / max(1, c['tp'] + c['fn'])
        precision = c['tp'] / max(1, c['tp'] + c['fp'])
        br = c['boundary_recalled'] / max(1, c['boundary_truth'])
        bp = c['boundary_precise'] / max(1, c['boundary_pred'])
        result[oid] = dict(**c, recall=recall, iou=c['tp'] / max(1, c['tp'] + c['fn'] + c['fp']),
                           leakage=1 - precision, boundary_f1=2 * br * bp / max(1e-12, br + bp),
                           missing_geometry_fraction=c['missing'] / max(1, c['pixels']))
    return result
