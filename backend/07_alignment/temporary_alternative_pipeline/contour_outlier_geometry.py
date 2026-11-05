"""Exact surface-distance filtering for tiny isolated contour fragments."""
from __future__ import annotations

from collections import deque

import numpy as np
import trimesh
from scipy.spatial import cKDTree


CLUSTER_DISTANCE_M = 0.10
MAX_CLUSTER_SIZE = 4


def _point_segment_distance_squared(points: np.ndarray, starts: np.ndarray,
                                    ends: np.ndarray) -> np.ndarray:
    direction = ends - starts
    denominator = np.einsum("ij,ij->i", direction, direction)
    parameter = np.divide(
        np.einsum("ij,ij->i", points - starts, direction), denominator,
        out=np.zeros(len(points), dtype=np.float64), where=denominator > 1e-20,
    )
    closest = starts + np.clip(parameter, 0.0, 1.0)[:, None] * direction
    return np.einsum("ij,ij->i", points - closest, points - closest)


def _segment_pair_distance_squared(a0: np.ndarray, a1: np.ndarray,
                                   b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa = np.einsum("ij,ij->i", u, u)
    bb = np.einsum("ij,ij->i", u, v)
    cc = np.einsum("ij,ij->i", v, v)
    dd = np.einsum("ij,ij->i", u, w)
    ee = np.einsum("ij,ij->i", v, w)
    distance_squared = np.minimum.reduce((
        _point_segment_distance_squared(a0, b0, b1),
        _point_segment_distance_squared(a1, b0, b1),
        _point_segment_distance_squared(b0, a0, a1),
        _point_segment_distance_squared(b1, a0, a1),
    ))
    denominator = aa * cc - bb * bb
    valid = denominator > 1e-20
    s = np.divide(bb * ee - cc * dd, denominator,
                  out=np.zeros_like(denominator), where=valid)
    t = np.divide(aa * ee - bb * dd, denominator,
                  out=np.zeros_like(denominator), where=valid)
    interior = valid & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    between = w + s[:, None] * u - t[:, None] * v
    interior_distance = np.einsum("ij,ij->i", between, between)
    return np.where(interior, np.minimum(distance_squared, interior_distance), distance_squared)


def _segment_intersects_triangle(starts: np.ndarray, ends: np.ndarray,
                                 triangles: np.ndarray) -> np.ndarray:
    direction = ends - starts
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    valid = np.abs(determinant) > 1e-12
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
    tvec = starts - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", direction, qvec) * inverse
    t = np.einsum("ij,ij->i", edge2, qvec) * inverse
    return (valid & (u >= -1e-10) & (v >= -1e-10)
            & (u + v <= 1.0 + 1e-10) & (t >= -1e-10) & (t <= 1.0 + 1e-10))


def triangle_pair_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return exact surface distances for corresponding triangle pairs."""
    distances_squared = np.full(len(first), np.inf, dtype=np.float64)
    for corner in range(3):
        point = first[:, corner]
        closest = trimesh.triangles.closest_point(second, point)
        distances_squared = np.minimum(
            distances_squared,
            np.einsum("ij,ij->i", closest - point, closest - point),
        )
        point = second[:, corner]
        closest = trimesh.triangles.closest_point(first, point)
        distances_squared = np.minimum(
            distances_squared,
            np.einsum("ij,ij->i", closest - point, closest - point),
        )

    edges = ((0, 1), (1, 2), (2, 0))
    intersections = np.zeros(len(first), dtype=bool)
    for first_edge in edges:
        a0, a1 = first[:, first_edge[0]], first[:, first_edge[1]]
        for second_edge in edges:
            distances_squared = np.minimum(
                distances_squared,
                _segment_pair_distance_squared(
                    a0, a1, second[:, second_edge[0]], second[:, second_edge[1]]
                ),
            )
        intersections |= _segment_intersects_triangle(a0, a1, second)
    for second_edge in edges:
        intersections |= _segment_intersects_triangle(
            second[:, second_edge[0]], second[:, second_edge[1]], first
        )
    distances_squared[intersections] = 0.0
    return np.sqrt(np.maximum(distances_squared, 0.0))


def nearby_triangle_pairs(triangles: np.ndarray, threshold: float) -> np.ndarray:
    if len(triangles) < 2:
        return np.empty((0, 2), dtype=np.int64)
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None, :], axis=2).max(axis=1)
    mins, maxs = triangles.min(axis=1), triangles.max(axis=1)
    neighbors = cKDTree(centers).query_ball_point(
        centers, radii + radii.max() + threshold
    )
    pairs = []
    for first_index, candidates in enumerate(neighbors):
        candidates = np.asarray([j for j in candidates if j > first_index], dtype=np.int64)
        if not len(candidates):
            continue
        gaps = np.maximum(
            0.0,
            np.maximum(mins[first_index] - maxs[candidates],
                       mins[candidates] - maxs[first_index]),
        )
        keep = np.einsum("ij,ij->i", gaps, gaps) < threshold * threshold
        pairs.extend((first_index, int(j)) for j in candidates[keep])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def isolated_small_groups(triangles: np.ndarray, threshold: float = CLUSTER_DISTANCE_M
                          ) -> tuple[np.ndarray, list[list[int]]]:
    """Find isolated red components of 1–4 mutually close triangles."""
    pairs = nearby_triangle_pairs(triangles, threshold)
    if len(pairs):
        distances = triangle_pair_distances(triangles[pairs[:, 0]], triangles[pairs[:, 1]])
        pairs = pairs[distances < threshold]
    adjacency = [[] for _ in triangles]
    for first, second in pairs:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))

    removed = np.zeros(len(triangles), dtype=bool)
    groups, visited = [], np.zeros(len(triangles), dtype=bool)
    close_set = {tuple(pair) for pair in pairs.tolist()}
    for start in range(len(triangles)):
        if visited[start]:
            continue
        queue, component = deque([start]), []
        visited[start] = True
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        if len(component) > MAX_CLUSTER_SIZE:
            continue
        all_pairs_close = all(
            (min(first, second), max(first, second)) in close_set
            for index, first in enumerate(component)
            for second in component[index + 1:]
        )
        has_external_neighbor = any(
            neighbor not in component
            for member in component for neighbor in adjacency[member]
        )
        if all_pairs_close and not has_external_neighbor:
            assert len(component) < 5 and not has_external_neighbor
            removed[component] = True
            groups.append(component)
    return removed, groups


def check_isolated_small_groups() -> None:
    base = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    assert np.allclose(triangle_pair_distances(base, base + [0, 0, 0.05]), [0.05])
    assert np.allclose(triangle_pair_distances(base, base + [0, 0, 0.2]), [0.2])

    def triangle(x: float) -> np.ndarray:
        return np.array([[x, 0, 0], [x + 0.001, 0, 0], [x, 0.001, 0]])

    samples = np.array([triangle(x) for x in (
        0, 0.02, 0.04, 0.06, 1, 1.01, 1.02, 1.03, 1.04, 2, 2.05, 2.12, 3
    )])
    removed, groups = isolated_small_groups(samples)
    assert [len(group) for group in groups] == [4, 1]
    assert np.flatnonzero(removed).tolist() == [0, 1, 2, 3, 12]
