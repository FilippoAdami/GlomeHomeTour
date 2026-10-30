#!/usr/bin/env python3
"""Color edge-connected wall regions with area-weighted, multi-seed growth."""
from __future__ import annotations

import argparse
import colorsys
import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np
import open3d as o3d

NORMAL_ANGLE_DEG = 35.0
MAX_PLANE_DISTANCE = 0.15
MIN_REGION_TRIANGLES = 10000
DEFAULT_SEED_COUNT = 128
DEFAULT_RANDOM_SEED = 42


def _wall_mesh_paths(instances_dir: Path) -> list[Path]:
    manifest_path = instances_dir / "structural_instances_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Structural instance manifest is missing: {manifest_path}")
    paths = []
    for item in json.loads(manifest_path.read_text()).get("instances", []):
        if str(item.get("role", "")).lower() != "wall":
            continue
        path = Path(item.get("file_path", item.get("file_name", "")))
        path = path if path.is_file() else instances_dir / path.name
        if not path.is_file():
            raise ValueError(f"Wall mesh listed in manifest is missing: {path}")
        paths.append(path)
    if not paths:
        raise ValueError(f"No wall meshes listed in {manifest_path}")
    return paths


def _triangle_adjacency(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build CSR face adjacency from shared edges, including non-manifold edges."""
    face_ids = np.repeat(np.arange(len(triangles), dtype=np.int32), 3)
    edges = triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2).copy()
    edges.sort(axis=1)
    order = np.lexsort((face_ids, edges[:, 1], edges[:, 0]))
    edges, face_ids = edges[order], face_ids[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(edges)]

    group_sizes = ends - starts
    manifold = np.flatnonzero(group_sizes == 2)
    first = face_ids[starts[manifold]]
    second = face_ids[starts[manifold] + 1]
    sources = [first, second]
    destinations = [second, first]
    for group in np.flatnonzero(group_sizes > 2):
        members = face_ids[starts[group]:ends[group]]
        others = members[1:]
        sources.extend((np.full(len(others), members[0], dtype=np.int32), others))
        destinations.extend((others, np.full(len(others), members[0], dtype=np.int32)))
    if not len(first) and len(sources) == 2:
        return np.zeros(len(triangles) + 1, dtype=np.int64), np.empty(0, dtype=np.int32)

    source = np.concatenate(sources)
    destination = np.concatenate(destinations)
    order = np.argsort(source, kind="stable")
    counts = np.bincount(source, minlength=len(triangles))
    offsets = np.r_[0, np.cumsum(counts)]
    return offsets, destination[order]


def _grow_regions(
    vertices: np.ndarray,
    triangles: np.ndarray,
    seed_count: int = DEFAULT_SEED_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a reproducible multi-seed grow and merge over edge-adjacent triangles."""
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    normals = np.zeros_like(cross)
    normals[valid] = cross[valid] / double_area[valid, None]
    areas = double_area * 0.5
    centroids = corners.mean(axis=1)
    offsets, neighbors = _triangle_adjacency(triangles)
    labels = np.full(len(triangles), -1, dtype=np.int32)
    cosine_limit = np.cos(np.deg2rad(NORMAL_ANGLE_DEG))
    parent: list[int] = []
    normal_sums: list[np.ndarray] = []
    point_sums: list[np.ndarray] = []
    area_sums: list[float] = []
    frontier: deque[tuple[int, int]] = deque()

    def find(region: int) -> int:
        while parent[region] != region:
            parent[region] = parent[parent[region]]
            region = parent[region]
        return region

    def add_seed(face: int) -> None:
        region = len(parent)
        area = float(areas[face])
        parent.append(region)
        normal_sums.append(normals[face] * area)
        point_sums.append(centroids[face] * area)
        area_sums.append(area)
        labels[face] = region
        frontier.append((region, face))

    def plane(region: int) -> tuple[np.ndarray, np.ndarray]:
        region = find(region)
        normal = normal_sums[region]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        return normal, point_sums[region] / area_sums[region]

    def similarity(first: int, second: int) -> tuple[float, float] | None:
        first, second = find(first), find(second)
        if first == second or area_sums[first] <= 0.0 or area_sums[second] <= 0.0:
            return None
        normal_a, point_a = plane(first)
        normal_b, point_b = plane(second)
        cosine = min(1.0, abs(float(normal_a @ normal_b)))
        angle = float(np.rad2deg(np.arccos(cosine)))
        delta = point_b - point_a
        # shortcut: angled infinite planes lack one unique offset; use local correspondences if merges remain unreliable.
        distance = min(abs(float(normal_a @ delta)), abs(float(normal_b @ delta)))
        if distance > MAX_PLANE_DISTANCE:
            return None
        return angle, distance

    def merge(first: int, second: int, require_normal_match: bool = True) -> int | None:
        first, second = find(first), find(second)
        if first == second:
            return first
        fit = similarity(first, second)
        if fit is None or (require_normal_match and fit[0] > NORMAL_ANGLE_DEG):
            return None
        if area_sums[first] < area_sums[second]:
            first, second = second, first
        if float(normal_sums[first] @ normal_sums[second]) < 0.0:
            normal_sums[second] = -normal_sums[second]
        parent[second] = first
        normal_sums[first] += normal_sums[second]
        point_sums[first] += point_sums[second]
        area_sums[first] += area_sums[second]
        return first

    def add_face(region: int, face: int) -> None:
        region = find(region)
        region_normal, _ = plane(region)
        normal = normals[face]
        if float(normal @ region_normal) < 0.0:
            normal = -normal
        area = float(areas[face])
        labels[face] = region
        normal_sums[region] += normal * area
        point_sums[region] += centroids[face] * area
        area_sums[region] += area
        frontier.append((region, face))

    rng = np.random.default_rng(random_seed)
    valid_faces = np.flatnonzero(valid)
    if len(valid_faces):
        seed_count = min(max(1, seed_count), len(valid_faces))
        weights = areas[valid_faces]
        seeds = rng.choice(valid_faces, size=seed_count, replace=False, p=weights / weights.sum())
        for seed in seeds:
            add_seed(int(seed))

    while True:
        while frontier:
            region, face = frontier.popleft()
            region = find(region)
            labels[face] = region
            region_normal, region_point = plane(region)
            region_offset = -float(region_normal @ region_point)
            for neighbor in neighbors[offsets[face]:offsets[face + 1]]:
                neighbor = int(neighbor)
                if not valid[neighbor]:
                    continue
                other = int(labels[neighbor])
                if other >= 0:
                    merged = merge(region, other)
                    if merged is not None and merged != region:
                        region = merged
                        frontier.append((region, face))
                        frontier.append((region, neighbor))
                    continue
                if abs(float(normals[neighbor] @ region_normal)) < cosine_limit:
                    continue
                distance = np.max(np.abs(corners[neighbor] @ region_normal + region_offset))
                if distance <= MAX_PLANE_DISTANCE:
                    add_face(region, neighbor)

        remaining = np.flatnonzero(valid & (labels < 0))
        if not len(remaining):
            break
        count = min(max(1, seed_count), len(remaining))
        weights = areas[remaining]
        for seed in rng.choice(remaining, size=count, replace=False, p=weights / weights.sum()):
            add_seed(int(seed))

    # Degenerate faces cannot define a plane; attach them to adjacent support or each other.
    for seed in np.flatnonzero(~valid & (labels < 0)):
        adjacent = labels[neighbors[offsets[seed]:offsets[seed + 1]]]
        adjacent = adjacent[adjacent >= 0]
        if len(adjacent):
            roots = np.asarray([find(int(region)) for region in adjacent])
            ids, counts = np.unique(roots, return_counts=True)
            labels[seed] = ids[np.argmax(counts)]
            continue
        region = len(parent)
        parent.append(region)
        normal_sums.append(np.zeros(3))
        point_sums.append(np.zeros(3))
        area_sums.append(0.0)
        labels[seed] = region
        stack = [int(seed)]
        while stack:
            face = stack.pop()
            for neighbor in neighbors[offsets[face]:offsets[face + 1]]:
                neighbor = int(neighbor)
                if valid[neighbor] or labels[neighbor] >= 0:
                    continue
                labels[neighbor] = region
                stack.append(neighbor)

    # Absorb small patches only into an adjacent plane with a compatible normal.
    roots = np.asarray([find(region) for region in range(len(parent))], dtype=np.int32)
    region_labels = roots[labels]
    counts = np.bincount(region_labels, minlength=len(parent))
    edge_sources = np.repeat(np.arange(len(labels), dtype=np.int32), np.diff(offsets))
    region_edges = np.column_stack((region_labels[edge_sources], region_labels[neighbors]))
    region_edges = region_edges[region_edges[:, 0] != region_edges[:, 1]]
    region_edges.sort(axis=1)
    region_edges = np.unique(region_edges, axis=0)
    adjacency = [set() for _ in parent]
    for first, second in region_edges:
        first, second = int(first), int(second)
        adjacency[first].add(second)
        adjacency[second].add(first)

    def merge_clusters(
        first: int, second: int, require_normal_match: bool
    ) -> tuple[int, set[int]] | None:
        first, second = find(first), find(second)
        merged = merge(first, second, require_normal_match)
        if merged is None:
            return None
        affected = {find(item) for item in adjacency[first] | adjacency[second]}
        affected.difference_update((first, second, merged))
        for item in affected:
            adjacency[item].discard(first)
            adjacency[item].discard(second)
            adjacency[item].add(merged)
        adjacency[merged] = affected
        if first != merged:
            adjacency[first].clear()
        if second != merged:
            adjacency[second].clear()
        counts[merged] = counts[first] + counts[second]
        if first != merged:
            counts[first] = 0
        if second != merged:
            counts[second] = 0
        return merged, affected

    pending: list[tuple[int, int, int]] = []
    versions = np.zeros(len(parent), dtype=np.int32)

    def queue_small(region: int) -> None:
        region = find(region)
        if 0 < counts[region] < MIN_REGION_TRIANGLES:
            versions[region] += 1
            heapq.heappush(pending, (int(counts[region]), region, int(versions[region])))

    for region in range(len(parent)):
        queue_small(region)
    while pending:
        count, region, version = heapq.heappop(pending)
        if (region != find(region) or versions[region] != version
                or counts[region] != count or count >= MIN_REGION_TRIANGLES):
            continue
        candidates = []
        for neighbor in adjacency[region]:
            neighbor = find(neighbor)
            fit = similarity(region, neighbor)
            if fit is not None:
                candidates.append((fit[0], fit[1], -int(counts[neighbor]), neighbor))
        if not candidates:
            continue
        result = None
        for *_, neighbor in sorted(candidates):
            result = merge_clusters(region, neighbor, require_normal_match=False)
            if result is not None:
                break
        if result is None:
            continue
        merged, affected = result
        queue_small(merged)
        for item in affected:
            queue_small(item)

    # Keep merges local; evolving centroid/normal averages can otherwise chain distant patches.
    # Recheck every boundary after small-cluster fits have changed.
    final_edges = sorted(
        (region, find(neighbor))
        for region in range(len(parent)) if find(region) == region
        for neighbor in adjacency[region]
        if region < find(neighbor)
    )
    final_pending = deque(final_edges)
    while final_pending:
        first, second = final_pending.popleft()
        first, second = find(first), find(second)
        if first == second or second not in adjacency[first]:
            continue
        result = merge_clusters(first, second, require_normal_match=True)
        if result is None:
            continue
        merged, _ = result
        final_pending.extend((merged, neighbor) for neighbor in sorted(adjacency[merged]))

    roots = np.asarray([find(region) for region in range(len(parent))], dtype=np.int32)
    root_labels = roots[labels]
    black_regions = np.zeros(len(parent), dtype=bool)
    for region in range(len(parent)):
        if roots[region] != region or counts[region] <= 0 or counts[region] >= MIN_REGION_TRIANGLES:
            continue
        if not any(similarity(region, neighbor) is not None for neighbor in adjacency[region]):
            black_regions[region] = True
    black_faces = black_regions[root_labels]
    _, labels = np.unique(root_labels, return_inverse=True)
    return labels.astype(np.int32), normals, areas, black_faces


def _region_color(region_id: int) -> np.ndarray:
    hue = (region_id * 0.618033988749895) % 1.0
    saturation = 0.62 + 0.32 * ((region_id * 7) % 16) / 15
    brightness = 0.78 + 0.20 * ((region_id * 11) % 8) / 7
    return np.asarray(colorsys.hsv_to_rgb(hue, saturation, brightness), dtype=np.float64)


def _write_colored_regions(
    paths: list[Path], output_path: Path, labels_path: Path,
    seed_count: int, random_seed: int
) -> dict:
    vertex_chunks, color_chunks, label_chunks, region_records = [], [], [], []
    total_triangles = degenerate_triangles = next_region_id = 0
    output_triangle_offset = 0

    for path in paths:
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if not len(vertices) or not len(triangles):
            continue
        labels, normals, areas, black_faces = _grow_regions(
            vertices, triangles, seed_count, random_seed
        )
        # Region IDs are global across instance files; mesh topology stays instance-local.
        local_labels = labels
        region_count = int(labels.max()) + 1
        labels = local_labels + next_region_id
        label_chunks.append(labels.astype(np.int32, copy=False))
        region_base = next_region_id
        next_region_id += region_count
        corners = vertices[triangles]
        vertex_chunks.append(corners.reshape(-1, 3))
        color_table = np.asarray([_region_color(region_base + i) for i in range(region_count)])
        face_colors = color_table[local_labels]
        black_regions = np.zeros(region_count, dtype=bool)
        black_regions[np.unique(local_labels[black_faces])] = True
        face_colors[black_faces] = 0.0
        color_chunks.append(np.repeat(face_colors, 3, axis=0))

        counts = np.bincount(local_labels, minlength=region_count)
        region_areas = np.bincount(local_labels, weights=areas, minlength=region_count)
        degenerate_counts = np.bincount(local_labels, weights=(areas <= 0), minlength=region_count)
        face_min, face_max = corners.min(axis=1), corners.max(axis=1)
        bounds_min = np.full((region_count, 3), np.inf)
        bounds_max = np.full((region_count, 3), -np.inf)
        np.minimum.at(bounds_min, local_labels, face_min)
        np.maximum.at(bounds_max, local_labels, face_max)
        seed_order = np.lexsort((np.arange(len(areas)), -areas, local_labels))
        sorted_labels = local_labels[seed_order]
        seed_faces = seed_order[np.r_[True, sorted_labels[1:] != sorted_labels[:-1]]]
        for local_id, seed in enumerate(seed_faces):
            region_id = region_base + local_id
            normal = normals[seed].tolist()
            region_records.append({
                "region_id": int(region_id),
                "source_file": path.name,
                "source_mesh_output_triangle_start": int(output_triangle_offset),
                "source_triangle_count": int(len(triangles)),
                "triangle_count": int(counts[local_id]),
                "area": float(region_areas[local_id]),
                "bounds_min": bounds_min[local_id].tolist(),
                "bounds_max": bounds_max[local_id].tolist(),
                "representative_triangle_normal": normal,
                "representative_triangle_plane_offset": float(-np.dot(normals[seed], corners[seed].mean(axis=0)))
                if areas[seed] > 0 else None,
                "degenerate_triangle_count": int(degenerate_counts[local_id]),
                "painted_black": bool(black_regions[local_id]),
                "color_rgb": [0, 0, 0] if black_regions[local_id]
                else np.rint(_region_color(region_id) * 255).astype(int).tolist(),
            })
        total_triangles += len(triangles)
        degenerate_triangles += int((areas <= 0).sum())
        output_triangle_offset += len(triangles)

    if not vertex_chunks:
        raise ValueError("Wall meshes contain no triangles")
    output_vertices = np.vstack(vertex_chunks)
    output_colors = np.vstack(color_chunks)
    output_triangles = np.arange(len(output_vertices), dtype=np.int32).reshape(-1, 3)
    output_mesh = o3d.geometry.TriangleMesh()
    output_mesh.vertices = o3d.utility.Vector3dVector(output_vertices)
    output_mesh.triangles = o3d.utility.Vector3iVector(output_triangles)
    output_mesh.vertex_colors = o3d.utility.Vector3dVector(output_colors)
    if not o3d.io.write_triangle_mesh(str(output_path), output_mesh, write_vertex_colors=True):
        raise OSError(f"Could not write colored region mesh: {output_path}")
    np.save(labels_path, np.concatenate(label_chunks))
    return {
        "algorithm": "area_seeded_multi_source_region_growing_v7",
        "seed_count_per_batch": seed_count,
        "random_seed": random_seed,
        "seed_sampling": "area_weighted_without_replacement",
        "normal_angle_threshold_deg": NORMAL_ANGLE_DEG,
        "max_cluster_plane_distance_scene_units": MAX_PLANE_DISTANCE,
        "small_region_merge_min_triangles": MIN_REGION_TRIANGLES,
        "small_region_merge_rule": "closest average normal among edge-adjacent clusters within the closest orthogonal plane-distance limit",
        "small_region_black_rule": "paint regions below the triangle threshold black when no edge-adjacent cluster passes the plane-distance limit",
        "final_cluster_merge_rule": "iteratively merge edge-adjacent clusters within the normal-angle and closest orthogonal plane-distance limits",
        "wall_meshes": [str(path) for path in paths],
        "triangle_count": int(total_triangles),
        "degenerate_triangle_count": int(degenerate_triangles),
        "region_count": len(region_records),
        "regions": region_records,
        "colored_mesh": str(output_path),
        "triangle_labels_file": labels_path.name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Directory containing structural_instances_manifest.json and wall PLYs")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (defaults to --instances-dir)")
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT,
                        help=f"Area-weighted random seeds per growth batch (default: {DEFAULT_SEED_COUNT})")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED,
                        help=f"Seed for reproducible region growth (default: {DEFAULT_RANDOM_SEED})")
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("seed count must be positive")

    try:
        paths = _wall_mesh_paths(args.instances_dir)
        output_dir = args.output_dir or args.instances_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_mesh_path = output_dir / "wall_regions_colored.ply"
        labels_path = output_dir / "wall_regions_triangle_labels.npy"
        manifest_path = output_dir / "wall_regions_manifest.json"
        source_paths = {path.resolve() for path in paths}
        if any(path.resolve() in source_paths for path in (output_mesh_path, labels_path, manifest_path)):
            raise ValueError("Region output path would overwrite an input wall mesh")
        result = _write_colored_regions(
            paths, output_mesh_path, labels_path, args.seed_count, args.random_seed
        )
        manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(f"Colored {result['triangle_count']} wall triangles into {result['region_count']} regions")
    print(f"Mesh:     {output_mesh_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
