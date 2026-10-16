#!/usr/bin/env python3
"""Fit finite Manhattan wall planes from extracted wall meshes.

No room position, dimensions, or topology are assumed.  Each supported
coplanar span stays separate, so recesses and disconnected wall groups survive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

DEFAULT_DIR = Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover finite Manhattan wall planes.")
    parser.add_argument("--wall-mesh", type=Path, help="Optional single wall mesh (legacy input).")
    parser.add_argument("--instances-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--plane-dist-thresh", type=float, default=.10)
    parser.add_argument("--snap-thresh", type=float, default=.15)
    return parser.parse_args()


def _mesh_paths(args: argparse.Namespace) -> list[Path]:
    if args.wall_mesh:
        return [args.wall_mesh]
    manifest_path = args.instances_dir / "structural_instances_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Wall mesh was not supplied and manifest is missing: {manifest_path}")
    paths = []
    for item in json.loads(manifest_path.read_text()).get("instances", []):
        if str(item.get("role", "")).lower() != "wall":
            continue
        path = Path(item.get("file_path", item.get("file_name", "")))
        paths.append(path if path.is_file() else args.instances_dir / path.name)
    return paths


def _load_meshes(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, triangles, colors, base = [], [], [], 0
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Wall mesh not found: {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        v, t, c = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        if not len(v) or not len(t):
            continue
        vertices.append(v); triangles.append(t + base)
        colors.append(c if len(c) == len(v) else np.full((len(v), 3), .8))
        base += len(v)
    if not triangles:
        raise ValueError("Wall input contains no triangles")
    return np.vstack(vertices), np.vstack(triangles), np.vstack(colors)


def _basis(normals: np.ndarray, areas: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    horizontal = normals[:, [0, 2]]
    lengths = np.linalg.norm(horizontal, axis=1)
    keep = lengths > .85
    if not np.any(keep):
        raise ValueError("No sufficiently vertical wall faces")
    angles = np.mod(np.arctan2(horizontal[keep, 1], horizontal[keep, 0]), np.pi / 2)
    edges = np.linspace(0., np.pi / 2, 181)
    hist, _ = np.histogram(angles, bins=edges, weights=areas[keep])
    peak = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    # Axial mean refines the area-weighted histogram peak without random sampling.
    near = np.abs(np.angle(np.exp(4j * (angles - peak)))) <= np.deg2rad(12.)
    theta = float((.25 * np.angle(np.sum(areas[keep][near] * np.exp(4j * angles[near])))) % (np.pi / 2))
    return np.array([np.cos(theta), 0., np.sin(theta)]), np.array([-np.sin(theta), 0., np.cos(theta)]), theta


def _groups(values: np.ndarray, weights: np.ndarray, tolerance: float) -> list[np.ndarray]:
    order = np.argsort(values)
    out, group = [], [int(order[0])]
    mean, total = values[order[0]], weights[order[0]]
    for index in order[1:]:
        if abs(values[index] - mean) > tolerance:
            out.append(np.asarray(group, dtype=int)); group = [int(index)]
            mean, total = values[index], weights[index]
        else:
            group.append(int(index))
            mean = (mean * total + values[index] * weights[index]) / (total + weights[index])
            total += weights[index]
    out.append(np.asarray(group, dtype=int))
    return out


def _discover(vertices: np.ndarray, triangles: np.ndarray, normals: np.ndarray, centers: np.ndarray,
              areas: np.ndarray, a: np.ndarray, b: np.ndarray, offset_tol: float,
              span_gap: float) -> tuple[list[dict[str, Any]], np.ndarray]:
    planes, inlier = [], np.zeros(len(triangles), dtype=bool)
    for axis, normal, tangent, span_key in (("Dir A", a, b, "span_w_m"), ("Dir B", b, a, "span_u_m")):
        alignment = normals @ normal
        # Do not collapse the opposite faces of a thin wall into one plane.
        for sign in (-1., 1.):
            candidates = np.where(alignment * sign >= .85)[0]
            if not len(candidates):
                continue
            for local in _groups(centers[candidates] @ normal, areas[candidates], offset_tol):
                faces = candidates[local]
                intervals = vertices[triangles[faces]] @ tangent
                intervals = np.column_stack((intervals.min(axis=1), intervals.max(axis=1)))
                ordered = np.argsort(intervals[:, 0]); spans, group = [], [int(ordered[0])]
                right = intervals[ordered[0], 1]
                for index in ordered[1:]:
                    if intervals[index, 0] > right + span_gap:
                        spans.append(np.asarray(group, dtype=int)); group = [int(index)]
                    else:
                        group.append(int(index))
                    right = max(right, intervals[index, 1])
                spans.append(np.asarray(group, dtype=int))
                for span in spans:
                    support = faces[span]
                    points = vertices[triangles[support].reshape(-1)]
                    lo, hi = np.min(points @ tangent), np.max(points @ tangent)
                    y0, y1 = np.min(points[:, 1]), np.max(points[:, 1])
                    if hi - lo < .12-1e-9 or y1 - y0 < .12-1e-9:
                        continue
                    planes.append({"axis": axis, "offset_m": float(np.average(centers[support] @ normal, weights=areas[support])),
                                   span_key: [float(lo), float(hi)], "normal_unit": sign * normal,
                                   "bounds_y": [float(y0), float(y1)], "source_faces": support})
    # A wall plane needs area support.  This removes scan crumbs without using a
    # scene-sized absolute cutoff; substantial recesses remain proportional to
    # their parent wall set.
    minimum_area = max(.02, float(areas.sum()) * .005)
    planes = [plane for plane in planes if float(areas[plane["source_faces"]].sum()) >= minimum_area]
    merged = []
    for plane in sorted(planes, key=lambda p: -areas[p["source_faces"]].sum()):
        key = "span_w_m" if plane["axis"] == "Dir A" else "span_u_m"
        target = None
        for other in merged:
            if other["axis"] != plane["axis"] or np.dot(other["normal_unit"], plane["normal_unit"]) < .99:
                continue
            overlap = min(other[key][1], plane[key][1])-max(other[key][0], plane[key][0])
            if abs(other["offset_m"]-plane["offset_m"]) < span_gap and overlap >= .5*min(np.ptp(other[key]), np.ptp(plane[key])):
                target = other
                break
        if target is None:
            merged.append(plane)
        else:
            target[key] = [min(target[key][0], plane[key][0]), max(target[key][1], plane[key][1])]
            target["bounds_y"] = [min(target["bounds_y"][0], plane["bounds_y"][0]), max(target["bounds_y"][1], plane["bounds_y"][1])]
            target["source_faces"] = np.unique(np.r_[target["source_faces"], plane["source_faces"]])
    planes = merged
    for plane in planes:
        inlier[plane["source_faces"]] = True
    planes.sort(key=lambda p: (p["axis"], p["offset_m"], p.get("span_u_m", p.get("span_w_m"))[0]))
    for index, plane in enumerate(planes, 1):
        plane["id"], plane["name"] = index, f"wall_plane_{index:03d}"
    return planes, inlier


def _write(path: Path, vertices: np.ndarray, triangles: np.ndarray, colors: np.ndarray) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False,
                                        write_vertex_normals=True, write_vertex_colors=True):
        raise OSError(f"Failed to write wall mesh: {path}")


def main() -> int:
    args = parse_args()
    try:
        paths = _mesh_paths(args)
        if not paths:
            # Open terraces and partial scans still have useful surface geometry.
            direction = np.array([1., 0., 0.])
            longest = 0.
            instances = json.loads((args.instances_dir / "structural_instances_manifest.json").read_text())
            for item in instances.get("instances", []):
                if item.get("role") not in {"floor", "ceiling"}:
                    continue
                mesh = o3d.io.read_triangle_mesh(str(args.instances_dir / item["file_name"]))
                vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
                edges = np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
                edges, counts = np.unique(edges, axis=0, return_counts=True)
                for first, second in edges[counts == 1]:
                    vector = vertices[second] - vertices[first]
                    vector[1] = 0.
                    length = np.linalg.norm(vector)
                    if length > longest:
                        direction, longest = vector / length, length
            a = direction
            b = np.array([-a[2], 0., a[0]])
            args.output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {"orthogonal_basis": {"dir_A_east_west": a.tolist(), "dir_B_north_south": b.tolist()},
                        "wall_planes": [], "artifacts": {}}
            (args.output_dir / "wall_planes_manifest.json").write_text(json.dumps(manifest, indent=2))
            return 0
        vertices, triangles, colors = _load_meshes(paths)
        corners = vertices[triangles]
        cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        double_area = np.linalg.norm(cross, axis=1); valid = double_area > 1e-10
        if not np.any(valid):
            raise ValueError("Wall input has only degenerate triangles")
        normal, centers, areas = cross[valid] / double_area[valid, None], corners[valid].mean(axis=1), double_area[valid] / 2
        a, b, theta = _basis(normal, areas)
        scale = float(np.linalg.norm(np.ptp(vertices[:, [0, 2]], axis=0)))
        offset_tol = min(args.plane_dist_thresh, max(.025, scale * .02))
        planes, local_inlier = _discover(vertices, triangles[valid], normal, centers, areas, a, b,
                                          offset_tol, max(args.snap_thresh, min(.30, scale * .02)))
        if not planes:
            raise ValueError("No finite Manhattan wall planes were supported by the input")
        inlier = np.zeros(len(triangles), dtype=bool); inlier[np.flatnonzero(valid)[local_inlier]] = True
        # The Manhattan frame is a regularization aid, not permission to erase a
        # measured diagonal/curved architectural return.  Retain meaningful
        # vertical source support and explicitly report that it was not flattened.
        horizontal_normal = np.linalg.norm(normal[:, [0, 2]], axis=1)
        unmatched = (horizontal_normal > .85) & (np.maximum(np.abs(normal @ a), np.abs(normal @ b)) < .85)
        retained_unmodeled = np.zeros(len(unmatched), dtype=bool)
        if np.any(unmatched):
            selected = triangles[valid][unmatched]
            unique, remap = np.unique(selected, return_inverse=True)
            subset = o3d.geometry.TriangleMesh()
            subset.vertices = o3d.utility.Vector3dVector(vertices[unique])
            subset.triangles = o3d.utility.Vector3iVector(remap.reshape(-1, 3))
            labels, _, component_areas = subset.cluster_connected_triangles()
            labels = np.asarray(labels)
            supported = []
            for component in np.flatnonzero(np.asarray(component_areas) >= .02):
                points = vertices[np.unique(selected[labels == component])]
                horizontal_span = np.max(np.ptp(points @ np.array([a, b]).T, axis=0))
                covered = np.zeros(len(points), dtype=bool)
                for plane in planes:
                    normal_axis, tangent = (a, b) if plane["axis"] == "Dir A" else (b, a)
                    span = plane.get("span_w_m", plane.get("span_u_m"))
                    along = points @ tangent
                    covered |= ((np.abs(points @ normal_axis-plane["offset_m"]) < .12)
                                & (along >= span[0]-.12) & (along <= span[1]+.12)
                                & (points[:, 1] >= plane["bounds_y"][0]-.12)
                                & (points[:, 1] <= plane["bounds_y"][1]+.12))
                if horizontal_span >= .12-1e-9 and np.ptp(points[:, 1]) >= .12-1e-9 and not covered.all():
                    supported.append(component)
            retained_unmodeled[np.flatnonzero(unmatched)] = np.isin(np.asarray(labels), supported)
        inlier[np.flatnonzero(valid)[retained_unmodeled]] = True
        args.output_dir.mkdir(parents=True, exist_ok=True)
        kept = triangles[inlier]; unique, remap = np.unique(kept, return_inverse=True)
        cleaned = args.output_dir / "wall_cleaned_pieces.ply"
        _write(cleaned, vertices[unique], remap.reshape(-1, 3), colors[unique])

        quad_v, quad_t, quad_c, records = [], [], [], []
        for plane in planes:
            tangent = b if plane["axis"] == "Dir A" else a
            normal_axis = a if plane["axis"] == "Dir A" else b
            lo, hi = plane.get("span_w_m", plane.get("span_u_m")); y0, y1 = plane["bounds_y"]
            base = plane["offset_m"] * normal_axis; p0, p1 = base + lo * tangent, base + hi * tangent
            start = len(quad_v); quad_v.extend(([p0[0], y0, p0[2]], [p1[0], y0, p1[2]], [p1[0], y1, p1[2]], [p0[0], y1, p0[2]]))
            quad_t.extend(([start, start + 1, start + 2], [start, start + 2, start + 3]))
            source = plane.pop("source_faces")
            rgb = np.average(colors[np.unique(triangles[valid][source].reshape(-1))], axis=0); quad_c.extend([rgb] * 4)
            record = {key: value for key, value in plane.items()}
            n = record["normal_unit"]
            record["normal_unit"] = [round(float(v), 8) for v in n]
            record["plane_equation"] = [round(float(v), 8) for v in (*n, -np.dot(n, base))]
            record["bounds_y"] = [round(float(v), 4) for v in record["bounds_y"]]
            record["source"] = {"triangle_count": int(len(source)), "area_m2": round(float(areas[source].sum()), 4)}
            for key in ("offset_m", "span_u_m", "span_w_m"):
                if key in record:
                    record[key] = round(record[key], 4) if isinstance(record[key], float) else [round(v, 4) for v in record[key]]
            records.append(record)
        enclosing = args.output_dir / "wall_enclosing_planes.ply"
        _write(enclosing, np.asarray(quad_v), np.asarray(quad_t), np.asarray(quad_c))
        manifest = {"source_meshes": [str(p) for p in paths], "dominant_orientation_deg": round(float(np.degrees(theta)), 3),
                    "orthogonal_basis": {"dir_A_east_west": a.tolist(), "dir_B_north_south": b.tolist()},
                    "total_wall_planes_count": len(records), "wall_planes": records,
                    "unmodeled_wall_support": {"triangle_count": int(retained_unmodeled.sum()),
                                               "area_m2": round(float(areas[retained_unmodeled].sum()), 4),
                                               "reason": "vertical support outside the dominant Manhattan frame; retained in cleaned source"},
                    "filtering_stats": {"total_triangles": int(len(triangles)), "inlier_triangles": int(inlier.sum()),
                                        "inlier_fraction": round(float(inlier.mean()), 4),
                                        "inlier_area_m2": round(float((double_area[inlier] / 2).sum()), 3)},
                    "artifacts": {"cleaned_wall_pieces": str(cleaned), "enclosing_planes": str(enclosing)}}
        (args.output_dir / "wall_planes_manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Discovered {len(records)} finite wall planes from {len(paths)} wall mesh(es).")
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
