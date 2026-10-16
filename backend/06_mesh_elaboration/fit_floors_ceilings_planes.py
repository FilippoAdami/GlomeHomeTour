#!/usr/bin/env python3
"""Fit structural floor and ceiling planes listed by the extraction manifest.

The extractor owns instance discovery.  This step assumes neither names,
elevations, number of levels, nor a rectangular room footprint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and clean manifest-listed floor and ceiling meshes.")
    parser.add_argument("--instances-dir", type=Path, default=Path(
        "backend/current_scene/06_mesh_elaboration/extracted_structural_instances"))
    # Retained for callers. Bounds now come from each source mesh rather than a room rectangle.
    parser.add_argument("--wall-manifest", type=Path, default=Path(
        "backend/current_scene/06_mesh_elaboration/extracted_structural_instances/wall_planes_manifest.json"))
    parser.add_argument("--dist-thresh", type=float, default=0.10,
                        help="Maximum perpendicular residual (metres) for retained faces.")
    return parser.parse_args()


def _oriented(normals: np.ndarray, prefer_normal_up: bool) -> np.ndarray:
    result = np.asarray(normals, dtype=np.float64).copy()
    sign = np.where(result[:, 1] >= 0.0, 1.0, -1.0)
    if not prefer_normal_up:
        sign = -sign
    return result * sign[:, None]


def _weighted_plane(points: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
    centre = np.average(points, axis=0, weights=weights)
    delta = points - centre
    cov = (delta * weights[:, None]).T @ delta / max(float(weights.sum()), 1e-12)
    normal = np.linalg.eigh(cov)[1][:, 0]
    normal /= np.linalg.norm(normal)
    return normal, -float(normal @ centre)


def _candidates(normals: np.ndarray, areas: np.ndarray, limit: int = 16) -> Iterable[np.ndarray]:
    """Area leaders plus dominant normal modes, with no random sampling."""
    order = np.argsort(-areas, kind="stable")
    if len(order) > limit:
        order = np.unique(np.r_[order[:limit // 2], order[np.linspace(0, len(order) - 1, limit // 2, dtype=int)]])
    for index in order:
        yield normals[index]
    # Scan triangles can be individually small while a real plane still forms
    # the dominant direction mode.  Binning is only a candidate generator;
    # exact area/residual scoring below makes the final decision.
    bins, inverse = np.unique(np.rint(normals / 0.02).astype(np.int16), axis=0, return_inverse=True)
    support = np.bincount(inverse, weights=areas)
    for index in np.argsort(-support, kind="stable")[:limit]:
        members = inverse == index
        mode = np.average(normals[members], axis=0, weights=areas[members])
        yield mode / np.linalg.norm(mode)


def fit_robust_plane(
    points: np.ndarray,
    weights: np.ndarray | None = None,
    distance_threshold: float = 0.08,
    prefer_normal_up: bool = True,
    face_normals: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Deterministically fit the largest normal-compatible planar support.

    Area weighting makes dense triangulation irrelevant. Candidate normals must
    have broad support before their offset is scored, rejecting perpendicular
    beam flanks and wall scraps without world-coordinate assumptions.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < (3 if face_normals is None else 1):
        raise ValueError("insufficient non-degenerate surface support to fit a plane")
    weights = np.ones(len(points), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(points),) or np.any(weights <= 0):
        raise ValueError("weights must be positive and match points")
    normals = None
    if face_normals is None:
        seed, _ = _weighted_plane(points, weights)
        if (seed[1] < 0) == prefer_normal_up:
            seed = -seed
        candidates = (seed,)
    else:
        raw = np.asarray(face_normals, dtype=np.float64)
        if raw.shape != points.shape:
            raise ValueError("face_normals must match points")
        valid = np.linalg.norm(raw, axis=1) > 1e-12
        if not valid.any():
            raise ValueError("at least one non-degenerate normal is required")
        points, weights, raw = points[valid], weights[valid], raw[valid]
        raw /= np.linalg.norm(raw, axis=1)[:, None]
        normals = _oriented(raw, prefer_normal_up)
        candidates = _candidates(normals, weights)

    # 30° separates a surface from its orthogonal flanks. Offset support then
    # chooses the actual coplanar sheet among parallel beams and base surfaces.
    cosine = float(np.cos(np.deg2rad(30.0)))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for candidate in candidates:
        candidate = candidate / np.linalg.norm(candidate)
        compatible = np.ones(len(points), bool) if normals is None else normals @ candidate >= cosine
        if compatible.sum() < (3 if normals is None else 1):
            continue
        offsets, support_weights = points[compatible] @ candidate, weights[compatible]
        order = np.argsort(offsets, kind="stable")
        offsets, support_weights = offsets[order], support_weights[order]
        cumulative = np.r_[0.0, np.cumsum(support_weights)]
        left = np.searchsorted(offsets, offsets - 2.0 * distance_threshold, side="left")
        scores = cumulative[1:] - cumulative[left]
        right = int(np.argmax(scores))
        score = float(scores[right])
        if best is None or score > best[0]:
            start = int(left[right])
            centre = float(np.average(offsets[start:right + 1], weights=support_weights[start:right + 1]))
            best = (score, candidate.copy(), compatible &
                    (np.abs(points @ candidate - centre) <= distance_threshold))
    if best is None:
        raise ValueError("could not find a normal-compatible planar support")
    _, initial_normal, support = best
    initial_d = -float(np.average(points[support] @ initial_normal, weights=weights[support]))
    initial_score = float(weights[np.abs(points @ initial_normal+initial_d) <= distance_threshold].sum())
    if support.sum() < 3:
        return initial_normal, initial_d, initial_score/float(weights.sum())
    normal, d = _weighted_plane(points[support], weights[support])
    if (normal[1] < 0) == prefer_normal_up:
        normal, d = -normal, -d
    inliers = np.abs(points @ normal + d) <= distance_threshold
    normal, d = _weighted_plane(points[inliers], weights[inliers])
    if (normal[1] < 0) == prefer_normal_up:
        normal, d = -normal, -d
    refined_score = float(weights[np.abs(points @ normal+d) <= distance_threshold].sum())
    if refined_score < initial_score:
        normal, d = initial_normal, initial_d
        refined_score = initial_score
    ratio = refined_score / float(weights.sum())
    return normal, float(d), ratio


def _should_flatten(points: np.ndarray, weights: np.ndarray, normal: np.ndarray, d: float,
                    distance_threshold: float) -> bool:
    """Only flatten when horizontal residual is indistinguishable from fit noise."""
    residual = np.abs(points @ normal + d)
    y = float(np.average(points[:, 1], weights=weights))
    horizontal = np.abs(points[:, 1] - y)
    fitted_rms = float(np.sqrt(np.average(residual * residual, weights=weights)))
    horizontal_rms = float(np.sqrt(np.average(horizontal * horizontal, weights=weights)))
    return horizontal_rms < 1e-10 or (fitted_rms > 1e-10 and horizontal_rms <= fitted_rms*1.75)


def _subset_mesh(mesh: o3d.geometry.TriangleMesh, triangles: np.ndarray, keep: np.ndarray,
                 plane: tuple[np.ndarray, float] | None = None) -> o3d.geometry.TriangleMesh:
    """Keep source face order, colors, and first-use vertex order."""
    vertices, colors = np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors)
    selected = triangles[keep]
    unique, first, inverse = np.unique(selected, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    used = unique[order]
    output_triangles = remap[inverse].reshape(selected.shape)
    output_vertices = vertices[used].copy()
    if plane is not None:
        normal, d = plane
        output_vertices -= (output_vertices @ normal + d)[:, None] * normal
    output = o3d.geometry.TriangleMesh()
    output.vertices = o3d.utility.Vector3dVector(output_vertices)
    output.triangles = o3d.utility.Vector3iVector(output_triangles)
    if len(colors) == len(vertices):
        output.vertex_colors = o3d.utility.Vector3dVector(colors[used])
    output.compute_vertex_normals()
    return output


def _load_targets(instances_dir: Path) -> list[dict[str, Any]]:
    manifest_path = instances_dir / "structural_instances_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"structural instance manifest not found: {manifest_path}")
    with manifest_path.open() as handle:
        source = json.load(handle)
    result = []
    for item in source.get("instances", []):
        role, name, filename = item.get("role"), item.get("instance_name"), item.get("file_name")
        if role not in {"floor", "ceiling"}:
            continue
        if (not isinstance(name, str) or not isinstance(filename, str)
                or Path(name).name != name or Path(filename).name != filename
                or name in {"", ".", ".."}):
            raise ValueError("each floor/ceiling instance needs instance_name and file_name")
        path = instances_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {role} mesh for {name}: {path}")
        result.append({"name": name, "type": role, "file": path})
    return result


def main() -> int:
    args = parse_args()
    try:
        targets = _load_targets(args.instances_dir)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    manifest: dict[str, Any] = {"elements": []}
    for item in targets:
        mesh = o3d.io.read_triangle_mesh(str(item["file"]))
        triangles, vertices = np.asarray(mesh.triangles), np.asarray(mesh.vertices)
        if not len(triangles) or not len(vertices):
            print(f"error: {item['file']} contains no triangles", file=sys.stderr)
            return 2
        mesh.compute_triangle_normals()
        v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
        centres = (v0 + v1 + v2) / 3.0
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        valid = areas > 1e-12
        up = item["type"] == "floor"
        face_normals = np.asarray(mesh.triangle_normals)
        compatible = valid & (face_normals[:, 1]*(1. if up else -1.) > 0.)
        if areas[compatible].sum() < .2*areas[valid].sum():
            compatible = valid
        normal, d, support = fit_robust_plane(centres[compatible], areas[compatible], args.dist_thresh, up,
                                              face_normals[compatible])
        keep = valid & (np.abs(centres @ normal + d) <= args.dist_thresh)
        if not keep.any():
            print(f"error: fitting {item['name']} retained no faces", file=sys.stderr)
            return 2
        fitted_support = keep & compatible
        if _should_flatten(centres[fitted_support], areas[fitted_support], normal, d, args.dist_thresh):
            normal = np.array([0.0, 1.0 if up else -1.0, 0.0])
            d = -float(normal[1] * np.average(centres[fitted_support, 1], weights=areas[fitted_support]))
            keep = valid & (np.abs(centres @ normal + d) <= args.dist_thresh)
        # A single plane that explains less than the great majority is not a
        # safe substitute for a multi-slope source. Preserve it for the caller
        # to split rather than silently deleting a valid roof face.
        preserve_source = support < 0.80
        if preserve_source:
            keep = valid
        tilt = float(np.rad2deg(np.arccos(np.clip(abs(normal[1]), -1.0, 1.0))))
        mean_y = float(np.average(centres[fitted_support, 1], weights=areas[fitted_support]))
        cleaned_path = args.instances_dir / f"{item['name']}_cleaned.ply"
        enclosing_path = args.instances_dir / f"{item['name']}_enclosing_plane.ply"
        cleaned = _subset_mesh(mesh, triangles, keep)
        # Snapping the retained source footprint avoids creating a room-sized rectangle.
        enclosing = _subset_mesh(mesh, triangles, keep, None if preserve_source else (normal, d))
        for path, result in ((cleaned_path, cleaned), (enclosing_path, enclosing)):
            if not o3d.io.write_triangle_mesh(str(path), result, write_ascii=False, compressed=False,
                                             write_vertex_normals=True, write_vertex_colors=True):
                raise OSError(f"Failed to write structural surface: {path}")
        manifest["elements"].append({
            "name": item["name"], "type": item["type"], "source_file": str(item["file"]),
            "plane_equation": [float(normal[0]), float(normal[1]), float(normal[2]), float(d)],
            "normal_unit": normal.tolist(), "tilt": round(tilt, 4),
            "tilt_from_horizontal_deg": round(tilt, 4),
            "mean_elevation_y": round(mean_y, 6), "inlier_triangles": int(keep.sum()),
            "inlier_fraction": round(float(keep.sum() / len(triangles)), 6),
            "outlier_triangles_pruned": int((~keep).sum()),
            "inlier_area_m2": round(float(areas[keep].sum()), 6), "fit_confidence": round(support, 6),
            "preserve_source": preserve_source,
            "bounds": {"x": [float(vertices[:, 0].min()), float(vertices[:, 0].max())],
                       "y": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
                       "z": [float(vertices[:, 2].min()), float(vertices[:, 2].max())]},
            "artifacts": {"cleaned_mesh": str(cleaned_path), "enclosing_plane": str(enclosing_path)},
        })
        print(f"{item['name']}: {keep.sum()}/{len(triangles)} faces, tilt {tilt:.3f}°")
    output = args.instances_dir / "floors_ceilings_planes_manifest.json"
    with output.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Manifest saved to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
