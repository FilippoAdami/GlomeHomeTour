#!/usr/bin/env python3
"""Assign scan-level and floor-level names to aligned structural plane meshes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import open3d as o3d


HEIGHT_GAP_M = 0.5
HORIZONTAL_TOLERANCE_DEG = 5.0
COPLANAR_TOLERANCE_M = 0.05
DEFAULT_ALIGNMENT_DIR = Path(__file__).resolve().parents[1] / "current_scene" / "multi_room_alignment"


def _scan_key(path: Path) -> tuple[int, int | str]:
    match = re.fullmatch(r"scan_(\d+)", path.name)
    return (0, int(match.group(1))) if match else (1, path.name)


def _mesh_triangles(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if not len(faces) or vertices.ndim != 2 or not np.isfinite(vertices).all():
        raise ValueError(f"Mesh has no valid triangles: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh has invalid triangle indices: {path}")

    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(crosses, axis=1)
    valid = np.isfinite(double_areas) & (double_areas > 1e-12)
    triangles, crosses, double_areas = triangles[valid], crosses[valid], double_areas[valid]
    if not len(triangles):
        raise ValueError(f"Mesh has no non-degenerate triangles: {path}")
    return triangles, crosses / double_areas[:, None], double_areas / 2.0


def _plane_record(
    scan_id: str, source: Path, alignment_dir: Path,
    triangles: np.ndarray, areas: np.ndarray, role: str, plane_id: str,
) -> dict:
    weights = np.repeat(areas / 3.0, 3)
    points = triangles.reshape(-1, 3)
    center = np.average(points, axis=0, weights=weights)
    offsets = points - center
    covariance = (offsets * weights[:, None]).T @ offsets / weights.sum()
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    normal /= np.linalg.norm(normal)
    if normal[1] < 0:
        normal *= -1.0
    residual = offsets @ normal
    equation = np.r_[normal, -normal @ center]
    face_centers = triangles.mean(axis=1)
    return {
        "plane_id": plane_id,
        "scan_id": scan_id,
        "role": role,
        "source_mesh": str(source.relative_to(alignment_dir)),
        "mean_elevation_m": float(np.average(face_centers[:, 1], weights=areas)),
        "plane_equation": [float(value) for value in equation],
        "normal_unit": [float(value) for value in normal],
        "support_area_m2": float(areas.sum()),
        "support_triangle_count": int(len(triangles)),
        "fit_rms_m": float(np.sqrt(np.average(residual**2, weights=weights))),
    }


def _horizontal_mask(normals: np.ndarray) -> np.ndarray:
    return np.abs(normals[:, 1]) >= np.cos(np.deg2rad(HORIZONTAL_TOLERANCE_DEG))


def _largest_coplanar_set(
    triangles: np.ndarray, normals: np.ndarray, areas: np.ndarray, candidates: np.ndarray,
) -> np.ndarray:
    ids = np.flatnonzero(candidates)
    if not len(ids):
        return candidates
    normal = np.average(normals[ids], axis=0, weights=areas[ids])
    normal /= np.linalg.norm(normal)
    centers = triangles[ids].mean(axis=1)
    offsets = centers @ normal
    order = np.argsort(offsets, kind="stable")
    groups: list[list[int]] = [[int(order[0])]]
    for position in order[1:]:
        if offsets[position] - offsets[groups[-1][0]] > COPLANAR_TOLERANCE_M:
            groups.append([int(position)])
        else:
            groups[-1].append(int(position))
    selected = max(groups, key=lambda group: float(areas[ids[group]].sum()))
    result = np.zeros_like(candidates)
    result[ids[selected]] = True
    return result


def _floor_from_mesh(path: Path, scan_id: str, alignment_dir: Path, plane_id: str) -> dict:
    triangles, normals, areas = _mesh_triangles(path)
    horizontal = _horizontal_mask(normals)
    upward = horizontal & (normals[:, 1] > 0)
    selected = upward if upward.any() else horizontal & (normals[:, 1] < 0)
    selected = _largest_coplanar_set(triangles, normals, areas, selected)
    if not selected.any():
        raise ValueError(f"No horizontal floor support within {HORIZONTAL_TOLERANCE_DEG:g}°: {path}")
    return _plane_record(scan_id, path, alignment_dir, triangles[selected], areas[selected], "floor", plane_id)


def _slab_planes(path: Path, scan_id: str, alignment_dir: Path, index: int) -> tuple[dict | None, dict | None]:
    triangles, normals, areas = _mesh_triangles(path)
    horizontal = _horizontal_mask(normals)
    up = horizontal & (normals[:, 1] > 0)
    down = horizontal & (normals[:, 1] < 0)
    up = _largest_coplanar_set(triangles, normals, areas, up)
    down = _largest_coplanar_set(triangles, normals, areas, down)
    if up.any() and down.any():
        up_height = float(np.average(triangles[up].mean(axis=1)[:, 1], weights=areas[up]))
        down_height = float(np.average(triangles[down].mean(axis=1)[:, 1], weights=areas[down]))
        floor_mask, ceiling_mask = (up, down) if up_height > down_height else (down, up)
        floor = _plane_record(
            scan_id, path, alignment_dir, triangles[floor_mask], areas[floor_mask],
            "floor", f"{scan_id}:slab_floor:{index}",
        )
        ceiling = _plane_record(
            scan_id, path, alignment_dir, triangles[ceiling_mask], areas[ceiling_mask],
            "ceiling", f"{scan_id}:slab_ceiling:{index}",
        )
        return floor, ceiling

    if up.any():
        return (_plane_record(
            scan_id, path, alignment_dir, triangles[up], areas[up],
            "floor", f"{scan_id}:slab_floor:{index}",
        ), None)
    if down.any():
        return (None, _plane_record(
            scan_id, path, alignment_dir, triangles[down], areas[down],
            "ceiling", f"{scan_id}:slab_ceiling:{index}",
        ))
    return None, None


def _height_groups(planes: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for plane in sorted(planes, key=lambda item: (item["mean_elevation_m"], item["plane_id"])):
        if not groups or plane["mean_elevation_m"] - groups[-1][0]["mean_elevation_m"] > HEIGHT_GAP_M:
            groups.append([plane])
        else:
            groups[-1].append(plane)
    return groups


def _letter(index: int) -> str:
    result = ""
    while index >= 0:
        index, remainder = divmod(index, 26)
        result = chr(ord("a") + remainder) + result
        index -= 1
    return result


def _level_name(index: int) -> str:
    return "BASE" if index == 0 else f"+{index}"


def build_floor_level_metadata(alignment_dir: Path) -> dict:
    alignment_dir = Path(alignment_dir).resolve()
    scans_dir = alignment_dir / "scans"
    scan_dirs = sorted((path for path in scans_dir.glob("scan_*") if path.is_dir()), key=_scan_key)
    if not scan_dirs:
        raise FileNotFoundError(f"No aligned scan folders found: {scans_dir}")

    floors: list[dict] = []
    ceilings: list[dict] = []
    for scan_dir in scan_dirs:
        scan_id = scan_dir.name
        elements_dir = scan_dir / "mesh_elaboration" / "elements"
        if not elements_dir.is_dir():
            raise FileNotFoundError(f"Aligned regularized elements are missing: {elements_dir}")

        scan_floors = []
        for path in sorted(elements_dir.glob("floor_level_*.ply")):
            plane = _floor_from_mesh(path, scan_id, alignment_dir, f"{scan_id}:{path.stem}")
            scan_floors.append(plane)

        for index, path in enumerate(sorted(elements_dir.glob("slab_level_*.ply"))):
            floor, ceiling = _slab_planes(path, scan_id, alignment_dir, index)
            if floor:
                scan_floors.append(floor)
            if ceiling:
                ceilings.append(ceiling)

        if not scan_floors:
            raise ValueError(f"No aligned floor planes found for {scan_id}: {elements_dir}")
        floors.extend(scan_floors)

        for index, path in enumerate(sorted(elements_dir.glob("ceiling_group_*.ply"))):
            triangles, _, areas = _mesh_triangles(path)
            ceilings.append(_plane_record(
                scan_id, path, alignment_dir, triangles, areas, "ceiling",
                f"{scan_id}:ceiling_group:{index}",
            ))

    scan_baselines = []
    for scan_dir in scan_dirs:
        scan_floors = [plane for plane in floors if plane["scan_id"] == scan_dir.name]
        scan_baselines.append(min(scan_floors, key=lambda item: item["mean_elevation_m"]))
    scan_level_groups = _height_groups(scan_baselines)
    scan_level_by_scan = {}
    scan_levels = []
    for index, group in enumerate(scan_level_groups):
        name = _level_name(index)
        scan_ids = sorted({plane["scan_id"] for plane in group}, key=lambda scan_id: _scan_key(Path(scan_id)))
        for scan_id in scan_ids:
            scan_level_by_scan[scan_id] = name
        scan_levels.append({
            "name": name,
            "elevation_min_m": min(plane["mean_elevation_m"] for plane in group),
            "elevation_max_m": max(plane["mean_elevation_m"] for plane in group),
            "scan_ids": scan_ids,
        })

    for scan_dir in scan_dirs:
        scan_id = scan_dir.name
        scan_floors = [plane for plane in floors if plane["scan_id"] == scan_id]
        groups = _height_groups(scan_floors)
        for level_index, group in enumerate(groups):
            name = scan_level_by_scan[scan_id]
            if level_index:
                name += _letter(level_index - 1)
            for plane in group:
                plane["floor_level_name"] = name
                plane["scan_level_name"] = scan_level_by_scan[scan_id]

    floor_height_groups = _height_groups(floors)
    for index, group in enumerate(floor_height_groups):
        for plane in group:
            plane["height_group_index"] = index

    for ceiling in ceilings:
        below = [plane for plane in floors
                 if plane["scan_id"] == ceiling["scan_id"]
                 and plane["mean_elevation_m"] <= ceiling["mean_elevation_m"]]
        associated = max(below, key=lambda plane: plane["mean_elevation_m"], default=None)
        ceiling["floor_level_name"] = associated["floor_level_name"] if associated else None
        ceiling["scan_level_name"] = associated["scan_level_name"] if associated else None

    return {
        "schema_version": 1,
        "source": "per-scan aligned Phase 06 regularized element meshes",
        "source_alignment_dir": str(alignment_dir),
        "height_gap_threshold_m": HEIGHT_GAP_M,
        "floor_level_count": len(floor_height_groups),
        "scan_level_count": len(scan_level_groups),
        "scan_count": len(scan_dirs),
        "floor_plane_count": len(floors),
        "ceiling_plane_count": len(ceilings),
        "scan_levels": scan_levels,
        "floor_height_groups": [
            {
                "height_group_index": index,
                "elevation_min_m": min(plane["mean_elevation_m"] for plane in group),
                "elevation_max_m": max(plane["mean_elevation_m"] for plane in group),
                "plane_ids": sorted(plane["plane_id"] for plane in group),
            }
            for index, group in enumerate(floor_height_groups)
        ],
        "floor_planes": floors,
        "ceiling_planes": ceilings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR,
                        help="Phase 07 output directory containing scans/.")
    args = parser.parse_args()

    metadata = build_floor_level_metadata(args.alignment_dir)
    output = args.alignment_dir.parent / "artifacts" / "floor_level_metadata.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
