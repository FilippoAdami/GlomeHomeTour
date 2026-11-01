"""Preserve validated floor openings across Phase 06 rebuilds."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

CACHE_NAME = "phase06_floor_openings.json"
MIN_OUTLINE_OVERLAP = 0.75
OPENING_CLEARANCE_M = 0.001


def _footprints(rings: list) -> tuple[object, object]:
    polygons = []
    for ring in rings:
        points = np.asarray(ring, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("floor opening cache contains an invalid X/Z ring")
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 1e-8:
            raise ValueError("floor opening cache contains a degenerate X/Z ring")
        polygons.append(polygon)
    if not polygons:
        raise ValueError("floor opening cache contains no rings")

    parents = []
    for index, polygon in enumerate(polygons):
        containers = [other for other, candidate in enumerate(polygons)
                      if other != index and candidate.area > polygon.area
                      and candidate.contains(polygon)]
        parents.append(min(containers, key=lambda other: polygons[other].area) if containers else None)

    depths = []
    for index in range(len(polygons)):
        depth, parent, seen = 0, parents[index], {index}
        while parent is not None:
            if parent in seen:
                raise ValueError("floor opening cache contains nested rings with a cycle")
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        depths.append(depth)

    components = []
    for index, polygon in enumerate(polygons):
        if depths[index] % 2:
            continue
        holes = [polygons[child].exterior.coords for child, parent in enumerate(parents)
                 if parent == index and depths[child] == depths[index] + 1]
        components.append(Polygon(polygon.exterior.coords, holes))
    footprint = unary_union(components)
    outline = unary_union([polygon for polygon, depth in zip(polygons, depths) if depth % 2 == 0])
    if footprint.is_empty or outline.is_empty or not footprint.is_valid:
        raise ValueError("floor opening cache does not form a valid footprint")
    return footprint, outline


def _rings(geometry) -> list[list[list[float]]]:
    result = []
    for polygon in _polygon_parts(geometry):
        if polygon.area <= 1e-8:
            continue
        result.append(np.asarray(polygon.exterior.coords[:-1]).tolist())
        result.extend(np.asarray(ring.coords[:-1]).tolist() for ring in polygon.interiors)
    return result


def _polygon_parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _read_cache(path: Path) -> list[dict] | None:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: ignoring unreadable floor opening cache: {path}")
        return None
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("constraints"), list):
        print(f"Warning: ignoring invalid floor opening cache: {path}")
        return None
    return [item for item in data["constraints"] if isinstance(item, dict)]


def apply_saved_openings(records: list[dict], cache_path: Path) -> set[tuple[int, int]]:
    changed = set()
    constraints = _read_cache(cache_path)
    if not constraints:
        return changed
    for level in records:
        for area in level["areas"]:
            current, outline = _footprints(area["floor_polygon_rings_world_xz_m"])
            openings = []
            for item in constraints:
                if item.get("level") != level["level"] or item.get("area_id") != area["area_id"]:
                    continue
                try:
                    previous, previous_outline = _footprints(item.get("rings_world_xz_m", []))
                except (TypeError, ValueError):
                    continue
                overlap = outline.intersection(previous_outline).area
                if (overlap / outline.area < MIN_OUTLINE_OVERLAP
                        or overlap / previous_outline.area < MIN_OUTLINE_OVERLAP):
                    continue
                safe_outline = outline.buffer(-OPENING_CLEARANCE_M)
                if safe_outline.is_empty:
                    continue
                for opening in _polygon_parts(previous_outline.difference(previous)):
                    if opening.area > 1e-8 and safe_outline.covers(opening):
                        openings.append(opening)
            if openings:
                restored = current.difference(unary_union(openings))
                if current.area - restored.area > 1e-6 and restored.is_valid:
                    area["floor_polygon_rings_world_xz_m"] = _rings(restored)
                    changed.add((int(level["level"]), int(area["area_id"])))
                    print(f"Preserved previous opening(s) for floor {level['level']}, area {area['area_id']}")
    return changed


def update_opening_cache(records: list[dict], cache_path: Path) -> None:
    constraints = _read_cache(cache_path)
    if constraints is None:
        return
    additions = []
    for level in records:
        for area in level["areas"]:
            rings = area["floor_polygon_rings_world_xz_m"]
            try:
                footprint, outline = _footprints(rings)
            except (TypeError, ValueError):
                continue
            openings = outline.difference(footprint)
            if openings.area <= 1e-6:
                continue
            safe_outline = outline.buffer(-OPENING_CLEARANCE_M)
            if safe_outline.is_empty:
                continue
            openings = [opening for opening in _polygon_parts(openings)
                        if safe_outline.covers(opening)]
            if not openings:
                continue
            known = []
            for item in constraints:
                if item.get("level") != level["level"] or item.get("area_id") != area["area_id"]:
                    continue
                try:
                    previous, previous_outline = _footprints(item.get("rings_world_xz_m", []))
                    known.append(previous_outline.difference(previous))
                except (TypeError, ValueError):
                    continue
            openings_geometry = unary_union(openings)
            if known and openings_geometry.difference(unary_union(known)).area <= max(
                    1e-4, openings_geometry.area * 0.10):
                continue
            additions.append({
                "level": int(level["level"]),
                "area_id": int(area["area_id"]),
                "rings_world_xz_m": rings,
            })
    if not additions:
        return

    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        temporary.write_text(json.dumps({"version": 1, "constraints": constraints + additions}, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)

