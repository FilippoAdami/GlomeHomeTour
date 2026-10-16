#!/usr/bin/env python3
"""Report structural levels from regularized surfaces and validated slab artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from shapely.geometry import shape


ROLE_NAMES = {"floor", "ceiling", "wall"}
LEVEL_ELEVATION_TOLERANCE_M = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report structural levels from regularized surfaces and validated slab artifacts.")
    parser.add_argument("--instances-dir", type=Path, default=Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances"), help="Directory containing regularized structural instances")
    parser.add_argument("--scene-size-file", type=Path, default=Path("backend/current_scene/01_poses_refinment/scene_size.txt"), help="Optional scene_size.txt consistency reference")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances"), help="Output directory")
    return parser.parse_args()


def load_scene_size(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        try:
            data[key] = float(value) if "." in value else int(value)
        except ValueError:
            data[key] = value
    return data


def plane_y(plane: np.ndarray, xz: tuple[float, float]) -> float:
    if abs(plane[1]) < 1e-8:
        raise ValueError("structural plane must not be vertical")
    return float(-(plane[0] * xz[0] + plane[2] * xz[1] + plane[3]) / plane[1])


def footprint(value: Any):
    try:
        geometry = shape(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("footprint_xz must be GeoJSON") from exc
    if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty or not geometry.is_valid:
        raise ValueError("footprint_xz must be a valid non-empty GeoJSON Polygon or MultiPolygon")
    return geometry


def validate_unit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("unit must be an object")
    name, role = value.get("unit_name"), value.get("structural_role")
    if not isinstance(name, str) or not name or role not in ROLE_NAMES:
        raise ValueError("unit requires a non-empty name and floor, ceiling, or wall role")
    if role == "wall":
        return value
    plane = np.asarray(value.get("plane_equation"), dtype=float)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError(f"unit {name} has an invalid plane_equation")
    if role in {"floor", "ceiling"} and abs(plane[1]) < 1e-8:
        raise ValueError(f"unit {name} has a vertical {role} plane")
    elevation = value.get("mean_elevation_y")
    if not isinstance(elevation, (int, float)) or not np.isfinite(elevation):
        raise ValueError(f"unit {name} has an invalid mean_elevation_y")
    return {**value, "plane": plane, "elevation": float(elevation), "geometry": footprint(value.get("footprint_xz"))}


def nearest_overlapping_ceiling(floor: dict[str, Any], ceilings: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for ceiling in ceilings:
        overlap = floor["geometry"].intersection(ceiling["geometry"])
        if overlap.is_empty or overlap.area <= 1e-8:
            continue
        centroid = overlap.centroid
        xz = (float(centroid.x), float(centroid.y))
        floor_y, ceiling_y = plane_y(floor["plane"], xz), plane_y(ceiling["plane"], xz)
        if ceiling_y > floor_y:
            candidates.append((ceiling_y - floor_y, ceiling, xz, floor_y, ceiling_y, float(overlap.area)))
    if not candidates:
        return None
    clear_height, ceiling, xz, floor_y, ceiling_y, overlap_area = min(candidates, key=lambda item: item[0])
    return {"ceiling_unit": ceiling["unit_name"], "clear_height_m": clear_height, "overlap_centroid_xz": list(xz), "floor_elevation_y_at_overlap": floor_y, "ceiling_elevation_y_at_overlap": ceiling_y, "overlap_area_m2": overlap_area}


def group_floors(floors: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for floor in sorted(floors, key=lambda unit: unit["elevation"]):
        if not groups or floor["elevation"] - groups[-1][-1]["elevation"] > LEVEL_ELEVATION_TOLERANCE_M:
            groups.append([floor])
        else:
            groups[-1].append(floor)
    return groups


def find_pattern_manifest(args: argparse.Namespace) -> Path | None:
    for directory in (args.output_dir, args.instances_dir):
        candidate = directory / "plane_patterns_manifest.json"
        if candidate.exists():
            return candidate
    return None


def validate_slabs(slabs: Any, units: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(slabs, list):
        raise ValueError("slabs must be a list")
    validated, names = [], set()
    for slab in slabs:
        if not isinstance(slab, dict):
            raise ValueError("slab must be an object")
        name, floor_name, ceiling_name = slab.get("slab_name"), slab.get("floor"), slab.get("ceiling")
        if not isinstance(name, str) or not name or Path(name).name != name or name in names:
            raise ValueError("slab_name must be a unique safe name")
        if units.get(floor_name, {}).get("structural_role") != "floor" or units.get(ceiling_name, {}).get("structural_role") != "ceiling":
            raise ValueError(f"slab {name} must reference a floor and ceiling unit")
        source = Path(slab.get("file_path", ""))
        if source.suffix != ".ply" or not source.is_file() or slab.get("is_watertight") is not True:
            raise ValueError(f"slab {name} has no validated watertight PLY artifact")
        footprint(slab.get("footprint_xz"))
        plane = np.asarray(slab.get("floor_plane_equation"), dtype=float)
        if plane.shape != (4,) or not np.all(np.isfinite(plane)):
            raise ValueError(f"slab {name} has invalid floor_plane_equation")
        names.add(name)
        validated.append({**slab, "source": source, "output_name": f"{name}.ply"})
    return validated


def main() -> int:
    args = parse_args()
    manifest_path = find_pattern_manifest(args)
    if manifest_path is None:
        print("Error: plane_patterns_manifest.json not found in output or instances directory")
        return 1
    try:
        pattern_manifest = json.loads(manifest_path.read_text())
        raw_units = pattern_manifest.get("units")
        if not isinstance(raw_units, list):
            raise ValueError("units must be a list")
        unit_list = [validate_unit(unit) for unit in raw_units]
        units = {unit["unit_name"]: unit for unit in unit_list}
        if len(units) != len(unit_list):
            raise ValueError("unit_name values must be unique")
        slabs = validate_slabs(pattern_manifest.get("slabs", []), units)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Error: Invalid pattern manifest {manifest_path}: {exc}")
        return 1

    floors = [unit for unit in unit_list if unit["structural_role"] == "floor"]
    ceilings = [unit for unit in unit_list if unit["structural_role"] == "ceiling"]
    levels = []
    for level_index, floor_group in enumerate(group_floors(floors)):
        floor_reports = []
        for floor in floor_group:
            floor_reports.append({
                "unit_name": floor["unit_name"],
                "mean_elevation_y": floor["elevation"],
                "plane_equation": floor["plane_equation"],
                "footprint_xz": floor["footprint_xz"],
                "regularized_artifact": floor.get("regularized_artifact"),
                "pattern_classification": floor.get("pattern_classification"),
                "ceiling_match": nearest_overlapping_ceiling(floor, ceilings),
            })
        levels.append({"level_index": level_index, "mean_floor_elevation_y": float(np.mean([floor["elevation"] for floor in floor_group])), "floor_units": floor_reports})

    scene_info = load_scene_size(args.scene_size_file)
    report = {"source_pattern_manifest": str(manifest_path), "consistency_report": {"scene_size_file": str(args.scene_size_file), "scene_size_floors": scene_info.get("floors"), "detected_floor_units": len(floors), "detected_levels": len(levels)}, "levels": levels, "slabs": [{key: value for key, value in slab.items() if key not in {"source", "output_name"}} for slab in slabs]}

    levels_manifest = args.output_dir / "structural_levels_manifest.json"
    try:
        with tempfile.TemporaryDirectory(dir=args.output_dir.parent, prefix=".structural_levels_") as temporary:
            temporary_dir = Path(temporary)
            for slab in slabs:
                staged = temporary_dir / slab["output_name"]
                destination = args.output_dir / slab["output_name"]
                if slab["source"].resolve() == destination.resolve():
                    staged = slab["source"]
                else:
                    shutil.copy2(slab["source"], staged)
                mesh = o3d.io.read_triangle_mesh(str(staged))
                if len(mesh.triangles) == 0 or not mesh.is_watertight():
                    raise ValueError(f"slab {slab['slab_name']} artifact is not watertight")
            staged_manifest = temporary_dir / levels_manifest.name
            staged_manifest.write_text(json.dumps(report, indent=2))
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for slab in slabs:
                staged = temporary_dir / slab["output_name"]
                if staged.exists():
                    staged.replace(args.output_dir / slab["output_name"])
            staged_manifest.replace(levels_manifest)
    except (OSError, ValueError) as exc:
        print(f"Error: Failed to publish structural levels: {exc}")
        return 1

    print(f"Reported {len(levels)} levels from {len(floors)} floor units and {len(slabs)} slabs.")
    print(f"Manifest saved to: {levels_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
