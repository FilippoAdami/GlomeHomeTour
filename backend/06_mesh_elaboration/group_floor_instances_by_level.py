#!/usr/bin/env python3
"""Group extracted floor instances into the declared number of scene levels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags


def read_floor_count(path: Path) -> int:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object with a 'floors' field in {path}")
        value = data.get("floors")
    else:
        matches = [line.split(":", 1)[1].strip() for line in path.read_text(encoding="utf-8").splitlines()
                   if line.split(":", 1)[0].strip() == "floors" and ":" in line]
        if len(matches) != 1:
            raise ValueError(f"Expected one 'floors:' entry in {path}")
        value = matches[0]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid floor count in {path}: {value!r}") from exc
    if not np.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"Floor count must be a positive integer in {path}: {value!r}")
    return int(number)


def _height_groups(heights: np.ndarray, count: int) -> list[list[int]]:
    """Minimum within-group height variance with non-empty, height-ordered groups."""
    n = len(heights)
    if count > n:
        raise ValueError(f"Cannot form {count} floor levels from only {n} extracted floor instances")
    values = heights - heights[0]
    prefix = np.r_[0.0, np.cumsum(values)]
    prefix_sq = np.r_[0.0, np.cumsum(values * values)]

    def cost(start: int, end: int) -> float:
        size = end - start
        total = prefix[end] - prefix[start]
        return max(0.0, float(prefix_sq[end] - prefix_sq[start] - total * total / size))

    scores = np.full((count + 1, n + 1), np.inf)
    previous = np.full((count + 1, n + 1), -1, dtype=np.int64)
    scores[0, 0] = 0.0
    for groups in range(1, count + 1):
        for end in range(groups, n + 1):
            for start in range(groups - 1, end):
                score = scores[groups - 1, start] + cost(start, end)
                if score < scores[groups, end]:
                    scores[groups, end] = score
                    previous[groups, end] = start

    result = []
    end = n
    for groups in range(count, 0, -1):
        start = int(previous[groups, end])
        result.append(list(range(start, end)))
        end = start
    return list(reversed(result))


def _load_instances(instances_dir: Path, role: str, required: bool) -> list[dict]:
    manifest_path = instances_dir / "structural_instances_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("instances"), list):
        raise ValueError(f"Invalid structural instance manifest: {manifest_path}")
    items = []
    seen = set()
    for entry in manifest.get("instances", []):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid instance entry in {manifest_path}: {entry!r}")
        if entry.get("role") != role:
            continue
        filename = entry.get("file_name")
        if not isinstance(filename, str) or Path(filename).name != filename or not re.fullmatch(rf"{role}_inst\d+\.ply", filename):
            raise ValueError(f"Unsafe or invalid {role} instance filename in {manifest_path}: {filename!r}")
        if filename in seen:
            raise ValueError(f"Duplicate {role} instance in {manifest_path}: {filename}")
        seen.add(filename)
        path = instances_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"{role.title()} instance PLY listed in the manifest is missing: {path}")
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
            raise ValueError(f"{role.title()} instance has empty or non-finite geometry: {path}")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError(f"{role.title()} instance has invalid triangle indices: {path}")
        colors = np.asarray(mesh.vertex_colors)
        if len(colors) not in (0, len(vertices)):
            raise ValueError(f"{role.title()} instance has incomplete vertex colors: {path}")
        flags = read_ply_face_flags(path, "is_hole_fill")
        if flags is None:
            flags = np.zeros(len(triangles), dtype=np.uint8)
        elif len(flags) != len(triangles):
            raise ValueError(f"{role.title()} instance face flags do not match its triangles: {path}")
        # shortcut: each extracted PLY is indivisible, when one spans levels upgrade to per-triangle splitting.
        items.append({
            "file_name": filename,
            "role": role,
            "path": path,
            "vertices": vertices,
            "triangles": triangles,
            "colors": colors,
            "flags": flags,
            "height": float(np.median(vertices[:, 1])),
        })
    if not items and required:
        raise ValueError(f"No {role} instances found in {manifest_path}")
    return sorted(items, key=lambda item: (item["height"], item["file_name"]))


def _edge_keys(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return unique exact geometric edge keys; extracted PLYs retain source float32 coordinates."""
    pairs = np.asarray(vertices, dtype="<f4")[triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)]
    left, right = pairs[:, 0], pairs[:, 1]
    first_difference = np.argmax(left != right, axis=1)
    swap = np.take_along_axis(left, first_difference[:, None], axis=1)[:, 0] > \
        np.take_along_axis(right, first_difference[:, None], axis=1)[:, 0]
    ordered = np.concatenate((np.where(swap[:, None], right, left),
                              np.where(swap[:, None], left, right)), axis=1)
    ordered[ordered == 0] = 0  # Normalize signed zero before byte-key comparison.
    keys = ordered.reshape(-1).view(np.dtype((np.void, ordered.dtype.itemsize * 6)))
    return np.unique(keys)


def _write_merged_floor(path: Path, items: list[dict], include_surface_roles: bool = False) -> None:
    vertices, triangles, flags, colors, surface_roles = [], [], [], [], []
    have_colors = any(len(item["colors"]) for item in items)
    offset = 0
    for item in items:
        vertices.append(item["vertices"])
        triangles.append(item["triangles"] + offset)
        flags.append(item["flags"])
        if include_surface_roles:
            role_value = 0 if item["role"] == "floor" else 1
            surface_roles.append(np.full(len(item["triangles"]), role_value, dtype=np.uint8))
        if have_colors:
            rgb = item["colors"]
            colors.append(np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                          if len(rgb) else np.full((len(item["vertices"]), 3), 128, dtype=np.uint8))
        offset += len(item["vertices"])
    attributes = {"is_hole_fill": ("uchar", np.concatenate(flags))}
    if include_surface_roles:
        attributes["surface_role"] = ("uchar", np.concatenate(surface_roles))
    written = export_ply_with_face_attributes(
        path, np.concatenate(vertices), np.concatenate(triangles),
        np.concatenate(colors) if have_colors else None,
        face_attributes=attributes,
    )
    if not written or not path.is_file():
        raise OSError(f"Could not write aggregated floor PLY: {path}")


def group_floor_instances(instances_dir: Path, scene_size_file: Path) -> Path:
    floor_count = read_floor_count(scene_size_file)
    instances = _load_instances(instances_dir, "floor", required=True)
    ramps = _load_instances(instances_dir, "ramp", required=False)
    groups = _height_groups(np.asarray([item["height"] for item in instances]), floor_count)
    output_dir = instances_dir / "intermediate_floor_levels"
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = []
    names = []
    floor_edges = [_edge_keys(item["vertices"], item["triangles"]) for item in instances] if ramps else []
    ramp_edges = {item["file_name"]: _edge_keys(item["vertices"], item["triangles"]) for item in ramps}
    with tempfile.TemporaryDirectory(prefix=".floor_levels_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        for level, indices in enumerate(groups):
            members = [instances[index] for index in indices]
            member_edges = np.unique(np.concatenate([floor_edges[index] for index in indices])) if ramps else None
            level_ramps = ([ramp for ramp in ramps if np.intersect1d(
                member_edges, ramp_edges[ramp["file_name"]], assume_unique=True).size]
                if ramps else [])
            aggregate_parts = members + level_ramps
            filename = f"floor_level_{level:03d}_intermediate.ply"
            staged_path = temp_dir / filename
            if len(aggregate_parts) == 1:
                shutil.copyfile(members[0]["path"], staged_path)
            else:
                _write_merged_floor(staged_path, aggregate_parts, include_surface_roles=bool(level_ramps))
            names.append(filename)
            levels.append({
                "level": level,
                "elevation_y_median": float(np.median([item["height"] for item in members])),
                "instances": [{"file_name": item["file_name"], "elevation_y_median": item["height"]}
                              for item in members],
                "ramps": [{"file_name": item["file_name"], "elevation_y_median": item["height"]}
                          for item in level_ramps],
                "output_file": filename,
                "num_floor_vertices": int(sum(len(item["vertices"]) for item in members)),
                "num_ramp_vertices": int(sum(len(item["vertices"]) for item in level_ramps)),
                "num_vertices": int(sum(len(item["vertices"]) for item in aggregate_parts)),
                "num_floor_triangles": int(sum(len(item["triangles"]) for item in members)),
                "num_ramp_triangles": int(sum(len(item["triangles"]) for item in level_ramps)),
                "num_triangles": int(sum(len(item["triangles"]) for item in aggregate_parts)),
            })
        output_manifest = {"floor_count": floor_count, "levels": levels}
        if any(level["ramps"] for level in levels):
            output_manifest["surface_role_values"] = {"0": "floor", "1": "ramp", "2": "ceiling"}
        (temp_dir / "floor_level_intermediate_manifest.json").write_text(
            json.dumps(output_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        for filename in names:
            os.replace(temp_dir / filename, output_dir / filename)
        os.replace(temp_dir / "floor_level_intermediate_manifest.json",
                   output_dir / "floor_level_intermediate_manifest.json")

    keep = set(names)
    for path in output_dir.iterdir():
        if path.is_file() and re.fullmatch(r"floor_level_\d{3}_intermediate\.ply", path.name) and path.name not in keep:
            path.unlink()
    print(f"Grouped {len(instances)} floor instances into {floor_count} levels: {output_dir}")
    for level in levels:
        members = ", ".join(item["file_name"] for item in level["instances"])
        ramp_names = ", ".join(item["file_name"] for item in level["ramps"])
        print(f"  Level {level['level']:03d} (Y={level['elevation_y_median']:.3f}): {members}; ramps: {ramp_names or 'none'}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Step 1 output directory containing the structural instance manifest and floor PLYs")
    parser.add_argument("--scene-size-file", type=Path, required=True,
                        help="scene_size.txt or JSON containing the expected floor count")
    args = parser.parse_args(argv)
    try:
        group_floor_instances(args.instances_dir, args.scene_size_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not group floor instances: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
