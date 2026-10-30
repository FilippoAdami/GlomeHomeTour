#!/usr/bin/env python3
"""Append nearby lower ceiling instances to each intermediate floor-level mesh."""
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

MAX_BELOW_M = 0.5


def _read_instance(instances_dir: Path, filename: str, role: str) -> dict:
    if Path(filename).name != filename or not re.fullmatch(rf"{role}_inst\d+\.ply", filename):
        raise ValueError(f"Invalid {role} instance filename: {filename!r}")
    path = instances_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Instance PLY listed in structural manifest is missing: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f"Instance has empty or non-finite geometry: {path}")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"Instance has invalid triangle indices: {path}")
    rgb = np.asarray(mesh.vertex_colors)
    if len(rgb) == len(vertices):
        colors = np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    elif len(rgb) == 0:
        colors = None
    else:
        raise ValueError(f"Instance has incomplete vertex colors: {path}")
    flags = read_ply_face_flags(path, "is_hole_fill")
    if flags is None:
        flags = np.zeros(len(triangles), dtype=np.uint8)
    elif len(flags) != len(triangles):
        raise ValueError(f"Instance face flags do not match its triangles: {path}")
    triangle_y = vertices[triangles, 1].mean(axis=1)
    return {"file_name": filename, "role": role, "path": path, "vertices": vertices,
            "triangles": triangles, "colors": colors, "flags": flags,
            "triangle_y": triangle_y}


def _write_mesh(path: Path, parts: list[tuple[dict, int]], mark_surface_roles: bool) -> None:
    vertices, triangles, flags, colors, surface_roles = [], [], [], [], []
    have_colors = any(part[0]["colors"] is not None for part in parts)
    offset = 0
    for item, role_value in parts:
        vertices.append(item["vertices"])
        triangles.append(item["triangles"] + offset)
        flags.append(item["flags"])
        if have_colors:
            colors.append(item["colors"] if item["colors"] is not None else
                          np.full((len(item["vertices"]), 3), 128, dtype=np.uint8))
        if mark_surface_roles:
            surface_roles.append(np.full(len(item["triangles"]), role_value, dtype=np.uint8))
        offset += len(item["vertices"])

    face_attributes = {"is_hole_fill": ("uchar", np.concatenate(flags))}
    if mark_surface_roles:
        face_attributes["surface_role"] = ("uchar", np.concatenate(surface_roles))
    written = export_ply_with_face_attributes(
        path, np.concatenate(vertices), np.concatenate(triangles),
        np.concatenate(colors) if have_colors else None,
        face_attributes=face_attributes,
    )
    if not written or not path.is_file():
        raise OSError(f"Could not write updated floor-level PLY: {path}")


def add_ceilings_to_floor_levels(instances_dir: Path) -> Path:
    output_dir = instances_dir / "intermediate_floor_levels"
    levels_path = output_dir / "floor_level_intermediate_manifest.json"
    structural_path = instances_dir / "structural_instances_manifest.json"
    levels_manifest = json.loads(levels_path.read_text(encoding="utf-8"))
    structural_manifest = json.loads(structural_path.read_text(encoding="utf-8"))
    if (not isinstance(levels_manifest, dict) or not isinstance(levels_manifest.get("levels"), list)
            or not isinstance(structural_manifest, dict)
            or not isinstance(structural_manifest.get("instances"), list)):
        raise ValueError("Invalid floor-level or structural instance manifest")

    floor_files, ramp_files, ceiling_names = set(), set(), []
    for item in structural_manifest["instances"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid instance entry in structural instance manifest")
        role = item.get("role")
        if role not in {"floor", "ramp", "ceiling"}:
            continue
        filename = item.get("file_name")
        if not isinstance(filename, str) or not re.fullmatch(rf"{role}_inst\d+\.ply", filename):
            raise ValueError(f"Invalid {role} instance filename: {filename!r}")
        names = floor_files if role == "floor" else ramp_files if role == "ramp" else ceiling_names
        if filename in names:
            raise ValueError(f"Duplicate {role} instance in structural manifest: {filename}")
        if role == "floor":
            floor_files.add(filename)
        elif role == "ramp":
            ramp_files.add(filename)
        else:
            ceiling_names.append(filename)
    ceiling_names.sort()
    if not floor_files or not levels_manifest["levels"]:
        raise ValueError("No intermediate floor levels or extracted floor instances are available")

    cache: dict[str, dict] = {}

    def load(filename: str, role: str) -> dict:
        if filename not in cache:
            cache[filename] = _read_instance(instances_dir, filename, role)
        return cache[filename]

    ceilings = [load(name, "ceiling") for name in ceiling_names]
    previous_matches = False
    updated_levels = []
    outputs: list[tuple[str, list[tuple[dict, int]], bool]] = []

    for source_level in levels_manifest["levels"]:
        if not isinstance(source_level, dict) or not isinstance(source_level.get("instances"), list):
            raise ValueError("Invalid level entry in floor-level manifest")
        output_file = source_level.get("output_file")
        if not isinstance(output_file, str) or not re.fullmatch(r"floor_level_\d{3}_intermediate\.ply", output_file):
            raise ValueError(f"Invalid intermediate floor filename: {output_file!r}")
        if any(not isinstance(item, dict) for item in source_level["instances"]):
            raise ValueError(f"Invalid source floor entries for {output_file}")
        floor_names = [item.get("file_name") for item in source_level["instances"]]
        if not floor_names or any(not isinstance(name, str) or name not in floor_files for name in floor_names):
            raise ValueError(f"Invalid source floor instances for {output_file}")
        if len(set(floor_names)) != len(floor_names):
            raise ValueError(f"Duplicate source floor instances for {output_file}")
        floor_parts = [load(name, "floor") for name in floor_names]
        ramp_entries = source_level.get("ramps", [])
        if not isinstance(ramp_entries, list) or any(not isinstance(item, dict) for item in ramp_entries):
            raise ValueError(f"Invalid source ramp entries for {output_file}")
        ramp_names = [item.get("file_name") for item in ramp_entries]
        if any(not isinstance(name, str) or name not in ramp_files for name in ramp_names):
            raise ValueError(f"Invalid source ramp instances for {output_file}")
        if len(set(ramp_names)) != len(ramp_names):
            raise ValueError(f"Duplicate source ramp instances for {output_file}")
        ramp_parts = [load(name, "ramp") for name in ramp_names]
        floor_y = float(np.median(np.concatenate([item["triangle_y"] for item in floor_parts])))
        matched = []
        for ceiling in ceilings:
            difference = floor_y - float(np.median(ceiling["triangle_y"]))
            if 0.0 <= difference <= MAX_BELOW_M:
                matched.append((ceiling, difference))
        previous_ceilings = source_level.get("ceilings", [])
        if not isinstance(previous_ceilings, list) or any(not isinstance(item, dict) for item in previous_ceilings):
            raise ValueError(f"Invalid previous ceiling assignments for {output_file}")
        matched_names = [item.get("file_name") for item in previous_ceilings]
        previous_matches |= bool(matched_names)
        updated = dict(source_level)
        updated["floor_triangle_centroid_y_median"] = floor_y
        updated["ceilings"] = [
            {"file_name": ceiling["file_name"],
             "triangle_centroid_y_median": float(np.median(ceiling["triangle_y"])),
             "below_floor_m": difference}
            for ceiling, difference in matched
        ]
        updated["num_floor_vertices"] = int(sum(len(item["vertices"]) for item in floor_parts))
        updated["num_ramp_vertices"] = int(sum(len(item["vertices"]) for item in ramp_parts))
        updated["num_ceiling_vertices"] = int(sum(len(item["vertices"]) for item, _ in matched))
        updated["num_vertices"] = (updated["num_floor_vertices"] + updated["num_ramp_vertices"]
                                    + updated["num_ceiling_vertices"])
        updated["num_floor_triangles"] = int(sum(len(item["triangles"]) for item in floor_parts))
        updated["num_ramp_triangles"] = int(sum(len(item["triangles"]) for item in ramp_parts))
        updated["num_ceiling_triangles"] = int(sum(len(item["triangles"]) for item, _ in matched))
        updated["num_triangles"] = (updated["num_floor_triangles"] + updated["num_ramp_triangles"]
                                     + updated["num_ceiling_triangles"])
        updated_levels.append(updated)
        if matched:
            outputs.append((output_file, [(item, 0) for item in floor_parts] +
                            [(item, 1) for item in ramp_parts] +
                            [(item, 2) for item, _ in matched], True))
        elif matched_names:
            # Restore the pre-ceiling aggregate if a previous run had appended ceilings.
            outputs.append((output_file, [(item, 0) for item in floor_parts] +
                            [(item, 1) for item in ramp_parts], bool(ramp_parts)))

    if not any(level["ceilings"] for level in updated_levels) and not previous_matches:
        print(f"No ceiling instances within {MAX_BELOW_M:.2f} m below any floor; intermediate PLYs unchanged")
        return output_dir

    levels_manifest["levels"] = updated_levels
    levels_manifest["ceiling_match_max_below_m"] = MAX_BELOW_M
    if any(level["ceilings"] for level in updated_levels) or any(level.get("ramps") for level in updated_levels):
        levels_manifest["surface_role_values"] = {"0": "floor", "1": "ramp", "2": "ceiling"}
    else:
        levels_manifest.pop("surface_role_values", None)

    with tempfile.TemporaryDirectory(prefix=".floor_ceilings_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        for filename, parts, mark_roles in outputs:
            if not mark_roles and len(parts) == 1:
                shutil.copyfile(parts[0][0]["path"], temp_dir / filename)
            else:
                _write_mesh(temp_dir / filename, parts, mark_roles)
        (temp_dir / levels_path.name).write_text(
            json.dumps(levels_manifest, indent=2) + "\n", encoding="utf-8")
        for filename, _, _ in outputs:
            os.replace(temp_dir / filename, output_dir / filename)
        os.replace(temp_dir / levels_path.name, levels_path)

    print(f"Updated {sum(bool(level['ceilings']) for level in updated_levels)} floor levels with nearby ceilings")
    for level in updated_levels:
        if level["ceilings"]:
            names = ", ".join(item["file_name"] for item in level["ceilings"])
            print(f"  Level {level['level']:03d}: {names}")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing extracted instances and intermediate floors")
    args = parser.parse_args(argv)
    try:
        add_ceilings_to_floor_levels(args.instances_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not add ceiling instances to floor levels: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
