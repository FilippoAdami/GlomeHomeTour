#!/usr/bin/env python3
"""Remove off-plane outliers from floor-level intermediate meshes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags
from planar_outlier_filter import (
    FIT_DISTANCE_M,
    KEEP_DISTANCE_M,
    MIN_SUPPORT_FRACTION,
    filter_planar_outliers,
)


SURFACE_ROLES = (0, 1, 2)


def _clean_mesh(path: Path) -> tuple[dict, dict]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():
        raise ValueError(f"Intermediate floor mesh has empty or non-finite geometry: {path}")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"Intermediate floor mesh has invalid triangle indices: {path}")

    roles = read_ply_face_flags(path, "surface_role")
    if roles is None:
        roles = np.zeros(len(triangles), dtype=np.uint8)
    flags = read_ply_face_flags(path, "is_hole_fill")
    if flags is None:
        flags = np.zeros(len(triangles), dtype=np.uint8)
    if len(roles) != len(triangles) or len(flags) != len(triangles):
        raise ValueError(f"Intermediate floor face attributes do not match its triangles: {path}")
    if not np.isin(roles, SURFACE_ROLES).all():
        raise ValueError(f"Intermediate floor mesh contains an unknown surface role: {path}")

    keep = np.zeros(len(triangles), dtype=bool)
    vertical_removed = other_outlier_removed = degenerate_removed = 0
    for role in SURFACE_ROLES:
        face_ids = np.flatnonzero(roles == role)
        if not len(face_ids):
            continue
        role_keep, role_stats = filter_planar_outliers(vertices, triangles[face_ids])
        keep[face_ids] = role_keep
        vertical_removed += role_stats["far_vertical_triangles_removed"]
        other_outlier_removed += role_stats["other_outlier_triangles_removed"]
        degenerate_removed += role_stats["degenerate_triangles_removed"]

    selected = np.flatnonzero(keep)
    if not len(selected):
        raise ValueError(f"Cleaning removed every triangle from floor level: {path}")
    used, inverse = np.unique(triangles[selected], return_inverse=True)
    output_vertices = vertices[used]
    output_triangles = inverse.reshape(-1, 3)
    colors = np.asarray(mesh.vertex_colors)
    output_colors = None
    if len(colors) == len(vertices):
        output_colors = np.rint(np.clip(colors[used], 0.0, 1.0) * 255).astype(np.uint8)
    output_flags, output_roles = flags[selected], roles[selected].astype(np.uint8)
    return {
        "vertices": output_vertices,
        "triangles": output_triangles,
        "colors": output_colors,
        "flags": output_flags,
        "roles": output_roles,
    }, {
        "source_triangles": int(len(triangles)),
        "vertical_triangles_removed": vertical_removed,
        "other_outlier_triangles_removed": other_outlier_removed,
        "outlier_triangles_removed": vertical_removed + other_outlier_removed,
        "degenerate_triangles_removed": degenerate_removed,
        "output_triangles": int(len(output_triangles)),
        "output_vertices": int(len(output_vertices)),
        "output_floor_triangles": int(np.count_nonzero(output_roles == 0)),
        "output_ramp_triangles": int(np.count_nonzero(output_roles == 1)),
        "output_ceiling_triangles": int(np.count_nonzero(output_roles == 2)),
        "output_floor_vertices": int(len(np.unique(output_triangles[output_roles == 0]))
                                       if np.any(output_roles == 0) else 0),
        "output_ramp_vertices": int(len(np.unique(output_triangles[output_roles == 1]))
                                      if np.any(output_roles == 1) else 0),
        "output_ceiling_vertices": int(len(np.unique(output_triangles[output_roles == 2]))
                                         if np.any(output_roles == 2) else 0),
    }


def clean_floor_level_intermediates(instances_dir: Path) -> Path:
    output_dir = instances_dir / "intermediate_floor_levels"
    manifest_path = output_dir / "floor_level_intermediate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("levels"), list):
        raise ValueError(f"Invalid floor-level manifest: {manifest_path}")

    updated_levels, outputs = [], []
    with tempfile.TemporaryDirectory(prefix=".clean_floor_levels_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        for source_level in manifest["levels"]:
            if not isinstance(source_level, dict):
                raise ValueError("Invalid floor-level manifest entry")
            filename = source_level.get("output_file")
            if not isinstance(filename, str) or not re.fullmatch(r"floor_level_\d{3}_intermediate\.ply", filename):
                raise ValueError(f"Invalid intermediate floor filename: {filename!r}")
            data, stats = _clean_mesh(output_dir / filename)
            output = temp_dir / filename
            if not export_ply_with_face_attributes(
                output, data["vertices"], data["triangles"], data["colors"],
                face_attributes={
                    "is_hole_fill": ("uchar", data["flags"]),
                    "surface_role": ("uchar", data["roles"]),
                },
            ):
                raise OSError(f"Could not write cleaned floor-level PLY: {output}")
            updated = dict(source_level)
            updated.update(stats)
            updated["num_vertices"] = stats["output_vertices"]
            updated["num_triangles"] = stats["output_triangles"]
            updated["num_floor_vertices"] = stats["output_floor_vertices"]
            updated["num_ramp_vertices"] = stats["output_ramp_vertices"]
            updated["num_ceiling_vertices"] = stats["output_ceiling_vertices"]
            updated["num_floor_triangles"] = stats["output_floor_triangles"]
            updated["num_ramp_triangles"] = stats["output_ramp_triangles"]
            updated["num_ceiling_triangles"] = stats["output_ceiling_triangles"]
            updated_levels.append(updated)
            outputs.append(filename)

        manifest["levels"] = updated_levels
        manifest["triangle_cleaning"] = {
            "method": "per-role dominant-plane distance filter",
            "plane_fit_distance_m": FIT_DISTANCE_M,
            "maximum_kept_plane_distance_m": KEEP_DISTANCE_M,
            "minimum_plane_support_fraction": MIN_SUPPORT_FRACTION,
            "behavior": "Removes degenerate triangles and faces farther than 0.20 m from the supported role plane, regardless of orientation; vertical faces within 0.20 m are retained. Plane pruning is skipped when support is below 50%.",
        }
        manifest["surface_role_values"] = {"0": "floor", "1": "ramp", "2": "ceiling"}
        (temp_dir / manifest_path.name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in outputs:
            os.replace(temp_dir / filename, output_dir / filename)
        os.replace(temp_dir / manifest_path.name, manifest_path)

    for level in updated_levels:
        print(f"Level {level['level']:03d}: removed {level['vertical_triangles_removed']} far vertical and "
              f"{level['other_outlier_triangles_removed']} other outlier triangles")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing intermediate floor levels")
    args = parser.parse_args(argv)
    try:
        clean_floor_level_intermediates(args.instances_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not clean floor-level intermediates: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
