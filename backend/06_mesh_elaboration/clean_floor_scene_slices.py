#!/usr/bin/env python3
"""Remove very long, thin hole-fill triangles from per-floor scene slices."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
from plyfile import PlyData

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR.parent / "04_3DGS_to_mesh"))
from fill_mesh_holes import export_ply_with_face_attributes

MIN_LONGEST_EDGE_M = 0.30
MIN_ASPECT_RATIO = 20.0
FACE_ATTRIBUTES = {
    "is_hole_fill": "uchar",
    "surface_role": "uchar",
    "is_fitted_plane": "uchar",
    "source_class_id": "int",
    "floor_area_id": "int",
    "ceiling_group_id": "int",
}


def _elongated_fill_faces(vertices: np.ndarray, triangles: np.ndarray,
                          hole_fill: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    candidates = np.flatnonzero((hole_fill == 1) & (fitted == 0))
    remove = np.zeros(len(triangles), dtype=bool)
    if not len(candidates):
        return remove

    points = vertices[triangles[candidates]]
    edge_a = points[:, 1] - points[:, 0]
    edge_b = points[:, 2] - points[:, 1]
    edge_c = points[:, 0] - points[:, 2]
    longest = np.maximum.reduce((np.linalg.norm(edge_a, axis=1),
                                 np.linalg.norm(edge_b, axis=1),
                                 np.linalg.norm(edge_c, axis=1)))
    area = 0.5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                         points[:, 2] - points[:, 0]), axis=1)
    # This is the longest edge divided by the altitude opposite it.
    aspect_ratio = np.divide(longest**2, 2.0 * area,
                             out=np.full(len(area), np.inf), where=area > 1e-12)
    remove[candidates] = (longest >= MIN_LONGEST_EDGE_M) & (aspect_ratio >= MIN_ASPECT_RATIO)
    return remove


def clean_floor_scene_slices(instances_dir: Path) -> Path:
    level_dir = instances_dir / "intermediate_floor_levels"
    manifest_path = level_dir / "floor_level_scene_slices_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("levels"), list):
        raise ValueError(f"Invalid floor scene slice manifest: {manifest_path}")

    outputs = []
    with tempfile.TemporaryDirectory(prefix=".clean_floor_slices_", dir=level_dir) as temp_name:
        temp_dir = Path(temp_name)
        for record in manifest["levels"]:
            filename = record.get("output_file") if isinstance(record, dict) else None
            if not isinstance(filename, str) or not re.fullmatch(r"floor_level_\d{3}_scene_slice\.ply", filename):
                raise ValueError(f"Invalid scene slice output filename: {filename!r}")
            source_path = level_dir / filename
            ply = PlyData.read(str(source_path), known_list_len={"face": {"vertex_indices": 3}})
            vertex_data = ply["vertex"].data
            face_data = ply["face"].data
            vertices = np.column_stack([vertex_data[name] for name in ("x", "y", "z")]).astype(np.float64)
            triangles = np.asarray(face_data["vertex_indices"], dtype=np.int64)
            face_property_names = face_data.dtype.names
            face_attributes = {
                name: face_data[name].copy() if name in face_property_names else None
                for name in FACE_ATTRIBUTES
            }
            if (not len(vertices) or not len(triangles)
                    or any(values is None or len(values) != len(triangles)
                           for values in face_attributes.values())):
                raise ValueError(f"Scene slice is missing required face attributes: {source_path}")

            remove = _elongated_fill_faces(vertices, triangles,
                                           face_attributes["is_hole_fill"],
                                           face_attributes["is_fitted_plane"])
            keep = ~remove
            used, inverse = np.unique(triangles[keep], return_inverse=True)
            colors = (np.column_stack([vertex_data[name] for name in ("red", "green", "blue")])
                      if all(name in vertex_data.dtype.names for name in ("red", "green", "blue")) else None)
            output_path = temp_dir / filename
            if not export_ply_with_face_attributes(
                output_path, vertices[used], inverse.reshape(-1, 3),
                colors[used] if colors is not None else None,
                face_attributes={
                    name: (kind, face_attributes[name][keep])
                    for name, kind in FACE_ATTRIBUTES.items()
                },
            ):
                raise OSError(f"Could not write cleaned scene slice: {output_path}")
            record["elongated_filled_triangles_removed_this_run"] = int(np.count_nonzero(remove))
            outputs.append(filename)

        manifest["elongated_fill_cleanup"] = {
            "minimum_longest_edge_m": MIN_LONGEST_EDGE_M,
            "minimum_aspect_ratio": MIN_ASPECT_RATIO,
            "scope": "source faces where is_hole_fill == 1 and is_fitted_plane == 0",
        }
        temp_manifest = temp_dir / manifest_path.name
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in outputs:
            os.replace(temp_dir / filename, level_dir / filename)
        os.replace(temp_manifest, manifest_path)

    for record in manifest["levels"]:
        print(f"Level {record['level']:03d}: removed "
              f"{record['elongated_filled_triangles_removed_this_run']:,} elongated filled triangles "
              f"from {record['output_file']}")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Stage 06 output directory containing floor scene slices")
    args = parser.parse_args(argv)
    try:
        clean_floor_scene_slices(args.instances_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: Could not clean floor scene slices: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
