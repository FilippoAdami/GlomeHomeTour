#!/usr/bin/env python3
"""Report whether each fitted floor has coherent fine-scale color texture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from planar_pattern_regularization import _plane_frame, detect_visual_texture

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import read_ply_face_flags

FLOOR_ROLE = 0


def _face_property(path: Path, name: str, count: int) -> np.ndarray:
    values = read_ply_face_flags(path, name)
    if values is None or len(values) != count:
        raise ValueError(f"{path} is missing a valid face property {name!r}")
    return np.asarray(values)


def detect_floor_textures(instances_dir: Path) -> Path:
    level_dir = instances_dir / "intermediate_floor_levels"
    manifest_path = level_dir / "floor_level_intermediate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    levels = manifest.get("levels") if isinstance(manifest, dict) else None
    if not isinstance(levels, list) or not levels:
        raise ValueError(f"No fitted floor levels found in {manifest_path}")

    results = []
    for level in levels:
        level_id = level.get("level")
        source_name = level.get("fitted_output_file")
        if (not isinstance(level_id, int) or level_id < 0 or not isinstance(source_name, str)
                or source_name != f"floor_level_{level_id:03d}_fitted.ply"):
            raise ValueError(f"Invalid fitted floor entry in level manifest: {level!r}")
        source_path = level_dir / source_name
        mesh = o3d.io.read_triangle_mesh(str(source_path))
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int64)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
        if not len(vertices) or not len(faces) or len(colors) != len(vertices):
            raise ValueError(f"Floor support PLY must contain geometry and vertex RGB: {source_path}")
        roles = _face_property(source_path, "surface_role", len(faces))
        previews = _face_property(source_path, "is_fitted_plane", len(faces))
        hole_fills = _face_property(source_path, "is_hole_fill", len(faces))
        area_ids = _face_property(source_path, "floor_area_id", len(faces))
        triangles = vertices[faces]
        crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        triangle_areas = np.linalg.norm(crosses, axis=1) * 0.5
        centers = triangles.mean(axis=1)
        triangle_colors = colors[faces].mean(axis=1)
        observed_floor = (roles == FLOOR_ROLE) & (previews == 0) & (hole_fills == 0)

        area_results = []
        for area in level.get("floor_areas", []):
            area_id = area.get("area_id")
            equation = np.asarray(area.get("plane_equation"), dtype=np.float64)
            if not isinstance(area_id, int) or equation.shape != (4,):
                raise ValueError(f"Invalid fitted plane metadata for floor {level_id}")
            selected = observed_floor & (area_ids == area_id) & (triangle_areas > 1e-12)
            normal, axis_u, axis_v, _ = _plane_frame(equation)
            uv = np.column_stack((centers[selected] @ axis_u, centers[selected] @ axis_v))
            diagnosis = detect_visual_texture(uv, triangle_colors[selected], triangle_areas[selected])
            area_results.append({"area_id": area_id, **diagnosis})
        if not area_results:
            raise ValueError(f"Floor {level_id} has no fitted areas to inspect")
        results.append({
            "level": level_id,
            "support_source_file": source_name,
            "support_filter": {"surface_role": FLOOR_ROLE, "is_fitted_plane": 0, "is_hole_fill": 0},
            "texture_present": any(area["present"] for area in area_results),
            "areas": area_results,
        })

    output_path = level_dir / "floor_texture_presence_manifest.json"
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(json.dumps({
        "detector": "fine_scale_color_structure_presence",
        "does_not_synthesize_or_apply_texture": True,
        "criteria": {
            "minimum_observed_cell_fraction": 0.50,
            "minimum_highpass_color_residual_p90": 0.01,
            "minimum_spatial_autocorrelation": 0.25,
            "minimum_median_texture_events_per_m": 2.0,
            "minimum_fraction_of_lines_with_4_events_per_m": 0.10,
        },
        "levels": results,
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, output_path)
    for result in results:
        answer = "YES" if result["texture_present"] else "NO"
        print(f"Floor level {result['level']:03d}: {answer}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path,
                        default=Path("backend/current_scene/scan_0/mesh_elaboration"))
    args = parser.parse_args()
    detect_floor_textures(args.instances_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
