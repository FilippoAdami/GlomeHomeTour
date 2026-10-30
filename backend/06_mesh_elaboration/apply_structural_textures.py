#!/usr/bin/env python3
"""
Apply Structural Textures.

Discovers and applies authentic seamless textures to regularized structural geometry:
1. Floor parquet synthesis (quilted pattern & chromatic alignment)
2. Ceiling beam texture synthesis (periodic beam strip tiles)
3. Wall plaster lighting regularization & color unification
4. Re-paints regularized surfaces, slabs, walls, and all_structural_regularized.ply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recognize_and_regularize_patterns import (
    PatternClassification,
    analyze_and_synthesize_planar_texture,
    detect_wall_color_splits,
    estimate_scene_chromatic_adaptation,
    export_combined,
    export_ply,
    paint_surface,
    regularize_wall_lighting_and_outliers,
    synthesize_ceiling_unit_tile,
    synthesize_floor_unit_tile,
    unify_adjacent_wall_colors,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_3DGS_to_mesh"))
from fill_mesh_holes import read_ply_face_flags


def parse_args():
    parser = argparse.ArgumentParser(description="Apply structural textures to regularized planes.")
    parser.add_argument("--instances-dir", type=Path, default=Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances"))
    parser.add_argument("--output-dir", type=Path, default=Path("backend/current_scene/06_mesh_elaboration/extracted_structural_instances"))
    parser.add_argument("--enhanced-projections-dir", type=Path,
                        help="Use SEESR-enhanced perpendicular projections as source colors before normal texture refinement")
    return parser.parse_args()


def load_enhanced_projections(projections_dir: Path):
    if not projections_dir:
        return {}
    manifest_path = projections_dir / "plane_texture_projections.json"
    manifest = json.loads(manifest_path.read_text())
    images_dir = projections_dir / "projected_seesr_3step"
    result = {}
    for unit in manifest["units"]:
        path = images_dir / unit["filename"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Enhanced projection missing: {path}")
        result[unit["unit_name"]] = (unit, image[:, :, ::-1].copy())
    return result


def sample_enhanced_projection(projection, points):
    import cv2

    metadata, image = projection
    points = np.asarray(points, dtype=np.float64)
    u_axis = np.asarray(metadata["u_axis"])
    v_axis = np.asarray(metadata["v_axis"])
    scale = metadata.get("pixel_scale")
    if scale is None:
        scale = min(
            (metadata["width"] - 1) / max(1e-9, metadata["max_u"] - metadata["min_u"]),
            (metadata["height"] - 1) / max(1e-9, metadata["max_v"] - metadata["min_v"]),
        )
    px = (points @ u_axis - metadata["min_u"]) * scale + metadata.get("offset_x", 0)
    py = (metadata["max_v"] - points @ v_axis) * scale + metadata.get("offset_y", 0)
    px = np.clip(px, 0, metadata["width"] - 1).astype(np.float32)
    py = np.clip(py, 0, metadata["height"] - 1).astype(np.float32)
    sampled = np.empty((len(points), 3), dtype=np.uint8)
    for start in range(0, len(points), 16000):
        stop = min(start + 16000, len(points))
        map_x = px[start:stop].reshape(-1, 1)
        map_y = py[start:stop].reshape(-1, 1)
        sampled[start:stop] = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR).reshape(-1, 3)
    return sampled.astype(np.float64) / 255.0


def main():
    args = parse_args()
    pattern_manifest_path = args.output_dir / "plane_patterns_manifest.json"
    if not pattern_manifest_path.is_file():
        pattern_manifest_path = args.instances_dir / "plane_patterns_manifest.json"
    if not pattern_manifest_path.is_file():
        print("Error: plane_patterns_manifest.json not found")
        return 1

    wall_path = args.instances_dir / "wall_planes_manifest.json"
    surface_path = args.instances_dir / "floors_ceilings_planes_manifest.json"
    if not wall_path.is_file() or not surface_path.is_file():
        print("Error: Required fitted geometry manifest not found")
        return 1

    manifest = json.loads(pattern_manifest_path.read_text())
    walls = json.loads(wall_path.read_text())
    elements = json.loads(surface_path.read_text()).get("elements", [])
    basis = walls["orthogonal_basis"]
    a, b = np.array(basis["dir_A_east_west"]), np.array(basis["dir_B_north_south"])
    wall_planes = walls.get("wall_planes", [])
    enhanced_projections = load_enhanced_projections(args.enhanced_projections_dir) if args.enhanced_projections_dir else {}

    # Global Chromatic Adaptation & White Balance Calibration
    wall_source_path = walls.get("artifacts", {}).get("cleaned_wall_pieces")
    wall_source = o3d.io.read_triangle_mesh(str(wall_source_path)) if wall_source_path and Path(wall_source_path).is_file() else o3d.geometry.TriangleMesh()
    wall_v = np.asarray(wall_source.vertices)
    wall_c = np.asarray(wall_source.vertex_colors) if len(wall_source.vertex_colors) == len(wall_v) else np.full((len(wall_v), 3), 0.6)
    scene_gain = estimate_scene_chromatic_adaptation(wall_c)

    units_by_name: Dict[str, Dict[str, Any]] = {}
    for element in elements:
        name = element["name"]
        source = Path(element["source_file"])
        if not source.is_file():
            continue
        mesh = o3d.io.read_triangle_mesh(str(source))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        colors = np.asarray(mesh.vertex_colors)
        if not len(colors):
            colors = np.full((len(vertices), 3), 0.6)
        if name in enhanced_projections:
            colors = sample_enhanced_projection(enhanced_projections[name], vertices)
        colors = np.clip(colors * scene_gain, 0.0, 1.0)
        equation = np.array(element["plane_equation"], dtype=float)
        equation /= np.linalg.norm(equation[:3])
        role = element["type"]

        face_flags = read_ply_face_flags(source, "is_hole_fill")
        if face_flags is not None and len(face_flags) == len(faces):
            orig_face_mask = (face_flags == 0)
        else:
            orig_face_mask = np.ones(len(faces), dtype=bool)

        if np.any(orig_face_mask):
            orig_v_idx = np.unique(faces[orig_face_mask])
            orig_vertices = vertices[orig_v_idx]
            orig_colors = colors[orig_v_idx]
            orig_tree = cKDTree(orig_vertices)
        else:
            orig_vertices = vertices
            orig_colors = colors
            orig_tree = cKDTree(vertices)

        # Retrieve pattern classification from manifest
        u_manifest = next((u for u in manifest.get("units", []) if u.get("unit_name") == name), None)
        p_class = PatternClassification(
            pattern_type=u_manifest["pattern_classification"]["pattern_type"] if u_manifest else "simple_and_plain",
            confidence=1.0,
            orientation="horizontal" if role != "wall" else "vertical",
            parameters=u_manifest["pattern_classification"].get("parameters", {}) if u_manifest else {},
            summary=""
        )

        unit_dict = {
            "name": name,
            "role": role,
            "equation": equation,
            "classification": p_class,
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
            "orig_vertices": orig_vertices,
            "orig_colors": orig_colors,
            "orig_tree": orig_tree,
            "color_tree": cKDTree(vertices),
            "preserve_source": bool(element.get("preserve_source")),
        }

        if role == "floor":
            unit_dict["seamless_tile"] = synthesize_floor_unit_tile(unit_dict, a, b)
            unit_dict["dir_A"] = a
            unit_dict["dir_B"] = b
            unit_dict["tile_res_m"] = 0.005
        elif role == "ceiling" and p_class.pattern_type == "beams":
            b_tile, s_off = synthesize_ceiling_unit_tile(unit_dict)
            unit_dict["seamless_tile"] = b_tile
            unit_dict["beam_s_offset"] = s_off
            unit_dict["tile_res_m"] = 0.005

        units_by_name[name] = unit_dict

    assembled = []

    # 1. Paint and re-export floor & ceiling regularized surfaces
    for u_info in manifest.get("units", []):
        name = u_info.get("unit_name")
        if u_info.get("structural_role") == "wall":
            continue
        unit = units_by_name.get(name)
        if not unit:
            continue
        ply_path = Path(u_info.get("regularized_artifact", args.output_dir / f"{name}_regularized.ply"))
        if not ply_path.is_file():
            continue
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        colors = paint_surface(unit, v)
        export_ply(ply_path, v, t, colors)

    # 2. Paint and re-export slabs
    for s_info in manifest.get("slabs", []):
        slab_path = Path(s_info.get("file_path", args.output_dir / f"{s_info.get('slab_name')}.ply"))
        if not slab_path.is_file():
            continue
        mesh = o3d.io.read_triangle_mesh(str(slab_path))
        slab_v, slab_t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        floor = units_by_name.get(s_info.get("floor"))
        ceiling = units_by_name.get(s_info.get("ceiling"))
        slab_c = np.zeros((len(slab_v), 3), dtype=np.float64)
        top_mask = slab_v[:, 1] >= (slab_v[:, 1].min() + slab_v[:, 1].max()) / 2.0
        if np.any(top_mask) and floor:
            slab_c[top_mask] = paint_surface(floor, slab_v[top_mask])
        if np.any(~top_mask) and ceiling:
            slab_c[~top_mask] = paint_surface(ceiling, slab_v[~top_mask])
        export_ply(slab_path, slab_v, slab_t, slab_c)
        assembled.append((slab_v, slab_t, slab_c))

    # Add remaining floor/ceiling surfaces to assembled
    for u_info in manifest.get("units", []):
        if u_info.get("structural_role") == "wall":
            continue
        ply_path = Path(u_info.get("regularized_artifact", args.output_dir / f"{u_info.get('unit_name')}_regularized.ply"))
        if ply_path.is_file() and not any(s.get("floor") == u_info.get("unit_name") for s in manifest.get("slabs", [])):
            mesh = o3d.io.read_triangle_mesh(str(ply_path))
            assembled.append((np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)))

    # 3. Paint walls with outlier removal, regularized lighting, and color unification
    wall_source_meshes = []
    if len(wall_source.vertices) > 0:
        wall_source_meshes.append(wall_source)
    for p in sorted(args.instances_dir.glob("wall_inst*.ply")):
        wm = o3d.io.read_triangle_mesh(str(p))
        if len(wm.vertices) > 0:
            wall_source_meshes.append(wm)

    all_v, all_c, all_t, base_idx = [], [], [], 0
    for wm in wall_source_meshes:
        mv = np.asarray(wm.vertices)
        mc = np.asarray(wm.vertex_colors)
        mt = np.asarray(wm.triangles)
        if len(mv) == 0:
            continue
        if len(mc) != len(mv):
            mc = np.full((len(mv), 3), 0.6)
        mc = np.clip(mc * scene_gain, 0.0, 1.0)
        all_v.append(mv)
        all_c.append(mc)
        if len(mt) > 0:
            all_t.append(mt + base_idx)
        base_idx += len(mv)

    if all_v:
        wall_full_v = np.vstack(all_v)
        wall_full_c = np.vstack(all_c)
        wall_full_t = np.vstack(all_t) if all_t else np.empty((0, 3), dtype=np.int32)
    else:
        wall_full_v = np.empty((0, 3), dtype=np.float64)
        wall_full_c = np.empty((0, 3), dtype=np.float64)
        wall_full_t = np.empty((0, 3), dtype=np.int32)

    wall_tree = cKDTree(wall_full_v) if len(wall_full_v) > 0 else None

    if len(wall_full_t) > 0:
        wall_v0 = wall_full_v[wall_full_t[:, 0]]
        wall_v1 = wall_full_v[wall_full_t[:, 1]]
        wall_v2 = wall_full_v[wall_full_t[:, 2]]
        wall_centers = (wall_v0 + wall_v1 + wall_v2) / 3.0
        wall_fnorm = np.cross(wall_v1 - wall_v0, wall_v2 - wall_v0)
        wall_fnorm /= np.maximum(1e-12, np.linalg.norm(wall_fnorm, axis=1, keepdims=True))
    else:
        wall_centers = np.empty((0, 3), dtype=np.float64)
        wall_fnorm = np.empty((0, 3), dtype=np.float64)

    prepared_walls = []
    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    wall_entries = [u for u in manifest.get("units", []) if u.get("structural_role") == "wall"]
    wall_meshes = []

    for u_wall in wall_entries:
        wall_name = u_wall["unit_name"]
        ply_path = Path(u_wall.get("regularized_artifact", args.output_dir / f"{wall_name}_regularized.ply"))
        if not ply_path.is_file():
            continue
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(t):
            continue

        wp_eq = np.array(u_wall["plane_equation"], dtype=np.float64)
        wp_n = wp_eq[:3] / max(1e-12, np.linalg.norm(wp_eq[:3]))
        wp_d = float(wp_eq[3])

        wp_pts = np.empty((0, 3), dtype=np.float64)
        wp_cols = np.empty((0, 3), dtype=np.float64)

        if len(wall_full_t) > 0:
            dists = np.abs(wall_centers @ wp_n + wp_d)
            inward_mask = (dists < 0.15) & (wall_fnorm @ wp_n > 0.15)
            if np.count_nonzero(inward_mask) < 4:
                inward_mask = (dists < 0.15)
            if np.count_nonzero(inward_mask) >= 4:
                inward_v_idx = np.unique(wall_full_t[inward_mask])
                wp_pts = wall_full_v[inward_v_idx]
                wp_cols = wall_full_c[inward_v_idx]

        if len(wp_pts) < 4 and wall_tree is not None:
            _, near_idx = wall_tree.query(v, k=1)
            wp_pts = wall_full_v[near_idx]
            wp_cols = wall_full_c[near_idx]

        if len(wp_pts) and wall_name in enhanced_projections:
            wp_cols = sample_enhanced_projection(enhanced_projections[wall_name], wp_pts)

        if len(wp_pts) < 4:
            wp_pts = v
            wp_cols = np.full((len(v), 3), 0.6)

        t_axis = np.cross(up_axis, wp_n)
        t_axis = a if np.linalg.norm(t_axis) < 1e-4 else t_axis / np.linalg.norm(t_axis)

        clean_pts, reg_cols, inl_u, inl_v, diag = regularize_wall_lighting_and_outliers(
            wp_pts, wp_cols, t_axis, up_axis
        )
        split_info = detect_wall_color_splits(inl_v, inl_u, reg_cols)

        prepared_walls.append({
            "name": wall_name,
            "plane_equation": wp_eq,
            "v": v,
            "t": t,
            "ply_path": ply_path,
            "wp_n": wp_n,
            "t_axis": t_axis,
            "clean_pts": clean_pts,
            "reg_cols": reg_cols,
            "diag": diag,
            "split_info": split_info,
            "u_wall": u_wall,
        })

    prepared_walls = unify_adjacent_wall_colors(prepared_walls)
    target_wall_white = np.array([0.88, 0.88, 0.86])
    for pw in prepared_walls:
        if not pw.get("split_info", {}).get("is_split"):
            pw["unified_color"] = target_wall_white

    for pw in prepared_walls:
        v, t = pw["v"], pw["t"]
        wall_unit = {
            "name": pw["name"],
            "role": "wall",
            "equation": pw["plane_equation"],
            "vertices": pw["clean_pts"],
            "colors": pw["reg_cols"],
            "orig_vertices": pw["clean_pts"],
            "orig_colors": pw["reg_cols"],
            "orig_tree": cKDTree(pw["clean_pts"]) if len(pw["clean_pts"]) >= 4 else wall_tree,
            "color_tree": cKDTree(pw["clean_pts"]) if len(pw["clean_pts"]) >= 4 else wall_tree,
            "u_axis": pw["t_axis"],
            "unified_color": pw.get("unified_color") if pw.get("unified_color") is not None else target_wall_white,
            "split_info": pw.get("split_info"),
            "classification": PatternClassification(
                pattern_type="simple_and_plain",
                confidence=1.0,
                orientation="vertical",
                parameters={},
                summary="Regularized vertical wall plane",
            ),
        }

        colors = analyze_and_synthesize_planar_texture(wall_unit, v)
        export_ply(pw["ply_path"], v, t, colors)
        wall_meshes.append((v, t, colors))

        # Update pattern parameters in manifest
        pw["u_wall"]["pattern_classification"]["parameters"] = {
            "unified_color": (pw.get("unified_color") if pw.get("unified_color") is not None else target_wall_white).tolist(),
            "is_split": pw["split_info"]["is_split"],
            "split_type": pw["split_info"]["split_type"],
            "delta_E": pw["split_info"]["delta_E"],
            "inlier_ratio": pw["diag"].get("inlier_ratio", 1.0),
        }

    # 4. Re-export wall_regularized.ply and all_structural_regularized.ply
    if wall_meshes:
        export_combined(args.output_dir / "wall_regularized.ply", wall_meshes)
        assembled.extend(wall_meshes)

    extraction_path = args.instances_dir / "structural_instances_manifest.json"
    if extraction_path.is_file():
        for item in json.loads(extraction_path.read_text()).get("instances", []):
            if item.get("role") != "stair":
                continue
            stair_file = args.instances_dir / item["file_name"]
            if not stair_file.is_file():
                continue
            mesh = o3d.io.read_triangle_mesh(str(stair_file))
            sv, sf, sc = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
            if not len(sc):
                sc = np.full((len(sv), 3), 0.6)
            assembled.append((sv, sf, sc))

    if assembled:
        export_combined(args.output_dir / "all_structural_regularized.ply", assembled)

    pattern_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Successfully applied textures to regularized structural surfaces in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
