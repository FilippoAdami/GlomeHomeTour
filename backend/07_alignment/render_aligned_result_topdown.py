#!/usr/bin/env python3
"""Renders publication-quality orthographic top-down views of the unified multi-room alignment result.

Generates:
1. aligned_result_3d_doors_topdown.png: 3D raycast surface render with Phase 05 doors in SOLID RED.
2. aligned_result_3d_textured_topdown.png: 3D raycast surface render with realistic phototextures.
3. aligned_result_topdown_annotated.png: Architectural floor plan with 3D shaded surface underlay,
   door dimension callouts, room labels, and Manhattan coordinate grid.
4. aligned_result_topdown_dual.png: Side-by-side comparison of textured model and red doors model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd


def render_orthographic_mesh_topdown(
    mesh_path: Path,
    resolution: int = 2048,
    is_red_doors: bool = False,
    light_dir: np.ndarray = np.array([0.4, 0.8, -0.4]),
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Renders orthographic top-down view of mesh using Open3D RaycastingScene with directional lighting."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_triangle_normals()

    # Filter downward-facing ceiling faces above 2.0m height
    fn = np.asarray(mesh.triangle_normals)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    fc = verts[faces].mean(axis=1)

    is_ceiling = (fn[:, 1] < -0.2) & (fc[:, 1] > 2.0)
    keep_faces = ~is_ceiling

    used_verts, inv = np.unique(faces[keep_faces], return_inverse=True)
    sub_m = o3d.geometry.TriangleMesh()
    sub_m.vertices = o3d.utility.Vector3dVector(verts[used_verts])
    sub_m.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
    if mesh.has_vertex_colors():
        sub_m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used_verts])
    sub_m.compute_vertex_normals()
    sub_m.compute_triangle_normals()

    bmin = sub_m.get_min_bound()
    bmax = sub_m.get_max_bound()
    center = (bmin + bmax) / 2.0

    margin = 0.06
    extent_x = (bmax[0] - bmin[0]) * (1.0 + margin)
    extent_z = (bmax[2] - bmin[2]) * (1.0 + margin)
    half_size = max(extent_x, extent_z) / 2.0

    pixel_size = (2.0 * half_size) / resolution
    x = (np.arange(resolution) + 0.5 - resolution / 2.0) * pixel_size + center[0]
    z = (np.arange(resolution) + 0.5 - resolution / 2.0) * pixel_size + center[2]

    xx, zz = np.meshgrid(x, z)
    origins = np.stack([xx, np.full_like(xx, bmax[1] + 5.0), zz], axis=-1)
    directions = np.zeros_like(origins)
    directions[..., 1] = -1.0

    rays = np.concatenate([origins, directions], axis=-1).reshape(-1, 6).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sub_m))
    hits = scene.list_intersections(o3d.core.Tensor(rays))

    ray_ids = hits["ray_ids"].numpy().astype(np.int64)
    ids = hits["primitive_ids"].numpy().astype(np.int64)
    distances = hits["t_hit"].numpy()
    uvs = hits["primitive_uvs"].numpy()

    order = np.lexsort((distances, ray_ids))
    tri_normals = np.asarray(sub_m.triangle_normals)

    # Pass outward shell through (downward facing normals)
    outward = tri_normals[ids, 1] > 0.05
    eligible = order[~outward[order]]
    eligible_rays = ray_ids[eligible]
    selected = eligible[np.r_[True, eligible_rays[1:] != eligible_rays[:-1]]]
    selected_rays = ray_ids[selected]
    selected_ids = ids[selected]

    vert_colors = np.asarray(sub_m.vertex_colors)
    sub_faces = np.asarray(sub_m.triangles)
    uv = uvs[selected]
    w = np.column_stack((1.0 - uv.sum(axis=1), uv))
    face_colors = vert_colors[sub_faces[selected_ids]]
    base_colors = np.einsum("ij,ijk->ik", w, face_colors)

    if not is_red_doors:
        # Apply sRGB gamma correction for phototextures
        base_colors = np.where(
            base_colors <= 0.0031308,
            12.92 * base_colors,
            1.055 * np.power(np.maximum(base_colors, 0.0), 1.0 / 2.4) - 0.055,
        )

    l_dir = light_dir / np.linalg.norm(light_dir)
    hit_normals = tri_normals[selected_ids]
    ndotl = np.clip(np.einsum("ij,j->i", hit_normals, l_dir), 0.0, 1.0)

    ambient = 0.42 if is_red_doors else 0.55
    diffuse = 0.58 if is_red_doors else 0.45
    shading = ambient + diffuse * ndotl

    shaded_colors = base_colors * shading[:, None]
    shaded_colors = np.clip(shaded_colors, 0.0, 1.0)

    bg_color = np.array([0.08, 0.08, 0.09])
    img = np.tile(bg_color, (resolution, resolution, 1))
    img.reshape(-1, 3)[selected_rays] = shaded_colors

    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Compute bounds for extent: [x_min, x_max, neg_z_min, neg_z_max]
    x_min = float(center[0] - half_size)
    x_max = float(center[0] + half_size)
    neg_z_min = float(-center[2] - half_size)
    neg_z_max = float(-center[2] + half_size)
    extent = (x_min, x_max, neg_z_min, neg_z_max)

    return img_uint8, extent


def create_annotated_topdown_plan(
    render_img: np.ndarray,
    extent: Tuple[float, float, float, float],
    door_entries: List[Dict[str, Any]],
    transforms_dict: Dict[str, np.ndarray],
    output_png: Path,
) -> None:
    """Creates a high-resolution annotated architectural floor plan combining the 3D surface render, callouts, and stats."""
    x_min, x_max, neg_z_min, neg_z_max = extent

    fig, ax = plt.subplots(figsize=(16, 16), facecolor="#101012")
    ax.set_facecolor("#101012")

    # Display shaded 3D mesh surface as underlay
    ax.imshow(render_img, extent=[x_min, x_max, neg_z_min, neg_z_max], origin="upper", zorder=1)

    # Room bounding labels
    room_centers = {
        "ROOM 1: BEDROOM & MEZZANINE (SCAN 0)": (0.0, 0.0),
        "ROOM 2: HALLWAY (SCAN 1)": (3.2, 3.2),
        "ROOM 3: BATHROOM (SCAN 2)": (3.3, -1.0),
    }
    for rlabel, (rx, rz) in room_centers.items():
        ax.text(
            rx, rz, rlabel,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#f0f0f0",
            bbox=dict(boxstyle="round,pad=0.4", fc="#18181c", ec="#555566", lw=1.2, alpha=0.88),
            zorder=5,
        )

    # Annotate door instances with leader arrows
    for i, entry in enumerate(door_entries):
        sname = entry["scan"]
        T = transforms_dict.get(sname, np.eye(4))
        c_local = np.array(entry["centroid_local"])
        c_global = (c_local @ T[:3, :3].T) + T[:3, 3]
        cx = c_global[0]
        cz_fwd = -c_global[2]

        if "Bedroom" in entry["description"]:
            offset_x, offset_y = -40, 20
            ha = "right"
        elif "North" in entry["description"]:
            offset_x, offset_y = 50, -18
            ha = "left"
        elif "SW Habitable" in entry["description"]:
            offset_x, offset_y = -48, -32
            ha = "right"
        elif "Mid-W" in entry["description"]:
            offset_x, offset_y = -48, 2
            ha = "right"
        elif "East" in entry["description"]:
            offset_x, offset_y = 48, 2
            ha = "left"
        elif "South Entrance" in entry["description"]:
            offset_x, offset_y = 48, 20
            ha = "left"
        elif "Bathroom" in entry["description"]:
            offset_x, offset_y = 0, -32
            ha = "center"
        else:
            offset_x, offset_y = 35, 10
            ha = "left"

        door_text = (
            f"DOOR #{i+1}: {entry['description']}\n"
            f"Span Width: {entry['width_m']*100:.1f} cm | Height: {entry['height_m']:.2f} m (Area={entry['area_m2']:.1f}m²)"
        )
        ax.annotate(
            door_text,
            (cx, cz_fwd),
            textcoords="offset points", xytext=(offset_x, offset_y),
            ha=ha, va="center",
            fontsize=8.0, fontweight="bold", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.35", fc="#250006", ec="#ff2244", lw=1.5, alpha=0.92),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.10", color="#ff4466", lw=1.5),
            zorder=9,
        )

    # Statistics card
    single_doors = [d for d in door_entries if d["width_m"] >= 0.8 and d["width_m"] <= 1.3]
    avg_single_w = float(np.mean([d["width_m"] for d in single_doors]))
    all_w = float(np.mean([d["width_m"] for d in door_entries]))

    stats_box = (
        f"PHASE 05 EMBEDDING-GEMMA DOOR CLASSES & DIMENSIONS\n"
        f"=====================================================\n"
        f"Scan 0 Door Class: Class 22 (Bedroom Exit Door)\n"
        f"Scan 1 Door Class: Class 5  (All Hallway Doors & North Nook)\n"
        f"Scan 2 Door Class: Class 11 (Bathroom Entrance Door)\n"
        f"-----------------------------------------------------\n"
        f"Average Single Door Span (5 doors): {avg_single_w*100:.1f} cm ({avg_single_w:.3f} m)\n"
        f"Average Span Across All 7 Elements: {all_w*100:.1f} cm ({all_w:.3f} m)\n"
        f"-----------------------------------------------------\n"
        f"1. Scan 0 Bedroom Exit Door:       100.0 cm span (H=2.32m)\n"
        f"2. Scan 1 North 3-Door Nook:       143.6 cm cluster (H=2.62m)\n"
        f"3. Scan 1 SW Habitable Room Door:  117.4 cm span (H=2.29m)\n"
        f"4. Scan 1 East Portal:             114.8 cm span (H=2.30m)\n"
        f"5. Scan 1 Mid-W Habitable Door:    126.0 cm span (H=2.27m)\n"
        f"6. Scan 1 South Entrance Lintel:    24.5 cm span (H=1.08m)\n"
        f"7. Scan 2 Bathroom Entrance Door:  104.1 cm span (H=2.88m)"
    )

    ax.text(
        0.02, 0.98, stats_box,
        transform=ax.transAxes,
        fontsize=8.0, fontfamily="monospace", fontweight="bold",
        color="#ffffff", va="top", ha="left",
        bbox=dict(boxstyle="square,pad=0.6", fc="#180005", ec="#ff2244", lw=2.0, alpha=0.95),
        zorder=10,
    )

    ax.set_title("Unified Multi-Room 3D Mesh Result with Phase 05 Door Classes in SOLID RED", color="#ffffff", fontsize=15, pad=14, fontweight="bold")
    ax.set_xlabel("Global Manhattan X (meters)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Global Manhattan -Z / Forward (meters)", color="#cccccc", fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(neg_z_min, neg_z_max)
    ax.grid(True, color="#252528", linestyle=":", lw=0.8)
    ax.tick_params(colors="#aaaaaa")

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved annotated 3D floor plan to: {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render top-down views of aligned result.")
    parser.add_argument("--scene-dir", type=str, default="backend/current_scene")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    multi_room_dir = scene_dir / "multi_room_alignment"
    red_mesh_p = multi_room_dir / "aligned_sealed_meshes_with_doors_red.ply"
    tex_mesh_p = multi_room_dir / "aligned_sealed_meshes.ply"
    transforms_json = scene_dir / "aligned_scans_transforms.json"

    with open(transforms_json, "r") as f:
        manifest = json.load(f)
    transforms_dict = {
        sname: np.array(sdata["transform_matrix"])
        for sname, sdata in manifest["scans"].items()
    }

    # Load door summary
    from visualize_doors_alignment import get_door_classes_from_phase5, load_instances_summary_for_doors
    door_classes = get_door_classes_from_phase5(scene_dir)
    door_entries = load_instances_summary_for_doors(scene_dir, door_classes)

    # 1. Render Red Doors 3D Mesh Topdown
    print("Rendering 3D Mesh with Solid Red Doors...")
    out_red_png = scene_dir / "aligned_result_3d_doors_topdown.png"
    red_img, extent = render_orthographic_mesh_topdown(red_mesh_p, resolution=2048, is_red_doors=True)
    cv2.imwrite(str(out_red_png), cv2.cvtColor(red_img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_red_png}")

    # 2. Render Textured 3D Mesh Topdown
    print("Rendering Textured 3D Mesh...")
    out_tex_png = scene_dir / "aligned_result_3d_textured_topdown.png"
    tex_img, _ = render_orthographic_mesh_topdown(tex_mesh_p, resolution=2048, is_red_doors=False)
    cv2.imwrite(str(out_tex_png), cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_tex_png}")

    # 3. Create Annotated Floor Plan with 3D Shaded Underlay
    print("Creating Annotated Floor Plan...")
    out_annotated_png = scene_dir / "aligned_result_topdown_annotated.png"
    create_annotated_topdown_plan(red_img, extent, door_entries, transforms_dict, out_annotated_png)

    # 4. Create Side-by-Side Dual View (Textured + Red Doors Annotated)
    print("Creating Dual Comparison View...")
    im_tex = cv2.imread(str(out_tex_png))
    im_ann = cv2.imread(str(out_annotated_png))
    h = min(im_tex.shape[0], im_ann.shape[0])
    w_tex = int(im_tex.shape[1] * (h / im_tex.shape[0]))
    w_ann = int(im_ann.shape[1] * (h / im_ann.shape[0]))
    r_tex = cv2.resize(im_tex, (w_tex, h), interpolation=cv2.INTER_AREA)
    r_ann = cv2.resize(im_ann, (w_ann, h), interpolation=cv2.INTER_AREA)
    dual_img = np.hstack([r_tex, r_ann])
    out_dual_png = scene_dir / "aligned_result_topdown_dual.png"
    cv2.imwrite(str(out_dual_png), dual_img)
    print(f"Saved: {out_dual_png}")

    # 5. Copy all artifacts to brain artifact directory
    brain_dir = Path("/home/monday/.gemini/antigravity/brain/463322cb-961f-4ef6-844a-d36dd919babd")
    for fname in [
        "aligned_result_3d_doors_topdown.png",
        "aligned_result_3d_textured_topdown.png",
        "aligned_result_topdown_annotated.png",
        "aligned_result_topdown_dual.png",
    ]:
        src = scene_dir / fname
        dst = brain_dir / fname
        shutil.copy2(src, dst)
        print(f"Copied {fname} to artifacts dir.")


if __name__ == "__main__":
    main()
