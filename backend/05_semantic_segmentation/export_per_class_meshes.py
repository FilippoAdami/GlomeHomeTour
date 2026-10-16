#!/usr/bin/env python3
"""
Exports per-class isolated sub-meshes for Blender outliner inspection and multi-class visualization.

Outputs:
  - per_class_meshes/class_{cid:03d}_{role}_{n_inst}inst_{tris}tris.ply for each active class
  - per_class_meshes/scene_clay_backdrop.ply (entire room in soft neutral clay)
  - per_class_meshes/blender_import_layers.py (one-click Blender loader)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import open3d as o3d

from detect_structural_classes import get_class_color, export_colored_ply


def export_per_class_meshes(
    mesh_path: Path,
    structural_json_path: Path,
    instances_csv_path: Path,
    output_dir: Path,
    include_clay_backdrop: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base mesh from: {mesh_path}...")
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    v_colors = (np.asarray(mesh.vertex_colors) * 255.0).astype(np.uint8)
    print(f"  Loaded {len(verts):,} vertices, {len(tris):,} triangles in {time.perf_counter() - t0:.2f}s")

    with open(structural_json_path) as f:
        struct_data = json.load(f)
    classifications = struct_data["classifications"]
    active_classes = sorted([int(k) for k in classifications.keys()])

    palette = np.array([get_class_color(c)[1] for c in active_classes], dtype=np.float32) * 255.0
    tri_c = v_colors[tris].mean(axis=1)
    d_palette = np.linalg.norm(tri_c[:, None, :] - palette[None, :, :], axis=2)
    labels = np.array(active_classes)[d_palette.argmin(axis=1)]

    # Read instances count per class
    inst_counts = {}
    inst_details = {}
    if instances_csv_path.is_file():
        with open(instances_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["Class ID"])
                inst_counts[cid] = inst_counts.get(cid, 0) + 1
                inst_details.setdefault(cid, []).append(int(row["Triangle Count"]))

    print(f"\nExporting {len(active_classes)} isolated class meshes to {output_dir}/...")
    exported_files = []

    for cid in active_classes:
        c_info = classifications.get(str(cid), classifications.get(cid, {}))
        role = c_info.get("structural_role", "object")
        n_inst = inst_counts.get(cid, 1)

        c_mask = labels == cid
        c_tris = tris[c_mask]
        if len(c_tris) == 0:
            continue

        used_v_idx, new_tri_idx = np.unique(c_tris, return_inverse=True)
        sub_verts = verts[used_v_idx]
        sub_tris = new_tri_idx.reshape(c_tris.shape)
        sub_colors = v_colors[used_v_idx]

        tri_count = len(sub_tris)
        fname = f"class_{cid:03d}_{role}_{n_inst}inst_{tri_count}tris.ply"
        out_ply = output_dir / fname

        export_colored_ply(out_ply, sub_verts, sub_tris, vertex_colors_rgb=sub_colors)
        exported_files.append((fname, cid, role, n_inst, tri_count))

    if include_clay_backdrop:
        clay_path = output_dir / "scene_clay_backdrop.ply"
        clay_colors = np.full((len(verts), 3), 210, dtype=np.uint8)
        export_colored_ply(clay_path, verts, tris, vertex_colors_rgb=clay_colors)
        print(f"Exported neutral clay backdrop: {clay_path.name}")

    # Generate Blender Python import script
    blender_script_path = output_dir / "blender_import_layers.py"
    blender_code = f"""# Blender Python Script: One-Click Outliner Layer Importer
# In Blender: Scripting Tab -> New -> Paste this code -> Run Script (Alt+P)
# Or run from shell: blender --python blender_import_layers.py

import bpy
from pathlib import Path

meshes_dir = Path(r"{output_dir.resolve()}")
coll_name = "GlomeHomeTour_Classes"

# Create or get dedicated collection
if coll_name in bpy.data.collections:
    coll = bpy.data.collections[coll_name]
else:
    coll = bpy.data.collections.new(coll_name)
    bpy.context.scene.collection.children.link(coll)

# Import each class PLY
ply_files = sorted(list(meshes_dir.glob("class_*.ply")))
print(f"Importing {{len(ply_files)}} class layers into '{{coll_name}}'...")

for ply_path in ply_files:
    bpy.ops.wm.ply_import(filepath=str(ply_path))
    obj = bpy.context.selected_objects[0]
    obj.name = ply_path.stem
    # Link to collection
    for old_coll in obj.users_collection:
        old_coll.objects.unlink(obj)
    coll.objects.link(obj)

# Import clay backdrop (muted by default)
clay_file = meshes_dir / "scene_clay_backdrop.ply"
if clay_file.is_file():
    bpy.ops.wm.ply_import(filepath=str(clay_file))
    clay_obj = bpy.context.selected_objects[0]
    clay_obj.name = "Scene_Clay_Backdrop"
    for old_coll in clay_obj.users_collection:
        old_coll.objects.unlink(clay_obj)
    coll.objects.link(clay_obj)
    # Hide in viewport by default
    clay_obj.hide_viewport = False

print("Done! You can now toggle visibility of any class in the Blender Outliner.")
"""
    with open(blender_script_path, "w") as f:
        f.write(blender_code)
    print(f"Generated Blender importer script: {blender_script_path.name}")

    # Generate README table
    readme_path = output_dir / "README.md"
    readme_lines = [
        "# Isolated 3D Class Meshes\n",
        f"Contains **{len(exported_files)} isolated PLY sub-meshes**, one for each segmented physical class.\n",
        "## Usage in Blender:",
        "1. Open Blender.",
        "2. Run `per_class_meshes/blender_import_layers.py` (Scripting tab -> Run Script).",
        "3. Every class will appear in the Outliner tree with its name, instance count, and triangle count.",
        "4. Click the **eye icon 👁** to toggle individual classes on or off with zero distraction from other classes!\n",
        "## Exported Classes Table:\n",
        "| File Name | Class ID | Role | Instances | Triangles |",
        "| :--- | :---: | :--- | :---: | :---: |",
    ]
    for fname, cid, role, n_inst, tri_count in exported_files:
        readme_lines.append(f"| `{fname}` | **#{cid}** | `{role}` | {n_inst} | {tri_count:,} |")

    with open(readme_path, "w") as f:
        f.write("\n".join(readme_lines) + "\n")
    print(f"Generated README summary: {readme_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Export per-class isolated 3D meshes")
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/step2_mesh_projection/mesh_projected_refined.ply"),
    )
    parser.add_argument(
        "--structural-json",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/step3_structural_detection/structural_classes.json"),
    )
    parser.add_argument(
        "--instances-csv",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/step3_structural_detection/instances_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backend/current_scene/06c_semantic_segmentation/per_class_meshes"),
    )
    args = parser.parse_args()

    export_per_class_meshes(
        mesh_path=args.mesh_path,
        structural_json_path=args.structural_json,
        instances_csv_path=args.instances_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
