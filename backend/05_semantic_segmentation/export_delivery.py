#!/usr/bin/env python3
"""
Deliver regularized, clean semantic segmentation outputs aligned to upstream and downstream contracts.

Deliverables:
1. mesh_segmented.ply:
   Exact geometric copy of the Stage 04 sealed mesh (mesh.ply), with:
   - Face order and positions matching mesh.ply; boundary vertices split for exact colors
   - Triangle-level class colors (deterministic 2D panoptic palette)
   - Triangle face properties: int class_id, int instance_id
2. Instances PLYs/:
   - Structural/:
       Wall/
       Floor/
       Ceiling/
       Stairs/
       Columns/
       Ramps/
   - Non-Structural/:
       <class_id>/ (e.g. 1/, 2/, ... in ascending numeric order)
   Every instance PLY has authentic RGB colors from the sealed mesh.
3. class_metadata.json:
   Mapping class IDs to structural role, confidence, and physical details.
4. segmentation.json:
   Full manifest of instances, bounding boxes, counts, areas, and relative file paths.
5. 3d_masks/:
   Downstream compatibility symlinks pointing into Instances PLYs/.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from detect_structural_classes import get_class_color


STRUCTURAL_ROLE_DIRS = {
    "wall": "Wall",
    "floor": "Floor",
    "ceiling": "Ceiling",
    "stairs": "Stairs",
    "column": "Columns",
    "columns": "Columns",
    "ramp": "Ramps",
    "ramps": "Ramps",
}


def export_segmented_mesh_ply(
    output_path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_labels: np.ndarray,
    triangle_instances: np.ndarray,
    active_classes: List[int],
) -> None:
    """
    Exports binary little-endian PLY mesh with vertex class colors and custom face properties:
      property int class_id
      property int instance_id
    """
    num_verts = len(vertices)
    num_faces = len(triangles)

    if np.shape(triangle_labels) != (num_faces,) or np.shape(triangle_instances) != (num_faces,):
        raise ValueError('class and instance labels must match the source mesh face order')

    # Compute triangle RGB class colors
    tri_colors = np.full((num_faces, 3), 128, dtype=np.uint8)
    for cid in active_classes:
        m = triangle_labels == cid
        if np.any(m):
            _, col_rgb = get_class_color(cid)
            tri_colors[m] = (np.array(col_rgb) * 255.0).round().astype(np.uint8)

    # Split display vertices at class boundaries so interpolation cannot invent mixed colors.
    classes, columns = np.unique(triangle_labels, return_inverse=True)
    if num_faces:
        keys = triangles.ravel().astype(np.int64) * len(classes) + np.repeat(columns, 3)
        unique, remapped = np.unique(keys, return_inverse=True)
        vertices = vertices[unique // len(classes)]
        triangles = remapped.reshape(-1, 3)
        palette = np.full((len(classes), 3), 128, dtype=np.uint8)
        _, first = np.unique(triangle_labels, return_index=True)
        palette[:] = tri_colors[first]
        v_colors = palette[unique % len(classes)]
    else:
        v_colors = np.full((num_verts, 3), 128, dtype=np.uint8)
    num_verts = len(vertices)

    v_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    v_data = np.zeros(num_verts, dtype=v_dtype)
    v_data["x"] = vertices[:, 0]
    v_data["y"] = vertices[:, 1]
    v_data["z"] = vertices[:, 2]
    v_data["red"] = v_colors[:, 0]
    v_data["green"] = v_colors[:, 1]
    v_data["blue"] = v_colors[:, 2]

    f_dtype = np.dtype([
        ("n_verts", "u1"),
        ("v0", "<i4"),
        ("v1", "<i4"),
        ("v2", "<i4"),
        ("class_id", "<i4"),
        ("instance_id", "<i4"),
    ])
    f_data = np.zeros(num_faces, dtype=f_dtype)
    f_data["n_verts"] = 3
    f_data["v0"] = triangles[:, 0]
    f_data["v1"] = triangles[:, 1]
    f_data["v2"] = triangles[:, 2]
    f_data["class_id"] = triangle_labels.astype(np.int32)
    f_data["instance_id"] = triangle_instances.astype(np.int32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {num_faces}\n"
        "property list uchar int vertex_indices\n"
        "property int class_id\n"
        "property int instance_id\n"
        "end_header\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v_data.tobytes())
        f.write(f_data.tobytes())
    print(f"Exported segmented scene mesh with class & instance metadata: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def export_single_instance_ply(
    output_path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
) -> None:
    """Exports a single instance sub-mesh with real RGB colors."""
    num_verts = len(vertices)
    num_faces = len(triangles)
    v_colors = np.clip(vertex_colors, 0, 255).astype(np.uint8)

    v_dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ])
    v_data = np.zeros(num_verts, dtype=v_dtype)
    v_data["x"] = vertices[:, 0]
    v_data["y"] = vertices[:, 1]
    v_data["z"] = vertices[:, 2]
    v_data["red"] = v_colors[:, 0]
    v_data["green"] = v_colors[:, 1]
    v_data["blue"] = v_colors[:, 2]

    f_dtype = np.dtype([
        ("n_verts", "u1"),
        ("v0", "<i4"),
        ("v1", "<i4"),
        ("v2", "<i4"),
    ])
    f_data = np.zeros(num_faces, dtype=f_dtype)
    f_data["n_verts"] = 3
    f_data["v0"] = triangles[:, 0]
    f_data["v1"] = triangles[:, 1]
    f_data["v2"] = triangles[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {num_faces}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(v_data.tobytes())
        f.write(f_data.tobytes())


def deliver_stage05_outputs(
    sealed_mesh_path: Path,
    triangle_labels: np.ndarray,
    instances: Dict[str, Dict[str, Any]],
    classifications: Dict[int, Dict[str, Any]],
    output_dir: Path,
    panoptic_npz_dir: Optional[Path] = None,
    panoptic_overlays_dir: Optional[Path] = None,
    create_legacy_symlinks: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates the authoritative regularized delivery package for Stage 05.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=================================================================")
    print(f"DELIVERING REGULARIZED STAGE 05 OUTPUTS")
    print(f"Destination: {output_dir}")
    print(f"=================================================================")

    # 1. Load sealed source mesh to obtain authentic geometry and RGB colors
    t0 = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(str(sealed_mesh_path))
    sealed_verts = np.asarray(mesh.vertices, dtype=np.float32)
    sealed_tris = np.asarray(mesh.triangles, dtype=np.int32)
    num_faces = len(sealed_tris)

    if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(sealed_verts):
        sealed_v_colors = (np.asarray(mesh.vertex_colors) * 255.0).round().astype(np.uint8)
    else:
        sealed_v_colors = np.full((len(sealed_verts), 3), 180, dtype=np.uint8)

    if np.shape(triangle_labels) != (num_faces,):
        raise ValueError('triangle labels must match the sealed source mesh')
    sorted_instances = sorted(instances.items(),
        key=lambda item: (item[1]['class_id'], -len(item[1].get('triangles', []))))
    # Class-local instance IDs agree with the manifest and compatibility filenames.
    triangle_instance_ids = np.zeros(num_faces, dtype=np.int32)
    class_counters = {}
    for inst_key, inst in sorted_instances:
        tri_idx = inst.get("triangles")
        if tri_idx is not None and len(tri_idx) > 0:
            cid = int(inst['class_id'])
            if (np.any(tri_idx < 0) or np.any(tri_idx >= num_faces)
                    or np.any(triangle_labels[tri_idx] != cid) or np.any(triangle_instance_ids[tri_idx])):
                raise ValueError(f'instance {inst_key} has overlapping or mismatched source faces')
            class_counters[cid] = class_counters.get(cid, 0) + 1
            triangle_instance_ids[tri_idx] = class_counters[cid]

    active_classes = sorted(list(classifications.keys()))

    # 3. Export authoritative full scene mesh: mesh_segmented.ply
    mesh_segmented_path = output_dir / "mesh_segmented.ply"
    export_segmented_mesh_ply(
        output_path=mesh_segmented_path,
        vertices=sealed_verts,
        triangles=sealed_tris,
        triangle_labels=triangle_labels,
        triangle_instances=triangle_instance_ids,
        active_classes=active_classes,
    )

    # 4. Prepare clean directory structure for Instances PLYs/
    instances_root = output_dir / "Instances PLYs"
    if instances_root.exists():
        shutil.rmtree(instances_root)
    structural_root = instances_root / "Structural"
    non_structural_root = instances_root / "Non-Structural"

    for folder_name in ["Wall", "Floor", "Ceiling", "Stairs", "Columns", "Ramps"]:
        (structural_root / folder_name).mkdir(parents=True, exist_ok=True)
    non_structural_root.mkdir(parents=True, exist_ok=True)

    legacy_masks_root = output_dir / "3d_masks"
    if legacy_masks_root.exists():
        shutil.rmtree(legacy_masks_root)
    if create_legacy_symlinks:
        legacy_masks_root.mkdir(parents=True, exist_ok=True)

    instances_manifest = []
    class_instance_counters: Dict[int, int] = {}
    role_instance_counters: Dict[str, int] = {}

    print(f"\nExporting {len(sorted_instances)} isolated instance PLYs with authentic sealed mesh colors...")
    for inst_key, inst in sorted_instances:
        cid = int(inst["class_id"])
        c_meta = classifications.get(cid, classifications.get(str(cid), {}))
        role = c_meta.get("structural_role", "object")

        tri_idx = inst.get("triangles")
        if tri_idx is None or len(tri_idx) == 0:
            continue

        class_instance_counters[cid] = class_instance_counters.get(cid, 0) + 1
        inst_idx = class_instance_counters[cid]

        # Extract instance sub-mesh geometry & authentic vertex colors
        inst_tris_global = sealed_tris[tri_idx]
        used_v_idx, remapped_tris = np.unique(inst_tris_global, return_inverse=True)
        sub_verts = sealed_verts[used_v_idx]
        sub_tris = remapped_tris.reshape(inst_tris_global.shape)
        sub_colors = sealed_v_colors[used_v_idx]

        fname = f"instance_{inst_idx:03d}.ply"

        if role in STRUCTURAL_ROLE_DIRS:
            target_dir = structural_root / STRUCTURAL_ROLE_DIRS[role]
            folder = STRUCTURAL_ROLE_DIRS[role]
            role_instance_counters[folder] = role_instance_counters.get(folder, 0) + 1
            fname = f"instance_{role_instance_counters[folder]:03d}.ply"
        else:
            target_dir = non_structural_root / str(cid)

        target_dir.mkdir(parents=True, exist_ok=True)
        target_ply = target_dir / fname
        export_single_instance_ply(target_ply, sub_verts, sub_tris, sub_colors)

        # Create downstream Stage 06 compatibility directory/symlink
        if create_legacy_symlinks:
            leg_class_dir = legacy_masks_root / f"class_{cid:03d}_{role}"
            leg_class_dir.mkdir(parents=True, exist_ok=True)
            leg_ply = leg_class_dir / f"instance_{inst_idx:03d}.ply"
            if leg_ply.is_symlink() or leg_ply.exists():
                leg_ply.unlink(missing_ok=True)
            try:
                rel_target = os.path.relpath(target_ply, leg_class_dir)
                leg_ply.symlink_to(rel_target)
            except Exception:
                shutil.copy2(target_ply, leg_ply)

        instances_manifest.append({
            "instance_key": inst_key,
            "instance_index": inst_idx,
            "class_id": cid,
            "structural_role": role,
            "triangle_count": int(len(tri_idx)),
            "area_m2": round(float(inst.get("area_m2", 0.0)), 4),
            "h_mean": round(float(inst.get("h_mean", 0.0)), 3),
            "h_span": round(float(inst.get("h_span", 0.0)), 3),
            "bounds_min": [round(float(x), 3) for x in inst.get("bounds_min", [0, 0, 0])],
            "bounds_max": [round(float(x), 3) for x in inst.get("bounds_max", [0, 0, 0])],
            "spans": [round(float(x), 3) for x in inst.get("spans", [0, 0, 0])],
            "relative_path": str(target_ply.relative_to(output_dir)),
        })

    # 5. Export class_metadata.json
    packaged_classes = {}
    for cid in active_classes:
        c_info = classifications.get(cid, classifications.get(str(cid), {}))
        packaged_classes[str(cid)] = {
            "class_id": int(cid),
            "structural_role": c_info.get("structural_role", "object"),
            "confidence": c_info.get("confidence", 0.80),
            "details": c_info.get("details", ""),
            "instances_count": class_instance_counters.get(cid, 0),
        }

    class_meta_path = output_dir / "class_metadata.json"
    class_meta_path.write_text(json.dumps({"classes": packaged_classes}, indent=2) + "\n", encoding="utf-8")
    print(f"Exported class metadata: {class_meta_path}")

    # 6. Export top-level segmentation.json manifest
    segmentation_manifest = {
        "scene_mesh": str(sealed_mesh_path),
        "mesh_segmented": str(mesh_segmented_path.relative_to(output_dir)),
        "instances_root": "Instances PLYs",
        "num_classes": len(active_classes),
        "total_instances": len(instances_manifest),
        "structural_summary": {
            "wall": [cid for cid, m in packaged_classes.items() if m["structural_role"] == "wall"],
            "floor": [cid for cid, m in packaged_classes.items() if m["structural_role"] == "floor"],
            "ceiling": [cid for cid, m in packaged_classes.items() if m["structural_role"] == "ceiling"],
            "stairs": [cid for cid, m in packaged_classes.items() if m["structural_role"] == "stairs"],
            "columns": [cid for cid, m in packaged_classes.items() if m["structural_role"] in ["columns", "column"]],
            "ramps": [cid for cid, m in packaged_classes.items() if m["structural_role"] in ["ramps", "ramp"]],
            "non_structural": [cid for cid, m in packaged_classes.items() if m["structural_role"] == "object"],
        },
        "classes": packaged_classes,
        "instances": instances_manifest,
    }
    seg_manifest_path = output_dir / "segmentation.json"
    seg_manifest_path.write_text(json.dumps(segmentation_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported top-level segmentation manifest: {seg_manifest_path}")

    # 7. Optionally link or copy 2D panoptic data (NPZ archives & overlays)
    if panoptic_npz_dir is not None and Path(panoptic_npz_dir).is_dir():
        dest_npz = output_dir / "panoptic_data/npz_data"
        dest_npz.mkdir(parents=True, exist_ok=True)
        for npz_file in Path(panoptic_npz_dir).glob("*.npz"):
            dest_file = dest_npz / npz_file.name
            if not dest_file.exists():
                try:
                    dest_file.symlink_to(os.path.relpath(npz_file, dest_npz))
                except Exception:
                    shutil.copy2(npz_file, dest_file)
        source_summary = Path(panoptic_npz_dir).parent / 'full_pipeline_benchmark_summary.json'
        archives = list(Path(panoptic_npz_dir).glob('*_final.npz'))
        if not source_summary.is_file() and archives:
            source_summary = archives[0].resolve().parent.parent / source_summary.name
        target_summary = dest_npz.parent / source_summary.name
        if source_summary.is_file() and source_summary.resolve() != target_summary.resolve():
            shutil.copy2(source_summary, target_summary)

    if panoptic_overlays_dir is not None and Path(panoptic_overlays_dir).is_dir():
        dest_ov = output_dir / "panoptic_data/overlays"
        dest_ov.mkdir(parents=True, exist_ok=True)
        for ov_file in Path(panoptic_overlays_dir).glob("*.png"):
            dest_file = dest_ov / ov_file.name
            if not dest_file.exists():
                try:
                    dest_file.symlink_to(os.path.relpath(ov_file, dest_ov))
                except Exception:
                    shutil.copy2(ov_file, dest_file)

    elapsed = time.perf_counter() - t0
    print(f"Delivered {len(instances_manifest)} instances across {len(active_classes)} classes in {elapsed:.2f}s!")
    return segmentation_manifest
