#!/usr/bin/env python3
"""
Axis-Align Structural Mesh.

Rotates the regularized structural mesh and its components around the vertical Y-axis
so that the dominant Manhattan wall orientations lie strictly along the canonical Cartesian axes:
  - Dir A (East-West) -> [1, 0, 0] (+X)
  - Dir B (North-South) -> [0, 0, 1] (+Z)

Saves axis_alignment_transform.json and updates all_structural_regularized.ply and manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True, help="Directory containing regularized instances and manifests")
    parser.add_argument("--output-dir", type=Path, help="Directory to save axis-aligned outputs (defaults to instances-dir)")
    return parser.parse_args()


def compute_axis_rotation(dominant_orientation_deg: float) -> np.ndarray:
    """
    Rotation in XZ plane by -dominant_orientation_deg to rotate Dir A = [cos(theta), 0, sin(theta)]
    directly to canonical +X = [1, 0, 0] and Dir B = [-sin(theta), 0, cos(theta)] to canonical +Z = [0, 0, 1].
    """
    theta_rad = np.radians(dominant_orientation_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    R = np.array([
        [cos_t, 0.0, sin_t],
        [0.0,   1.0, 0.0],
        [-sin_t, 0.0, cos_t]
    ], dtype=np.float64)
    return R


def transform_mesh_file(path: Path, R: np.ndarray, t: np.ndarray):
    if not path.is_file():
        return
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.vertices):
        return
    verts = np.asarray(mesh.vertices)
    transformed_verts = (R @ verts.T).T + t
    mesh.vertices = o3d.utility.Vector3dVector(transformed_verts)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)


def axis_align(instances_dir: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_manifest_path = instances_dir / "wall_planes_manifest.json"
    if not wall_manifest_path.is_file():
        print("Warning: wall_planes_manifest.json not found; skipping axis-alignment")
        return {}

    wall_data = json.loads(wall_manifest_path.read_text())
    dom_deg = float(wall_data.get("dominant_orientation_deg", 0.0))
    print(f"Dominant wall orientation: {dom_deg:.3f}°")

    R = compute_axis_rotation(dom_deg)
    t = np.zeros(3, dtype=np.float64)

    # Transform all_structural_regularized.ply
    shell_path = instances_dir / "all_structural_regularized.ply"
    if shell_path.is_file():
        print(f"Axis-aligning structural shell: {shell_path}...")
        transform_mesh_file(shell_path, R, t)

    # Transform regularized structural PLY files in instances_dir (keeping raw extracted instances intact)
    ply_patterns = [
        "*_regularized.ply",
        "slab_inst*.ply",
        "wall_enclosing_planes.ply",
        "wall_cleaned_pieces.ply"
    ]
    for pattern in ply_patterns:
        for ply_file in instances_dir.glob(pattern):
            transform_mesh_file(ply_file, R, t)

    # Transform manifests:
    # 1. wall_planes_manifest.json
    wall_data["dominant_orientation_deg_original"] = dom_deg
    wall_data["dominant_orientation_deg"] = 0.0
    wall_data["orthogonal_basis"] = {
        "dir_A_east_west": [1.0, 0.0, 0.0],
        "dir_B_north_south": [0.0, 0.0, 1.0]
    }
    for plane in wall_data.get("wall_planes", []):
        n = np.array(plane["normal_unit"], dtype=np.float64)
        rot_n = R @ n
        # snap near-zero components
        rot_n[np.abs(rot_n) < 1e-4] = 0.0
        plane["normal_unit"] = [round(float(v), 8) for v in rot_n]
        if "plane_equation" in plane:
            pe = np.array(plane["plane_equation"], dtype=np.float64)
            d = pe[3] - float(rot_n @ t)
            plane["plane_equation"] = [round(float(v), 8) for v in (*rot_n, d)]
    (instances_dir / "wall_planes_manifest.json").write_text(json.dumps(wall_data, indent=2) + "\n")

    # 2. floors_ceilings_planes_manifest.json
    fc_manifest_path = instances_dir / "floors_ceilings_planes_manifest.json"
    if fc_manifest_path.is_file():
        fc_data = json.loads(fc_manifest_path.read_text())
        for elem in fc_data.get("elements", []):
            if "normal_unit" in elem:
                rot_n = R @ np.array(elem["normal_unit"], dtype=np.float64)
                rot_n[np.abs(rot_n) < 1e-4] = 0.0
                elem["normal_unit"] = [round(float(v), 8) for v in rot_n]
            if "plane_equation" in elem:
                pe = np.array(elem["plane_equation"], dtype=np.float64)
                rot_norm = R @ pe[:3]
                d = pe[3] - float(rot_norm @ t)
                elem["plane_equation"] = [round(float(v), 8) for v in (*rot_norm, d)]
        fc_manifest_path.write_text(json.dumps(fc_data, indent=2) + "\n")

    # 3. plane_patterns_manifest.json
    pat_manifest_path = instances_dir / "plane_patterns_manifest.json"
    if pat_manifest_path.is_file():
        pat_data = json.loads(pat_manifest_path.read_text())
        for u in pat_data.get("units", []):
            if "plane_equation" in u:
                pe = np.array(u["plane_equation"], dtype=np.float64)
                rot_norm = R @ pe[:3]
                d = pe[3] - float(rot_norm @ t)
                u["plane_equation"] = [round(float(v), 8) for v in (*rot_norm, d)]
            if "normal_unit" in u:
                rot_n = R @ np.array(u["normal_unit"], dtype=np.float64)
                rot_n[np.abs(rot_n) < 1e-4] = 0.0
                u["normal_unit"] = [round(float(v), 8) for v in rot_n]
        pat_manifest_path.write_text(json.dumps(pat_data, indent=2) + "\n")

    transform_record = {
        "rotation_deg_y": float(-dom_deg),
        "rotation_matrix": R.tolist(),
        "translation_m": t.tolist(),
        "status": "axis_aligned_to_canonical_cartesian_frame"
    }
    transform_path = instances_dir / "axis_alignment_transform.json"
    transform_path.write_text(json.dumps(transform_record, indent=2) + "\n")
    print(f"Axis-aligned structural mesh and saved transform to {transform_path}")
    return transform_record


def main():
    args = parse_args()
    out = args.output_dir if args.output_dir else args.instances_dir
    axis_align(args.instances_dir, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
