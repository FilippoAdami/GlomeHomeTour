#!/usr/bin/env python3
"""Project SeeSR-enhanced cross-section pixels back onto their source mesh faces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh


def srgb_to_linear(colors: np.ndarray) -> np.ndarray:
    return np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--cross-sections-dir", type=Path, required=True)
    parser.add_argument("--enhanced-images-dir", type=Path, required=True)
    parser.add_argument("--projection-map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-view-cosine", type=float, default=0.25,
                        help="Discard hits with |normal·view| below this threshold")
    args = parser.parse_args()
    if not 0.0 <= args.min_view_cosine < 1.0:
        parser.error("--min-view-cosine must be in [0, 1)")

    mesh = o3d.io.read_triangle_mesh(str(args.mesh_path))
    if not len(mesh.triangles):
        raise ValueError(f"No triangles in source mesh: {args.mesh_path}")
    fixed = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False
    )
    trimesh.repair.fix_normals(fixed, multibody=True)
    faces = np.asarray(fixed.faces, dtype=np.int64)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    colors = np.asarray(mesh.vertex_colors).copy()
    if colors.shape != (len(mesh.vertices), 3):
        colors = np.full((len(mesh.vertices), 3), 0.6, dtype=np.float64)

    cross_manifest = json.loads((args.cross_sections_dir / "cross_sections_manifest.json").read_text())
    color_sum = np.zeros_like(colors, dtype=np.float64)
    weight_sum = np.zeros(len(colors), dtype=np.float64)
    view_stats = []
    for view in cross_manifest["views"]:
        name = view["view_name"]
        map_path = args.projection_map_dir / f"{name}.npz"
        image_path = args.enhanced_images_dir / Path(view["image_file"]).name
        with np.load(map_path) as projection:
            triangle_ids = projection["triangle_ids"].reshape(-1).astype(np.int64)
            barycentric = projection["barycentric"].reshape(-1, 3).astype(np.float64)
            cosine = projection["view_cosine"].reshape(-1).astype(np.float64)
        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        if len(triangle_ids) != len(pixels):
            raise ValueError(f"Projection map/image size mismatch for {name}")

        valid = (triangle_ids >= 0) & (triangle_ids < len(faces))
        angle_ok = cosine >= args.min_view_cosine
        use = valid & angle_ok
        face_vertices = faces[triangle_ids[use]]
        weights = barycentric[use] * cosine[use, None] ** 2
        linear_pixels = srgb_to_linear(pixels[use])
        for corner in range(3):
            contribution = weights[:, corner]
            np.add.at(color_sum, face_vertices[:, corner], linear_pixels * contribution[:, None])
            np.add.at(weight_sum, face_vertices[:, corner], contribution)
        view_stats.append({
            "view_name": name,
            "valid_pixels": int(valid.sum()),
            "angle_rejected_pixels": int((valid & ~angle_ok).sum()),
            "projected_pixels": int(use.sum()),
        })

    supported = weight_sum > 1e-8
    colors[supported] = color_sum[supported] / weight_sum[supported, None]
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    mesh.compute_vertex_normals()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh = args.output_dir / "mesh_seesr_reprojected.ply"
    if not o3d.io.write_triangle_mesh(str(output_mesh), mesh, write_ascii=False):
        raise OSError(f"Failed to write {output_mesh}")
    report = {
        "source_mesh": str(args.mesh_path.resolve()),
        "enhanced_images_dir": str(args.enhanced_images_dir.resolve()),
        "projection_map_dir": str(args.projection_map_dir.resolve()),
        "output_mesh": str(output_mesh.resolve()),
        "min_view_cosine": args.min_view_cosine,
        "supported_vertices": int(supported.sum()),
        "total_vertices": len(colors),
        "views": view_stats,
    }
    (args.output_dir / "reprojection_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reprojected enhanced cross-sections to {output_mesh}")
    print(f"Updated {supported.sum()} of {len(colors)} mesh vertices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
