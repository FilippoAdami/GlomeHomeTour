#!/usr/bin/env python3
"""GlomeHomeTour: Fast Surfel Cloud to 3D Polygonal Mesh (.glb) Exporter.

Converts oriented 2D Gaussian surfels (centers, normals, tangents, scales, and colors)
into a continuous 3D quad polygonal mesh (.glb / .ply) loadable in standard 3D viewers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

# Ensure backend is in sys.path
_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from initialization import SurfelCloud


def export_surfel_cloud_to_glb(
    ply_path: Path,
    out_glb_path: Path,
    voxel_downsample_m: float = 0.02,
    max_surfels: int | None = None,
) -> Path:
    """Read binary surfel PLY and export an oriented quad mesh in GLB format."""
    print(f"Loading surfels from: {ply_path}")
    cloud = SurfelCloud.from_ply(
        ply_path,
        voxel_downsample_m=voxel_downsample_m,
        max_surfels=max_surfels,
    )
    n = len(cloud)
    print(f"Constructing polygonal mesh for {n:,} surfel quads...")

    p = cloud.positions
    u = cloud.tangent_u
    v = cloud.tangent_v
    su = cloud.scales_2d[:, 0:1]
    sv = cloud.scales_2d[:, 1:2]
    c = (cloud.colors_rgb * 255).astype(np.uint8)

    # 4 vertices per oriented surfel quad
    v0 = p - su * u - sv * v
    v1 = p + su * u - sv * v
    v2 = p + su * u + sv * v
    v3 = p - su * u + sv * v

    vertices = np.empty((n * 4, 3), dtype=np.float32)
    vertices[0::4] = v0
    vertices[1::4] = v1
    vertices[2::4] = v2
    vertices[3::4] = v3

    colors = np.empty((n * 4, 4), dtype=np.uint8)
    colors[:, 3] = 255
    colors[0::4, :3] = c
    colors[1::4, :3] = c
    colors[2::4, :3] = c
    colors[3::4, :3] = c

    base_idx = np.arange(n, dtype=np.int32) * 4
    faces = np.empty((n * 2, 3), dtype=np.int32)
    faces[0::2, 0] = base_idx
    faces[0::2, 1] = base_idx + 1
    faces[0::2, 2] = base_idx + 2
    faces[1::2, 0] = base_idx
    faces[1::2, 1] = base_idx + 2
    faces[1::2, 2] = base_idx + 3

    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Assembling Trimesh ({len(vertices):,} vertices, {len(faces):,} faces)...")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)

    print(f"Exporting standalone binary GLB to: {out_glb_path}")
    mesh.export(str(out_glb_path), file_type="glb")
    file_mb = out_glb_path.stat().st_size / (1024 * 1024)
    print(f"Success! Exported {out_glb_path.name} ({file_mb:.2f} MB)")
    return out_glb_path


def main():
    parser = argparse.ArgumentParser(description="Export Surfel Cloud to .glb 3D mesh")
    parser.add_argument(
        "--ply",
        type=str,
        default="backend/scenes/bedroom_complete_depth_results/GS_input/bedroom_complete_surfels.ply",
        help="Path to surfel PLY file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="backend/scenes/bedroom_complete_depth_results/bedroom_surfel_mesh.glb",
        help="Path to output .glb file",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel downsample grid in meters (e.g. 0.02 = 2cm)",
    )
    args = parser.parse_args()
    export_surfel_cloud_to_glb(Path(args.ply), Path(args.out), voxel_downsample_m=args.voxel_size)


if __name__ == "__main__":
    main()
