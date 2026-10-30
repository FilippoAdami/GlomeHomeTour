#!/usr/bin/env python3
"""Experimental multi-view TSDF reconstruction of TripoSplat Gaussian PLY."""
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parent
PLY = ROOT / "triposplat_rocm.ply"
OUT = ROOT / "triposplat_tsdf.glb"
RES = 96
IMAGE = 192
N_VIEWS = 14
TRUNC = 0.045


def main():
    started = time.perf_counter()
    v = PlyData.read(PLY)["vertex"].data
    xyz = np.column_stack([v[k] for k in ("x", "y", "z")]).astype(np.float32)
    scales = np.exp(np.column_stack([v[k] for k in ("scale_0", "scale_1", "scale_2")])).astype(np.float32)
    opacity = (1 / (1 + np.exp(-np.clip(v["opacity"], -20, 20)))).astype(np.float32)
    sh = np.column_stack([v[k] for k in ("f_dc_0", "f_dc_1", "f_dc_2")]).astype(np.float32)
    colors = np.clip((sh * 0.28209479177387814 + 0.5) * 255, 0, 255).astype(np.uint8)
    quat = np.column_stack([v[k] for k in ("rot_1", "rot_2", "rot_3", "rot_0")]).astype(np.float32)
    rot = Rotation.from_quat(quat).as_matrix().astype(np.float32)
    cov = rot @ (scales[:, :, None] ** 2 * np.transpose(rot, (0, 2, 1)))

    lo, hi = np.percentile(xyz, [0.2, 99.8], axis=0)
    mid = (lo + hi) / 2
    extent = float(np.max(hi - lo) * 1.14)
    lo, hi = mid - extent / 2, mid + extent / 2
    grid_axes = [np.linspace(lo[i], hi[i], RES, dtype=np.float32) for i in range(3)]
    volume_sum = np.zeros((RES, RES, RES), np.float32)
    volume_weight = np.zeros_like(volume_sum)
    views = []
    phi = (1 + 5**0.5) / 2
    dirs = []
    for i in range(12):
        z = 1 - 2 * (i + 0.5) / 12
        a = 2 * np.pi * i / phi
        dirs.append(np.array([np.sqrt(1-z*z)*np.cos(a), np.sqrt(1-z*z)*np.sin(a), z], np.float32))
    dirs += [np.array([1, 0, 0], np.float32), np.array([0, 1, 0], np.float32)]

    # Low opacity splats add background haze; keep the stronger geometric support.
    keep = opacity > 0.15
    p, s, c, o, cv = xyz[keep], scales[keep], colors[keep], opacity[keep], cov[keep]
    pixel_scale = (IMAGE - 1) / extent
    yy, xx = np.mgrid[0:RES, 0:RES]
    gy, gx, gz = np.meshgrid(*grid_axes, indexing="ij")

    for direction in dirs:
        forward = -direction
        right = np.cross(forward, np.array([0, 0, 1], np.float32))
        if np.linalg.norm(right) < 1e-5:
            right = np.cross(forward, np.array([0, 1, 0], np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        basis = np.stack([right, up, forward])
        uv = p @ basis[:2].T
        dep = p @ forward
        cv2d = basis[:2] @ cv @ basis[:2].T
        radius = np.ceil(2.5 * np.sqrt(np.maximum(np.linalg.eigvalsh(cv2d)[:, -1], 0)) * pixel_scale).astype(np.int32)
        px = np.rint((uv[:, 0] / extent + 0.5) * (IMAGE - 1)).astype(np.int32)
        py = np.rint((0.5 - uv[:, 1] / extent) * (IMAGE - 1)).astype(np.int32)
        sig_depth = np.sqrt(np.maximum(np.einsum('i,nij,j->n', forward, cv, forward), 1e-12))
        front_depth = dep - 2.0 * sig_depth
        depth = np.full((IMAGE, IMAGE), np.inf, np.float32)
        valid = (px >= 0) & (px < IMAGE) & (py >= 0) & (py < IMAGE)
        px, py, radius, front_depth = px[valid], py[valid], radius[valid], front_depth[valid]
        maxr = int(min(radius.max(initial=0), 20))
        for dy in range(-maxr, maxr + 1):
            for dx in range(-maxr, maxr + 1):
                d2 = dx*dx + dy*dy
                sel = (radius >= np.sqrt(d2)) & (px+dx >= 0) & (px+dx < IMAGE) & (py+dy >= 0) & (py+dy < IMAGE)
                if np.any(sel):
                    np.minimum.at(depth, (py[sel]+dy, px[sel]+dx), front_depth[sel])
        mask = np.isfinite(depth)
        if not np.any(mask):
            continue
        # Orthographic depth fusion: project every voxel into this rendered depth map.
        pos = np.stack([gx, gy, gz], axis=-1)
        proj = pos @ basis[:2].T
        vd = pos @ forward
        ix = np.rint((proj[..., 0] / extent + 0.5) * (IMAGE - 1)).astype(np.int32)
        iy = np.rint((0.5 - proj[..., 1] / extent) * (IMAGE - 1)).astype(np.int32)
        ok = (ix >= 0) & (ix < IMAGE) & (iy >= 0) & (iy < IMAGE)
        observed = np.full(vd.shape, np.nan, np.float32)
        observed[ok] = depth[iy[ok], ix[ok]]
        sdf = observed - vd
        used = np.isfinite(sdf) & (sdf > -TRUNC)
        volume_sum[used] += np.clip(sdf[used] / TRUNC, -1, 1)
        volume_weight[used] += 1
        views.append({"direction": direction.tolist(), "covered_pixels": int(mask.sum())})

    tsdf = np.ones_like(volume_sum)
    seen = volume_weight > 0
    tsdf[seen] = volume_sum[seen] / volume_weight[seen]
    verts, faces, normals, values = marching_cubes(tsdf, level=0, spacing=tuple((hi-lo)/(RES-1)))
    verts += lo
    # Nearest Gaussian DC color is a rough vertex-color bake, not a recovered UV texture.
    from scipy.spatial import cKDTree
    color_idx = cKDTree(xyz).query(verts, workers=-1)[1]
    sampled_colors = colors[color_idx]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.column_stack([sampled_colors, np.full(len(verts), 255, np.uint8)])
    )
    vertex_colors = mesh.visual.vertex_colors.copy()
    mesh.export(OUT, file_type="glb")
    reloaded = trimesh.load(OUT, force="mesh")
    elapsed = time.perf_counter() - started
    report = {
        "input": str(PLY.name), "output": str(OUT.name), "runtime_seconds": elapsed,
        "method": "14 orthographic Gaussian-splat depth maps -> weighted TSDF -> marching cubes",
        "settings": {"volume_resolution": RES, "depth_resolution": IMAGE, "truncation_world_units": TRUNC,
                     "splat_opacity_threshold": 0.15, "splat_front_support_sigma": 2.0},
        "views": views, "tsdf_observed_fraction": float(seen.mean()),
        "vertices": len(mesh.vertices), "faces": len(mesh.faces), "watertight": bool(mesh.is_watertight),
        "vertex_color_range_rgb": [vertex_colors[:, :3].min(axis=0).tolist(), vertex_colors[:, :3].max(axis=0).tolist()],
        "exported_color_range_rgb": [reloaded.visual.vertex_colors[:, :3].min(axis=0).tolist(), reloaded.visual.vertex_colors[:, :3].max(axis=0).tolist()],
        "bounds": mesh.bounds.tolist(), "glb_bytes": OUT.stat().st_size,
        "notes": "Gaussian-derived orthographic depth approximation; vertex colors from nearest Gaussian DC color. Not a calibrated renderer or UV texture reconstruction."
    }
    (ROOT / "triposplat_tsdf.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
