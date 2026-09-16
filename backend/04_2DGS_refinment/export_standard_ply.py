#!/usr/bin/env python3
"""Export Material2DGSModel checkpoint to standard 3DGS/2DGS PLY format.

This produces a canonical Gaussian Splatting .ply file compatible with:
- PlayCanvas SuperSplat (https://playcanvas.com/supersplat/editor)
- Antimatter15 WebGL Viewer (https://antimatter15.com/splat)
- Nerfstudio / gsplat / SIBR / Postshot / Polycam / Luma
- Blender 3D Gaussian Splatting Add-ons
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

SH_C0 = 0.28209479177387814


def export_model_to_standard_3dgs_ply(
    checkpoint_or_model: Union[str, Path, dict],
    output_ply_path: Union[str, Path],
) -> Path:
    out_path = Path(output_ply_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(checkpoint_or_model, (str, Path)):
        ckpt = torch.load(str(checkpoint_or_model), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
    elif hasattr(checkpoint_or_model, "state_dict"):
        state = checkpoint_or_model.state_dict()
    elif isinstance(checkpoint_or_model, dict) and "model_state_dict" in checkpoint_or_model:
        state = checkpoint_or_model["model_state_dict"]
    else:
        state = checkpoint_or_model

    xyz = state["_xyz"].detach().cpu().numpy().astype(np.float32)
    n = xyz.shape[0]

    # Quaternions: (w, x, y, z)
    rot_raw = state["_rotation"].detach().cpu().numpy().astype(np.float32)
    rot_norm = np.maximum(np.linalg.norm(rot_raw, axis=-1, keepdims=True), 1e-6)
    rot = rot_raw / rot_norm

    # Scales: (sigma_u, sigma_v) in log space
    log_scales_2d = state["_scaling"].detach().cpu().numpy().astype(np.float32)
    # 2DGS flat normal scale (1e-4 in log space is -9.21)
    log_scale_normal = np.full((n, 1), -9.21034, dtype=np.float32)
    log_scales = np.hstack([log_scales_2d, log_scale_normal])

    # Opacity: already in logit space in model._opacity
    opacity_logit = state["_opacity"].detach().cpu().numpy().astype(np.float32)
    if opacity_logit.ndim == 1:
        opacity_logit = opacity_logit[:, None]

    # Albedo: logit -> sigmoid -> SH DC
    albedo_logit = state["_albedo"].detach().cpu().numpy().astype(np.float32)
    albedo_rgb = 1.0 / (1.0 + np.exp(-np.clip(albedo_logit, -15.0, 15.0)))
    f_dc = (albedo_rgb - 0.5) / SH_C0

    # Higher-order spherical harmonics (e.g. SH Degree 1)
    f_rest = None
    n_rest = 0
    if "_features_rest" in state and state["_features_rest"] is not None:
        raw_rest = state["_features_rest"].detach().cpu().numpy().astype(np.float32)
        if raw_rest.shape[-1] > 0:
            f_rest = raw_rest
            n_rest = f_rest.shape[-1]

    # Compute unit normals from rotation quaternion
    w, x, y, z = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    normals = np.column_stack([nx, ny, nz]).astype(np.float32)

    # Header for standard 3DGS format
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    for i in range(n_rest):
        header_lines.append(f"property float f_rest_{i}")
    header_lines.extend([
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header\n",
    ])
    header = "\n".join(header_lines)

    # 17 + n_rest float32 properties per vertex: x,y,z, nx,ny,nz, f_dc0,1,2, [f_rest], opacity, scale0,1,2, rot0,1,2,3
    total_props = 17 + n_rest
    vertex_data = np.empty((n, total_props), dtype=np.float32)
    vertex_data[:, 0:3] = xyz
    vertex_data[:, 3:6] = normals
    vertex_data[:, 6:9] = f_dc
    col = 9
    if n_rest > 0 and f_rest is not None:
        vertex_data[:, col:col + n_rest] = f_rest
        col += n_rest
    vertex_data[:, col:col + 1] = opacity_logit
    col += 1
    vertex_data[:, col:col + 3] = log_scales
    col += 3
    vertex_data[:, col:col + 4] = rot

    with open(out_path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())

    print(f"[3DGS Export] Successfully wrote standard Gaussian Splatting PLY: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB, {n:,} splats, SH degree: {1 if n_rest == 9 else 0})")
    return out_path


if __name__ == "__main__":
    ckpt = Path("backend/scenes/bedroom_complete_depth_results/2dgs_output/material_2dgs_checkpoint.pt")
    out = Path("backend/scenes/bedroom_complete_depth_results/2dgs_output/bedroom_standard_3dgs.ply")
    export_model_to_standard_3dgs_ply(ckpt, out)
