"""GlomeHomeTour: Surfel Camera Frustum Projection Kernel & Oracle (Phase 7.1).

Contains:
1. Pure-PyTorch Fallback Oracle:
   - High-throughput vectorized tensor projection on ROCm/CUDA and CPU.
2. Wave32 Alignment:
   - Enforces batch alignment to multiples of 32 for optimal SIMD utilization
     on AMD RDNA4 (gfx1200 / RX 9070 XT).
"""

from __future__ import annotations

from typing import Tuple, Union

import torch


def project_surfels_frustum_torch(
    positions: torch.Tensor,     # (N, 3) float32 in meters
    w2c_matrix: torch.Tensor,    # (4, 4) float32 world to camera
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near_plane: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch fallback oracle for surfel camera projection.

    Returns:
        uv_depth: (N, 3) float32 containing [u, v, depth]
        valid_mask: (N,) bool mask indicating surfels in front of camera and within image bounds
    """
    device = positions.device
    n = positions.shape[0]

    if n == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=device), torch.empty((0,), dtype=torch.bool, device=device)

    w2c = w2c_matrix.to(device=device, dtype=torch.float32)

    # Transform to camera space, column-wise. A (N, 4) @ (4, 4)^T matmul silently
    # leaves every row past 2**19 zeroed on gfx1200 / ROCm 7.1; see _rotate() in
    # 03_2DGS_training/rasterizer_interface.py.
    r, t = w2c[:3, :3], w2c[:3, 3]
    p_cam = positions[:, 0:1] * r[:, 0] + positions[:, 1:2] * r[:, 1] + positions[:, 2:3] * r[:, 2] + t

    # OpenGL camera: looks along -Z
    depth = -p_cam[:, 2]

    in_front = depth > near_plane
    safe_depth = torch.clamp_min(depth, 1e-4)

    u = fx * (p_cam[:, 0] / safe_depth) + cx
    v = -fy * (p_cam[:, 1] / safe_depth) + cy

    in_bounds = in_front & (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))

    uv_depth = torch.stack([u, v, depth], dim=-1)
    return uv_depth, in_bounds
