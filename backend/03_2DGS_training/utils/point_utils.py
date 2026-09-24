import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

def _rotate(points, M):
    """points @ M for (N, 3) @ (3, 3), in elementwise column form.

    torch.matmul silently leaves every output row from index 2**19 onward as
    zeros on gfx1200 / ROCm 7.1 (see backend/CLAUDE.md). At 1920x1080 this
    path has N = 2.07M, so ~75% of the normal map would be quietly zeroed.
    """
    return points[:, 0:1] * M[0] + points[:, 1:2] * M[1] + points[:, 2:3] * M[2]

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = _rotate(points, intrins.inverse().T @ c2w[:3,:3].T)
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    cross_prod = torch.cross(dx, dy, dim=-1)
    # The guard has to be relative, not absolute. |dx x dy| scales with the
    # metric spacing between neighbouring pixels' 3D points, so it shrinks with
    # the square of the render resolution: at r=2 a wall gives ~1e-4, at native
    # 1080p the same wall gives ~1e-6. An absolute 1e-5 cut therefore zeroed
    # 97-99% of surf_normal at exactly the resolution the geometry is finalised
    # at, which silently reduced the normal-consistency loss to the constant 1
    # (no gradient). Scale by the edge lengths that produced the cross product.
    scale = dx.norm(dim=-1, keepdim=True) * dy.norm(dim=-1, keepdim=True)
    degenerate = torch.norm(cross_prod, dim=-1, keepdim=True) <= 1e-4 * scale
    normal_map = torch.where(degenerate, torch.zeros_like(cross_prod),
                             F.normalize(cross_prod, dim=-1, eps=1e-20))
    output[1:-1, 1:-1, :] = normal_map
    return output