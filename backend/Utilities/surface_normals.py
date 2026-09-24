"""Windowed plane-fit surface normals, shared by seed orientation and prior supervision.

2dgs_combined_pipeline.md Stage 0 step 4: back-project a depth map to camera-space
points, fit a plane in a small window (~3-5 px radius) excluding neighbours whose
3D distance exceeds a depth-scaled threshold, rather than a plain cross-product
(which smears normals across depth discontinuities). The doc also notes normals
are needed twice -- seed orientation and per-image supervision -- "both come from
the same computation": both 02_depth_estimation/depth_priors.py's
`compute_surface_normals` and 03_2DGS_training/utils/depth_prior.py's prior cache
builder call `fit_plane_normals` below.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

def _smallest_eigpair_3x3(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Smallest eigenvalue and its unit eigenvector for a batch of symmetric 3x3s.

    `torch.linalg.eigh` batched over ~1e5-1e6 3x3 matrices raises
    ``hipErrorInvalidConfiguration`` on gfx1200/ROCm 7.1 (no such issue on
    small batches or CPU) -- so this uses the closed-form trigonometric
    solution for symmetric 3x3 eigenvalues instead of a solver call, in the
    same spirit as backend/CLAUDE.md's matmul-avoidance workaround.

    cov: (N, 3, 3). Returns (eigenvalue (N,), eigenvector (N, 3)).
    """
    a11, a22, a33 = cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]
    a12, a13, a23 = cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2]

    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p2 = (b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * (a12 * a12 + a13 * a13 + a23 * a23))
    p = torch.sqrt((p2 / 6.0).clamp_min(1e-18))

    # det(B) for B = (A - qI) / p, expanded directly rather than via a 3x3
    # matmul kernel.
    detB = (b11 * b22 * b33 + 2.0 * a12 * a23 * a13
            - b11 * a23 * a23 - b22 * a13 * a13 - b33 * a12 * a12) / (p ** 3).clamp_min(1e-18)
    r = (detB / 2.0).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0

    eig_min = q + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)  # smallest root

    # Eigenvector of the smallest eigenvalue via cross product of two rows of
    # (A - eig_min I); pick whichever pair of rows is least parallel per pixel.
    m11, m22, m33 = a11 - eig_min, a22 - eig_min, a33 - eig_min
    row0 = torch.stack([m11, a12, a13], dim=-1)
    row1 = torch.stack([a12, m22, a23], dim=-1)
    row2 = torch.stack([a13, a23, m33], dim=-1)
    c01 = torch.cross(row0, row1, dim=-1)
    c02 = torch.cross(row0, row2, dim=-1)
    c12 = torch.cross(row1, row2, dim=-1)
    cands = torch.stack([c01, c02, c12], dim=1)  # (N, 3, 3)
    norms = cands.norm(dim=-1)  # (N, 3)
    best = norms.argmax(dim=-1)  # (N,)
    vec = cands[torch.arange(cands.shape[0], device=cov.device), best]

    degenerate = p2 < 1e-12  # near-isotropic covariance: no well-defined plane
    eig_min = torch.where(degenerate, torch.zeros_like(eig_min), eig_min)
    vec = torch.where(degenerate[:, None], torch.zeros_like(vec), vec)
    return eig_min, vec


def fit_plane_normals(
    points: torch.Tensor,
    valid: torch.Tensor | None,
    radius: int = 2,
    dist_thresh_abs: float = 0.03,
    dist_thresh_rel: float = 0.03,
    depth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-pixel plane normal, fit residual (metres), and inlier count.

    points: (H, W, 3) camera- or world-space points, any consistent frame.
    valid: (H, W) bool, True where the source depth sample itself is usable
        (missing/zero depth). None means every pixel is a candidate neighbour.
    depth: (H, W) scale reference for the distance tolerance; defaults to
        each point's distance from the origin of `points`' frame.

    Returns (normal (H,W,3) unit -- sign ambiguous, caller orients it;
    residual (H,W), inf where fewer than 3 inliers; count (H,W) int).
    """
    H, W, _ = points.shape
    device = points.device
    k = 2 * radius + 1

    pts = points.permute(2, 0, 1)[None]  # (1, 3, H, W)
    pad = F.pad(pts, (radius, radius, radius, radius), mode="replicate")
    win = F.unfold(pad, kernel_size=k).reshape(3, k * k, H, W).permute(2, 3, 1, 0)  # (H, W, k*k, 3)

    center = points[:, :, None, :]
    delta = win - center
    dist = delta.norm(dim=-1)  # (H, W, k*k)

    ref_depth = depth if depth is not None else points.norm(dim=-1)
    tol = dist_thresh_abs + dist_thresh_rel * ref_depth[:, :, None]

    if valid is not None:
        vpad = F.pad(valid.float()[None, None], (radius, radius, radius, radius), mode="constant", value=0.0)
        vwin = F.unfold(vpad, kernel_size=k).reshape(k * k, H, W).permute(1, 2, 0).bool()
    else:
        vwin = torch.ones_like(dist, dtype=torch.bool)

    inlier = vwin & (dist <= tol)
    count = inlier.sum(dim=-1)

    w = inlier.float()
    wsum = w.sum(dim=-1, keepdim=True).clamp_min(1.0)
    centroid = (win * w[..., None]).sum(dim=2) / wsum
    centered = (win - centroid[:, :, None, :]) * w[..., None]

    # 3x3 weighted covariance, built elementwise (never matmul/einsum on an
    # (N, 3)-shaped tensor -- see backend/CLAUDE.md's gfx1200 BLAS row-ceiling note).
    cxx = (centered[..., 0] * centered[..., 0]).sum(-1)
    cyy = (centered[..., 1] * centered[..., 1]).sum(-1)
    czz = (centered[..., 2] * centered[..., 2]).sum(-1)
    cxy = (centered[..., 0] * centered[..., 1]).sum(-1)
    cxz = (centered[..., 0] * centered[..., 2]).sum(-1)
    cyz = (centered[..., 1] * centered[..., 2]).sum(-1)
    cov = torch.stack([
        torch.stack([cxx, cxy, cxz], dim=-1),
        torch.stack([cxy, cyy, cyz], dim=-1),
        torch.stack([cxz, cyz, czz], dim=-1),
    ], dim=-2).reshape(-1, 3, 3)

    residual_flat, normal_flat = _smallest_eigpair_3x3(cov)
    residual_flat = residual_flat.clamp_min(0.0)

    count_flat = count.reshape(-1)
    residual_flat = torch.where(count_flat >= 3, torch.sqrt(residual_flat / count_flat.clamp_min(3.0)),
                                 torch.full_like(residual_flat, float("inf")))
    normal_flat = torch.where((count_flat >= 3)[:, None], normal_flat, torch.zeros_like(normal_flat))

    normal = F.normalize(normal_flat, dim=-1, eps=1e-8).reshape(H, W, 3)
    residual = residual_flat.reshape(H, W)
    return normal, residual, count


def orient_towards(normal: torch.Tensor, ray_dir: torch.Tensor) -> torch.Tensor:
    """Flip `normal` (..., 3) so it points back against `ray_dir` (camera -> surface)."""
    flip = (normal * ray_dir).sum(-1, keepdim=True) > 0
    return torch.where(flip, -normal, normal)


def grazing_angle_factor(normal: torch.Tensor, ray_dir: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    """|cos| between the surface normal and the viewing ray -- low near grazing incidence."""
    return (normal * ray_dir).sum(-1).abs().clamp_min(floor)


def residual_confidence_factor(residual: torch.Tensor, scale: float = 0.02) -> torch.Tensor:
    """Maps plane-fit residual (metres) to (0, 1], decaying smoothly past `scale`."""
    return torch.exp(-residual.clamp_min(0.0) / scale)
