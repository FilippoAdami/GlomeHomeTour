"""GlomeHomeTour Backend: 2DGS Reconstruction Loss Functions.

Implements differentiable L1 color loss, SSIM loss, and Surface Normal
consistency loss against monocular depth priors.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_gaussian_window(window_size: int, sigma: float, channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate 1D Gaussian convolution kernel shaped (C, 1, 1, window_size)."""
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    gauss_1d = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    gauss_1d = gauss_1d / torch.clamp(torch.sum(gauss_1d), min=1e-6)
    return gauss_1d.view(1, 1, 1, window_size).repeat(channels, 1, 1, 1)


def _gaussian_blur(x: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur: two 1D grouped convs instead of one 11x11.

    MIOpen's grouped 11x11 kernel costs ~180 ms per training step at 1080p on
    gfx1200, which was the single largest cost in the loop -- larger than the
    whole rasterizer. Separating it is ~5x less work and hits a fast path.
    """
    channels = x.shape[1]
    pad = window.shape[-1] // 2
    x = F.conv2d(x, window, padding=(0, pad), groups=channels)
    return F.conv2d(x, window.transpose(2, 3), padding=(pad, 0), groups=channels)


def l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute mean L1 color loss, optionally masked.

    Args:
        pred: (B, C, H, W)
        target: (B, C, H, W)
        mask: (B, 1, H, W) or (B, C, H, W) binary/float mask in [0, 1]

    Returns:
        Scalar L1 loss.
    """
    diff = torch.abs(pred - target)
    if mask is not None:
        if mask.shape[1] != diff.shape[1] and mask.shape[1] == 1:
            mask = mask.expand(-1, diff.shape[1], -1, -1)
        valid_elems = torch.clamp(torch.sum(mask), min=1e-6)
        return torch.sum(diff * mask) / valid_elems
    return torch.mean(diff)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    mask: Optional[torch.Tensor] = None,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute Structural Similarity Index Measure (SSIM) loss: 1.0 - SSIM(pred, target).

    Args:
        pred: (B, C, H, W) in range [0, 1]
        target: (B, C, H, W) in range [0, 1]
        window_size: Gaussian kernel window size (default 11)
        sigma: Gaussian standard deviation (default 1.5)
        mask: Optional (B, 1, H, W) binary mask
        c1: Stability constant for luminance
        c2: Stability constant for contrast

    Returns:
        Scalar SSIM loss in range [0, 2].
    """
    b, channels, h, w = pred.shape
    if min(h, w) < window_size:
        # Fallback to L1 if image is smaller than window
        return l1_loss(pred, target, mask=mask)

    window = _create_gaussian_window(window_size, sigma, channels, pred.dtype, pred.device)

    # One batched blur over all five moments (means, second moments, cross term).
    moments = _gaussian_blur(
        torch.cat([pred, target, pred * pred, target * target, pred * target], dim=0),
        window,
    )
    mu1, mu2, e11, e22, e12 = moments.split(b, dim=0)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    # Variances and covariance
    sigma1_sq = e11 - mu1_sq
    sigma2_sq = e22 - mu2_sq
    sigma12 = e12 - mu1_mu2

    # Numerical bounds clamping
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

    # SSIM formula
    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-6)

    loss_map = 1.0 - ssim_map

    if mask is not None:
        if mask.shape[1] != channels and mask.shape[1] == 1:
            mask = mask.expand(-1, channels, -1, -1)
        # Apply same filtering to mask to match receptive field
        mask_filtered = _gaussian_blur(mask, window)
        mask_valid = (mask_filtered > 0.5).to(dtype=pred.dtype)
        valid_count = torch.clamp(torch.sum(mask_valid), min=1e-6)
        return torch.sum(loss_map * mask_valid) / valid_count

    return torch.mean(loss_map)


def normal_loss(
    pred_normal: torch.Tensor,
    target_normal: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute surface normal consistency loss: 1.0 - <N_pred, N_target>.

    Args:
        pred_normal: (B, 3, H, W) surface normals
        target_normal: (B, 3, H, W) prior surface normals
        mask: Optional (B, 1, H, W) binary mask

    Returns:
        Scalar normal consistency loss in range [0, 2].
    """
    # Normalize inputs
    p_norm = F.normalize(pred_normal, dim=1, eps=1e-6)
    t_norm = F.normalize(target_normal, dim=1, eps=1e-6)

    # Cosine similarity dot product: (B, 1, H, W)
    cos_sim = torch.sum(p_norm * t_norm, dim=1, keepdim=True)
    cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)

    loss_map = 1.0 - cos_sim

    if mask is not None:
        valid_count = torch.clamp(torch.sum(mask), min=1e-6)
        return torch.sum(loss_map * mask) / valid_count

    return torch.mean(loss_map)


def depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked L1 against the Depth Anything 3 metric depth prior.

    Anchors geometry where photometric loss is ambiguous (blank walls), which
    is exactly where monocular 2DGS otherwise grows floaters. Pixels with a
    non-positive prior are always excluded.

    Args:
        pred_depth: (B, 1, H, W) rendered expected depth in meters.
        gt_depth: (B, 1, H, W) prior metric depth in meters.
        mask: Optional (B, 1, H, W) extra validity mask (e.g. rendered coverage).

    Returns:
        Scalar L1 depth loss.
    """
    valid = (gt_depth > 0).to(dtype=pred_depth.dtype)
    if mask is not None:
        valid = valid * mask
    denom = torch.clamp(torch.sum(valid), min=1e-6)
    return torch.sum(torch.abs(pred_depth - gt_depth) * valid) / denom


class ReconstructionLoss(nn.Module):
    """Composite objective function for 2DGS training:

    L = (1 - lambda_ssim) * L_1 + lambda_ssim * L_ssim
        + gamma_normal * L_normal + gamma_depth * L_depth
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        gamma_normal: float = 0.05,
        gamma_depth: float = 0.1,
        window_size: int = 11,
    ) -> None:
        super().__init__()
        self.lambda_ssim = float(lambda_ssim)
        self.gamma_normal = float(gamma_normal)
        self.gamma_depth = float(gamma_depth)
        self.window_size = window_size

    def forward(
        self,
        pred_rgb: torch.Tensor,
        gt_rgb: torch.Tensor,
        pred_normal: Optional[torch.Tensor] = None,
        gt_normal: Optional[torch.Tensor] = None,
        pred_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        coverage_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Evaluate composite loss.

        Args:
            pred_rgb: (B, 3, H, W) rendered RGB
            gt_rgb: (B, 3, H, W) ground truth keyframe RGB
            pred_normal: Optional (B, 3, H, W) rendered normal map
            gt_normal: Optional (B, 3, H, W) Depth Anything v3 prior normal map
            pred_depth: Optional (B, 1, H, W) rendered expected depth
            gt_depth: Optional (B, 1, H, W) Depth Anything v3 metric depth prior
            mask: Optional (B, 1, H, W) valid depth / foreground mask
            coverage_mask: Optional (B, 1, H, W) rendered-coverage mask for the
                depth term only (rendered depth is meaningless where alpha ~ 0)

        Returns:
            total_loss: scalar tensor for backpropagation.
            metrics: dictionary of individual loss values.
        """
        loss_color_l1 = l1_loss(pred_rgb, gt_rgb, mask=mask)
        loss_color_ssim = ssim_loss(pred_rgb, gt_rgb, window_size=self.window_size, mask=mask)

        loss_total = (1.0 - self.lambda_ssim) * loss_color_l1 + self.lambda_ssim * loss_color_ssim
        metrics = {
            "loss_total": loss_total,
            "loss_l1": loss_color_l1,
            "loss_ssim": loss_color_ssim,
        }

        if pred_normal is not None and gt_normal is not None:
            loss_norm = normal_loss(pred_normal, gt_normal, mask=mask)
            loss_total = loss_total + self.gamma_normal * loss_norm
            metrics["loss_normal"] = loss_norm
            metrics["loss_total"] = loss_total
        else:
            metrics["loss_normal"] = torch.tensor(0.0, device=pred_rgb.device, dtype=pred_rgb.dtype)

        if pred_depth is not None and gt_depth is not None and self.gamma_depth > 0.0:
            d_mask = mask if coverage_mask is None else (
                coverage_mask if mask is None else coverage_mask * mask
            )
            loss_depth = depth_loss(pred_depth, gt_depth, mask=d_mask)
            loss_total = loss_total + self.gamma_depth * loss_depth
            metrics["loss_depth"] = loss_depth
            metrics["loss_total"] = loss_total
        else:
            metrics["loss_depth"] = torch.tensor(0.0, device=pred_rgb.device, dtype=pred_rgb.dtype)

        return loss_total, metrics
