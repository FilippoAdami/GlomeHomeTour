#
# Stage 0.6 of 2dgs_combined_pipeline.md: multi-scale structure-tensor
# wavelength maps (Lambda_min), the reference signal SAD-2DGS densification
# splits against.
#
# The question this answers, per pixel, is "what is the finest feature the
# image actually carries here?". A surfel whose on-screen tangential extent is
# larger than that wavelength cannot represent the feature no matter how its
# colour is optimised -- it has to be subdivided. A surfel smaller than it is
# already sufficient, and splitting it only buys VRAM cost and floaters.
#
# That is a strictly better densification signal than gradient magnitude, which
# is what 3DGS/2DGS ship with: gradient magnitude also rises on a correctly
# resolved high-contrast edge, on a badly exposed frame, and on any surfel the
# optimiser happens to be moving, none of which are under-resolution.
#
# Units: Lambda_min comes out in *pixels of the image it was computed from*.
# Everything downstream renders at a different resolution to that, so callers
# must rescale -- see DepthPriors.get_wavelength(). A feature's wavelength in
# pixels scales linearly with render width; getting this wrong is silent, since
# a too-large Lambda_min simply means nothing ever splits.
#

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# L=4 levels, sigma_l = SIGMA_BASE^l (doc: sigma_l = 1.5^l).
NUM_LEVELS = 4
SIGMA_BASE = 1.5
# Integration scale of the tensor smoothing, rho_l = TENSOR_RHO_FACTOR * sigma_l.
TENSOR_RHO_FACTOR = 3.0
EPS = 1e-8
# A textureless wall has no gradient at any scale, so lambda_1 is 0 there and
# 1/sqrt(lambda_1) diverges. "No feature to resolve" is the correct reading --
# eta = extent / Lambda_min goes to 0 and nothing splits -- but the raw value
# overflows float16 storage and poisons any mean taken over the map, so it is
# capped at a wavelength no indoor image can carry.
LAMBDA_MIN_CLAMP_PX = (1.0, 1000.0)


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (C, H, W) image. Reflect padding at borders."""
    k = _gaussian_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    c = img.shape[0]
    x = img[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"),
                 k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"),
                 k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x[0]


def _central_gradients(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(d/dx, d/dy) of a (C, H, W) image by central differences, edges replicated."""
    p = F.pad(img[None], (1, 1, 1, 1), mode="replicate")[0]
    gx = (p[:, 1:-1, 2:] - p[:, 1:-1, :-2]) * 0.5
    gy = (p[:, 2:, 1:-1] - p[:, :-2, 1:-1]) * 0.5
    return gx, gy


@torch.no_grad()
def wavelength_map(image: torch.Tensor) -> torch.Tensor:
    """Lambda_min (H, W), in pixels of `image`, for a (3, H, W) image in [0, 1].

    Follows 2dgs_combined_pipeline.md §0.6 step for step:
      S_l   = G_rho * sum_c grad I_l,c grad I_l,c^T   (per-scale structure tensor)
      S^_l  = S_l / (tr S_l + eps)                    (orientation only, no contrast)
      E_l   = ||I_{l-1} - I_l||_2                     (bandpass energy at that scale)
      S_bar = sum_l E_l^3 w_l^2 S^_l / sum_l E_l^3,   w_l = 1 / (2 pi sigma_l)
      Lambda_min = 1 / (sqrt(lambda_1(S_bar)) + eps)

    Note the asymmetry in S_bar: w_l^2 is in the numerator only, so the trace of
    S_bar is the E^3-weighted mean of w_l^2 rather than 1. That is what carries
    the scale information -- a pixel whose energy sits at the coarsest level
    ends up with a large Lambda_min (~2 pi sigma_3 = 21 px), one whose energy
    sits at the finest with a small one (~2 pi sigma_0 = 6 px). Normalising it
    away would leave Lambda_min in [1, 1.41] everywhere and the whole map
    useless.
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) image, got {tuple(image.shape)}")

    prev = image
    num = torch.zeros((3,) + image.shape[1:], device=image.device, dtype=image.dtype)
    den = torch.zeros(image.shape[1:], device=image.device, dtype=image.dtype)

    for level in range(NUM_LEVELS):
        sigma = SIGMA_BASE ** level
        img_l = gaussian_blur(image, sigma)

        gx, gy = _central_gradients(img_l)
        # Summed over colour channels: a red/green edge at constant luminance is
        # still a feature the splats have to resolve, and a greyscale tensor
        # would miss it.
        s = torch.stack([(gx * gx).sum(0), (gx * gy).sum(0), (gy * gy).sum(0)])
        s = gaussian_blur(s, TENSOR_RHO_FACTOR * sigma)
        s_hat = s / (s[0] + s[2] + EPS)

        # E_l is integrated over the same neighbourhood as the tensor it
        # weights. The doc writes it per-pixel, but a per-pixel bandpass energy
        # is exactly zero at every node of an oscillating feature -- on a sine
        # grating that is every half period -- and there all L levels vanish
        # together, leaving 0/0 and a clamped Lambda_min spike sitting in the
        # middle of perfectly well-resolved texture. Since the node spacing
        # scales with 1/period, those spikes are *denser* on fine texture, which
        # inverts the whole map: measured on a 6 px grating vs a 30 px one, the
        # unsmoothed form reported 338 px and 80 px respectively, i.e. it called
        # the fine grating the coarser of the two. With this smoothing the same
        # two read 7 px and 21 px, which is 2*pi*sigma at the scale each one
        # actually lives at. Real photographs hide the defect behind sensor
        # noise; a flat wall or a clean gradient does not.
        energy = gaussian_blur((prev - img_l).pow(2).sum(0, keepdim=True),
                               TENSOR_RHO_FACTOR * sigma)[0].clamp_min(0).sqrt()
        e3 = energy.pow(3)
        w = 1.0 / (2.0 * math.pi * sigma)

        num += (e3 * (w * w))[None] * s_hat
        den += e3
        prev = img_l

    s_bar = num / (den + EPS)[None]

    # Dominant eigenvalue of the symmetric 2x2 [[a, b], [b, c]], closed form.
    # No eigensolver call: torch.linalg.eigh batched over ~1e6 small matrices
    # raises hipErrorInvalidConfiguration on this card (see depth_prior.py).
    a, b, c = s_bar[0], s_bar[1], s_bar[2]
    half_tr = (a + c) * 0.5
    disc = torch.sqrt(((a - c) * 0.5) ** 2 + b * b)
    lambda_1 = (half_tr + disc).clamp_min(0.0)

    return (1.0 / (lambda_1.sqrt() + EPS)).clamp(*LAMBDA_MIN_CLAMP_PX)
