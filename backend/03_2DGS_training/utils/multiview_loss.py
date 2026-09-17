#
# PGSR-style multi-view planar photometric consistency loss.
#
# 2DGS already regularises geometry *within* a view (normal_loss ties the rendered
# normal to the depth-derived one, dist_loss penalises depth distortion). What it
# has no term for is agreement *between* views, which is PGSR's actual
# contribution: treat each pixel's rendered depth + normal as a local plane, use
# the plane-induced homography to find where that pixel lands in a neighbouring
# view, and penalise the colour difference.
#
# Deliberately gather-based: for every sampled pixel of the current view we look
# up the *ground-truth* neighbour image at the warped location. That needs no
# extra render pass, so the loss costs a few ms instead of doubling iteration
# time, at the price of comparing against captured rather than rendered colour
# (which is what we want anyway - the GT image is the harder constraint).
#

import os

import torch

from utils.graphics_utils import fov2focal
from utils.loss_utils import compute_saturation_mask


_NAN_DEBUG = os.environ.get("GLOME_NAN_DEBUG") == "1"
# Populated only under _NAN_DEBUG. A finite loss here can still emit NaN
# gradients, so the forward values alone do not identify the culprit -- both
# directions have to be recorded per intermediate.
FWD_BLAME = {}
GRAD_BLAME = {}


def _dbg(t, name):
    """Record non-finite forward values and backward gradients for `t`."""
    if not _NAN_DEBUG:
        return t
    n_fwd = int((~torch.isfinite(t)).sum())
    if n_fwd:
        FWD_BLAME[name] = max(FWD_BLAME.get(name, 0), n_fwd)
    if t.requires_grad:
        def _hook(g, _n=name):
            bad = int((~torch.isfinite(g)).sum())
            if bad:
                GRAD_BLAME[_n] = max(GRAD_BLAME.get(_n, 0), bad)
        t.register_hook(_hook)
    return t


def build_capture_order(cameras):
    """Cameras sorted into capture order, plus name -> index.

    Scene shuffles the training list, and the COLMAP loader reuses one intrinsics
    id for every frame, so neither list position nor `uid` identifies a frame.
    Both dataset readers sort by `image_name` before loading, and these captures
    are sequential video, so sorting by name recovers capture order - which for
    video is also spatial adjacency.
    """
    order = sorted(cameras, key=lambda cam: cam.image_name)
    return order, {cam.image_name: i for i, cam in enumerate(order)}


def pick_neighbors(view, order, index, num_neighbors=2, stride=2):
    """The nearest frames either side of `view` in capture order."""
    i = index.get(view.image_name)
    if i is None:
        return []
    picks = []
    for step in range(1, num_neighbors + 1):
        for j in (i - step * stride, i + step * stride):
            if 0 <= j < len(order) and j != i:
                picks.append(order[j])
        if len(picks) >= num_neighbors:
            break
    return picks[:num_neighbors]


def _intrinsics(view, device):
    """Pinhole K matching the rasteriser's convention (centred principal point)."""
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    return torch.tensor([[fx, 0.0, view.image_width / 2.0],
                         [0.0, fy, view.image_height / 2.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def multiview_photometric_loss(view, neighbors, depth, normal, alpha,
                               num_samples=8000, alpha_thresh=0.5,
                               min_depth=0.05, veto_weight=0.5,
                               saturation_threshold=0.98,
                               saturation_percentile=99.8,
                               saturation_max_chroma_diff=0.15):
    """Plane-induced-homography photometric loss between `view` and `neighbors`.

    depth  : (1, H, W) rendered surf_depth (z in camera space)
    normal : (3, H, W) rendered normal, world frame (as the renderer returns it)
    alpha  : (1, H, W) rendered accumulated alpha, used to skip empty pixels
    veto_weight : weight on max-error neighbor loss vs mean loss (0.0 = mean, 0.5 = 50/50)
    saturation_threshold : dynamic saturation threshold floor (<= 0 disables saturation masking)
    """
    if not neighbors:
        return torch.zeros((), device=depth.device)

    device = depth.device
    # Named img_h/img_w, not H/W: H is the homography below, and the collision is
    # silent (broadcasting turns "uv <= H - 1" into a shape error at best).
    img_h, img_w = depth.shape[-2:]

    # Sample pixels that actually have geometry; a full-frame warp would build a
    # 3x3 matrix per pixel (~2M for 1080p) for no accuracy gain.
    # ponytail: stochastic pixel subset. Switch to patch/census windows if the
    # per-pixel photometric term turns out too noisy to converge.
    solid = (alpha[0] > alpha_thresh) & (depth[0] > min_depth)

    gt = view.original_image.to(device)
    if saturation_threshold > 0:
        sat_mask = compute_saturation_mask(
            gt,
            min_threshold=saturation_threshold,
            percentile=saturation_percentile,
            max_chroma_diff=saturation_max_chroma_diff
        )
        solid = solid & (~sat_mask)

    flat = solid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    if flat.numel() < 256:
        return torch.zeros((), device=device)
    if flat.numel() > num_samples:
        flat = flat[torch.randperm(flat.numel(), device=device)[:num_samples]]

    ys = (flat // img_w).float()
    xs = (flat % img_w).float()
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)          # (P, 3)

    K = _intrinsics(view, device)
    K_inv = torch.inverse(K)

    d = depth.reshape(-1)[flat].unsqueeze(-1)                         # (P, 1)
    # Back-project to camera space: X = z * K^-1 [u v 1]
    X_cam = d * (pix @ K_inv.T)                                       # (P, 3)

    # world_view_transform is W2C transposed (row-vector convention), so its
    # rotation block is R_c2w and n_cam = R_c2w^T n_world.
    R_c2w = view.world_view_transform[:3, :3].to(device)
    n_world = normal.reshape(3, -1)[:, flat].T                        # (P, 3)
    n_cam = n_world @ R_c2w                                           # (P, 3)
    n_cam = torch.nn.functional.normalize(n_cam, dim=-1)

    # Plane offset q = n . X. Pixels whose plane passes through the camera are
    # degenerate for the homography, so drop them.
    q = (n_cam * X_cam).sum(dim=-1, keepdim=True)
    usable = q.abs().squeeze(-1) > 1e-4
    # `usable` drops these pixels from the loss, but only *after* the division
    # below has already produced inf for them -- and 0 * inf is NaN in backward.
    # The loss value stays finite, so no loss-level guard can see it; it surfaces
    # only as NaN gradients that silently rot the scene. Substitute a finite
    # denominator before dividing; its value is irrelevant since these pixels
    # are masked out anyway.
    q_safe = torch.where(q.abs() > 1e-4, q, torch.full_like(q, 1e-4))
    _dbg(X_cam, "X_cam"); _dbg(n_cam, "n_cam"); _dbg(q, "q"); _dbg(q_safe, "q_safe")
    if usable.sum() < 256:
        return torch.zeros((), device=device)

    src_rgb = gt.reshape(3, -1)[:, flat].T                            # (P, 3)

    W2C_cur = view.world_view_transform.T.to(device)
    losses = []
    for nb in neighbors:
        W2C_nb = nb.world_view_transform.T.to(device)
        T_rel = W2C_nb @ torch.inverse(W2C_cur)                       # cur cam -> nb cam
        R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]
        K_nb = _intrinsics(nb, device)

        # H = K_nb (R - t n^T / q) K_cur^-1, one 3x3 per sampled pixel.
        M = R_rel.unsqueeze(0) - torch.einsum("i,pj->pij", t_rel, n_cam / q_safe)
        H = K_nb.unsqueeze(0) @ M @ K_inv.unsqueeze(0)                # (P, 3, 3)

        _dbg(M, "M"); _dbg(H, "H")
        warped = torch.einsum("pij,pj->pi", H, pix)
        z = warped[:, 2]
        uv = warped[:, :2] / torch.where(z.abs() < 1e-8,
                                         torch.full_like(z, 1e-8), z).unsqueeze(-1)
        _dbg(warped, "warped"); _dbg(z, "z"); _dbg(uv, "uv")

        valid = usable & (z > 1e-6)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= img_w - 1)
        valid &= (uv[:, 1] >= 0) & (uv[:, 1] <= img_h - 1)
        if valid.sum() < 256:
            continue

        # grid_sample wants normalised coords in (-1, 1), shape (N, 1, P, 2).
        grid = torch.stack([uv[:, 0] / (img_w - 1) * 2.0 - 1.0,
                            uv[:, 1] / (img_h - 1) * 2.0 - 1.0], dim=-1)
        sampled = torch.nn.functional.grid_sample(
            nb.original_image.to(device).unsqueeze(0),
            grid.unsqueeze(0).unsqueeze(0),
            mode="bilinear", padding_mode="border", align_corners=True)
        nb_rgb = sampled.squeeze(0).squeeze(1).T                       # (P, 3)
        _dbg(grid, "grid"); _dbg(nb_rgb, "nb_rgb")

        if saturation_threshold > 0:
            nb_max = nb_rgb.max(dim=-1).values
            nb_min = nb_rgb.min(dim=-1).values
            nb_is_sat = (nb_max >= saturation_threshold) & ((nb_max - nb_min) <= saturation_max_chroma_diff)
            valid = valid & (~nb_is_sat)

        if valid.sum() < 256:
            continue

        losses.append((src_rgb[valid] - nb_rgb[valid]).abs().mean())

    if not losses:
        return torch.zeros((), device=device)

    all_losses = torch.stack(losses)
    if veto_weight <= 0.0 or all_losses.numel() == 1:
        return all_losses.mean()
    mean_loss = all_losses.mean()
    max_loss = all_losses.max()
    return (1.0 - veto_weight) * mean_loss + veto_weight * max_loss
