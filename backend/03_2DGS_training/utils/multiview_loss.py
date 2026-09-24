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
# The comparison itself is PGSR's NCC over a greyscale patch, not a per-pixel
# colour difference. Both endpoints of the warp sit in a phone capture with
# auto-exposure and a moving light field, so an absolute colour difference is
# partly measuring exposure; NCC is invariant to affine intensity change and
# measures only whether the *structure* lines up, which is what the homography
# is being supervised on. The whole patch is warped by the pixel's own
# homography, i.e. PGSR's assumption that the patch is locally planar.
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


def _patch_offsets(patch, device):
    """(K, 2) pixel offsets of a `patch` x `patch` window centred on 0."""
    r = patch // 2
    d = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    oy, ox = torch.meshgrid(d, d, indexing="ij")
    return torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=-1)


def _grey(img):
    """(3, H, W) -> (1, H, W) luma. NCC compares structure, so colour is noise here."""
    return (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)


def _sample(img, uv, img_w, img_h):
    """Bilinear lookup of `img` (C, H, W) at pixel coords `uv` (..., 2) -> (..., C)."""
    grid = torch.stack([uv[..., 0] / (img_w - 1) * 2.0 - 1.0,
                        uv[..., 1] / (img_h - 1) * 2.0 - 1.0], dim=-1)
    shape = grid.shape[:-1]
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid.reshape(1, 1, -1, 2),
        mode="bilinear", padding_mode="border", align_corners=True)
    return out.reshape(img.shape[0], -1).T.reshape(*shape, img.shape[0])


def _ncc(ref, tgt, eps=1e-8):
    """Per-patch normalised cross-correlation of (P, K) intensity patches -> (P,).

    `eps` only has to keep the division finite, so it is well below any real
    patch variance. A larger one silently shrinks the correlation instead: patch
    variance here is ~1e-2, so an eps of 1e-4 inside the square root drags a
    perfect match down to ~0.5. Degenerate low-variance patches are excluded by
    the caller's `textured` mask, not by this epsilon.
    """
    ref_c = ref - ref.mean(dim=-1, keepdim=True)
    tgt_c = tgt - tgt.mean(dim=-1, keepdim=True)
    cov = (ref_c * tgt_c).mean(dim=-1)
    denom = (torch.sqrt(ref_c.pow(2).mean(dim=-1) + eps)
             * torch.sqrt(tgt_c.pow(2).mean(dim=-1) + eps))
    return (cov / denom).clamp(-1.0, 1.0)


def multiview_photometric_loss(view, neighbors, depth, normal, alpha,
                               num_samples=2048, alpha_thresh=0.5,
                               min_depth=0.05, veto_weight=0.5,
                               ncc_patch=11,
                               saturation_threshold=0.98,
                               saturation_percentile=99.8,
                               saturation_max_chroma_diff=0.15):
    """Plane-induced-homography NCC loss between `view` and `neighbors`.

    depth  : (1, H, W) rendered unbiased depth (z in camera space)
    normal : (3, H, W) rendered normal, world frame (as the renderer returns it)
    alpha  : (1, H, W) rendered accumulated alpha, used to skip empty pixels
    veto_weight : weight on max-error neighbor loss vs mean loss (0.0 = mean, 0.5 = 50/50)
    ncc_patch : NCC window side in pixels at full resolution; halved on the
        downsampled stages so the window keeps covering the same scene area
        (Dash renders at a divisor, and an 11 px window at 420 px wide is a
        different physical patch than at 1080 px).
    saturation_threshold : dynamic saturation threshold floor (<= 0 disables saturation masking)
    """
    if not neighbors:
        return torch.zeros((), device=depth.device)

    device = depth.device
    # Named img_h/img_w, not H/W: H is the homography below, and the collision is
    # silent (broadcasting turns "uv <= H - 1" into a shape error at best).
    img_h, img_w = depth.shape[-2:]

    # Patch side, scaled to the rendered resolution (see docstring). Kept odd so
    # the window stays centred on its pixel.
    patch = ncc_patch if img_w >= 900 else max(5, (ncc_patch // 2) | 1)

    # Sample pixels that actually have geometry; a full-frame warp would build a
    # 3x3 matrix per pixel (~2M for 1080p) for no accuracy gain. Each sample now
    # costs `patch**2` texture lookups, hence far fewer of them than the
    # per-pixel version used.
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

    # Reference patches, sampled once and shared across neighbours.
    gt_grey = _grey(gt)
    offsets = _patch_offsets(patch, device)                           # (K, 2)
    pix_patch = torch.cat([pix[:, None, :2] + offsets[None],
                           torch.ones(pix.shape[0], offsets.shape[0], 1, device=device)],
                          dim=-1)                                     # (P, K, 3)
    ref_patch = _sample(gt_grey, pix_patch[..., :2], img_w, img_h).squeeze(-1)  # (P, K)
    # A patch that is flat in the reference has no structure to correlate, so its
    # NCC is dominated by sensor noise. Those are exactly the textureless pixels
    # the depth prior is responsible for, not this loss.
    textured = ref_patch.std(dim=-1) > 0.01

    W2C_cur = view.world_view_transform.T.to(device)
    losses = []
    for nb in neighbors:
        W2C_nb = nb.world_view_transform.T.to(device)
        T_rel = W2C_nb @ torch.inverse(W2C_cur)                       # cur cam -> nb cam
        R_rel, t_rel = T_rel[:3, :3], T_rel[:3, 3]
        K_nb = _intrinsics(nb, device)

        # H = K_nb (R + t n^T / q) K_cur^-1, one 3x3 per sampled pixel.
        #
        # Plus, not minus. For a point X on the plane, x_nb = R X + t, and
        # n . X = q lets the translation be written as t * (n . X) / q, giving
        # (R + t n^T / q) X. A minus here applies the baseline backwards: it warps
        # to the pixel the neighbour would see if it sat on the *other* side of
        # the reference camera, so the loss was pulling geometry towards a
        # mirrored correspondence. The sign convention of n itself does not
        # matter -- flipping n flips q with it, and n/q is unchanged.
        M = R_rel.unsqueeze(0) + torch.einsum("i,pj->pij", t_rel, n_cam / q_safe)
        H = K_nb.unsqueeze(0) @ M @ K_inv.unsqueeze(0)                # (P, 3, 3)

        _dbg(M, "M"); _dbg(H, "H")
        # Whole patch through the pixel's own homography (PGSR: the patch is
        # assumed to lie on the same plane as its centre).
        warped = torch.einsum("pij,pkj->pki", H, pix_patch)            # (P, K, 3)
        z = warped[..., 2]
        uv = warped[..., :2] / torch.where(z.abs() < 1e-8,
                                           torch.full_like(z, 1e-8), z).unsqueeze(-1)
        _dbg(warped, "warped"); _dbg(z, "z"); _dbg(uv, "uv")

        centre = offsets.shape[0] // 2
        valid = usable & textured & (z[:, centre] > 1e-6)
        # The whole patch has to land inside the neighbour; a patch clamped
        # against the border correlates with the border, not with the scene.
        in_view = ((uv[..., 0] >= 0) & (uv[..., 0] <= img_w - 1)
                   & (uv[..., 1] >= 0) & (uv[..., 1] <= img_h - 1)).all(dim=-1)
        valid &= in_view
        if valid.sum() < 64:
            continue

        nb_img = nb.original_image.to(device)
        nb_patch = _sample(_grey(nb_img), uv, img_w, img_h).squeeze(-1)  # (P, K)
        _dbg(uv, "uv_patch"); _dbg(nb_patch, "nb_patch")

        if saturation_threshold > 0:
            nb_rgb = _sample(nb_img, uv[:, centre], img_w, img_h)       # (P, 3)
            nb_max = nb_rgb.max(dim=-1).values
            nb_min = nb_rgb.min(dim=-1).values
            nb_is_sat = (nb_max >= saturation_threshold) & ((nb_max - nb_min) <= saturation_max_chroma_diff)
            valid = valid & (~nb_is_sat)

        if valid.sum() < 64:
            continue

        # (1 - NCC) / 2 in [0, 1]: 0 when the patches match, 1 when inverted.
        ncc = _ncc(ref_patch[valid], nb_patch[valid])
        losses.append(((1.0 - ncc) * 0.5).mean())

    if not losses:
        return torch.zeros((), device=device)

    all_losses = torch.stack(losses)
    if veto_weight <= 0.0 or all_losses.numel() == 1:
        return all_losses.mean()
    mean_loss = all_losses.mean()
    max_loss = all_losses.max()
    return (1.0 - veto_weight) * mean_loss + veto_weight * max_loss
