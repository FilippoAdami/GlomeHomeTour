#
# Per-surfel, per-view geometry used by two things in
# 2dgs_combined_pipeline.md that both need to know where a surfel lands on
# screen and how big it is there:
#
#   §4.1 Frequency violation -- a surfel's projected tangential extent against
#        the local feature wavelength Lambda_min, which decides what splits.
#   §5.3 Asymmetric free-space carving -- a surfel's depth against the rendered
#        unbiased surface depth along the same ray, which decides what is a
#        floater.
#
# Pure functions on tensors, no model and no rasteriser, so they can be checked
# against hand-computed pinhole geometry (tests/test_sad_densify.py).
#
# Two conventions to keep straight:
#   * `view.world_view_transform` is getWorld2View2(...).T, so world->camera is
#     `p @ M[:3, :3] + M[3, :3]`, not `M @ p`.
#   * Intrinsics come from the FoV with the principal point at the image
#     centre, because that is what diff-surfel-rasterization renders -- it has
#     no principal-point input. Using COLMAP's own cx/cy here would measure
#     surfel extents against a camera the renderer never uses (same reasoning
#     as depth_prior._intrinsics_from_fov).
#

from __future__ import annotations

import math

import torch

from utils.point_utils import _rotate


def intrinsics(view):
    """(fx, fy, cx, cy) in pixels at the view's current render resolution."""
    w, h = int(view.image_width), int(view.image_height)
    fx = w / (2.0 * math.tan(view.FoVx * 0.5))
    fy = h / (2.0 * math.tan(view.FoVy * 0.5))
    return fx, fy, w * 0.5, h * 0.5


def project(xyz: torch.Tensor, view):
    """(u, v, z) for (N, 3) world points: pixel coordinates and camera-space depth.

    `_rotate` rather than matmul throughout: torch.matmul zeroes output rows
    past 2**19 on gfx1200/ROCm 7.1, and a scene here carries ~5e5 surfels,
    right on that boundary.
    """
    m = view.world_view_transform
    p_cam = _rotate(xyz, m[:3, :3]) + m[3, :3]
    fx, fy, cx, cy = intrinsics(view)
    z = p_cam[:, 2]
    zc = z.clamp_min(1e-6)
    u = fx * p_cam[:, 0] / zc + cx
    v = fy * p_cam[:, 1] / zc + cy
    return u, v, z


def tangential_screen_extents(xyz, scaling, rotation_matrices, view):
    """(ext_u, ext_v) in pixels: how far each surfel's tangent axes reach on screen.

    `rotation_matrices` is (N, 3, 3) with columns [t_u, t_v, n], the layout
    build_rotation produces and the rest of the model assumes.

    This is the doc's v_u = J W R [s_u t_u] evaluated as a finite difference
    rather than by building the Jacobian: project the surfel centre and the
    tip of each scaled tangent axis, and take the distance between them. Same
    quantity to first order, and it stays exact through the perspective divide
    on the near, oblique surfels where J's linearisation is worst.
    """
    u0, v0, _ = project(xyz, view)
    out = []
    for axis in (0, 1):
        tip = xyz + scaling[:, axis:axis + 1] * rotation_matrices[:, :, axis]
        u1, v1, _ = project(tip, view)
        out.append(torch.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2))
    return out[0], out[1]


def sample_map(u, v, values, default=0.0):
    """Nearest-neighbour lookup of a (1, H, W) or (H, W) map at float pixel coords.

    Points outside the image get `default`. Rounded rather than bilinear: these
    maps (Lambda_min, unbiased depth) are discontinuous at object boundaries,
    and blending across one invents a value that is wrong on both sides.
    """
    if values.dim() == 3:
        values = values[0]
    h, w = values.shape
    ui = u.round().long()
    vi = v.round().long()
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    flat = values.reshape(-1)
    idx = (vi.clamp(0, h - 1) * w + ui.clamp(0, w - 1))
    return torch.where(inside, flat[idx], torch.full_like(u, default)), inside


def frequency_violation(ext_u, ext_v, lambda_min):
    """(eta_u, eta_v) = projected tangential extent / local feature wavelength.

    eta > 1 means the surfel is wider on screen than the finest feature the
    image carries there, i.e. it cannot represent that feature however its
    colour is optimised.
    """
    denom = lambda_min.clamp_min(1e-6)
    return ext_u / denom, ext_v / denom


def freespace_classify(z, depth_at_pixel, inside, margin):
    """(free, on_surface) masks for §5.3.

    free        -- the surfel sits in front of the surface this ray hit, by
                   more than the margin: nothing should be there.
    on_surface  -- it sits on that surface, within the margin.

    A surfel *behind* the surface is in neither: it is occluded from this view,
    which is evidence of nothing. That asymmetry is the whole point of the
    criterion -- it is why an unobstructed side view can veto a light-bloom
    protrusion that every head-on view happily accepts.
    """
    valid = inside & (depth_at_pixel > 0) & (z > 0)
    free = valid & (z < depth_at_pixel - margin)
    on_surface = valid & ((z - depth_at_pixel).abs() <= margin)
    return free, on_surface
