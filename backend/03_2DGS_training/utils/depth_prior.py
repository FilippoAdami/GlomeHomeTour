#
# Monocular depth/normal priors for 2DGS training.
#
# Stage 2 (02_depth_estimation) already produces one metrically-aligned DA3 depth
# map per keyframe, but until now training only ever saw them indirectly, through
# the surfel cloud they were fused into. Once the optimiser starts moving surfels
# nothing holds a textureless wall in place: the photometric loss is flat there
# and the multi-view loss is flat there for the same reason. These per-pixel
# priors are that anchor.
#
# Two pieces live here:
#   * `build_confidence_cache` -- run once per scene. Reprojects every depth map
#     into a handful of geometrically-chosen neighbours and records, per pixel,
#     how many of them agree. That count is the confidence C_d, and it is what
#     demotes windows, mirrors and flying edge pixels, where the depth model is
#     confidently wrong in a way no single view can detect.
#   * `DepthPriors` + the two loss functions -- the training-time side.
#
# Deliberately NOT reusing 02_depth_estimation's filter_multiview_consistency():
# that one works on the fused 3D cloud in the ARCore/OpenGL frame, and its output
# is a per-point keep/drop decision. Here the consensus has to be per-pixel and
# has to be measured in exactly the frame and intrinsics convention the
# rasteriser renders with, or the confidence would weight the loss against a
# camera model the loss does not use.
#

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn.functional as F

_utilities_path = Path(__file__).resolve().parents[2] / "Utilities"
if str(_utilities_path) not in sys.path:
    sys.path.insert(0, str(_utilities_path))
from surface_normals import (  # noqa: E402
    fit_plane_normals, orient_towards, grazing_angle_factor, residual_confidence_factor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_tensor import wavelength_map  # noqa: E402

# Confidence cache resolution divisor. Priors are smooth; half resolution is
# also exactly what the r=2 training stage renders at, so that stage needs no
# resampling at all.
CACHE_DIVISOR = 2
# Neighbours reprojected into, per frame, and the geometric band they are picked
# from. Below MIN_BASELINE two views see the same error identically and agreement
# means nothing; above MAX_BASELINE occlusion dominates and disagreement means
# nothing either.
NUM_CONSENSUS_VIEWS = 6
MIN_BASELINE_M = 0.10
MAX_BASELINE_M = 2.00
MAX_VIEW_ANGLE_DEG = 60.0
# Confidence for a pixel no neighbour could observe at all (image borders,
# mostly). Unverified is not the same as contradicted, so it is not zero -- but
# it does not get a verified pixel's weight either.
UNVERIFIED_CONF = 0.5
# Agreement tolerance, absolute + relative. Matches the band
# 02_depth_estimation/initialization.py uses for the same test on the cloud.
DEPTH_TOL_ABS_M = 0.03
DEPTH_TOL_REL = 0.03
# 2dgs_combined_pipeline.md Stage 0 step 4: plane fit over a ~3-5 px window,
# same agreement band as the multi-view consensus test above (§4 reuses it
# rather than introducing a second knob).
PLANE_FIT_RADIUS = 2
# Scale at which the plane-fit residual factor in C_n decays to ~1/e.
PLANE_FIT_RESIDUAL_SCALE_M = 0.02


def _intrinsics_from_fov(fovx, fovy, width, height):
    """K in the rasteriser's convention: principal point at the image centre.

    COLMAP's own cx/cy are a few pixels off-centre here, but diff-surfel-
    rasterization has no principal-point input and renders centred. Measuring
    consensus with COLMAP's cx/cy would score the depth maps against a camera
    the renderer never uses.
    """
    fx = width / (2.0 * math.tan(fovx * 0.5))
    fy = height / (2.0 * math.tan(fovy * 0.5))
    return fx, fy, width * 0.5, height * 0.5


# --------------------------------------------------------------------------
# Cache construction (offline, once per scene)
# --------------------------------------------------------------------------

def _pick_consensus_neighbors(centers, forwards, i):
    """Indices of frames that see frame `i`'s surface from a usefully different place."""
    d = np.linalg.norm(centers - centers[i], axis=1)
    cos = forwards @ forwards[i]
    ok = ((d >= MIN_BASELINE_M) & (d <= MAX_BASELINE_M)
          & (cos >= math.cos(math.radians(MAX_VIEW_ANGLE_DEG))))
    ok[i] = False
    cand = np.nonzero(ok)[0]
    if cand.size == 0:
        return cand
    # Nearest first, not widest first. Measured on this capture: at a 0.3 m
    # baseline 72% of frame i's pixels land inside neighbour j and agree to
    # ~1 cm; pick the widest pairs in the band instead and most of the frame
    # falls outside the neighbour's frustum, so the consensus is computed from
    # whatever survives -- a much noisier estimate of the same quantity.
    return cand[np.argsort(d[cand])[:NUM_CONSENSUS_VIEWS]]


def _wavelength_for_frame(frame, height, width, device):
    """Lambda_min (H, W) at cache resolution, or None if the image is missing.

    Missing images are not fatal: SAD densification falls back to the gradient
    criterion frame by frame, which is what every run before Stage 0.6 existed
    used for all of them.
    """
    path = frame.get("image_path")
    if not path or not os.path.exists(path):
        return None
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    arr = arr.permute(2, 0, 1).to(device)
    # Resized to the cache grid *before* the tensor, not after: Lambda_min is a
    # length in pixels, so computing it at one resolution and resampling the map
    # to another would keep the numbers of the first. Everything downstream
    # rescales from the cache resolution, and only that one.
    if arr.shape[-2:] != (height, width):
        arr = F.interpolate(arr[None], size=(height, width), mode="area")[0]
    return wavelength_map(arr)


@torch.no_grad()
def build_confidence_cache(frames, out_dir, device="cuda", verbose=True):
    """Write one ``<name>.npz`` of (half-res depth, per-pixel consensus count) per frame.

    `frames` is a list of dicts with keys: name, depth_path, w2c (4x4 world->camera,
    OpenCV convention), fovx, fovy, width, height.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w2c = np.stack([f["w2c"] for f in frames]).astype(np.float64)
    c2w = np.linalg.inv(w2c)
    centers = c2w[:, :3, 3]
    forwards = c2w[:, :3, 2]          # OpenCV: camera looks down +Z

    depths, shapes = [], []
    for f in frames:
        d = np.load(f["depth_path"]).astype(np.float32)
        d = torch.from_numpy(d)[None, None]
        d = F.interpolate(d, scale_factor=1.0 / CACHE_DIVISOR, mode="nearest")[0, 0]
        depths.append(d.to(device))
        shapes.append(d.shape)

    H, W = shapes[0]
    fx, fy, cx, cy = _intrinsics_from_fov(frames[0]["fovx"], frames[0]["fovy"], W, H)
    vv, uu = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32),
                            torch.arange(W, device=device, dtype=torch.float32),
                            indexing="ij")
    ray_x, ray_y = (uu - cx) / fx, (vv - cy) / fy

    w2c_t = torch.from_numpy(w2c).float().to(device)
    c2w_t = torch.from_numpy(c2w).float().to(device)

    stats = {"frames": len(frames), "mean_conf": 0.0, "mean_conf_n": 0.0,
             "mean_lambda_min_px": 0.0, "no_lambda_frames": 0, "no_neighbor_frames": 0}
    for i, f in enumerate(frames):
        d_i = depths[i]
        # Camera-space point cloud for frame i's own view, reused below both to
        # reproject into neighbours (C_d) and to plane-fit normals (C_n).
        pc_hw = torch.stack([ray_x * d_i, ray_y * d_i, d_i], dim=-1)  # (H, W, 3)
        pc = pc_hw.reshape(-1, 3)

        # Stage 0 §4 step 4: back-project depth to camera-space points, fit a
        # plane in a small window excluding neighbours past a depth-scaled
        # distance -- not the plain finite-difference cross product, which
        # smears normals across depth discontinuities.
        normal_cam, residual, _count = fit_plane_normals(
            pc_hw, d_i > 0.2, radius=PLANE_FIT_RADIUS,
            dist_thresh_abs=DEPTH_TOL_ABS_M, dist_thresh_rel=DEPTH_TOL_REL, depth=d_i,
        )
        ray_dir_cam = F.normalize(pc_hw, dim=-1, eps=1e-8)
        normal_cam = orient_towards(normal_cam, ray_dir_cam)
        grazing = grazing_angle_factor(normal_cam, ray_dir_cam)
        residual_factor = residual_confidence_factor(residual, scale=PLANE_FIT_RESIDUAL_SCALE_M)

        # Rotating a normal only needs the rotation block of c2w (translation
        # does not act on directions); elementwise column form as elsewhere.
        R_i = c2w_t[i, :3, :3]
        normal_flat = normal_cam.reshape(-1, 3)
        normal_world = (normal_flat[:, 0:1] * R_i[:, 0] + normal_flat[:, 1:2] * R_i[:, 1]
                        + normal_flat[:, 2:3] * R_i[:, 2])
        normal_world = F.normalize(normal_world, dim=-1, eps=1e-8).reshape(H, W, 3)

        nbrs = _pick_consensus_neighbors(centers, forwards, i)
        if nbrs.size == 0:
            # No usable neighbour: trust the depth map rather than zeroing it,
            # otherwise a short or slow segment of the capture silently loses
            # its prior entirely.
            conf = torch.full_like(d_i, 1.0)
            stats["no_neighbor_frames"] += 1
        else:
            # Elementwise column form, never matmul: N here is H*W = 518k at
            # half res and 2.07M at full, both at or past the gfx1200 BLAS
            # row ceiling documented in backend/CLAUDE.md.
            R, t = c2w_t[i, :3, :3], c2w_t[i, :3, 3]
            pw = pc[:, 0:1] * R[:, 0] + pc[:, 1:2] * R[:, 1] + pc[:, 2:3] * R[:, 2] + t

            agree = torch.zeros(pw.shape[0], device=device)
            observed = torch.zeros(pw.shape[0], device=device)
            for j in nbrs:
                Rj, tj = w2c_t[j, :3, :3], w2c_t[j, :3, 3]
                pj = pw[:, 0:1] * Rj[:, 0] + pw[:, 1:2] * Rj[:, 1] + pw[:, 2:3] * Rj[:, 2] + tj
                z = pj[:, 2]
                in_front = z > 0.2
                zc = z.clamp_min(1e-4)
                u = fx * pj[:, 0] / zc + cx
                v = fy * pj[:, 1] / zc + cy
                inside = in_front & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
                # grid_sample wants [-1, 1]; nearest so an edge pixel is never
                # blended with a background one.
                gx = (u / (W - 1) * 2 - 1).clamp(-1, 1)
                gy = (v / (H - 1) * 2 - 1).clamp(-1, 1)
                grid = torch.stack([gx, gy], dim=-1).reshape(1, H, W, 2)
                obs = F.grid_sample(depths[j][None, None], grid, mode="nearest",
                                    align_corners=True).reshape(-1)
                tol = DEPTH_TOL_ABS_M + DEPTH_TOL_REL * z
                seen = inside & (obs > 0.2)
                observed += seen.float()
                agree += (seen & ((obs - z).abs() <= tol)).float()
            # Denominator is the neighbours that actually saw the pixel, not all
            # of them. A pixel outside a neighbour's frustum carries no evidence
            # either way; charging it as a disagreement would penalise the image
            # borders and every frame at the end of a pan.
            conf = torch.where(observed > 0, agree / observed.clamp_min(1.0),
                               torch.full_like(agree, UNVERIFIED_CONF)).reshape(H, W)

        # C_n = C_d x grazing-angle factor x plane-fit-residual factor (§4 step 4).
        conf_n = (conf * grazing * residual_factor).clamp(0.0, 1.0)

        # Stage 0.6: local feature wavelength, the reference SAD densification
        # measures a surfel's screen extent against. Computed from the image,
        # not the depth -- it asks what the *texture* can resolve, which is a
        # different question from where the surface is.
        lambda_min = _wavelength_for_frame(f, H, W, device)
        if lambda_min is None:
            stats["no_lambda_frames"] += 1
        else:
            stats["mean_lambda_min_px"] += float(lambda_min.mean()) / len(frames)

        stats["mean_conf"] += float(conf.mean()) / len(frames)
        stats["mean_conf_n"] += float(conf_n.mean()) / len(frames)
        arrays = dict(
            depth=d_i.cpu().numpy().astype(np.float16),
            conf=(conf * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
            normal=normal_world.cpu().numpy().astype(np.float16),
            conf_n=(conf_n * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy(),
        )
        if lambda_min is not None:
            arrays["lambda_min"] = lambda_min.cpu().numpy().astype(np.float16)
        np.savez_compressed(out_dir / f"{f['name']}.npz", **arrays)
        if verbose and (i % 25 == 0 or i == len(frames) - 1):
            print(f"  priors {i + 1}/{len(frames)}  mean conf so far "
                  f"{stats['mean_conf'] * len(frames) / (i + 1):.3f}  "
                  f"mean conf_n so far {stats['mean_conf_n'] * len(frames) / (i + 1):.3f}", flush=True)

    (out_dir / "prior_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# --------------------------------------------------------------------------
# Training-time side
# --------------------------------------------------------------------------

class DepthPriors:
    """All cached priors, resident on the CPU, resampled to the render size on demand.

    ~1.5 MB per frame at half resolution, so a 400-frame scene costs ~600 MB of
    host RAM and nothing on the GPU between uses. Held on the CPU rather than the
    GPU because VRAM is the binding constraint here (surfels), host RAM is not,
    and the per-iteration upload is one 1.5 MB transfer.
    """

    def __init__(self, prior_dir):
        self.dir = Path(prior_dir)
        self.depth, self.conf, self.normal, self.conf_n = {}, {}, {}, {}
        self.lambda_min = {}
        for p in sorted(self.dir.glob("*.npz")):
            with np.load(p) as z:
                self.depth[p.stem] = torch.from_numpy(z["depth"].astype(np.float32))
                self.conf[p.stem] = torch.from_numpy(z["conf"].astype(np.float32) / 255.0)
                if "lambda_min" in z:
                    self.lambda_min[p.stem] = torch.from_numpy(z["lambda_min"].astype(np.float32))
                if "normal" in z:
                    # (H, W, 3) -> (3, H, W), matching rend_normal's layout.
                    self.normal[p.stem] = torch.from_numpy(z["normal"].astype(np.float32)).permute(2, 0, 1)
                    self.conf_n[p.stem] = torch.from_numpy(z["conf_n"].astype(np.float32) / 255.0)

    def __len__(self):
        return len(self.depth)

    def get(self, view):
        """(depth, confidence) as (1, H, W) cuda tensors at `view`'s render size."""
        d = self.depth.get(view.image_name)
        if d is None:
            return None, None
        H, W = int(view.image_height), int(view.image_width)
        d = d[None, None].to("cuda", non_blocking=True)
        c = self.conf[view.image_name][None, None].to("cuda", non_blocking=True)
        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            # Nearest for confidence: bilinear would invent intermediate
            # confidences along the exact edges the filter exists to reject.
            c = F.interpolate(c, size=(H, W), mode="nearest")
        return d[0], c[0]

    def get_normal(self, view):
        """(normal, C_n) as (3, H, W) / (1, H, W) cuda tensors at render size.

        Cached at build time from the depth map's own plane fit (Stage 0 §4),
        not recomputed per iteration -- pose refinement can drift the camera
        this prior was captured from by the same amount the cached depth
        prior already tolerates, so this stays consistent with `get()`.
        """
        n = self.normal.get(view.image_name)
        if n is None:
            return None, None
        H, W = int(view.image_height), int(view.image_width)
        n = n[None].to("cuda", non_blocking=True)
        cn = self.conf_n[view.image_name][None, None].to("cuda", non_blocking=True)
        if n.shape[-2:] != (H, W):
            n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
            n = F.normalize(n, dim=1, eps=1e-8)
            cn = F.interpolate(cn, size=(H, W), mode="nearest")
        return n[0], cn[0]

    def get_wavelength(self, view):
        """Lambda_min as a (1, H, W) cuda tensor, in pixels *at the render size*.

        The rescale is the whole point of this method. Lambda_min is a length in
        pixels of the cache image; a feature 6 px across at half resolution is
        12 px across at full. SAD compares it against a surfel's projected
        extent, which the rasteriser measures in render pixels, so returning the
        cached numbers unscaled would make every surfel look twice as
        well-resolved at native resolution as it is -- and silently, since the
        only symptom is that splitting quietly stops. This is the same
        resolution-scaling trap that killed surf_normal at 1080p (see
        project_history.md, 2026-09-21 prior audit).
        """
        lm = self.lambda_min.get(view.image_name)
        if lm is None:
            return None
        H, W = int(view.image_height), int(view.image_width)
        cache_h, cache_w = lm.shape[-2:]
        lm = lm[None, None].to("cuda", non_blocking=True)
        if (cache_h, cache_w) != (H, W):
            lm = F.interpolate(lm, size=(H, W), mode="bilinear", align_corners=False)
            lm = lm * (0.5 * (W / cache_w + H / cache_h))
        return lm[0]


def prior_weight(base, iteration, decay_from, decay_until, floor):
    """`base`, decayed linearly to `base * floor` over [decay_from, decay_until].

    The prior is right about where a wall is and wrong in the last centimetre;
    the photometric and multi-view terms are the reverse. Handing control over
    from one to the other is the point of the decay -- without it the depth
    model's bias is baked into the final geometry.
    """
    if decay_until <= decay_from or iteration <= decay_from:
        return base
    if iteration >= decay_until:
        return base * floor
    t = (iteration - decay_from) / float(decay_until - decay_from)
    return base * (1.0 + t * (floor - 1.0))


def depth_prior_loss(rendered_depth, prior_depth, conf, mask):
    """Confidence-weighted L1 between rendered and prior depth, in metres."""
    w = conf * mask
    denom = w.sum()
    if denom < 1.0:
        return torch.zeros((), device=rendered_depth.device)
    return ((rendered_depth - prior_depth).abs() * w).sum() / denom


def normal_prior_loss(rendered_normal, prior_normal, conf, mask):
    """Confidence-weighted cosine loss between rendered and prior normals.

    Both are expected in world space and alpha-scaled the same way, i.e. exactly
    the pair `rend_normal` / `surf_normal` that the built-in normal consistency
    loss compares -- only the target is the prior's geometry instead of the
    render's own.
    """
    w = conf * mask
    denom = w.sum()
    if denom < 1.0:
        return torch.zeros((), device=rendered_normal.device)
    cos = (rendered_normal * prior_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    return ((1.0 - cos) * w).sum() / denom


def depth_convergence_loss(expected_depth, unbiased_depth, mask):
    """Pull expected depth onto the unbiased surface where alpha says there is one.

    `unbiased_depth` is the rasteriser's Eq. (9) channel (Unbiased Depth for 2DGS,
    arXiv 2503.06587): the depth of the first splat whose accumulated
    O_i = sum_j (alpha_j + eps) * G_j crosses the threshold. That is the real surface
    criterion this loss used to approximate with the median buffer, which only stood
    in for it while the HIP kernel did not expose O_i.

    The target is detached, so the gradient lands entirely on the expected-depth side
    -- the side that carries the bias.
    """
    if not mask.any():
        return torch.zeros((), device=expected_depth.device)
    diff = (expected_depth - unbiased_depth.detach()).abs()
    return diff[mask].mean()


# --------------------------------------------------------------------------
# CLI: build the cache for a workspace
# --------------------------------------------------------------------------

DEPTH_MAPS_REL = os.path.join("02_depth_estimation", "depth", "depth_maps")
PRIOR_DIR_REL = os.path.join("03_2DGS_training", "depth_priors")


def frames_from_colmap(source_path, depth_maps_dir):
    """Frame records for every COLMAP image that has a depth map, in the training frame."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
    from utils.graphics_utils import getWorld2View2, focal2fov

    sparse = Path(source_path) / "sparse" / "0"
    extr = read_extrinsics_binary(str(sparse / "images.bin"))
    intr = read_intrinsics_binary(str(sparse / "cameras.bin"))
    depth_maps_dir = Path(depth_maps_dir)

    frames = []
    for key in extr:
        e = extr[key]
        i = intr[e.camera_id]
        name = os.path.basename(e.name).split(".")[0]
        dp = depth_maps_dir / f"{name}.npy"
        if not dp.exists():
            continue
        if i.model == "SIMPLE_PINHOLE":
            fl_x = fl_y = i.params[0]
        elif i.model == "PINHOLE":
            fl_x, fl_y = i.params[0], i.params[1]
        else:
            raise ValueError(f"unsupported COLMAP camera model {i.model}")
        R = np.transpose(qvec2rotmat(e.qvec))
        T = np.array(e.tvec)
        # The image itself, for the Stage 0.6 wavelength map. COLMAP records the
        # filename with its extension; the depth map does not, hence e.name here
        # rather than `name`.
        image_path = Path(source_path) / "images" / os.path.basename(e.name)
        frames.append({
            "name": name,
            "depth_path": str(dp),
            "image_path": str(image_path) if image_path.exists() else None,
            "w2c": getWorld2View2(R, T).astype(np.float64),
            "fovx": focal2fov(fl_x, i.width),
            "fovy": focal2fov(fl_y, i.height),
            "width": i.width,
            "height": i.height,
        })
    # Sorted so the neighbour search and the on-disk cache are reproducible.
    frames.sort(key=lambda f: f["name"])
    return frames


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build the per-pixel depth/normal prior cache")
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("-o", "--out_dir", default=None)
    ap.add_argument("--depth_maps", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    src = Path(a.source_path).resolve()
    depth_maps = Path(a.depth_maps) if a.depth_maps else src / DEPTH_MAPS_REL
    out_dir = Path(a.out_dir) if a.out_dir else src / PRIOR_DIR_REL

    if not depth_maps.is_dir():
        raise FileNotFoundError(f"{depth_maps} missing -- run 02_depth_estimation first")

    frames = frames_from_colmap(src, depth_maps)
    if not frames:
        raise RuntimeError(f"no COLMAP image matched a depth map in {depth_maps}")

    # Count *and* content. A cache built before the normal prior existed has one
    # npz per frame and no "normal" key in any of them, so a count-only check
    # accepts it -- and DepthPriors.get_normal then returns None for every frame,
    # silently disabling the normal prior loss for the whole run. Check the keys
    # the training side actually reads.
    cached = sorted(out_dir.glob("*.npz"))
    if len(cached) == len(frames) and not a.force:
        with np.load(cached[0]) as z:
            missing = [k for k in ("depth", "conf", "normal", "conf_n", "lambda_min")
                       if k not in z]
        if missing:
            print(f"[priors] cache in {out_dir} is stale (missing {', '.join(missing)}); rebuilding")
        else:
            print(f"[priors] {len(cached)} already cached in {out_dir}, skipping (--force to rebuild)")
            return 0

    print(f"[priors] {len(frames)} frames -> {out_dir}")
    stats = build_confidence_cache(frames, out_dir)
    print(f"[priors] done: mean confidence {stats['mean_conf']:.3f}, "
          f"{stats['no_neighbor_frames']} frames with no usable neighbour")
    # §7 diagnostic rows for the two cache quantities, reported rather than
    # enforced -- the doc's corrective actions are parameter changes for a
    # human to make, not something to silently apply mid-build.
    lam = stats["mean_lambda_min_px"]
    if stats["no_lambda_frames"] == len(frames):
        print("[priors] [warning] no images found; Lambda_min maps not built, "
              "SAD densification will fall back to the gradient criterion")
    else:
        print(f"[priors] mean Lambda_min {lam:.2f} px at cache resolution "
              f"({stats['no_lambda_frames']} frames without an image)")
        if lam < 1.0:
            print("[priors] [warning] mean Lambda_min < 1 px (§7): the structure "
                  "tensor is amplifying sensor noise; raise structure_tensor.SIGMA_BASE")
    if not 0.65 <= stats["mean_conf"] <= 0.98:
        print(f"[priors] [warning] mean C_d {stats['mean_conf']:.3f} outside the §7 "
              "healthy band 0.80-0.95; check the reprojection baseline and depth noise")
    # A mean confidence near zero means the reprojection convention is wrong,
    # not that the depth maps are bad -- fail rather than train on a prior that
    # is everywhere-zero-weighted.
    if stats["mean_conf"] < 0.05:
        raise RuntimeError(
            f"mean multi-view confidence is {stats['mean_conf']:.3f}; the depth maps "
            "and COLMAP poses do not agree at all. Check that both come from the "
            "same run before training against them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
