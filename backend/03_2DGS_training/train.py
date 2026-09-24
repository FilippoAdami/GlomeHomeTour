#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_saturation_mask, masked_l1_loss, masked_ssim, smooth_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.multiview_loss import build_capture_order, pick_neighbors, multiview_photometric_loss
from utils.pose_refine import TrackLoss
from utils.depth_prior import (DepthPriors, PRIOR_DIR_REL, prior_weight,
                               depth_prior_loss, normal_prior_loss,
                               depth_convergence_loss)

# Set GLOME_NAN_DEBUG=1 to trap the first non-finite gradient or parameter.
# Off by default: it adds a few reductions per step and aborts the run.
_NAN_DEBUG = os.environ.get("GLOME_NAN_DEBUG") == "1"
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# What 16 GB of VRAM holds at full resolution with the backward pass live.
VRAM_SURFEL_CEILING = 4_500_000


def growth_target(iteration, initial_points, budget, start, end):
    """Surfel count the schedule expects at `iteration`.

    Linear between `start` and `end`. Linear rather than front- or back-loaded
    because neither shape is measured here, and a straight line is the one that
    states its assumption honestly: the budget buys the same number of surfels
    per iteration throughout the growth window. Front-loading gives new surfels
    more iterations to be optimised but spends the budget while the render is
    still at the coarse stage, where the gradient cannot see the detail the
    surfels are meant to resolve; the two roughly cancel, so the shape is not
    worth a knob until something measures it.
    """
    if iteration <= start:
        return initial_points
    if iteration >= end:
        return budget
    u = (iteration - start) / float(end - start)
    return initial_points + (budget - initial_points) * u


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Surfel budget, proportional to the initialisation. Step 4 voxelises at a
    # fixed metric size, so the init count is already proportional to the surface
    # area of the property -- a budget of `growth_factor` x that scales from a
    # studio to a five-bedroom house without a per-scene number, which an absolute
    # budget (Taming's 1M) cannot. The VRAM ceiling is the only absolute here.
    initial_points = gaussians.get_xyz.shape[0]
    if opt.max_gaussians is not None and opt.max_gaussians < 0:
        opt.max_gaussians = min(int(opt.growth_factor * initial_points), VRAM_SURFEL_CEILING)
        print(f"max_gaussians: {opt.max_gaussians} "
              f"({opt.growth_factor}x {initial_points} init points)")

    # Both extras below index cameras by capture order: Scene shuffles the
    # training list, and the COLMAP loader hands every frame the same uid.
    train_cams = scene.getTrainCameras()
    mv_order, mv_index = build_capture_order(train_cams) if opt.lambda_multiview > 0.0 else ([], {})

    # Per-pixel depth/normal priors. Missing cache is not fatal: the run simply
    # falls back to the geometry the initialisation cloud carries, which is what
    # every run before these priors existed did.
    priors = None
    if opt.lambda_depth_prior > 0.0 or opt.lambda_normal_prior > 0.0:
        prior_dir = opt.depth_prior_dir or os.path.join(dataset.source_path, PRIOR_DIR_REL)
        if os.path.isdir(prior_dir):
            priors = DepthPriors(prior_dir)
            print(f"Depth priors: {len(priors)} frames from {prior_dir}")
        if not priors or len(priors) == 0:
            print(f"[warning] no depth prior cache at {prior_dir}; "
                  "depth/normal prior losses disabled. Run utils/depth_prior.py to build it.")
            priors = None

    exposure_optimizer = None
    if opt.optimize_exposure:
        for cam in train_cams:
            cam.enable_exposure_compensation()
        exposure_optimizer = torch.optim.Adam([cam.exposure for cam in train_cams],
                                              lr=opt.exposure_lr, eps=1e-15)
        print(f"Exposure compensation ON: {len(train_cams)} cameras")

    pose_optimizer, track_loss = None, None
    if opt.refine_poses_during_training:
        for cam in train_cams:
            cam.enable_pose_refinement()
        pose_optimizer = torch.optim.Adam([cam.pose_delta for cam in train_cams],
                                          lr=opt.pose_lr, eps=1e-15)
        track_loss = TrackLoss(dataset.source_path, train_cams)
        print(f"Pose refinement ON: {len(train_cams)} cameras, "
              f"track supervision on {track_loss.num_views} views")

    stage_first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)  # trusted, our own checkpoint; torch>=2.6 defaults weights_only=True
        stage_first_iter = first_iter
        gaussians.restore(model_params, opt)
        # Fresh gradient statistics for the new resolution stage
        if hasattr(gaussians, "xyz_gradient_accum") and gaussians.xyz_gradient_accum is not None:
            gaussians.xyz_gradient_accum.zero_()
        if hasattr(gaussians, "denom") and gaussians.denom is not None:
            gaussians.denom.zero_()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_dprior_for_log = 0.0
    ema_mv_for_log = 0.0
    ema_pos_grad_for_log = 0.0
    ema_composite_loss_for_log = 0.0
    prev_comp_loss = None
    rolling_loss_derivative = 0.0
    # §7 verification metrics. Collected over the run and checked against the
    # doc's healthy ranges at the end, where the numbers are comparable between
    # runs -- a threshold breach mid-schedule is often just the phase boundary.
    diagnostics = {"sad_added": 0, "freespace_pruned": 0,
                   "conv_at_3k": None, "mean_zncc": 0.0, "zncc_samples": 0,
                   "points_post_reset": None, "points_pre_reset": None}

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # §3 Phase 3: freeze p, q, s for the final stretch and optimise
        # appearance only. Has to come after update_learning_rate, which
        # rewrites the xyz group's lr from the decay schedule every iteration.
        if opt.freeze_geometry_last > 0 and iteration > opt.iterations - opt.freeze_geometry_last:
            for pg in gaussians.optimizer.param_groups:
                if pg["name"] in ("xyz", "scaling", "rotation"):
                    pg["lr"] = 0.0

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        if pose_optimizer is not None:
            # Rebuild this camera's matrices so the render is a function of the
            # current pose delta (otherwise the delta gets no gradient).
            viewpoint_cam.refresh_pose()

        iter_background = torch.rand(3, device="cuda") if dataset.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, iter_background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        # Photometric terms only. Every geometry term below reads render_pkg
        # directly, so the exposure parameters can never absorb a depth error.
        photo_image = viewpoint_cam.apply_exposure(image)
        valid_pix_mask = None
        if opt.saturation_threshold > 0:
            sat_mask = compute_saturation_mask(
                gt_image,
                min_threshold=opt.saturation_threshold,
                percentile=opt.saturation_percentile,
                max_chroma_diff=opt.saturation_max_chroma_diff
            )
            valid_pix_mask = ~sat_mask
            Ll1 = masked_l1_loss(photo_image, gt_image, mask=valid_pix_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - masked_ssim(photo_image, gt_image, mask=valid_pix_mask))
        else:
            Ll1 = l1_loss(photo_image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(photo_image, gt_image))

        # regularization
        rend_alpha = render_pkg['rend_alpha']
        # ponytail: HIP rasterizer occasionally emits NaN alpha on degenerate
        # surfels; clamp here rather than upstream, tighten if a real cause is found.
        rend_alpha = torch.nan_to_num(rend_alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        # The same degenerate surfels poison the normal buffers. Clean them here
        # rather than at each use: multiview_loss reads rend_normal too, and a
        # NaN there is just as fatal as one in normal_loss.
        render_pkg['rend_normal'] = torch.nan_to_num(render_pkg['rend_normal'], nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        render_pkg['surf_normal'] = torch.nan_to_num(render_pkg['surf_normal'], nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        valid_mask = rend_alpha > 0.05
        if valid_pix_mask is not None:
            valid_mask = valid_mask & valid_pix_mask.unsqueeze(0)

        # Every downstream geometry consumer reads the unbiased depth, not
        # surf_depth: the design doc's one depth definition, so the prior, the
        # multi-view warp and the export TSDF all agree on where the surface is.
        # It is zero where no ray accumulated enough opacity to cross the
        # threshold -- alpha alone does not exclude those, so they get their own
        # mask rather than contributing a full-depth error.
        unbiased_depth = render_pkg["rend_depth_unbiased"]
        surface_mask = valid_mask & (unbiased_depth > 0)

        # §4.1 / §5.3 per-view evidence. Both read this iteration's render and
        # this iteration's surfel positions, so they are gathered here rather
        # than at the density-control step, where the surfels have moved and
        # the render is gone. Both are no_grad and cost one projection of the
        # cloud each -- about 1% of an iteration at half a million surfels.
        if opt.sad_densify and priors is not None and iteration < opt.densify_until_iter:
            lam = priors.get_wavelength(viewpoint_cam)
            if lam is not None:
                gaussians.accumulate_sad_stats(viewpoint_cam, lam, visibility_filter)
        if opt.freespace_carve and iteration > opt.freespace_from_iter:
            gaussians.accumulate_freespace_stats(
                viewpoint_cam, unbiased_depth.detach(), opt.freespace_margin)

        if opt.lambda_normal > 0.0 and iteration > opt.normal_from_iter:
            rend_normal = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            # These are unit normals, so the dot product cannot leave [-1, 1].
            # nan_to_num above maps +-inf to +-3.4e38 rather than to zero, and
            # squaring those overflows back to +-inf, whose sum across channels
            # is NaN. Clamp closes the inf path; the isfinite mask below closes
            # the NaN one. Neither alone is enough.
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0).clamp(-1.0, 1.0))[None]
            # .mean() of an empty selection is NaN, which backward() then spreads
            # into every geometry parameter permanently (colours survive, since
            # they take no gradient from here). A view can render with no pixel
            # above the alpha threshold, so the mask really does come up empty.
            normal_sel = valid_mask & torch.isfinite(normal_error)
            normal_loss = (opt.lambda_normal * normal_error[normal_sel].mean()
                           if normal_sel.any() else torch.tensor(0.0, device="cuda"))
        else:
            normal_loss = torch.tensor(0.0, device="cuda")

        if opt.lambda_dist > 0.0 and iteration > 3000:
            rend_dist = render_pkg["rend_dist"]
            dist_loss = (opt.lambda_dist * rend_dist[valid_mask].mean()
                         if valid_mask.any() else torch.tensor(0.0, device="cuda"))
        else:
            dist_loss = torch.tensor(0.0, device="cuda")

        # Depth and normal priors. On from iteration 0: the initialisation cloud
        # is already metrically correct, so the rendered normals are meaningful
        # immediately and there is nothing to warm up.
        depth_prior_term = torch.tensor(0.0, device="cuda")
        normal_prior_term = torch.tensor(0.0, device="cuda")
        if priors is not None:
            prior_depth, prior_conf = priors.get(viewpoint_cam)
            if prior_depth is not None:
                decay = prior_weight(1.0, iteration, opt.prior_decay_from_iter,
                                     opt.prior_decay_until_iter, opt.prior_weight_floor)
                pmask = surface_mask.float()
                if opt.lambda_depth_prior > 0.0:
                    depth_prior_term = (opt.lambda_depth_prior * decay) * depth_prior_loss(
                        unbiased_depth, prior_depth, prior_conf, pmask)
                if opt.lambda_normal_prior > 0.0:
                    prior_normal, prior_conf_n = priors.get_normal(viewpoint_cam)
                    if prior_normal is not None:
                        # Alpha-scaled the same way as surf_normal in the
                        # renderer, so the prior normal and the rendered
                        # normal are compared like for like.
                        pn = torch.nan_to_num(prior_normal, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
                        pn = pn * rend_alpha.detach()
                        normal_prior_term = (opt.lambda_normal_prior * decay) * normal_prior_loss(
                            render_pkg["rend_normal"], pn, prior_conf_n, pmask)

        if opt.lambda_depth_conv > 0.0 and iteration > opt.depth_conv_from_iter:
            depth_conv_term = opt.lambda_depth_conv * depth_convergence_loss(
                render_pkg["rend_depth_expected"], unbiased_depth, surface_mask)
        else:
            depth_conv_term = torch.tensor(0.0, device="cuda")

        # DN-Splatter edge-aware smoothness: penalise depth curvature, weighted
        # down wherever the image has an edge. The depth prior is only as smooth
        # as the depth model, and it decays; this is what keeps a wall flat
        # between the prior handing over and the multi-view term taking hold.
        if opt.lambda_depth_smooth > 0.0:
            depth_smooth_term = opt.lambda_depth_smooth * smooth_loss(
                unbiased_depth * surface_mask.float(), gt_image)
        else:
            depth_smooth_term = torch.tensor(0.0, device="cuda")

        if opt.lambda_multiview > 0.0 and iteration > opt.mv_from_iter:
            neighbors = pick_neighbors(viewpoint_cam, mv_order, mv_index, opt.mv_num_neighbors)
            if pose_optimizer is not None:
                # Each camera caches world_view_transform = f(pose_delta). The cached
                # tensor carries the autograd graph from whichever iteration last
                # refreshed it, so reusing a neighbour's stale matrix backprops through
                # a freed graph. Rebuild them here, as line 93 does for the current cam.
                for nb in neighbors:
                    nb.refresh_pose()
            multiview_loss = opt.lambda_multiview * multiview_photometric_loss(
                viewpoint_cam, neighbors, unbiased_depth,
                render_pkg["rend_normal"], rend_alpha,
                veto_weight=opt.mv_veto_weight,
                saturation_threshold=opt.saturation_threshold,
                saturation_percentile=opt.saturation_percentile,
                saturation_max_chroma_diff=opt.saturation_max_chroma_diff
            )
        else:
            multiview_loss = torch.tensor(0.0, device="cuda")

        if track_loss is not None:
            track_reg = opt.lambda_track * track_loss(viewpoint_cam)
        else:
            track_reg = torch.tensor(0.0, device="cuda")

        exposure_reg = torch.tensor(0.0, device="cuda")
        if exposure_optimizer is not None:
            exposure_reg = opt.lambda_exposure * (viewpoint_cam.exposure ** 2).sum()

        total_loss = (loss + dist_loss + normal_loss + multiview_loss + track_reg
                      + depth_prior_term + normal_prior_term + depth_conv_term
                      + depth_smooth_term
                      + exposure_reg)
        # The progress bar logs `loss` (photometric only), never total_loss, so a
        # NaN in any regulariser used to sail through invisibly for thousands of
        # iterations and only surface at the end as an all-NaN PLY. Fail loudly
        # on the first bad step instead.
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite total_loss at iteration {iteration}: photo={loss.item()} "
                f"dist={dist_loss.item()} normal={normal_loss.item()} "
                f"mv={multiview_loss.item()} track={track_reg.item()} "
                f"dprior={depth_prior_term.item()} nprior={normal_prior_term.item()} "
                f"dconv={depth_conv_term.item()} dsmooth={depth_smooth_term.item()}")
        total_loss.backward(retain_graph=_NAN_DEBUG)

        # A finite total_loss does not imply finite gradients: masking
        # normal_error with isfinite zeroes those entries in backward, and
        # 0 * inf upstream is NaN. The loss guard above cannot see that, which
        # is how a "clean" run still wrote a 63%-NaN PLY. Opt-in, it costs a
        # handful of reductions per step.
        if _NAN_DEBUG:
            _bad = [(_g["name"], _g["params"][0]) for _g in gaussians.optimizer.param_groups
                    if _g["params"][0].grad is not None
                    and not torch.isfinite(_g["params"][0].grad).all()]
            if _bad:
                # Re-run each term alone on the retained graph to find which one
                # poisons the parameter. Guessing the culprit from the loss values
                # is not possible -- they are all finite.
                _blame = {}
                for _tname, _t in (("photo", loss), ("normal", normal_loss),
                                   ("mv", multiview_loss), ("dist", dist_loss),
                                   ("track", track_reg), ("dprior", depth_prior_term),
                                   ("nprior", normal_prior_term), ("dconv", depth_conv_term)):
                    if not _t.requires_grad:
                        _blame[_tname] = "no-grad"
                        continue
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    _t.backward(retain_graph=True)
                    _blame[_tname] = {
                        _n: int((~torch.isfinite(_p.grad)).sum())
                        for _n, _p in ((_g["name"], _g["params"][0])
                                       for _g in gaussians.optimizer.param_groups)
                        if _p.grad is not None and not torch.isfinite(_p.grad).all()
                    } or "clean"
                # Dump the offending surfel indices so they can be matched
                # against scene geometry (opacity grad is one entry per surfel).
                for _g in gaussians.optimizer.param_groups:
                    if _g["name"] == "opacity" and _g["params"][0].grad is not None:
                        _idx = (~torch.isfinite(_g["params"][0].grad)).any(dim=1).nonzero().squeeze(-1)
                        # Positions, not indices: densify/prune renumbers the
                        # surfels every 100 iterations, so an index no longer
                        # refers to anything in the initial cloud.
                        np.save(os.path.join(os.path.dirname(__file__), "nan_surfels.npy"),
                                gaussians.get_xyz[_idx].detach().cpu().numpy())
                import utils.multiview_loss as _mv
                _mvblame = f" mv fwd={_mv.FWD_BLAME} mv grad={_mv.GRAD_BLAME}"
                raise RuntimeError(
                    f"non-finite GRADIENT at iteration {iteration}: "
                    + ", ".join(f"{_n}={int((~torch.isfinite(_p.grad)).sum())}/{_p.grad.numel()}"
                                for _n, _p in _bad)
                    + f"; total_loss={total_loss.item()} normal={normal_loss.item()} "
                    f"mv={multiview_loss.item()} dist={dist_loss.item()}"
                    f"; per-term blame: {_blame}" + _mvblame)

        iter_end.record()

        with torch.no_grad():
            if viewspace_point_tensor.grad is not None and visibility_filter.any():
                pos_grad_norm = torch.norm(viewspace_point_tensor.grad[visibility_filter], dim=-1).mean().item()
                ema_pos_grad_for_log = 0.1 * pos_grad_norm + 0.9 * ema_pos_grad_for_log

            comp_loss = loss.item() + opt.lambda_multiview * multiview_loss.item()
            if prev_comp_loss is not None:
                rolling_loss_derivative = 0.1 * (comp_loss - prev_comp_loss) + 0.9 * rolling_loss_derivative
            prev_comp_loss = comp_loss
            ema_composite_loss_for_log = 0.1 * comp_loss + 0.9 * ema_composite_loss_for_log

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_dprior_for_log = 0.4 * depth_prior_term.item() + 0.6 * ema_dprior_for_log
            ema_mv_for_log = 0.4 * multiview_loss.item() + 0.6 * ema_mv_for_log

            # §7 rows. L_conv is read unweighted (the table's threshold is on
            # the loss, not on lambda x loss), and the ZNCC row is recovered
            # from the same division -- the multi-view term is mean(1 - ZNCC),
            # so 1 - it is the mean correlation the doc wants to see above 0.45.
            if iteration >= 3000 and diagnostics["conv_at_3k"] is None and opt.lambda_depth_conv > 0:
                diagnostics["conv_at_3k"] = depth_conv_term.item() / opt.lambda_depth_conv
            if opt.lambda_multiview > 0 and multiview_loss.item() > 0:
                diagnostics["mean_zncc"] += 1.0 - multiview_loss.item() / opt.lambda_multiview
                diagnostics["zncc_samples"] += 1

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "dprior": f"{ema_dprior_for_log:.{5}f}",
                    "mv": f"{ema_mv_for_log:.{5}f}",
                    "|∇μ|": f"{ema_pos_grad_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/depth_prior_loss', ema_dprior_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_prior_loss', normal_prior_term.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/pos_grad_norm', ema_pos_grad_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/composite_loss', ema_composite_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/loss_slope', rolling_loss_derivative, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Density control, in two windows:
            #   growth  (< densify_until_iter): budgeted top-k densification onto
            #           a linear target curve, plus pruning
            #   cleanup (>= densify_until_iter): pruning only, so floaters that
            #           only become identifiable late still get removed
            # The old code ran both inside the growth window, which left the last
            # ~2/3 of training with no prune at all -- floaters accumulated
            # monotonically, which is exactly the observed symptom.
            if iteration < opt.iterations:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # Keep accruing even in the cleanup tail: the TIDI view-evidence
                # counters are what the tail's floater prune reads.
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, cam_id=viewpoint_cam.uid)

            growing = iteration < opt.densify_until_iter
            # High-frequency gradient shock damping on stage resumption
            in_stage_warmup = (stage_first_iter > 0) and (iteration < stage_first_iter + opt.densify_warmup_per_stage)
            allow_densify = growing and (not in_stage_warmup)
            interval = opt.densification_interval if growing else opt.cleanup_interval

            # Targeted Phase 2 opacity reset
            if opt.opacity_reset_iter > 0 and iteration == opt.opacity_reset_iter:
                print(f"\n[ITER {iteration}] Targeted Phase 2 opacity reset: clamping opacities to <= 0.01; "
                      f"culling suspended for {opt.opacity_recovery_iters} steps to allow recovery.")
                diagnostics["points_pre_reset"] = gaussians.get_xyz.shape[0]
                gaussians.reset_opacity()

            in_opacity_recovery = (opt.opacity_reset_iter > 0) and (
                opt.opacity_reset_iter <= iteration < opt.opacity_reset_iter + opt.opacity_recovery_iters
            )
            effective_opacity_cull = 0.0 if in_opacity_recovery else opt.opacity_cull

            is_recovery_cull_step = (opt.opacity_reset_iter > 0) and (
                iteration == opt.opacity_reset_iter + opt.opacity_recovery_iters
            )

            if (iteration > opt.densify_from_iter and iteration % interval == 0) or is_recovery_cull_step:
                # Screen-space/world-space size prune on its own gate. It used to
                # ride on `iteration > opacity_reset_interval`, so setting the reset
                # interval to "never" (999999) silently disabled the size prune too --
                # two unrelated guards accidentally coupled.
                # §5.2: 0.15 * min(H, W) of the frame actually being rendered,
                # so the criterion means the same thing in every resolution
                # phase. The old absolute 20 px culled legitimate close-range
                # surfels at native resolution -- a 3.5 cm surfel half a metre
                # from the camera projects to ~70 px there.
                size_threshold = (opt.max_screen_frac * min(int(viewpoint_cam.image_height),
                                                            int(viewpoint_cam.image_width))
                                  if iteration > opt.size_prune_from_iter else None)
                target_count = int(growth_target(iteration, initial_points, opt.max_gaussians,
                                                 opt.densify_from_iter, opt.densify_until_iter)) if allow_densify else None
                n_before = gaussians.get_xyz.shape[0]
                gaussians.densify_and_prune(opt.densify_grad_threshold, effective_opacity_cull, scene.cameras_extent,
                                            size_threshold, opt.max_gaussians,
                                            allow_densification=allow_densify,
                                            view_evidence_cull=(iteration >= opt.view_evidence_from_iter),
                                            frustum_min=opt.view_evidence_frustum_min,
                                            min_obs=opt.view_evidence_min_obs,
                                            target_count=target_count,
                                            max_world_size=opt.max_world_size,
                                            sad=({"tau_split": opt.sad_tau_split,
                                                  "min_views": opt.sad_min_views}
                                                 if opt.sad_densify else None),
                                            freespace=({"min_free_views": opt.freespace_min_views,
                                                        "free_ratio": opt.freespace_ratio}
                                                       if opt.freespace_carve
                                                       and iteration > opt.freespace_from_iter
                                                       else None))
                n_after = gaussians.get_xyz.shape[0]
                sad_added = getattr(gaussians, "last_sad_added", 0)
                fs_pruned = getattr(gaussians, "last_freespace_pruned", 0)
                if sad_added or fs_pruned:
                    diagnostics["sad_added"] += sad_added
                    diagnostics["freespace_pruned"] += fs_pruned
                    if tb_writer is not None:
                        tb_writer.add_scalar('density/sad_added', sad_added, iteration)
                        tb_writer.add_scalar('density/freespace_pruned', fs_pruned, iteration)
                delta_n = n_after - n_before
                delta_pct = abs(delta_n) / max(n_before, 1) * 100.0
                if tb_writer is not None:
                    tb_writer.add_scalar('density/delta_n_pct', delta_pct, iteration)
                    tb_writer.add_scalar('density/total_points', n_after, iteration)
                if is_recovery_cull_step:
                    diagnostics["points_post_reset"] = n_after
                    print(f"\n[ITER {iteration}] Post-reset opacity cull complete: {n_before:,} -> {n_after:,} points "
                          f"({delta_n:+,} / -{delta_pct:.2f}%)")

            # Fallback interval-based reset when targeted reset is not configured
            if opt.opacity_reset_iter <= 0:
                within_reset_window = opt.opacity_reset_until_iter < 0 or iteration <= opt.opacity_reset_until_iter
                if growing and within_reset_window and (
                        (iteration % opt.opacity_reset_interval == 0 and iteration > opt.densify_from_iter)
                        or (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                for param_group in gaussians.optimizer.param_groups:
                    p = param_group['params'][0]
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if opt.sparse_adam:
                    gaussians.optimizer.step(visibility_filter, gaussians.get_xyz.shape[0])
                else:
                    gaussians.optimizer.step()
                if opt.max_world_size is not None and opt.max_world_size > 0:
                    gaussians._scaling.data.clamp_(max=math.log(opt.max_world_size))
                # Densify/prune rebuilds these tensors, so a NaN can also enter
                # here rather than through the gradient.
                if _NAN_DEBUG:
                    for _g in gaussians.optimizer.param_groups:
                        _p = _g["params"][0]
                        if not torch.isfinite(_p).all():
                            raise RuntimeError(
                                f"non-finite PARAM after step at iteration {iteration} "
                                f"in {_g['name']}: {int((~torch.isfinite(_p)).sum())} "
                                f"of {_p.numel()}")
                if pose_optimizer is not None:
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
                if exposure_optimizer is not None:
                    exposure_optimizer.step()
                    exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

    report_verification_metrics(diagnostics, gaussians.get_xyz.shape[0], dataset.model_path)
    print("\nTraining complete.")


def report_verification_metrics(diagnostics, final_points, model_path):
    """Print the §7 verification table for this run and persist it as JSON.

    Reported, never auto-corrected. Every corrective action in the doc's §7 is a
    hyperparameter change whose effect has to be judged on the next full run;
    applying one silently mid-flight would mean no two runs were comparable and
    the number that triggered it would never be seen again. The point of the
    table is that a bad run says so out loud instead of quietly shipping a
    135 MB PLY that nobody looked at -- which is what every run in
    project_history.md marked "not measured" did.
    """
    rows = []

    def row(name, value, lo, hi, advice, fmt="{:.4f}"):
        if value is None:
            rows.append((name, "not measured", "-", ""))
            return
        ok = (lo is None or value >= lo) and (hi is None or value <= hi)
        band = (f"{fmt.format(lo) if lo is not None else '-'}"
                f" .. {fmt.format(hi) if hi is not None else '-'}")
        rows.append((name, fmt.format(value), band, "" if ok else advice))

    zncc = (diagnostics["mean_zncc"] / diagnostics["zncc_samples"]
            if diagnostics["zncc_samples"] else None)
    survival = (diagnostics["points_post_reset"] / diagnostics["points_pre_reset"]
                if diagnostics["points_pre_reset"] else None)

    row("L_conv at iter 3,000", diagnostics["conv_at_3k"], None, 0.015,
        "surfels are stacking; raise lambda_depth_conv (0.05 -> 0.12)")
    row("mean patch ZNCC", zncc, 0.45, None,
        "near 0 means the homography sign or a rotation transpose is wrong (§7)")
    row("post-reset surfel survival", survival, 0.90, None,
        "recovery window too short; extend opacity_recovery_iters (800 -> 1,200)")
    row("final surfel count", float(final_points), None, 4.5e6,
        "over-splitting in textured regions; raise sad_tau_split (0.75 -> 0.85)",
        fmt="{:,.0f}")

    print("\n--- §7 verification metrics ---")
    width = max(len(r[0]) for r in rows)
    for name, value, band, advice in rows:
        flag = "  <-- " + advice if advice else ""
        print(f"  {name:<{width}}  {value:>12}   healthy: {band}{flag}")
    print(f"  {'SAD surfels added':<{width}}  {diagnostics['sad_added']:>12,}")
    print(f"  {'free-space carved':<{width}}  {diagnostics['freespace_pruned']:>12,}")

    try:
        import json
        payload = dict(diagnostics, final_points=final_points,
                       mean_zncc=zncc, post_reset_survival=survival)
        with open(os.path.join(model_path, "verification_metrics.json"), "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"[warning] could not write verification_metrics.json: {exc}")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
