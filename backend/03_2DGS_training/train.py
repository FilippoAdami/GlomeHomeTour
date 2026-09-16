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
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_saturation_mask, masked_l1_loss, masked_ssim
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

# Set GLOME_NAN_DEBUG=1 to trap the first non-finite gradient or parameter.
# Off by default: it adds a few reductions per step and aborts the run.
_NAN_DEBUG = os.environ.get("GLOME_NAN_DEBUG") == "1"
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # The voxel grid in step 4 already set the density; densification gets 1.5x
    # that headroom, hard-capped at what 16 GB of VRAM holds.
    if opt.max_gaussians is not None and opt.max_gaussians < 0:
        opt.max_gaussians = min(int(1.5 * gaussians.get_xyz.shape[0]), 4_500_000)
        print(f"max_gaussians: {opt.max_gaussians} (from {gaussians.get_xyz.shape[0]} init points)")

    # Both extras below index cameras by capture order: Scene shuffles the
    # training list, and the COLMAP loader hands every frame the same uid.
    train_cams = scene.getTrainCameras()
    mv_order, mv_index = build_capture_order(train_cams) if opt.lambda_multiview > 0.0 else ([], {})

    pose_optimizer, track_loss = None, None
    if opt.refine_poses_during_training:
        for cam in train_cams:
            cam.enable_pose_refinement()
        pose_optimizer = torch.optim.Adam([cam.pose_delta for cam in train_cams],
                                          lr=opt.pose_lr, eps=1e-15)
        track_loss = TrackLoss(dataset.source_path, train_cams)
        print(f"Pose refinement ON: {len(train_cams)} cameras, "
              f"track supervision on {track_loss.num_views} views")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)  # trusted, our own checkpoint; torch>=2.6 defaults weights_only=True
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

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
        valid_pix_mask = None
        if opt.saturation_threshold > 0:
            sat_mask = compute_saturation_mask(
                gt_image,
                min_threshold=opt.saturation_threshold,
                percentile=opt.saturation_percentile,
                max_chroma_diff=opt.saturation_max_chroma_diff
            )
            valid_pix_mask = ~sat_mask
            Ll1 = masked_l1_loss(image, gt_image, mask=valid_pix_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - masked_ssim(image, gt_image, mask=valid_pix_mask))
        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        rend_alpha = render_pkg['rend_alpha']
        # ponytail: HIP rasterizer occasionally emits NaN alpha on degenerate
        # surfels; clamp here rather than upstream, tighten if a real cause is found.
        rend_alpha = torch.nan_to_num(rend_alpha, nan=0.0)
        # The same degenerate surfels poison the normal buffers. Clean them here
        # rather than at each use: multiview_loss reads rend_normal too, and a
        # NaN there is just as fatal as one in normal_loss.
        render_pkg['rend_normal'] = torch.nan_to_num(render_pkg['rend_normal'], nan=0.0)
        render_pkg['surf_normal'] = torch.nan_to_num(render_pkg['surf_normal'], nan=0.0)
        valid_mask = rend_alpha > 0.05
        if valid_pix_mask is not None:
            valid_mask = valid_mask & valid_pix_mask.unsqueeze(0)

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
                viewpoint_cam, neighbors, render_pkg["surf_depth"],
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

        total_loss = loss + dist_loss + normal_loss + multiview_loss + track_reg
        # The progress bar logs `loss` (photometric only), never total_loss, so a
        # NaN in any regulariser used to sail through invisibly for thousands of
        # iterations and only surface at the end as an all-NaN PLY. Fail loudly
        # on the first bad step instead.
        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"non-finite total_loss at iteration {iteration}: photo={loss.item()} "
                f"dist={dist_loss.item()} normal={normal_loss.item()} "
                f"mv={multiview_loss.item()} track={track_reg.item()}")
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
                                   ("track", track_reg)):
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
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification: standard grad/opacity schedule, with a hard cap on
            # total surfel count (Taming3DGS-style controlled growth), enforced
            # centrally in GaussianModel.densify_and_prune.
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, cam_id=viewpoint_cam.uid)

                if iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    allow_densification = (iteration > opt.densify_from_iter)
                    view_evidence_cull = (iteration >= opt.view_evidence_from_iter)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                                                size_threshold, opt.max_gaussians,
                                                allow_densification=allow_densification,
                                                view_evidence_cull=view_evidence_cull,
                                                frustum_min=opt.view_evidence_frustum_min,
                                                min_obs=opt.view_evidence_min_obs)

                # A reset exists to give the densifier a clean slate. Without growth to
                # refill, it just drops every surfel to 0.01 -- under opacity_cull (0.05)
                # -- and the next prune 100 iterations later deletes whatever gradient
                # did not rescue. That ratchet took Phase 1 from 2.76M to 135k surfels.
                # Gate it on the same window that allows densification.
                if (iteration % opt.opacity_reset_interval == 0 and iteration > opt.densify_from_iter) \
                        or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
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

    print("\nTraining complete.")

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
