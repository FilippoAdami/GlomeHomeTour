#!/usr/bin/env python3
"""Step 5 -- progressive multi-resolution 2DGS training.

In:  ``<workspace>/`` (images, ``sparse/0/``, ``02_depth_estimation/depth/points3D_depth.ply``).
Out: ``<workspace>/03_2DGS_training/2dgs/point_cloud/iteration_<N>/point_cloud.ply``.

Four stages at 1/8 -> 1/4 -> 1/2 -> full resolution, chained through
``--start_checkpoint``. Coarse stages settle global structure cheaply, where an
iteration costs 1/64 of a full-resolution one; only the last stage pays full
price. Iteration counts are *cumulative* because ``train.py`` resumes at the
checkpoint's iteration number and runs to ``--iterations``.

Resolution is handed to ``train.py`` as ``-r 8/4/2/1``, which ``loadCam``
already applies when loading each camera. No ``images_2/4/8`` folders are
generated: they would be ~4 GB of duplicated JPEG for no gain.

The depth cloud becomes ``sparse/0/points3D.ply`` because that is the file
``readColmapSceneInfo`` initialises from; COLMAP's own sparse cloud is kept
beside it as ``points3D_colmap.ply``. That swap is the entire mechanism by
which step 4's work reaches training -- no loader change.

    python 03_2DGS_training/step_train.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap, subprocess_env

bootstrap()

from Utilities.pipeline_step import StepContext, is_done

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
TRAIN_SCRIPT = _backend_dir / "03_2DGS_training" / "train.py"
STAGE_DIRNAME = "03_2DGS_training"
DEPTH_STAGE_DIRNAME = "02_depth_estimation"
MODEL_DIRNAME = f"{STAGE_DIRNAME}/2dgs"

# (resolution divisor, cumulative iteration target). The stages deliberately avoid multiples of OPACITY_RESET_INTERVAL so a reset
# never lands on the iteration a stage stops and checkpoints at.
#
# Only one coarse stage survives. The r=8/r=4 stages came from the sparse-SfM
# lineage, where they let a few thousand blobs find global structure cheaply.
# Step 4 already delivers metrically-correct geometry at 1.5 cm spacing, so
# there is no global structure left to find -- and at r=8 a 1.5 cm surfel
# 3-Phase Multi-Scale Optimization Framework:
# Phase 1: Coarse Geometry & Manifold Locking (420p short-side / ~756p long-side, matching DA3 inference res).
#          Requires ~25 epochs/view (0 to 5,000 iters for ~200 cams). Active densification, full depth/normal priors.
# Phase 2: Structural Refinement & Density Settlement (750p short-side / ~1333p long-side, 1.4x scale).
#          Requires ~15 epochs/view (5,000 to 8,000 iters for ~200 cams). Priors decay to floor.
#          Targeted opacity reset at ~33% (step 6,000), 800-step recovery window (recovery to 6,800), cull at 6,800.
# Phase 3: High-Res Polish (Native full resolution, 1080p / 4K).
#          Requires ~10-12.5 epochs/view (8,000 to 10,500 iters for ~200 cams).
#          DENSIFICATION STRICTLY DISABLED; position LR decayed; SH color and specularity polish.
STAGES = ((420, 5_000), (750, 8_000), (1, 10_500))
TOTAL_ITERATIONS = STAGES[-1][1]

DENSIFY_WARMUP_ITERS = 200
# Growth ends strictly at the end of Phase 2 (step 8,000 for 200 cams).
# In Phase 3 (native full resolution), densification is completely disabled:
# high-frequency pixel residuals must NOT create new primitives in free space.
DENSIFY_UNTIL_ITER = 8_000
CLEANUP_INTERVAL = 250       # prune-only cadence in the tail
SIZE_PRUNE_FROM_ITER = 1_500  # oversized-surfel prune, on its own gate
DENSIFY_GRAD_THRESHOLD = 0.0001
DENSIFICATION_INTERVAL = 50
GROWTH_FACTOR = 1.6
DENSIFY_WARMUP_PER_STAGE = 300  # Damp gradient shock after stepping resolution

# Targeted opacity reset in Phase 2 with recovery window
OPACITY_RESET_INTERVAL = 999_999  # Interval-based reset disabled in favor of targeted reset
OPACITY_RESET_UNTIL_ITER = 0
OPACITY_RESET_ITER = 6_000
OPACITY_RECOVERY_ITERS = 800

NORMAL_FROM_ITER = 0

# SAD-2DGS structure-aware densification (2dgs_combined_pipeline.md §4) and
# asymmetric free-space carving (§5.3). Both need per-view evidence, so both are
# cheap per iteration and only act at the density-control steps.
SAD_TAU_SPLIT = 0.75
SAD_MIN_VIEWS = 4
FREESPACE_MARGIN_M = 0.04
FREESPACE_MIN_VIEWS = 3
FREESPACE_RATIO = 0.6
# Both gated on the rendered unbiased depth being trustworthy. Before that the
# surface is still settling and "in front of the surface" means very little.
FREESPACE_FROM_FRAC = 0.15          # of the total schedule
# §3: appearance-only polish at the end of Phase 3.
FREEZE_GEOMETRY_LAST = 1_000

LAMBDA_DEPTH_PRIOR = 0.5
LAMBDA_NORMAL_PRIOR = 0.05
# Hand over to photometric and multi-view terms across Phase 2:
PRIOR_DECAY_FROM = 5_000
PRIOR_DECAY_UNTIL = 8_000
PRIOR_WEIGHT_FLOOR = 0.1

LAMBDA_DEPTH_CONV = 0.1
DEPTH_CONV_FROM = 500

LAMBDA_DEPTH_SMOOTH = 0.05

LAMBDA_MULTIVIEW = 0.3
MV_FROM_ITER = 500

EXPOSURE_LR = 0.001
LAMBDA_EXPOSURE = 0.01

DEFAULT_POSE_LR = 0.0001
DEFAULT_LAMBDA_TRACK = 0.1


def count_cameras(workspace: Path) -> int:
    """Return the number of camera views in the workspace."""
    images_txt = workspace / "sparse" / "0" / "images.txt"
    if images_txt.exists():
        with open(images_txt, "r", encoding="utf-8") as f:
            lines = [l for l in f if not l.startswith("#")]
        if len(lines) >= 2:
            return len(lines) // 2
    transforms_json = workspace / "transforms.json"
    if transforms_json.exists():
        import json
        with open(transforms_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            frames = data.get("frames", [])
            if frames:
                return len(frames)
    images_dir = workspace / "images"
    if images_dir.exists():
        imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if imgs:
            return len(imgs)
    return 200


def compute_epoch_schedule(num_cameras: int, use_depth: bool = True) -> tuple[tuple[tuple[int, int], ...], dict[str, int]]:
    """Compute 3-phase multi-scale schedule scaled to camera view count.

    When use_depth=True (default, dense metric prior from Step 4):
      Dense metric initialization already locks room manifold, walls, and ceiling.
      Phase 1: Coarse Geometry & Manifold Locking (420p, matching DA3 depth prior, ~8 epochs/view).
      Phase 2: Structural Refinement & Density Settlement (~750p, 1.4x scale, ~6 epochs/view).
      Phase 3: High-Res Polish (Native full resolution, 1080p / 4K, ~4 epochs/view).
               DENSIFICATION STRICTLY DISABLED; opacity reset DISABLED.
      Converges cleanly in ~6k-7k iterations (~5 min on AMD Radeon RX 9060 XT).

    When use_depth=False (sparse SfM cloud fallback):
      Requires longer exploratory growth (~25/15/10 epochs) with targeted opacity reset.
    """
    n = max(num_cameras, 20)
    if use_depth:
        if 180 <= n <= 220:
            p1 = 3_000
            p2 = 5_000
            p3 = 6_500
        else:
            e1 = 8.0
            e2 = 6.0
            e3 = 4.0
            p1 = round(e1 * n)
            p2 = p1 + round(e2 * n)
            p3 = p2 + round(e3 * n)
        reset_iter = -1
        recovery_iters = 0
    else:
        if 180 <= n <= 220:
            p1 = 5_000
            p2 = 8_000
            p3 = 10_500
            reset_iter = 6_000
            recovery_iters = 800
        else:
            e1 = 25
            e2 = 15
            e3 = 10 if n > 250 else 12.5
            p1 = round(e1 * n)
            p2 = p1 + round(e2 * n)
            p3 = p2 + round(e3 * n)
            p2_len = p2 - p1
            reset_iter = p1 + round(0.33 * p2_len)
            recovery_iters = max(400, round(4 * n))

    stages = ((420, p1), (750, p2), (1, p3))
    schedule_params = {
        "total_iterations": p3,
        "densify_until_iter": p2,
        "prior_decay_from": p1,
        "prior_decay_until": p2,
        "opacity_reset_iter": reset_iter,
        "opacity_recovery_iters": recovery_iters,
        "densify_warmup_per_stage": max(150, min(round(1.5 * n), 400)),
    }
    return stages, schedule_params


def ply_point_count(path: Path) -> int:
    """`element vertex N` out of a PLY header, without loading the cloud."""
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
            if line.startswith(b"end_header"):
                break
    raise ValueError(f"no vertex count in {path}")


def densification_flags(target_iter: int, total: int = TOTAL_ITERATIONS,
                        budget: int | None = None,
                        densify_until_iter: int = DENSIFY_UNTIL_ITER,
                        opacity_reset_iter: int = OPACITY_RESET_ITER,
                        opacity_recovery_iters: int = OPACITY_RECOVERY_ITERS,
                        densify_warmup_per_stage: int = DENSIFY_WARMUP_PER_STAGE) -> list[str]:
    """Densification/pruning flags for the stage ending at ``target_iter``."""
    return ["--densify_from_iter", str(DENSIFY_WARMUP_ITERS),
            "--densify_until_iter", str(densify_until_iter),
            "--densify_grad_threshold", str(DENSIFY_GRAD_THRESHOLD),
            "--densification_interval", str(DENSIFICATION_INTERVAL),
            "--cleanup_interval", str(CLEANUP_INTERVAL),
            "--size_prune_from_iter", str(SIZE_PRUNE_FROM_ITER),
            "--growth_factor", str(GROWTH_FACTOR)] + (
            ["--max_gaussians", str(budget)] if budget else []) + [
            "--opacity_reset_interval", str(OPACITY_RESET_INTERVAL),
            "--opacity_reset_until_iter", str(OPACITY_RESET_UNTIL_ITER),
            "--opacity_reset_iter", str(opacity_reset_iter),
            "--opacity_recovery_iters", str(opacity_recovery_iters),
            "--densify_warmup_per_stage", str(densify_warmup_per_stage),
            "--normal_from_iter", str(NORMAL_FROM_ITER),
            "--sad_densify",
            "--sad_tau_split", str(SAD_TAU_SPLIT),
            "--sad_min_views", str(SAD_MIN_VIEWS),
            "--freespace_carve",
            "--freespace_margin", str(FREESPACE_MARGIN_M),
            "--freespace_min_views", str(FREESPACE_MIN_VIEWS),
            "--freespace_ratio", str(FREESPACE_RATIO),
            "--freespace_from_iter", str(int(FREESPACE_FROM_FRAC * total))] + (
            # Only the final phase freezes geometry; an earlier stage would
            # freeze it before the schedule is anywhere near done.
            ["--freeze_geometry_last", str(FREEZE_GEOMETRY_LAST)]
            if target_iter == total else [])


def prior_flags(prior_decay_from: int = PRIOR_DECAY_FROM,
                prior_decay_until: int = PRIOR_DECAY_UNTIL,
                mv_from_iter: int = MV_FROM_ITER,
                lambda_mv: float = LAMBDA_MULTIVIEW) -> list[str]:
    """Depth/normal prior, depth-convergence and exposure flags (same for every stage)."""
    return ["--lambda_depth_prior", str(LAMBDA_DEPTH_PRIOR),
            "--lambda_normal_prior", str(LAMBDA_NORMAL_PRIOR),
            "--prior_decay_from_iter", str(prior_decay_from),
            "--prior_decay_until_iter", str(prior_decay_until),
            "--prior_weight_floor", str(PRIOR_WEIGHT_FLOOR),
            "--lambda_depth_conv", str(LAMBDA_DEPTH_CONV),
            "--depth_conv_from_iter", str(DEPTH_CONV_FROM),
            "--lambda_depth_smooth", str(LAMBDA_DEPTH_SMOOTH),
            "--lambda_multiview", str(lambda_mv),
            "--mv_from_iter", str(mv_from_iter),
            "--optimize_exposure",
            "--exposure_lr", str(EXPOSURE_LR),
            "--lambda_exposure", str(LAMBDA_EXPOSURE)]


def build_depth_priors(workspace: Path, ctx: StepContext, force: bool = False) -> bool:
    """Build the per-pixel prior cache from stage 3's depth maps. Read-only on stage 3.

    Runs as a subprocess for the same reason the training stages do: it holds a
    few hundred depth maps on the GPU and we want that memory back before
    training allocates its surfels.
    """
    prior_dir = workspace / STAGE_DIRNAME / "depth_priors"
    cmd = [sys.executable, str(_backend_dir / "03_2DGS_training" / "utils" / "depth_prior.py"),
           "-s", str(workspace), "-o", str(prior_dir)] + (["--force"] if force else [])
    ctx.note(f"$ {' '.join(cmd)}")
    with ctx.timer("build_depth_priors"):
        proc = subprocess.run(cmd, cwd=str(TRAIN_SCRIPT.parent), env=subprocess_env())
    if proc.returncode != 0:
        # Not fatal: without the cache train.py warns and trains on the
        # initialisation cloud alone, which is what every run before the priors
        # existed did. Losing the walls is worse than losing the whole run only
        # if it happens silently, so it is recorded as a metric.
        ctx.note(f"[warning] prior cache build failed (exit {proc.returncode}); "
                 "training will fall back to cloud-only geometry")
        ctx.metric("depth_priors", "failed")
        return False
    ctx.metric("depth_priors", len(list(prior_dir.glob("*.npz"))))
    return True


def install_depth_cloud(
    workspace: Path,
    ctx: StepContext,
    use_depth: bool = True,
    voxel_downsample_m: float | None = None,
) -> None:
    sparse_dir = workspace / "sparse" / "0"
    target = sparse_dir / "points3D.ply"
    depth_ply = workspace / DEPTH_STAGE_DIRNAME / "depth" / "points3D_depth.ply"
    backup = sparse_dir / "points3D_colmap.ply"

    if not use_depth:
        ctx.note("Using the COLMAP sparse cloud (--no-depth-cloud)")
        return
    if not depth_ply.exists():
        raise FileNotFoundError(f"{depth_ply} missing -- run step 4 first")

    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)  # keep COLMAP's, it is the fallback init

    if voxel_downsample_m is not None and voxel_downsample_m > 0:
        from initialization import SurfelCloud
        cloud = SurfelCloud.from_ply(depth_ply, voxel_downsample_m=voxel_downsample_m)
        cloud.to_ply(target)
        ctx.note(f"Applied final voxel downsample ({len(cloud):,} surfels at {voxel_downsample_m*100:.1f}cm) before 2DGS training -> sparse/0/points3D.ply")
    else:
        shutil.copy2(depth_ply, target)
        ctx.note(f"Installed depth cloud {depth_ply.name} -> sparse/0/points3D.ply")

    ctx.metric("init_cloud_mb", round(target.stat().st_size / 1e6, 1))
    ctx.note(f"Init cloud: {depth_ply.name} -> sparse/0/points3D.ply "
             f"({target.stat().st_size / 1e6:.1f} MB), COLMAP's kept as {backup.name}")


def train(workspace: Path, ctx: StepContext, use_depth: bool = True,
          refine_poses_stage4: bool = True, pose_lr: float = DEFAULT_POSE_LR,
          lambda_track: float = DEFAULT_LAMBDA_TRACK,
          voxel_downsample_m: float | None = None,
          stages: tuple = STAGES, schedule_params: dict | None = None,
          checkpoint_interval: int = 500,
          stop_after_phase: int | None = None,
          use_priors: bool = True) -> None:
    model_dir = workspace / MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    install_depth_cloud(workspace, ctx, use_depth, voxel_downsample_m=voxel_downsample_m)
    if use_priors:
        build_depth_priors(workspace, ctx)

    total = stages[-1][1]
    if schedule_params is None:
        schedule_params = {
            "total_iterations": total,
            "densify_until_iter": DENSIFY_UNTIL_ITER,
            "prior_decay_from": PRIOR_DECAY_FROM,
            "prior_decay_until": PRIOR_DECAY_UNTIL,
            "opacity_reset_iter": OPACITY_RESET_ITER,
            "opacity_recovery_iters": OPACITY_RECOVERY_ITERS,
            "densify_warmup_per_stage": DENSIFY_WARMUP_PER_STAGE,
        }

    # Budget fixed once, from the initialisation cloud: each stage is a fresh
    # process, so deriving it per stage would compound (1.6x of an already-grown
    # count) and the cap would drift upward every restart.
    budget = min(int(GROWTH_FACTOR * ply_point_count(workspace / "sparse" / "0" / "points3D.ply")),
                 4_500_000)
    ctx.metric("surfel_budget", budget)
    ctx.note(f"Surfel budget: {budget:,} ({GROWTH_FACTOR}x init), growth ends at iter {schedule_params['densify_until_iter']}")
    ctx.metric("stages", [{"resolution": r, "until_iter": n} for r, n in stages])
    checkpoint: Path | None = None
    stage_start = 0

    for phase_idx, (resolution, target_iter) in enumerate(stages, start=1):
        stage_checkpoint = model_dir / f"chkpnt{target_iter}.pth"
        if stage_checkpoint.exists():
            ctx.note(f"Phase {phase_idx} (1/{resolution} to iteration {target_iter}) already completed ({stage_checkpoint.name} found), skipping.")
            checkpoint = stage_checkpoint
            stage_start = target_iter + 1
            if stop_after_phase is not None and phase_idx >= stop_after_phase:
                ctx.note(f"Stopping after Phase {phase_idx} as requested (--stop-after-phase {stop_after_phase}).")
                return
            continue

        if checkpoint_interval > 0:
            stage_chkpts = sorted(list(set(
                [i for i in range(checkpoint_interval, target_iter + 1, checkpoint_interval) if i >= stage_start]
                + [target_iter]
            )))
        else:
            stage_chkpts = [target_iter]

        cmd = [sys.executable, str(TRAIN_SCRIPT),
               "-s", str(workspace),
               "-m", str(model_dir),
               "-r", str(resolution),
               "--iterations", str(target_iter),
               "--position_lr_max_steps", str(total),
               "--test_iterations", str(target_iter),
               "--checkpoint_iterations"] + [str(i) for i in stage_chkpts] + [
               "--save_iterations"] + [str(i) for i in stage_chkpts] + [
               "--quiet"] + densification_flags(
                   target_iter, total, budget,
                   densify_until_iter=schedule_params["densify_until_iter"],
                   opacity_reset_iter=schedule_params["opacity_reset_iter"],
                   opacity_recovery_iters=schedule_params["opacity_recovery_iters"],
                   densify_warmup_per_stage=schedule_params["densify_warmup_per_stage"])
        if use_priors:
            cmd += prior_flags(
                prior_decay_from=schedule_params["prior_decay_from"],
                prior_decay_until=schedule_params["prior_decay_until"])

        # Enable TrackGS pose refinement ONLY in Stage 3 / native full resolution (1080p / 4K)
        if resolution == 1 and refine_poses_stage4:
            cmd += [
                "--refine_poses_during_training",
                "--pose_lr", str(pose_lr),
                "--lambda_track", str(lambda_track),
            ]

        if checkpoint is not None:
            cmd += ["--start_checkpoint", str(checkpoint)]

        ctx.note(f"--- phase {phase_idx} (1/{resolution}) to iteration {target_iter} ---")
        ctx.note(f"$ {' '.join(cmd)}")
        with ctx.timer(f"train_r{resolution}"):
            proc = subprocess.run(cmd, cwd=str(TRAIN_SCRIPT.parent), env=subprocess_env())
        if proc.returncode != 0:
            raise RuntimeError(
                f"train.py failed at phase {phase_idx} (1/{resolution}) (exit {proc.returncode}). "
                f"The previous stage's checkpoint is still in {model_dir}, so this "
                "stage can be retried without redoing the earlier ones.")

        checkpoint = model_dir / f"chkpnt{target_iter}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"train.py exited 0 but wrote no {checkpoint.name}; cannot chain the next stage")
        ctx.metric(f"checkpoint_r{resolution}_mb", round(checkpoint.stat().st_size / 1e6, 1))
        stage_start = target_iter + 1

        if stop_after_phase is not None and phase_idx >= stop_after_phase:
            ctx.note(f"Stopping after Phase {phase_idx} as requested (--stop-after-phase {stop_after_phase}).")
            return

    final_ply = model_dir / "point_cloud" / f"iteration_{total}" / "point_cloud.ply"
    if not final_ply.exists():
        raise FileNotFoundError(f"Training finished but {final_ply} is missing")
    ctx.metric("final_ply", str(final_ply))
    ctx.metric("final_ply_mb", round(final_ply.stat().st_size / 1e6, 1))
    ctx.note(f"Trained scene: {final_ply} ({final_ply.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--no-depth-cloud", action="store_true",
                        help="Initialise from COLMAP's sparse cloud instead of step 4's")
    parser.add_argument("--no-depth-priors", action="store_true",
                        help="Skip the per-pixel depth/normal prior losses (cloud-only geometry)")
    parser.add_argument("--no-pose-refine", action="store_true",
                        help="Disable TrackGS camera pose refinement in Stage 4 (1080p)")
    parser.add_argument("--pose-lr", type=float, default=DEFAULT_POSE_LR,
                        help=f"Pose learning rate in Stage 4 (default: {DEFAULT_POSE_LR})")
    parser.add_argument("--lambda-track", type=float, default=DEFAULT_LAMBDA_TRACK,
                        help=f"Track loss weight for pose refinement in Stage 4 (default: {DEFAULT_LAMBDA_TRACK})")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Test run: collapse to a single stage of this many iterations")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Test run: -r value for the single stage (1280 = 720p, since "
                             "train.py reads a non-{1,2,4,8} -r as a target width)")
    parser.add_argument("--calibrated-200", action="store_true",
                        help="Force the static calibrated schedule (5k, 8k, 10.5k) for 200 keyframes")
    parser.add_argument("--checkpoint-interval", type=int, default=500,
                        help="Save checkpoints and point clouds every N iterations (default: 500, 0 to disable)")
    parser.add_argument("--stop-after-phase", type=int, default=None, choices=[1, 2, 3],
                        help="Stop training after completing the specified phase (1, 2, or 3)")
    parser.add_argument("--voxel-downsample-m", type=float, default=None,
                        help="Optional final voxel downsample grid size (in meters) to apply to depth cloud before training")
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    # Must be absolute: train() runs train.py with cwd=03_2DGS_training/, so a
    # relative workspace would resolve against that directory instead of here.
    workspace = Path(args.workspace).resolve()

    if args.iterations is not None:
        stages = ((args.resolution if args.resolution is not None else 1, args.iterations),)
        schedule_params = {
            "total_iterations": args.iterations,
            "densify_until_iter": int(0.75 * args.iterations),
            "prior_decay_from": int(0.5 * args.iterations),
            "prior_decay_until": int(0.75 * args.iterations),
            "opacity_reset_iter": -1,
            "opacity_recovery_iters": 0,
            "densify_warmup_per_stage": 0,
        }
    elif args.calibrated_200:
        stages, schedule_params = compute_epoch_schedule(200, use_depth=not args.no_depth_cloud)
    else:
        num_cams = count_cameras(workspace)
        stages, schedule_params = compute_epoch_schedule(num_cams, use_depth=not args.no_depth_cloud)
        print(f"[train] Detected {num_cams} cameras -> dynamic epoch schedule: {stages}")

    if args.stop_after_phase is None or args.stop_after_phase >= len(stages):
        final_ply = (workspace / MODEL_DIRNAME / "point_cloud"
                     / f"iteration_{stages[-1][1]}" / "point_cloud.ply")
        if not args.force and is_done(workspace, "train", [final_ply]):
            print("[train] already done, skipping (use --force to re-run)")
            return 0

    if args.force:
        # Checkpoints are what chain the stages; a stale one silently resumes
        # the previous run instead of starting over.
        shutil.rmtree(workspace / MODEL_DIRNAME, ignore_errors=True)
        print(f"[train] --force: cleared {workspace / MODEL_DIRNAME}")

    with StepContext("train", workspace, artifacts_dir=workspace / STAGE_DIRNAME) as ctx:
        train(workspace, ctx, use_depth=not args.no_depth_cloud,
              refine_poses_stage4=not args.no_pose_refine,
              pose_lr=args.pose_lr,
              lambda_track=args.lambda_track,
              voxel_downsample_m=args.voxel_downsample_m,
              stages=stages, schedule_params=schedule_params,
              checkpoint_interval=args.checkpoint_interval,
              stop_after_phase=args.stop_after_phase,
              use_priors=not args.no_depth_priors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
