#!/usr/bin/env python3
"""Step 5 -- progressive multi-resolution 2DGS training.

In:  ``<workspace>/`` (images, ``sparse/0/``, ``depth/points3D_depth.ply``).
Out: ``<workspace>/2dgs/point_cloud/iteration_<N>/point_cloud.ply``.

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
MODEL_DIRNAME = "2dgs"

# (resolution divisor, cumulative iteration target). A stage boundary sits
# exactly on PHASE2_FROM so the densification phases never straddle a stage, and
# the others deliberately avoid multiples of OPACITY_RESET_INTERVAL so a reset
# never lands on the iteration a stage stops and checkpoints at.
#
# Only one coarse stage survives. The r=8/r=4 stages came from the sparse-SfM
# lineage, where they let a few thousand blobs find global structure cheaply.
# Step 4 already delivers metrically-correct geometry at 1.5 cm spacing, so
# there is no global structure left to find -- and at r=8 a 1.5 cm surfel
# covers a fraction of a pixel, so most of the cloud took no gradient while the
# pixel-footprint culls kept deleting it.
STAGES = ((2, 2_000), (1, 10_000))
TOTAL_ITERATIONS = STAGES[-1][1]

# Two-phase densification.
#
# Phase 1 (0 - 6k): no densification at all. The surfel cloud arrives already at
# the target density (1.5 cm grid, see step_depth.VOXEL_DOWNSAMPLE_M), so these
# iterations only fit what is already there -- positions, rotations, scales and
# colours -- under multi-view and normal consistency.
# Phase 2 (6k - 10k): fine densification at a strict gradient threshold, purely
# to resolve high-frequency texture the initial grid cannot carry.
PHASE2_FROM = 6_000
DENSIFY_GRAD_THRESHOLD = 0.0008
OPACITY_RESET_INTERVAL = 2_000

# Normal consistency is wanted from the start of phase 1, not from train.py's
# upstream default of 7000 (which, on a 10k schedule, would leave it off for
# everything but the tail). Multi-view consistency needs no flag: lambda_multiview
# is 0.3 with mv_from_iter 0, so it is already active from iteration 0.
NORMAL_FROM_ITER = 0

# TrackGS pose refinement is enabled exclusively in Stage 4 (resolution 1 / 1080p,
# iters 6,000 to 10,000) with tight landmark reprojection regularization to absorb
# sub-pixel handheld frame-to-frame jitter without deforming global metric geometry.
DEFAULT_POSE_LR = 0.0001
DEFAULT_LAMBDA_TRACK = 0.1


def densification_flags(target_iter: int, total: int = TOTAL_ITERATIONS) -> list[str]:
    """Densification/pruning flags for the stage ending at ``target_iter``.

    train.py nests the opacity reset *inside* the ``densify_until_iter`` gate::

        if iteration < opt.densify_until_iter:
            if iteration > opt.densify_from_iter and ...:  densify_and_prune(...)
            if iteration % opt.opacity_reset_interval == 0: reset_opacity()

    so ``densify_until_iter = 0`` would switch off the 2k pruning cycles as well.
    Phase 1 therefore suppresses densification by putting ``densify_from_iter``
    past the end of the run and leaves the gate open, which keeps the resets.
    """
    if target_iter > PHASE2_FROM:
        densify_from, densify_until = PHASE2_FROM, total
    else:
        densify_from, densify_until = total, PHASE2_FROM
    return ["--densify_from_iter", str(densify_from),
            "--densify_until_iter", str(densify_until),
            "--densify_grad_threshold", str(DENSIFY_GRAD_THRESHOLD),
            "--opacity_reset_interval", str(OPACITY_RESET_INTERVAL),
            "--normal_from_iter", str(NORMAL_FROM_ITER)]


def install_depth_cloud(
    workspace: Path,
    ctx: StepContext,
    use_depth: bool = True,
    voxel_downsample_m: float | None = None,
) -> None:
    sparse_dir = workspace / "sparse" / "0"
    target = sparse_dir / "points3D.ply"
    depth_ply = workspace / "depth" / "points3D_depth.ply"
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
          stages: tuple = STAGES) -> None:
    model_dir = workspace / MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    install_depth_cloud(workspace, ctx, use_depth, voxel_downsample_m=voxel_downsample_m)

    total = stages[-1][1]
    ctx.metric("stages", [{"resolution": r, "until_iter": n} for r, n in stages])
    checkpoint: Path | None = None

    for resolution, target_iter in stages:
        cmd = [sys.executable, str(TRAIN_SCRIPT),
               "-s", str(workspace),
               "-m", str(model_dir),
               "-r", str(resolution),
               "--iterations", str(target_iter),
               "--position_lr_max_steps", str(total),
               "--save_iterations", str(target_iter),
               "--test_iterations", str(target_iter),
               "--checkpoint_iterations", str(target_iter),
               "--quiet"] + densification_flags(target_iter, total)

        # Enable TrackGS pose refinement ONLY in Stage 4 (full resolution 1080p)
        if resolution == 1 and refine_poses_stage4:
            cmd += [
                "--refine_poses_during_training",
                "--pose_lr", str(pose_lr),
                "--lambda_track", str(lambda_track),
            ]

        if checkpoint is not None:
            cmd += ["--start_checkpoint", str(checkpoint)]

        ctx.note(f"--- stage 1/{resolution} to iteration {target_iter} ---")
        ctx.note(f"$ {' '.join(cmd)}")
        with ctx.timer(f"train_r{resolution}"):
            proc = subprocess.run(cmd, cwd=str(TRAIN_SCRIPT.parent), env=subprocess_env())
        if proc.returncode != 0:
            raise RuntimeError(
                f"train.py failed at stage 1/{resolution} (exit {proc.returncode}). "
                f"The previous stage's checkpoint is still in {model_dir}, so this "
                "stage can be retried without redoing the earlier ones.")

        checkpoint = model_dir / f"chkpnt{target_iter}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"train.py exited 0 but wrote no {checkpoint.name}; cannot chain the next stage")
        ctx.metric(f"checkpoint_r{resolution}_mb", round(checkpoint.stat().st_size / 1e6, 1))

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
    parser.add_argument("--voxel-downsample-m", type=float, default=None,
                        help="Optional final voxel downsample grid size (in meters) to apply to depth cloud before training")
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    stages = STAGES
    if args.iterations is not None:
        stages = ((args.resolution if args.resolution is not None else 1, args.iterations),)

    # Must be absolute: train() runs train.py with cwd=03_2DGS_training/, so a
    # relative workspace would resolve against that directory instead of here.
    workspace = Path(args.workspace).resolve()
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

    with StepContext("train", workspace) as ctx:
        train(workspace, ctx, use_depth=not args.no_depth_cloud,
              refine_poses_stage4=not args.no_pose_refine,
              pose_lr=args.pose_lr,
              lambda_track=args.lambda_track,
              voxel_downsample_m=args.voxel_downsample_m,
              stages=stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
