#!/usr/bin/env python3
"""Step 3 -- Depth and normal supervised gsplat training.

In:  ``<workspace>/`` (images, ``sparse/0/``, ``02_depth_estimation/depth/points3D_depth.ply``,
     ``02_depth_estimation/depth/depth_maps/``, ``02_depth_estimation/depth/normal_maps/``).
Out: ``<workspace>/03_FastGS_DNSplatter/point_cloud/iteration_<N>/point_cloud.ply``.

Uses Apache-2.0 gsplat rasterization and locally produced depth/normal maps.

    python 03_FastGS_DNSplatter/step_train.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import os
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
STAGE_DIRNAME = "03_FastGS_DNSplatter"
MODEL_DIRNAME = STAGE_DIRNAME
DEPTH_STAGE_DIRNAME = "02_depth_estimation"
TRAIN_SCRIPT = _backend_dir / STAGE_DIRNAME / "gsplat_train.py"

DEFAULT_ITERATIONS = 22_000
DEFAULT_LAMBDA_DEPTH = 0.15
DEFAULT_LAMBDA_NORMAL = 0.075
DEFAULT_DEPTH_LOSS = "pearson"


def install_depth_cloud(
    workspace: Path,
    ctx: StepContext,
    use_depth: bool = True,
) -> None:
    sparse_dir = workspace / "sparse" / "0"
    target = sparse_dir / "points3D.ply"
    depth_ply = workspace / DEPTH_STAGE_DIRNAME / "depth" / "points3D_depth.ply"
    backup = sparse_dir / "points3D_colmap.ply"

    if not use_depth:
        ctx.note("Using the COLMAP sparse cloud (--no-depth-cloud)")
        return
    if not depth_ply.exists():
        if target.exists():
            ctx.note(f"{depth_ply} not found; falling back to existing {target}")
            return
        raise FileNotFoundError(f"{depth_ply} missing -- run step 2 (depth estimation) first")

    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)  # keep COLMAP's point cloud as backup

    shutil.copy2(depth_ply, target)
    ctx.note(f"Installed depth cloud {depth_ply.name} -> sparse/0/points3D.ply")
    ctx.metric("init_cloud_mb", round(target.stat().st_size / 1e6, 1))


def train(
    workspace: Path,
    ctx: StepContext,
    iterations: int = DEFAULT_ITERATIONS,
    use_depth_cloud: bool = True,
    depth_supervision: bool = True,
    normal_supervision: bool = True,
    lambda_depth: float = DEFAULT_LAMBDA_DEPTH,
    lambda_normal: float = DEFAULT_LAMBDA_NORMAL,
    depth_loss: str = DEFAULT_DEPTH_LOSS,
    extra_train_args: list[str] | None = None,
) -> None:
    install_depth_cloud(workspace, ctx, use_depth=use_depth_cloud)

    model_dir = workspace / MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "-s", str(workspace),
        "-m", str(model_dir),
        "--iterations", str(iterations),
        "--verbose",
    ]

    if depth_supervision:
        cmd += [
            "--depth-supervision",
            "--lambda-depth", str(lambda_depth),
            "--depth-loss", depth_loss,
        ]

    if normal_supervision:
        cmd += [
            "--normal-supervision",
            "--lambda-normal", str(lambda_normal),
        ]

    if extra_train_args:
        cmd += extra_train_args

    ctx.note(f"Starting gsplat training for {iterations} iterations")
    ctx.note(f"$ {' '.join(cmd)}")

    with ctx.timer("train_gsplat"):
        proc = subprocess.run(cmd, cwd=str(TRAIN_SCRIPT.parent), env=subprocess_env())

    if proc.returncode != 0:
        raise RuntimeError(
            f"gsplat_train.py failed (exit {proc.returncode}). "
            f"Inspect outputs in {model_dir} before re-running.")

    if not (model_dir / "checkpoint.pt").is_file():
        raise FileNotFoundError(f"Training finished but gsplat checkpoint.pt is missing in {model_dir}")

    final_ply = model_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    if not final_ply.exists():
        # Fallback check if saved at standard point_cloud location
        candidates = list(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            final_ply = candidates[0]
        else:
            raise FileNotFoundError(f"Training finished but point_cloud.ply is missing in {model_dir}")

    ctx.metric("final_ply", str(final_ply))
    ctx.metric("final_ply_mb", round(final_ply.stat().st_size / 1e6, 1))
    ctx.note(f"Trained scene: {final_ply} ({final_ply.stat().st_size / 1e6:.1f} MB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE),
                        help=f"Workspace directory (default: {DEFAULT_WORKSPACE})")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help=f"Total training iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--no-depth-cloud", action="store_true",
                        help="Initialise from COLMAP's sparse cloud instead of step 2's depth cloud")
    parser.add_argument("--no-depth-supervision", action="store_true",
                        help="Disable DAv3 depth supervision")
    parser.add_argument("--no-normal-supervision", action="store_true",
                        help="Disable StableNormal normal supervision")
    parser.add_argument("--depth-loss", default=DEFAULT_DEPTH_LOSS,
                        choices=["pearson", "l1", "log_l1", "scale_shift"],
                        help=f"Depth loss function (default: {DEFAULT_DEPTH_LOSS})")
    parser.add_argument("--lambda-depth", type=float, default=DEFAULT_LAMBDA_DEPTH,
                        help=f"Depth loss weight (default: {DEFAULT_LAMBDA_DEPTH})")
    parser.add_argument("--lambda-normal", type=float, default=DEFAULT_LAMBDA_NORMAL,
                        help=f"Normal loss weight (default: {DEFAULT_LAMBDA_NORMAL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists")
    args, unknown = parser.parse_known_args(argv)

    workspace = Path(args.workspace).resolve()
    model_dir = workspace / MODEL_DIRNAME
    final_ply = model_dir / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"

    if not args.force and is_done(workspace, "train", [final_ply, model_dir / "checkpoint.pt"]):
        print("[train] already done, skipping (use --force to re-run)")
        return 0

    with StepContext("train", workspace, artifacts_dir=workspace / STAGE_DIRNAME) as ctx:
        train(
            workspace=workspace,
            ctx=ctx,
            iterations=args.iterations,
            use_depth_cloud=not args.no_depth_cloud,
            depth_supervision=not args.no_depth_supervision,
            normal_supervision=not args.no_normal_supervision,
            lambda_depth=args.lambda_depth,
            lambda_normal=args.lambda_normal,
            depth_loss=args.depth_loss,
            extra_train_args=unknown,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
