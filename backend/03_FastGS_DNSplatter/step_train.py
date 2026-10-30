#!/usr/bin/env python3
"""Step 3 -- Depth and normal supervised gsplat training.

In:  one scan or a parent of ``scan_N/`` folders, each with ``images/``,
     ``sparse/0/`` and stage 02's ``depth/`` maps and initialization cloud.
Out: ``<scan>/gaussian_splat/{checkpoint.pt,gaussian_splat.ply}``.

Uses Apache-2.0 gsplat rasterization and locally produced depth/normal maps.

    python 03_FastGS_DNSplatter/step_train.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap, subprocess_env

bootstrap()

from Utilities.pipeline_step import StepContext, is_done

DEFAULT_WORKSPACE = _backend_dir / "current_scene"
STAGE_DIRNAME = "03_FastGS_DNSplatter"
TRAIN_SCRIPT = _backend_dir / STAGE_DIRNAME / "gsplat_train.py"

DEFAULT_ITERATIONS = 22_000
DEFAULT_LAMBDA_DEPTH = 0.15
DEFAULT_LAMBDA_NORMAL = 0.075
DEFAULT_DEPTH_LOSS = "pearson"
DEFAULT_SOR_FLOATER_REMOVAL = True
DEFAULT_SOR_MIN_CLUSTER_SIZE = 500
DEFAULT_SOR_MIN_DISTANCE = 0.10


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
    sor_floater_removal: bool = DEFAULT_SOR_FLOATER_REMOVAL,
    sor_min_cluster_size: int = DEFAULT_SOR_MIN_CLUSTER_SIZE,
    sor_min_distance: float = DEFAULT_SOR_MIN_DISTANCE,
    extra_train_args: list[str] | None = None,
) -> None:
    model_dir = workspace / "gaussian_splat"

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "-s", str(workspace),
        "-m", str(model_dir),
        "--iterations", str(iterations),
        "--verbose",
    ]
    if not use_depth_cloud:
        cmd += ["--no-depth-cloud"]

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

    if sor_floater_removal:
        cmd += [
            "--sor-floater-removal",
            "--sor-min-cluster-size", str(sor_min_cluster_size),
            "--sor-min-distance", str(sor_min_distance),
        ]
    else:
        cmd += ["--no-sor-floater-removal"]

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

    final_ply = model_dir / "gaussian_splat.ply"
    if not final_ply.exists():
        raise FileNotFoundError(f"Training finished but expected point cloud is missing: {final_ply}")

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
    parser.add_argument("--sor-floater-removal", action=argparse.BooleanOptionalAction, default=DEFAULT_SOR_FLOATER_REMOVAL,
                        help=f"Remove floaters detached from significant clusters at the end of training (default: {DEFAULT_SOR_FLOATER_REMOVAL})")
    parser.add_argument("--sor-min-cluster-size", type=int, default=DEFAULT_SOR_MIN_CLUSTER_SIZE,
                        help=f"Minimum number of gaussians for a cluster to be considered significant (default: {DEFAULT_SOR_MIN_CLUSTER_SIZE})")
    parser.add_argument("--sor-min-distance", type=float, default=DEFAULT_SOR_MIN_DISTANCE,
                        help=f"Outlier distance in world units (metres if calibrated; default: {DEFAULT_SOR_MIN_DISTANCE})")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists")
    args, unknown = parser.parse_known_args(argv)

    workspace = Path(args.workspace).resolve()
    from step_colmap import discover_scenes

    try:
        scenes = [scene for scene, _ in discover_scenes(workspace)]
    except FileNotFoundError:
        scenes = [workspace]  # Standalone COLMAP scenes do not need an ingestion manifest.

    outputs = [output for scene in scenes for output in (
        scene / "gaussian_splat" / "gaussian_splat.ply",
        scene / "gaussian_splat" / "checkpoint.pt",
    )]

    if not args.force and is_done(workspace, "train", outputs):
        print("[train] already done, skipping (use --force to re-run)")
        return 0

    failures, scan_stats = [], []
    scan_workspace = len(scenes) == 1 and scenes[0] == workspace
    with StepContext("train", workspace, write_artifacts=scan_workspace) as ctx:
        for scene in scenes:
            model_dir = scene / "gaussian_splat"
            scene_outputs = [model_dir / "checkpoint.pt",
                             model_dir / "gaussian_splat.ply"]
            if not args.force and is_done(scene, "train", scene_outputs):
                ctx.note(f"{scene.name}: already done, skipping")
                scan_stats.append({"scan": scene.name, "status": "skipped"})
                continue
            try:
                with (nullcontext(ctx) if scene == workspace else
                      StepContext("train", scene, artifacts_dir=scene)) as scan_ctx:
                    train(
                        workspace=scene,
                        ctx=scan_ctx,
                        iterations=args.iterations,
                        use_depth_cloud=not args.no_depth_cloud,
                        depth_supervision=not args.no_depth_supervision,
                        normal_supervision=not args.no_normal_supervision,
                        lambda_depth=args.lambda_depth,
                        lambda_normal=args.lambda_normal,
                        depth_loss=args.depth_loss,
                        sor_floater_removal=args.sor_floater_removal,
                        sor_min_cluster_size=args.sor_min_cluster_size,
                        sor_min_distance=args.sor_min_distance,
                        extra_train_args=unknown,
                    )
                scan_stats.append({"scan": scene.name, "status": "complete"})
            except Exception as exc:
                if scene == workspace:
                    raise
                failures.append(scene.name)
                scan_stats.append({"scan": scene.name, "status": "failed", "error": str(exc)})
                ctx.note(f"{scene.name}: training failed: {exc}")
        ctx.metric("scans", scan_stats)
        if failures:
            raise RuntimeError(f"Training failed for scans: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
