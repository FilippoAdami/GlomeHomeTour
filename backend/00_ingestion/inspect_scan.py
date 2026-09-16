#!/usr/bin/env python3
"""GlomeHomeTour Backend: Scan Package Inspection & Step 1 Verification Tool.

Runs Step 1 (Ingestion, Quality Gate, Pose Synchronization) on any capture ZIP
archive or directory, printing detailed diagnostics. Pose refinement now lives
in stage 2 (``01_poses_refinment/convert_transforms_to_colmap.py``).

Usage:
    python backend/inspect_scan.py backend/scenes/bedroom.zip
    python backend/inspect_scan.py backend/scenes/bedroom.zip --stride 4
    python backend/inspect_scan.py backend/scenes/bedroom.zip --full
"""

import argparse
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
from Utilities.pipeline_paths import bootstrap
bootstrap()

import numpy as np

from keyframe_selector import DynamicKeyframeSelector, estimate_scene_depths
from package_loader import load_package
from pose_aligner import PoseAligner
from quality_gate import QualityGate
from depth_priors import (
    DepthPriorEstimator,
    GlobalDepthGraphOptimizer,
    GlobalDepthGraphResult,
    MetricDepthAligner,
)
from initialization import SurfelCloudInitializer


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and verify Step 1 (Ingestion & Quality Gate) on a mobile scan package."
    )
    parser.add_argument("package_path", type=str, help="Path to capture ZIP archive or directory.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Evaluate every N-th frame for faster inspection (default: 1 = evaluate all frames).",
    )
    parser.add_argument(
        "--blur-thresh",
        type=float,
        default=None,
        help="Override adaptive blur threshold with a fixed value.",
    )
    parser.add_argument(
        "--run-depth",
        action="store_true",
        help="Run Step 2: Depth Anything prior estimation and 2DGS surfel initialization.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Depth model preset or HuggingFace repo (default: depth-anything/DA3NESTED-GIANT-LARGE-1.1).",
    )
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=5,
        help="Keyframe stride for depth inference (default: 5; use 1 for all frames).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Window chunk size for multi-view streaming inference (default: 8, uses ~10-12 GB VRAM).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=3,
        help="Overlap frames between consecutive chunks for blending (default: 3).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.8,
        help="Minimum confidence threshold to prune empty-air/floating points (default: 0.8).",
    )
    parser.add_argument(
        "--skip-graph-opt",
        action="store_true",
        help="Skip global optical flow graph optimizer (DA3 multi-view is already globally aligned).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache directory for depth and confidence maps.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cached depth and confidence maps before running.",
    )
    parser.add_argument(
        "--target-surfels",
        type=int,
        default=150_000,
        help="Target number of initial surfels (default: 150000).",
    )
    parser.add_argument(
        "--dynamic-keyframes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic spatial and co-visibility keyframe selection (default: True).",
    )
    parser.add_argument(
        "--min-baseline",
        type=float,
        default=0.35,
        help="Minimum translation distance between anchor keyframes in meters (default: 0.35).",
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=18.0,
        help="Minimum geodesic rotation angle between anchor keyframes in degrees (default: 18.0).",
    )
    parser.add_argument(
        "--max-covisibility",
        type=float,
        default=0.78,
        help="Maximum co-visibility overlap ratio to prune redundant views (default: 0.78).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Maximum depth horizon in meters (default: None, dynamically estimated from scene distribution).",
    )
    parser.add_argument(
        "--min-consensus",
        type=int,
        default=1,
        help="Minimum number of other views corroborating a 3D point (default: 1, i.e. >= 2 total views).",
    )
    parser.add_argument(
        "--no-sor",
        action="store_true",
        help="Disable Statistical Outlier Removal (SOR).",
    )
    parser.add_argument(
        "--output-ply",
        type=str,
        default="backend/scenes/initial_surfels.ply",
        help="Output file path for initialized surfel cloud PLY.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.015,
        help="Voxel downsampling grid size in meters (default: 0.015, 1.5 cm).",
    )
    args = parser.parse_args()

    pkg_path = Path(args.package_path)
    if not pkg_path.exists():
        print(f"ERROR: Package not found at: {pkg_path}")
        sys.exit(1)

    print("=" * 70)
    print(" GLOME HOME TOUR: STEP 1 INGESTION & QUALITY PIPELINE TEST")
    print("=" * 70)
    print(f"Target scan: {pkg_path.resolve()}")
    print(f"File size:   {pkg_path.stat().st_size / (1024 * 1024):.2f} MB")
    print("-" * 70)

    # 1. Package Loading
    print("\n[1/3] Loading and validating capture package against shared/schemas/...")
    t0 = time.time()
    try:
        pkg = load_package(pkg_path)
    except Exception as e:
        print(f"FAIL: Package loading failed schema verification: {e}")
        sys.exit(1)

    load_time = time.time() - t0
    intr = pkg.intrinsics
    print(f" -> SUCCESS in {load_time:.2f}s!")
    print(f"    - Resolution:        {intr.w} x {intr.h} ({intr.camera_model})")
    print(f"    - Focal lengths:     fx={intr.fl_x:.2f}, fy={intr.fl_y:.2f}")
    print(f"    - Principal point:   cx={intr.cx:.2f}, cy={intr.cy:.2f}")
    print(f"    - Field of view:     camera_angle_x = {np.degrees(intr.camera_angle_x):.1f}°")
    print(f"    - Keyframe count:    {len(pkg.keyframes)}")
    print(f"    - Trajectory points: {len(pkg.trajectory)}")

    cov = pkg.coverage_summary
    print(f"    - Scan duration:     {cov.get('duration_s', 0):.1f} s")
    print(f"    - Depth source:      {cov.get('depth_source', 'unknown')}")
    print(f"    - Voxel stats:       occupied={cov.get('voxels', {}).get('occupied', 0)}, free={cov.get('voxels', {}).get('free', 0)}")

    # Keyframe subset if stride > 1
    eval_keyframes = pkg.keyframes[::args.stride]
    if args.stride > 1:
        print(f"\n[Note] Running with stride={args.stride}: evaluating {len(eval_keyframes)} / {len(pkg.keyframes)} keyframes.")

    # 2. Quality Gate Evaluation
    print("\n[2/3] Running Desktop Quality Gate (Blur, Exposure, Texture)...")
    t1 = time.time()
    gate = QualityGate(blur_threshold=args.blur_thresh)
    gate_res = gate.evaluate(eval_keyframes)
    gate_time = time.time() - t1

    blur_scores = [m.blur_score for m in gate_res.metrics]
    print(f" -> Evaluated in {gate_time:.2f}s:")
    print(f"    - Total keyframes evaluated: {gate_res.summary['total']}")
    print(f"    - Accepted keyframes:        {gate_res.summary['accepted']} ({gate_res.summary['accepted']/max(1, len(eval_keyframes))*100:.1f}%)")
    print(f"    - Rejected (motion blur):    {gate_res.summary['rejected_blur']}")
    print(f"    - Rejected (low texture):    {gate_res.summary['rejected_texture']}")
    print(f"    - Rejected (bad exposure):   {gate_res.summary['rejected_exposure']}")
    print(f"    - Blur score distribution:   min={np.min(blur_scores):.2f}, median={np.median(blur_scores):.2f}, max={np.max(blur_scores):.2f}")

    if gate_res.summary["accepted"] == 0:
        print("WARNING: All frames rejected by quality gate! Consider checking lighting or blur thresholds.")
        sys.exit(1)

    # 3. Pose Synchronization
    print("\n[3/3] Running Sub-ms Timestamp Synchronization & Quaternion SLERP...")
    t2 = time.time()
    aligner = PoseAligner(pkg.trajectory)
    synced_keyframes = aligner.synchronize_keyframes(gate_res.accepted_keyframes)
    sync_time = time.time() - t2
    print(f" -> SUCCESS in {sync_time*1000:.1f}ms! Synchronized {len(synced_keyframes)} keyframe poses to continuous VIO.")

    # Print sample trajectory & motion distance
    first_pose = synced_keyframes[0].transform_matrix
    last_pose = synced_keyframes[-1].transform_matrix
    walked_dist = np.linalg.norm(last_pose[:3, 3] - first_pose[:3, 3])
    print(f"    - First pose translation: [x={first_pose[0,3]:.2f}, y={first_pose[1,3]:.2f}, z={first_pose[2,3]:.2f}]")
    print(f"    - Last pose translation:  [x={last_pose[0,3]:.2f}, y={last_pose[1,3]:.2f}, z={last_pose[2,3]:.2f}]")
    print(f"    - Net endpoint distance:  {walked_dist:.2f} meters")

    # 4. Step 2: Depth Estimation & Surfel Cloud Initialization
    if args.run_depth:
        print("\n" + "=" * 70)
        print(" STEP 2: GEOMETRIC DEPTH PRIORS & MULTI-VIEW RECONSTRUCTION")
        print("=" * 70)
        print("[+] Estimating depth priors and initializing 2D Gaussian surfels...")
        t4 = time.time()

        # Cache directory for precomputed depth and confidence maps
        if args.cache_dir:
            cache_dir = Path(args.cache_dir)
        else:
            safe_model = args.model_name.split("/")[-1].lower()
            cache_dir = Path("backend/scenes/.depth_cache") / f"{pkg_path.stem}_{safe_model}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        if args.clear_cache:
            print(f"    - Clearing cached files in: {cache_dir.resolve()}")
            for f in cache_dir.glob("*.npy"):
                f.unlink()

        if args.dynamic_keyframes:
            print("\n    - Running Dynamic Spatial & Co-visibility Keyframe Selector...")
            selector = DynamicKeyframeSelector(
                min_translation_m=args.min_baseline,
                min_rotation_deg=args.min_angle,
                max_covisibility=args.max_covisibility,
            )
            candidate_pool = synced_keyframes[::args.depth_stride] if args.depth_stride > 1 else synced_keyframes
            # Without measured depth the selector assumes everything is
            # reference_depth_m away and rates disjoint views as overlapping.
            sel_res = selector.select_keyframes(
                candidate_pool, intr,
                scene_depths=estimate_scene_depths(candidate_pool, intr))
            eval_depth_kfs = sel_res.selected_keyframes
            print(f"    -> Selected {len(eval_depth_kfs)} anchor keyframes from {len(candidate_pool)} candidates ({sel_res.selection_ratio*100:.1f}%):")
            print(f"       * Motion accepted:              {sel_res.reasons.get('sufficient_motion', 0)}")
            print(f"       * Coverage gap prevention:      {sel_res.reasons.get('coverage_gap_prevention', 0)}")
            print(f"       * Pruned micro-motion:          {sel_res.reasons.get('redundant_motion', 0)}")
            print(f"       * Pruned redundant covisibility: {sel_res.reasons.get('redundant_covisibility', 0)}")
        else:
            eval_depth_kfs = synced_keyframes[::args.depth_stride]
            print(f"    - Processing {len(eval_depth_kfs)} keyframes (stride={args.depth_stride})...")
        print(f"    - Depth model: {args.model_name}")
        print(f"    - Confidence threshold: {args.min_conf}")

        # Check if all depths and confidences are cached and contain no NaNs
        all_cached = True
        raw_depth_maps = []
        conf_maps = []
        for kf in eval_depth_kfs:
            safe_id = kf.file_path.replace("/", "_").replace(".", "_")
            d_file = cache_dir / f"depth_{safe_id}.npy"
            c_file = cache_dir / f"conf_{safe_id}.npy"
            if not d_file.exists():
                all_cached = False
                break
            try:
                test_d = np.load(d_file)
                if np.isnan(test_d).any():
                    all_cached = False
                    break
            except Exception:
                all_cached = False
                break

        if all_cached:
            print(f"    - Found existing valid cached depth and confidence maps in {cache_dir}. Loading...")
            for kf in eval_depth_kfs:
                safe_id = kf.file_path.replace("/", "_").replace(".", "_")
                d_file = cache_dir / f"depth_{safe_id}.npy"
                c_file = cache_dir / f"conf_{safe_id}.npy"
                raw_depth_maps.append(np.load(d_file))
                conf_maps.append(np.load(c_file) if c_file.exists() else None)
        else:
            estimator = DepthPriorEstimator(model_name=args.model_name)
            print(f"    - Running multi-view streaming inference (chunk_size={args.chunk_size}, overlap={args.overlap})...")
            images = [kf.load_image_rgb() for kf in eval_depth_kfs]

            # Compute W2C matrices for DA3 conditioning
            w2c_matrices = np.stack(
                [np.linalg.inv(kf.transform_matrix) for kf in eval_depth_kfs], axis=0
            ).astype(np.float32)

            # Compute intrinsics 3x3 matrices
            k_mat = np.array(
                [
                    [intr.fl_x, 0.0, intr.cx],
                    [0.0, intr.fl_y, intr.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            ixt_matrices = np.repeat(k_mat[None, ...], len(eval_depth_kfs), axis=0)

            raw_depth_maps, conf_maps = estimator.estimate_depth_streaming(
                images,
                extrinsics=w2c_matrices,
                intrinsics=ixt_matrices,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            # Save to cache
            print(f"    - Saving results to cache: {cache_dir.resolve()}")
            for idx, kf in enumerate(eval_depth_kfs):
                safe_id = kf.file_path.replace("/", "_").replace(".", "_")
                np.save(cache_dir / f"depth_{safe_id}.npy", raw_depth_maps[idx].astype(np.float32))
                if conf_maps is not None and idx < len(conf_maps) and conf_maps[idx] is not None:
                    np.save(cache_dir / f"conf_{safe_id}.npy", conf_maps[idx].astype(np.float32))

        # Optional Global Depth Graph Optimizer
        aligned_depth_maps = raw_depth_maps
        if not args.skip_graph_opt:
            print("\n    - Running Global Depth Graph Optimizer (Lucas-Kanade + Loop Closures)...")
            t_opt0 = time.time()
            graph_optimizer = GlobalDepthGraphOptimizer()
            opt_res = graph_optimizer.optimize(eval_depth_kfs, raw_depth_maps, intr)
            t_opt = time.time() - t_opt0

            print(f"    -> Graph solved in {t_opt:.2f}s:")
            print(f"       * Sequential tracking edges: {opt_res.num_temporal_edges}")
            print(f"       * Spatial loop-closure edges: {opt_res.num_loop_edges}")
            if np.isnan(opt_res.rmse_after_m) or any(np.isnan(s) for s in opt_res.scales):
                print("    -> Warning: Graph optimization encountered numerical instability, using DA3 multi-view depths directly.")
                aligned_depth_maps = raw_depth_maps
            else:
                print(f"       * Cross-view depth discrepancy: {opt_res.rmse_before_m*100:.2f} cm -> {opt_res.rmse_after_m*100:.2f} cm")
                print(f"       * Scale factor range:        [{np.min(opt_res.scales):.3f}, {np.max(opt_res.scales):.3f}] (mean={np.mean(opt_res.scales):.3f})")
                print(f"       * Shift offset range:        [{np.min(opt_res.shifts):+.3f}, {np.max(opt_res.shifts):+.3f}] m (mean={np.mean(opt_res.shifts):+.3f}m)")
                aligned_depth_maps = opt_res.aligned_depth_maps
        else:
            print("\n    - Skipping Global Graph Optimizer (using DA3 multi-view trajectory alignment directly).")

        sparse_pts = None

        # Initialize Surfel Cloud with confidence masking, consensus gating, and SOR
        initializer = SurfelCloudInitializer(
            target_surfels=args.target_surfels,
            voxel_downsample_m=args.voxel_size,
            max_depth_m=args.max_depth,
            min_consensus=args.min_consensus,
            enable_sor=not args.no_sor,
        )
        print(f"\n    - Unprojecting point clouds into initial surfel volume (target: {args.target_surfels:,}, min_conf: {args.min_conf})...")
        cloud = initializer.initialize_from_keyframes(
            eval_depth_kfs,
            aligned_depth_maps,
            intr,
            sparse_points_3d=sparse_pts,
            conf_maps=conf_maps if (conf_maps and any(c is not None for c in conf_maps)) else None,
            min_conf=args.min_conf,
        )

        surfel_time = time.time() - t4
        print(f" -> SUCCESS in {surfel_time:.2f}s!")
        print(f"    - Initialized Surfel Count:  {len(cloud):,}")
        print(f"    - Position bounds X:         [{np.min(cloud.positions[:,0]):.2f}, {np.max(cloud.positions[:,0]):.2f}] m (span: {np.ptp(cloud.positions[:,0]):.2f}m)")
        print(f"    - Position bounds Y:         [{np.min(cloud.positions[:,1]):.2f}, {np.max(cloud.positions[:,1]):.2f}] m (span: {np.ptp(cloud.positions[:,1]):.2f}m)")
        print(f"    - Position bounds Z:         [{np.min(cloud.positions[:,2]):.2f}, {np.max(cloud.positions[:,2]):.2f}] m (span: {np.ptp(cloud.positions[:,2]):.2f}m)")
        print(f"    - Mean 2D Gaussian Scale:    {np.mean(cloud.scales_2d)*100:.2f} cm")
        print(f"    - Mean Opacity:              {np.mean(cloud.opacities):.2f}")

        # Export PLY file
        out_ply = Path(args.output_ply)
        cloud.to_ply(out_ply)
        print(f"    - Exported Surfel Cloud PLY: {out_ply.resolve()} ({out_ply.stat().st_size / (1024*1024):.2f} MB)")

    print("\n" + "=" * 70)
    print(" VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
