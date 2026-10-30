#!/usr/bin/env python3
"""SOR Floater Removal Ablation Study for 3D Gaussian Splatting.

Evaluates 9 combinations (3x3 grid) of:
- Minimum number of Gaussians for a cluster to be considered significant
- Minimum distance from the cluster to be considered outlier

Outputs a summary table with:
- Run ID
- Min cluster size (Gaussians)
- Min distance from cluster (m)
- Number of clusters identified
- Number of floaters detected
- Percentage of floaters removed
- Time taken per run (s)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure stage03 directory and gsplat_train are importable
_stage_dir = Path(__file__).resolve().parent
_backend_dir = _stage_dir.parent
if str(_stage_dir) not in sys.path:
    sys.path.insert(0, str(_stage_dir))
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from gsplat_train import remove_cluster_floaters

DEFAULT_CHECKPOINT = (
    _backend_dir / "current_scene" / "scan_0" / "gaussian_splat" / "checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = _backend_dir.parent / "agents_files" / "artifacts" / "sor_floater_ablation"


def run_ablation(
    checkpoint_path: Path,
    min_cluster_sizes: list[int],
    min_distances: list[float],
    min_samples: int = 10,
    output_dir: Path | None = None,
) -> list[dict]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    splats_dict = checkpoint["splats"]
    total_splats = len(splats_dict["means"])
    print(f"Loaded {total_splats} Gaussians.")

    runs_results = []
    run_idx = 1
    total_runs = len(min_cluster_sizes) * len(min_distances)

    print("\n" + "=" * 105)
    print(f"{'Run':<4} | {'Min Cluster Size':<17} | {'Min Distance (m)':<17} | {'Clusters Found':<15} | {'Floaters Found':<15} | {'Floaters (%)':<13} | {'Time (s)':<10}")
    print("-" * 105)

    for d in min_distances:
        for min_size in min_cluster_sizes:
            # Construct a ParameterDict copy for the test
            splats_test = torch.nn.ParameterDict({
                k: torch.nn.Parameter(v.clone()) for k, v in splats_dict.items()
            })

            t0 = time.time()
            filtered_splats, stats = remove_cluster_floaters(
                splats_test,
                min_cluster_size=min_size,
                min_distance=d,
                min_samples=min_samples,
            )
            elapsed = time.time() - t0

            record = {
                "run_id": run_idx,
                "min_cluster_size": min_size,
                "min_distance_m": d,
                "num_clusters_identified": stats["num_clusters_identified"],
                "num_floaters_detected": stats["num_floaters_detected"],
                "floaters_pct": round(stats["floaters_percent"], 2),
                "surviving_gaussians": stats["surviving_gaussians"],
                "total_gaussians": stats["total_gaussians"],
                "time_taken_s": round(elapsed, 3),
            }
            runs_results.append(record)

            print(
                f"{run_idx:<4} | "
                f"{min_size:<17} | "
                f"{d:<17.2f} | "
                f"{record['num_clusters_identified']:<15} | "
                f"{record['num_floaters_detected']:<15} | "
                f"{record['floaters_pct']:<13.2f} | "
                f"{record['time_taken_s']:<10.3f}"
            )
            run_idx += 1

    print("=" * 105)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "checkpoint": str(checkpoint_path),
                "total_gaussians": total_splats,
                "runs": runs_results,
            }, f, indent=2)
        print(f"\nSaved structured JSON results to: {json_path}")

        # Save Markdown Summary Table
        md_path = output_dir / "summary_table.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# SOR Floater Removal Ablation Study Summary\n\n")
            f.write(f"- **Scene Model**: `{checkpoint_path}`\n")
            f.write(f"- **Total Input Gaussians**: {total_splats:,}\n")
            f.write(f"- **Total Runs**: {total_runs} (3x3 grid)\n\n")
            f.write("| Run | Min Cluster Size (Gaussians) | Min Outlier Distance (m) | Clusters Identified | Floaters Detected | Floaters (%) | Time Taken (s) |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in runs_results:
                f.write(
                    f"| {r['run_id']} | {r['min_cluster_size']:,} | {r['min_distance_m']:.2f} m | "
                    f"{r['num_clusters_identified']} | {r['num_floaters_detected']:,} | "
                    f"{r['floaters_pct']:.2f}% | {r['time_taken_s']:.3f} s |\n"
                )
        print(f"Saved Markdown summary table to: {md_path}")

    return runs_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to gsplat checkpoint.pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--min-cluster-sizes",
        type=int,
        nargs="+",
        default=[50, 200, 500],
        help="List of 3 minimum cluster sizes (default: 50 200 500)",
    )
    parser.add_argument(
        "--min-distances",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        help="List of 3 minimum outlier distances in meters (default: 0.05 0.10 0.20)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for ablation artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args(argv)

    run_ablation(
        checkpoint_path=Path(args.checkpoint),
        min_cluster_sizes=args.min_cluster_sizes,
        min_distances=args.min_distances,
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
