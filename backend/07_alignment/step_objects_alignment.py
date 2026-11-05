#!/usr/bin/env python3
"""Stage 07 — Objects Alignment: rigid baseline registration and support placement.

Sequences:
  1. build_rigid_baseline: Rigid point-to-plane registration of Stage 05 object instances
     to the Stage 06 completed structural mesh (complete.ply).
  2. fit_contact_groups_inside: Infer nearby floor/wall supports and fit contact groups
     using non-elongated object faces for placement evidence.

CLI:
  .venv/bin/python 07_alignment/step_objects_alignment.py --workspace current_scene/scan_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys
import time
import traceback

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from Utilities.pipeline_paths import bootstrap
bootstrap()


def discover_scans(workspace: Path) -> list[Path]:
    """Find scans under workspace or treat workspace itself as a scan directory."""
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    scans = sorted(p for p in workspace.glob("scan_*") if p.is_dir())
    if scans:
        return scans
    if (workspace / "mesh.ply").is_file() or (workspace / "segmentation").is_dir():
        return [workspace]
    return [workspace]


def run_pipeline_for_scene(
    shell_mesh: Path,
    segmentation_output: Path,
    output_dir: Path,
    allow_rotation: bool = False,
) -> int:
    source = Path(__file__).resolve().parent
    output = output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    fit_dir = output / "structural_fit"

    baseline_flags = [
        "--segmentation-output", str(segmentation_output.absolute()),
        "--shell", str(shell_mesh.absolute()),
        "--output-dir", str(output),
    ]
    if allow_rotation:
        baseline_flags.append("--allow-rotation")

    stages = [
        ("build_rigid_baseline", baseline_flags),
        ("fit_contact_groups_inside", [
            "--alignment-dir", str(output),
            "--output-dir", str(fit_dir),
        ]),
    ]

    timings = []
    for name, flags in stages:
        start = time.perf_counter()
        previous_argv = sys.argv
        sys.argv = [str(source / f"{name}.py"), *flags]
        code = 0
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as result:
            code = result.code if isinstance(result.code, int) else 0 if result.code is None else 1
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            sys.argv = previous_argv

        timings.append({"stage": name, "wall_seconds": time.perf_counter() - start, "exit_code": code})
        if output.exists():
            (output / "step07_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        if code:
            return code
    (output / "plane_snap_report.json").unlink(missing_ok=True)
    (fit_dir / "plane_snap_report.json").unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path,
                        help="Workspace or scan directory (auto-resolves Stage 04-06 inputs)")
    parser.add_argument("--shell", type=Path,
                        help="Path to Stage 06 structural mesh (defaults to <scan>/mesh_elaboration/complete.ply)")
    parser.add_argument("--segmentation-output", type=Path,
                        help="Path to Stage 05 segmentation directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (defaults to <scan>/objects_alignment)")
    parser.add_argument("--allow-rotation", action="store_true",
                        help="Estimate 3D rotation in addition to translation during rigid alignment")

    args = parser.parse_args(argv)

    if args.workspace is not None:
        workspace = args.workspace
        try:
            scans = discover_scans(workspace)
        except Exception as exc:
            print(f"Error resolving scans in {workspace}: {exc}", file=sys.stderr)
            return 1

        overall_rc = 0
        for scan in scans:
            shell_mesh = args.shell or (scan / "mesh_elaboration" / "complete.ply")
            seg_dir = args.segmentation_output or (scan / "segmentation")
            output_dir = args.output_dir or (scan / "objects_alignment")

            if not shell_mesh.is_file():
                print(f"Error: Regularized shell not found at {shell_mesh}. Run Stage 06 first.", file=sys.stderr)
                return 1
            if not seg_dir.is_dir():
                print(f"Error: Segmentation output not found at {seg_dir}. Run Stage 05 first.", file=sys.stderr)
                return 1

            print(f"\n=======================================================")
            print(f"Running Stage 07 Objects Alignment for: {scan.name}")
            print(f"  Shell Mesh:   {shell_mesh}")
            print(f"  Segmentation: {seg_dir}")
            print(f"  Output:       {output_dir}")
            print(f"=======================================================")

            t0 = time.perf_counter()
            rc = run_pipeline_for_scene(
                shell_mesh=shell_mesh,
                segmentation_output=seg_dir,
                output_dir=output_dir,
                allow_rotation=args.allow_rotation,
            )
            elapsed = time.perf_counter() - t0
            print(f"Stage 07 Objects Alignment finished in {elapsed:.2f}s (rc={rc})")
            if rc != 0:
                overall_rc = rc
                break
        return overall_rc

    # Explicit input mode
    if not args.shell or not args.segmentation_output:
        parser.error("Either --workspace OR both of (--shell, --segmentation-output) must be specified.")

    output_dir = args.output_dir or Path("current_scene/scan_0/objects_alignment")
    return run_pipeline_for_scene(
        shell_mesh=args.shell,
        segmentation_output=args.segmentation_output,
        output_dir=output_dir,
        allow_rotation=args.allow_rotation,
    )


if __name__ == "__main__":
    sys.exit(main())
