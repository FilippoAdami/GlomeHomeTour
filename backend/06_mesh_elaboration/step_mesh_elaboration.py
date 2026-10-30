#!/usr/bin/env python3
"""Run structural extraction, fitting, regularization and texture synthesis in order.

Accepts either:
  1. A workspace / scan directory (discovering locked Stage 01-05 inputs):
       .venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py --workspace current_scene/scan_0
  2. Explicit input paths (backward-compatible override):
       .venv/bin/python 06_mesh_elaboration/step_mesh_elaboration.py \
         --mesh-path <path/to/mesh.ply> \
         --labels-path <path/to/triangle_class_labels.npy> \
         --structural-json <path/to/class_metadata.json> \
         --output-dir <path/to/output_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback
from typing import Optional

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from Utilities.pipeline_paths import bootstrap
bootstrap()

try:
    from Utilities.pipeline_step import StepContext, update_pipeline_stats
except ImportError:
    StepContext = None
    update_pipeline_stats = None

from mesh_elaboration_inputs import (
    discover_scans,
    resolve_gaussian_splat,
    resolve_labels,
    resolve_mesh,
    resolve_output_dir,
    resolve_scene,
    resolve_scene_size_file,
    resolve_segmentation_output,
    resolve_structural_metadata,
)


def run_pipeline_for_scene(
    scene_mesh: Path,
    labels_path: Path,
    structural_json: Path,
    output_dir: Path,
    cross_section_mesh: Optional[Path] = None,
    segmentation_output: Optional[Path] = None,
    scene_size_file: Optional[Path] = None,
    skip_cross_sections: bool = False,
    skip_texture: bool = False,
    enhance_textures: bool = False,
    only_step: Optional[str] = None,
) -> int:
    source = Path(__file__).resolve().parent
    output = output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)
    sealed_mesh = cross_section_mesh.absolute() if cross_section_mesh else scene_mesh.absolute()
    if not sealed_mesh.is_file():
        sealed_mesh = scene_mesh.absolute()

    extraction_flags = [
        "--mesh-path", str(scene_mesh.absolute()),
        "--labels-path", str(labels_path.absolute()),
        "--structural-json", str(structural_json.absolute()),
        "--output-dir", str(output),
    ]
    if segmentation_output and segmentation_output.is_dir():
        extraction_flags.extend(["--segmentation-output", str(segmentation_output.absolute())])
    wall_fit_flags = ["--instances-dir", str(output),
                      "--output", str(output / "wall_enclosing_shell.ply"),
                      "--manifest-out", str(output / "wall_planes_manifest.json")]

    stages = [
        ("extract_structural_meshes", [("extract_structural_meshes", extraction_flags)]),
        ("fit_enclosing_walls", [("enclose_wall_instances", wall_fit_flags)]),
    ]
    group_floor_flags = ["--instances-dir", str(output)]
    if scene_size_file is not None:
        group_floor_flags.extend(["--scene-size-file", str(scene_size_file.absolute())])
    floor_substeps = [
            ("group_floor_instances_by_level", group_floor_flags),
            ("add_ceilings_to_floor_levels", ["--instances-dir", str(output)]),
            ("clean_floor_level_intermediates", ["--instances-dir", str(output)]),
            ("fit_floor_level_planes", ["--instances-dir", str(output)]),
            ("aggregate_fit_ceiling_planes", ["--instances-dir", str(output)]),
            ("slice_scene_by_floor", [
                "--instances-dir", str(output),
                "--mesh-path", str(scene_mesh.absolute()),
                "--labels-path", str(labels_path.absolute()),
                "--class-metadata", str(structural_json.absolute()),
            ]),
            ("clean_floor_scene_slices", ["--instances-dir", str(output)]),
            ("project_floor_slices_onto_planes", ["--instances-dir", str(output), "--detect-holes"]),
            ("regularize_floor_projection_contours", ["--instances-dir", str(output)]),
            ("fit_floor_projection_contours_to_black", ["--instances-dir", str(output)]),
            ("overlay_fitted_floor_wall_contours", ["--instances-dir", str(output)]),
            ("regularize_floor_level_patterns", ["--instances-dir", str(output)]),
    ]
    floor_step_names = {name for name, _ in floor_substeps}
    if scene_size_file is not None:
        if only_step in (None, "floors"):
            stages.append(("floors", floor_substeps))
        elif only_step in floor_step_names:
            stages.append((only_step, [step for step in floor_substeps if step[0] == only_step]))
    elif only_step == "floors" or only_step in floor_step_names:
        print("Error: floor processing requires depth/scene_size.txt with a floor count",
              file=sys.stderr)
        return 1
    else:
        print("Skipping floors because no scene_size file is available")
    legacy_stages = [
        ("fit_floors_ceilings_planes", [("fit_floors_ceilings_planes", [
            "--instances-dir", str(output),
            "--wall-manifest", str(output / "wall_planes_manifest.json"),
            "--max-tilt-deg", "4.0",
        ])]),
        ("recognize_and_regularize_patterns", [("recognize_and_regularize_patterns", [
            "--instances-dir", str(output),
            "--output-dir", str(output),
            "--scene-mesh", str(scene_mesh.absolute()),
            "--skip-texture",
        ])]),
    ]
    # Floor-level artifacts supersede the older combined slab/texture workflow.
    if scene_size_file is None or only_step in {name for name, _ in legacy_stages}:
        stages.extend(legacy_stages)

    # Keep all elaborated products in the Phase 04 world frame.
    if not skip_texture and (scene_size_file is None or only_step == "apply_structural_textures"):
        stages.append(("apply_structural_textures", [("apply_structural_textures", ["--instances-dir", str(output), "--output-dir", str(output)])]))

    if only_step is not None:
        stages = [stage for stage in stages if stage[0] == only_step]

    timings = []
    for name, substeps in stages:
        start = time.perf_counter()
        substep_timings = []
        code = 0
        for script_name, flags in substeps:
            previous_argv = sys.argv
            sys.argv = [str(source / f"{script_name}.py"), *flags]
            substep_start = time.perf_counter()
            try:
                runpy.run_path(sys.argv[0], run_name="__main__")
            except SystemExit as result:
                code = result.code if isinstance(result.code, int) else 0 if result.code is None else 1
            except Exception:
                traceback.print_exc()
                code = 1
            finally:
                sys.argv = previous_argv
            substep_timings.append({"stage": script_name, "wall_seconds": time.perf_counter() - substep_start, "exit_code": code})
            if code:
                break

        timings.append({"stage": name, "wall_seconds": time.perf_counter() - start,
                        "exit_code": code, "substeps": substep_timings})
        if output.exists():
            (output / "step06_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
        if code:
            return code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path,
                        help="Workspace or scan directory (auto-resolves Stage 04 mesh and Stage 05 segmentation inputs)")
    parser.add_argument("--mesh-path", type=Path,
                        help="Explicit path to authoritative sealed mesh.ply (defaults to <scan>/mesh.ply)")
    parser.add_argument("--cross-section-mesh-path", type=Path,
                        help="Sealed colored mesh for cross-section views (defaults to mesh-path)")
    parser.add_argument("--labels-path", type=Path,
                        help="Explicit path to triangle_class_labels.npy (defaults to <scan>/segmentation/triangle_class_labels.npy)")
    parser.add_argument("--structural-json", type=Path,
                        help="Explicit path to class_metadata.json (defaults to <scan>/segmentation/class_metadata.json)")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory (defaults to <scan>/mesh_elaboration_steps)")
    parser.add_argument("--scene-size-file", type=Path,
                        help="Optional scene_size.txt or scene_size.json consistency reference")
    parser.add_argument("--segmentation-output", type=Path,
                        help="Optional Stage 05 segmentation output directory for mesh object alignment")
    parser.add_argument("--skip-cross-sections", action="store_true",
                        help="Skip generating unobstructed cross-section views")
    parser.add_argument("--skip-texture", action="store_true",
                        help="Skip texture pattern synthesis on structural planes")
    parser.add_argument("--enhance-textures", action="store_true",
                        help="Opt in to SEESR-enhanced plane projections before texture refinement")
    parser.add_argument("--only-step", choices=("extract_structural_meshes", "fit_enclosing_walls", "floors",
                                                  "group_floor_instances_by_level", "add_ceilings_to_floor_levels",
                                                  "clean_floor_level_intermediates", "fit_floor_level_planes",
                                                  "aggregate_fit_ceiling_planes", "slice_scene_by_floor",
                                                  "clean_floor_scene_slices", "project_floor_slices_onto_planes",
                                                  "regularize_floor_projection_contours",
                                                  "fit_floor_projection_contours_to_black",
                                                  "overlay_fitted_floor_wall_contours",
                                                  "regularize_floor_level_patterns"),
                        help="Run Phase 06 Step 1, Step 2, complete Step 3, or one floor substep")

    args = parser.parse_args(argv)

    # Mode 1: Workspace discovery mode
    if args.workspace is not None:
        workspace = args.workspace
        try:
            scans = discover_scans(workspace)
        except Exception as exc:
            print(f"Error resolving scans in {workspace}: {exc}", file=sys.stderr)
            return 1

        overall_rc = 0
        for scan in scans:
            try:
                mesh_path = resolve_mesh(scan, args.mesh_path)
                labels_path = resolve_labels(scan, args.labels_path)
                structural_json = resolve_structural_metadata(scan, args.structural_json)
                seg_output = resolve_segmentation_output(scan, args.segmentation_output)
                scene_size = resolve_scene_size_file(scan, args.scene_size_file)
                output_dir = resolve_output_dir(scan, args.output_dir, subdir="mesh_elaboration_steps")
                cross_mesh = args.cross_section_mesh_path or mesh_path
            except Exception as exc:
                print(f"Error resolving inputs for scan {scan}: {exc}", file=sys.stderr)
                return 1

            print(f"\n=======================================================")
            print(f"Running Stage 06 Mesh Elaboration for: {scan.name}")
            print(f"  Mesh:         {mesh_path}")
            print(f"  Labels:       {labels_path}")
            print(f"  Metadata:     {structural_json}")
            print(f"  Segmentation: {seg_output}")
            print(f"  Output:       {output_dir}")
            print(f"=======================================================")

            t0 = time.perf_counter()
            rc = run_pipeline_for_scene(
                scene_mesh=mesh_path,
                labels_path=labels_path,
                structural_json=structural_json,
                output_dir=output_dir,
                cross_section_mesh=cross_mesh,
                segmentation_output=seg_output,
                scene_size_file=scene_size,
                skip_cross_sections=args.skip_cross_sections,
                skip_texture=args.skip_texture,
                enhance_textures=args.enhance_textures,
                only_step=args.only_step,
            )
            elapsed = time.perf_counter() - t0
            if update_pipeline_stats is not None:
                try:
                    update_pipeline_stats(
                        workspace=scan,
                        step_name="mesh_elaboration",
                        total_time=elapsed,
                        frames_processed=1,
                        extra_metrics={"output_dir": str(output_dir)},
                    )
                except Exception:
                    pass

            if rc != 0:
                print(f"Error: Step mesh elaboration failed for scan {scan.name} with exit code {rc}")
                return rc
        return 0

    # Mode 2: Explicit paths mode (strict requirement check for backward compatibility)
    missing = []
    if not args.mesh_path:
        missing.append("--mesh-path")
    if not args.labels_path:
        missing.append("--labels-path")
    if not args.structural_json:
        missing.append("--structural-json")
    if not args.output_dir:
        missing.append("--output-dir")
    if missing:
        parser.error(f"When --workspace is not specified, the following options are required: {', '.join(missing)}")

    return run_pipeline_for_scene(
        scene_mesh=args.mesh_path,
        labels_path=args.labels_path,
        structural_json=args.structural_json,
        output_dir=args.output_dir,
        cross_section_mesh=args.cross_section_mesh_path,
        segmentation_output=args.segmentation_output,
        scene_size_file=args.scene_size_file,
        skip_cross_sections=args.skip_cross_sections,
        skip_texture=args.skip_texture,
        enhance_textures=args.enhance_textures,
        only_step=args.only_step,
    )


if __name__ == "__main__":
    sys.exit(main())
