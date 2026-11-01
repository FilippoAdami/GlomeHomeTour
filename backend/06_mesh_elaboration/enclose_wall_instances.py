#!/usr/bin/env python3
"""Build a watertight side shell from the wall instances produced by extraction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble_wall_region_planes as assemble  # noqa: E402
import fit_wall_planes as legacy_fit  # noqa: E402
import fit_wall_region_planes as fit  # noqa: E402
import grow_wall_regions as grow  # noqa: E402


def _compatibility_manifest(fit_result: dict, growth_result: dict,
                            assembly_result: dict, output_path: Path) -> dict:
    axes = np.asarray(fit_result.get("orthogonal_horizontal_normal_directions", []), dtype=float)
    if axes.shape != (3, 3):
        raise ValueError("Fitted wall regions do not provide a horizontal basis")
    yaw = fit_result.get("horizontal_direction_yaw_radians")
    if yaw is None:
        raise ValueError("Fitted wall regions do not provide a dominant orientation")

    wall_planes = []
    region_areas = {int(region["region_id"]): float(region["area"])
                    for region in growth_result["regions"]}
    for index, panel in enumerate(assembly_result["planes"]):
        family = 0 if panel["axis"] == "Dir A" else 1
        regions = panel["trim_regions"]
        spans = [region["span_m"] for region in regions]
        heights = [region["bounds_y"] for region in regions]
        span_key = "span_w_m" if family == 0 else "span_u_m"
        span = [min(bounds[0] for bounds in spans), max(bounds[1] for bounds in spans)]
        bounds_y = [min(bounds[0] for bounds in heights), max(bounds[1] for bounds in heights)]
        enclosing_area = float(sum((s[1] - s[0]) * (h[1] - h[0])
                                   for s, h in zip(spans, heights)))
        support_area = float(sum(region_areas.get(int(region_id), 0.0)
                                 for region_id in panel["source_region_ids"]))
        wall_planes.append({
            "id": index,
            "name": panel["name"],
            "axis": panel["axis"],
            "family": family,
            "normal_unit": panel["normal"],
            "offset_m": panel["offset_m"],
            "plane_equation": [*panel["equation"]["normal"], panel["equation"]["constant"]],
            span_key: span,
            "bounds_y": bounds_y,
            "trim_regions": regions,
            "source_region_ids": panel["source_region_ids"],
            "support_area_m2": support_area,
            "enclosing_area_m2": enclosing_area,
            "inferred_connector": panel["inferred_connector"],
            "pattern_classification": panel.get("pattern_classification"),
            "color_split": panel.get("color_split"),
            "regularization": panel.get("regularization"),
        })

    topology = assembly_result["topology"]
    return {
        "algorithm": "region_grown_robust_plane_fit_and_closed_shell_assembly",
        "dominant_orientation_deg": round(float(np.degrees(yaw) % 90.0), 3),
        "orthogonal_basis": {
            "dir_A_east_west": axes[0].tolist(),
            "dir_B_north_south": axes[1].tolist(),
        },
        "wall_planes": wall_planes,
        "total_wall_planes_count": len(wall_planes),
        "semantic_category_color_samples_excluded": assembly_result.get(
            "semantic_category_color_samples_excluded", 0
        ),
        "mesh_supported_corner_extensions": assembly_result.get(
            "mesh_supported_corner_extensions", []
        ),
        "wall_vertical_bounds_m": [assembly_result["floor_y"], assembly_result["ceiling_y"]],
        "horizontal_topology": topology.get("final_topology", topology),
        "inferred_floor_closure_planes": topology.get("inferred_floor_closure_planes", []),
        "plane_constraints": {
            "vertical_bounds_m": [assembly_result["floor_y"], assembly_result["ceiling_y"]],
            "side_topology": "closed at every height; uncapped at floor and ceiling",
            "support_ownership": "each source triangle has at most one fitted-region owner",
        },
        "artifacts": {"enclosing_shell": str(output_path)},
    }


def _empty_manifest(output_path: Path, wall_paths: list[Path] | None = None) -> dict:
    vertical_bounds = None
    if wall_paths:
        vertices, _, _ = legacy_fit._load_meshes(wall_paths)
        vertical_bounds = [float(vertices[:, 1].min()), float(vertices[:, 1].max())]
    return {
        "algorithm": "region_grown_robust_plane_fit_and_closed_shell_assembly",
        "dominant_orientation_deg": 0.0,
        "orthogonal_basis": {
            "dir_A_east_west": [1.0, 0.0, 0.0],
            "dir_B_north_south": [0.0, 0.0, 1.0],
        },
        "wall_planes": [],
        "total_wall_planes_count": 0,
        "wall_vertical_bounds_m": vertical_bounds,
        "horizontal_topology": {"height_bands": [], "is_closed": False},
        "plane_constraints": {"side_topology": "no supported wall panels"},
        "artifacts": {"enclosing_shell": None},
    }


def _atomic_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                dir=destination.parent)
    os.close(fd)
    staged = Path(name)
    try:
        shutil.copyfile(source, staged)
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _atomic_json(data: dict, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                dir=destination.parent, text=True)
    staged = Path(name)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
    except Exception:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _save_failure_diagnostics(
    diagnostics_root: Path, regions_dir: Path, wall_paths: list[Path],
    stage: str, error: str, traceback_text: str, events: list[str],
) -> Path | None:
    try:
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="wall_enclosure_failure_", dir=diagnostics_root))
        input_dir = run_dir / "input_wall_meshes"
        input_dir.mkdir()
        saved_inputs = []
        for index, source in enumerate(wall_paths):
            if not source.is_file():
                continue
            saved_name = f"{index:03d}_{source.name}"
            shutil.copy2(source, input_dir / saved_name)
            saved_inputs.append({"source_path": str(source), "saved_path": f"input_wall_meshes/{saved_name}"})

        copied = []
        for name in (
            "wall_regions_colored.ply", "wall_regions_triangle_labels.npy",
            "wall_regions_semantic_category_ids.npy",
            "wall_regions_manifest.json", "wall_region_planes_manifest.json",
            "wall_regions_with_planes.ply", "wall_region_planes.glb",
            "wall_loop_topology_diagnostic.json",
        ):
            source = regions_dir / name
            if source.is_file():
                shutil.copy2(source, run_dir / name)
                copied.append(name)

        log_text = "\n".join([*events, f"FAILURE stage={stage}: {error}", traceback_text]).rstrip() + "\n"
        (run_dir / "wall_enclosure.log").write_text(log_text)
        (run_dir / "failure.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "error": error,
            "traceback": traceback_text,
            "diagnostic_directory": str(run_dir),
            "source_wall_meshes": saved_inputs,
            "copied_intermediates": copied,
            "checkpoints": events,
        }, indent=2) + "\n")
        return run_dir
    except Exception as diagnostic_error:
        print(f"[wall enclosure] Could not preserve failure diagnostics: {diagnostic_error}",
              file=sys.stderr)
        return None


def build_wall_shell(instances_dir: Path, output_path: Path,
                     manifest_path: Path | None = None,
                     diagnostics_dir: Path | None = None) -> dict:
    instances_dir = instances_dir.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve() if manifest_path else None
    diagnostics_root = (diagnostics_dir or instances_dir / "wall_enclosure_diagnostics").resolve()
    if not instances_dir.is_dir():
        raise ValueError(f"Structural instances directory does not exist: {instances_dir}")
    if manifest_path == output_path:
        raise ValueError("Shell PLY and wall manifest must have different output paths")
    wall_paths: list[Path] = []
    staged_shell: Path | None = None
    try:
        wall_paths = grow._wall_mesh_paths(instances_dir)
    except ValueError as error:
        if "No wall meshes listed" not in str(error):
            raise
    input_manifest = json.loads((instances_dir / "structural_instances_manifest.json").read_text())
    input_names = {Path(item.get("file_path", item.get("file_name", ""))).name
                   for item in input_manifest.get("instances", [])}
    if output_path.name in input_names or output_path == (instances_dir / "structural_instances_manifest.json").resolve():
        raise ValueError(f"Output path would overwrite an input: {output_path}")
    if manifest_path and (manifest_path.name in input_names
                          or manifest_path == (instances_dir / "structural_instances_manifest.json").resolve()):
        raise ValueError(f"Manifest path would overwrite an input: {manifest_path}")
    if output_path.name in legacy_fit.WALL_FIT_OUTPUTS:
        raise ValueError(f"Choose a new shell path instead of a legacy wall-fit artifact: {output_path}")
    if manifest_path and manifest_path.name in legacy_fit.WALL_FIT_OUTPUTS \
            and manifest_path.name != "wall_planes_manifest.json":
        raise ValueError(f"Manifest path conflicts with a legacy wall-fit artifact: {manifest_path}")

    shell_source = None
    if not wall_paths:
        result = _empty_manifest(output_path)
    else:
        events: list[str] = []

        def checkpoint(message: str) -> None:
            line = f"{datetime.now(timezone.utc).isoformat()} {message}"
            events.append(line)
            print(f"[wall enclosure] {message}", flush=True)

        with tempfile.TemporaryDirectory(prefix=".wall_shell_", dir=instances_dir) as temp:
            regions_dir = Path(temp)
            regions_mesh = regions_dir / "wall_regions_colored.ply"
            labels_path = regions_dir / "wall_regions_triangle_labels.npy"
            region_manifest_path = regions_dir / "wall_regions_manifest.json"
            checkpoint(f"inputs: {len(wall_paths)} wall mesh(es); diagnostics root {diagnostics_root}")
            checkpoint("region growth started")
            try:
                growth_result = grow._write_colored_regions(
                    wall_paths, regions_mesh, labels_path,
                    grow.DEFAULT_SEED_COUNT, grow.DEFAULT_RANDOM_SEED,
                )
                region_manifest_path.write_text(json.dumps(growth_result, indent=2) + "\n")
            except Exception as error:
                checkpoint(f"region growth failed: {error}")
                saved = _save_failure_diagnostics(
                    diagnostics_root, regions_dir, wall_paths, "region_growth", str(error),
                    traceback.format_exc(), events,
                )
                if saved:
                    print(f"[wall enclosure] diagnostics saved to {saved}", flush=True)
                raise
            checkpoint(
                f"region growth complete: {growth_result['triangle_count']} triangles, "
                f"{growth_result['region_count']} regions, "
                f"{growth_result['degenerate_triangle_count']} degenerate triangles"
            )

            checkpoint("plane fitting started")
            try:
                _, _, fit_result = fit._process(regions_dir)
            except Exception as error:
                checkpoint(f"plane fitting failed: {error}")
                saved = _save_failure_diagnostics(
                    diagnostics_root, regions_dir, wall_paths, "plane_fitting", str(error),
                    traceback.format_exc(), events,
                )
                if saved:
                    print(f"[wall enclosure] diagnostics saved to {saved}", flush=True)
                raise
            checkpoint(
                f"plane fitting complete: {fit_result['plane_count']} planes, "
                f"{fit_result.get('unassigned_triangle_count', 0)} unassigned triangles"
            )

            if fit_result["plane_count"] == 0:
                checkpoint("no fitted planes; preserving available region-stage artifacts")
                saved = _save_failure_diagnostics(
                    diagnostics_root, regions_dir, wall_paths, "plane_fitting_no_planes",
                    "Plane fitting produced no supported wall planes", "", events,
                )
                if saved:
                    print(f"[wall enclosure] diagnostics saved to {saved}", flush=True)
                result = _empty_manifest(output_path, wall_paths)
            else:
                checkpoint("wall-shell assembly started")
                failure_stage = "wall_shell_assembly"
                try:
                    _, _, assembly_result = assemble._assemble(
                        regions_dir, diagnostics_path=regions_dir / "wall_loop_topology_diagnostic.json"
                    )
                    checkpoint(
                        f"wall-shell assembly complete: {len(assembly_result['planes'])} panels"
                    )
                    failure_stage = "wall_manifest_generation"
                    result = _compatibility_manifest(
                        fit_result, growth_result, assembly_result, output_path
                    )
                    shell_source = Path(assembly_result["shell_ply"])
                except ValueError as error:
                    checkpoint(f"{failure_stage.replace('_', ' ')} failed: {error}")
                    saved = _save_failure_diagnostics(
                        diagnostics_root, regions_dir, wall_paths, failure_stage, str(error),
                        traceback.format_exc(), events,
                    )
                    if saved:
                        print(f"[wall enclosure] diagnostics saved to {saved}", flush=True)
                    if "Plane manifest contains no supported wall planes" not in str(error):
                        raise
                    result = _empty_manifest(output_path, wall_paths)
                except Exception as error:
                    checkpoint(f"{failure_stage.replace('_', ' ')} failed: {error}")
                    saved = _save_failure_diagnostics(
                        diagnostics_root, regions_dir, wall_paths, failure_stage, str(error),
                        traceback.format_exc(), events,
                    )
                    if saved:
                        print(f"[wall enclosure] diagnostics saved to {saved}", flush=True)
                    raise
            if shell_source is not None:
                staged_shell = _atomic_copy(shell_source, output_path)

    result["shell_written"] = shell_source is not None
    staged_manifest: Path | None = None
    try:
        if manifest_path:
            staged_manifest = _atomic_json(result, manifest_path)
        legacy_fit._clear_wall_fit_outputs(instances_dir, wall_paths)
        if staged_shell:
            os.replace(staged_shell, output_path)
        else:
            output_path.unlink(missing_ok=True)
        if staged_manifest:
            os.replace(staged_manifest, manifest_path)
    finally:
        if staged_shell:
            staged_shell.unlink(missing_ok=True)
        if staged_manifest:
            staged_manifest.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-dir", type=Path, required=True,
                        help="Extraction directory with structural_instances_manifest.json and wall/floor/ceiling instances")
    parser.add_argument("--output", type=Path,
                        help="Final shell PLY (defaults to <instances-dir>/wall_enclosing_shell.ply)")
    parser.add_argument("--manifest-out", type=Path,
                        help="Optional wall-plane metadata for the later floor/ceiling footprint steps")
    parser.add_argument("--diagnostics-dir", type=Path,
                        help="Persistent directory for failure logs and intermediate wall artifacts")
    args = parser.parse_args(argv)
    output_path = args.output or args.instances_dir / "wall_enclosing_shell.ply"
    try:
        result = build_wall_shell(args.instances_dir, output_path, args.manifest_out,
                                  args.diagnostics_dir)
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if result["shell_written"]:
        print(f"Wrote enclosing wall shell: {output_path}")
    else:
        print("No supported wall shell was produced; the previous shell was removed.")
    if args.manifest_out:
        print(f"Wall metadata: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
