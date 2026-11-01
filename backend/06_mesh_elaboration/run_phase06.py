#!/usr/bin/env python3
"""Run all Phase 06 steps and publish the finalized structural PLY delivery."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

BACKEND_DIR = Path(__file__).resolve().parents[1]
STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(STAGE_DIR))

from Utilities.pipeline_paths import bootstrap

bootstrap()

from assemble_final_regularized_mesh import assemble_final_regularized_mesh
from finalize_single_elements import finalize_single_elements
from floor_opening_constraints import CACHE_NAME
from mesh_elaboration_inputs import (
    discover_scans,
    resolve_labels,
    resolve_mesh,
    resolve_scene,
    resolve_scene_size_file,
    resolve_segmentation_output,
    resolve_structural_metadata,
)
from step_mesh_elaboration import run_pipeline_for_scene


def _check_existing_delivery(destination: Path) -> None:
    """Only replace a prior delivery with the documented complete/elements layout."""
    if not destination.exists() and not destination.is_symlink():
        return
    if destination.is_symlink() or not destination.is_dir():
        raise FileExistsError(f"Refusing to replace an unmanaged Phase 06 output: {destination}")
    entries = {path.name for path in destination.iterdir()}
    elements = destination / "elements"
    if (entries != {"complete.ply", "elements"}
            or not (destination / "complete.ply").is_file()
            or (destination / "complete.ply").stat().st_size == 0
            or elements.is_symlink() or not elements.is_dir()):
        raise FileExistsError(f"Refusing to replace an unmanaged Phase 06 output: {destination}")
    element_paths = list(elements.iterdir())
    if not element_paths or any(not path.is_file() or path.suffix.lower() != ".ply"
                                for path in element_paths):
        raise FileExistsError(f"Refusing to replace an unmanaged Phase 06 output: {destination}")


def _publish_delivery(delivery: Path, destination: Path, backup: Path) -> None:
    had_delivery = destination.exists()
    if had_delivery:
        os.replace(destination, backup)
    try:
        os.replace(delivery, destination)
    except Exception:
        if had_delivery:
            os.replace(backup, destination)
        raise


def _publish_opening_cache(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _preserve_floor_projection_images(stage_output: Path, scan: Path, run_id: str) -> Path | None:
    source = stage_output / "intermediate_floor_levels"
    images = sorted(source.glob("*.png"))
    if not images:
        return None
    destination = scan / "mesh_elaboration_diagnostics" / "floor_projection" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    for image in images:
        shutil.copy2(image, destination / image.name)
    return destination


def run_phase06(workspace: Path) -> dict:
    scan = resolve_scene(workspace).resolve()
    mesh_path = resolve_mesh(scan)
    labels_path = resolve_labels(scan)
    structural_json = resolve_structural_metadata(scan)
    scene_size = resolve_scene_size_file(scan)
    if scene_size is None:
        raise FileNotFoundError(f"Phase 06 requires a scene-size file under {scan}")

    destination = scan / "mesh_elaboration"
    _check_existing_delivery(destination)

    with tempfile.TemporaryDirectory(prefix=".phase06_build_", dir=scan) as temp_name:
        temporary = Path(temp_name)
        run_id = temporary.name.removeprefix(".phase06_build_")
        staged_scan = temporary / "scan"
        staged_scan.mkdir()
        opening_cache = scan / CACHE_NAME
        staged_opening_cache = staged_scan / CACHE_NAME
        if opening_cache.is_file():
            shutil.copy2(opening_cache, staged_opening_cache)
        stage_output = staged_scan / "mesh_elaboration_steps"

        print("Steps 1–3: rebuilding structural instances, walls, floors, and ceilings")
        started = time.perf_counter()
        code = run_pipeline_for_scene(
            scene_mesh=mesh_path,
            labels_path=labels_path,
            structural_json=structural_json,
            output_dir=stage_output,
            segmentation_output=resolve_segmentation_output(scan),
            scene_size_file=scene_size,
        )
        print(f"Steps 1–3: {time.perf_counter() - started:.1f}s")
        # diagnostics = _preserve_floor_projection_images(
        #     stage_output, scan, run_id
        # )
        # if diagnostics is not None:
        #     print(f"Floor projection images -> {diagnostics}")
        if code:
            raise RuntimeError(f"Phase 06 Steps 1–3 failed with exit code {code}")

        started = time.perf_counter()
        finalize_single_elements(staged_scan)
        print(f"Step 4: finalize walls and ceilings: {time.perf_counter() - started:.1f}s")
        elements_dir = stage_output / "elements"
        started = time.perf_counter()
        assembly = assemble_final_regularized_mesh(
            staged_scan,
            elements_dir=elements_dir,
            # diagnostics_dir=(scan / "mesh_elaboration_diagnostics" / "slab_closure" / run_id),
        )
        print(f"Step 5: assemble final elements: {time.perf_counter() - started:.1f}s")

        topology = assembly["topology"]
        if topology["nonmanifold_edge_count"]:
            raise RuntimeError(f"Phase 06 assembly has non-manifold edges: {topology}")

        final_ply = Path(assembly["output_ply"])
        if not final_ply.is_file() or final_ply.stat().st_size == 0:
            raise OSError(f"Phase 06 did not produce a final PLY: {final_ply}")
        entries = list(elements_dir.iterdir())
        element_files = sorted(path for path in entries if path.is_file() and path.suffix.lower() == ".ply")
        if not element_files or len(element_files) != len(entries):
            raise OSError(f"Phase 06 element output is empty or contains non-PLY files: {elements_dir}")

        delivery = temporary / "delivery"
        delivery.mkdir()
        os.replace(final_ply, delivery / "complete.ply")
        os.replace(elements_dir, delivery / "elements")
        _publish_opening_cache(staged_opening_cache, opening_cache)
        _publish_delivery(delivery, destination, temporary / "previous_delivery")

    return {
        "workspace": str(scan),
        "output_dir": str(destination),
        "complete_ply": str(destination / "complete.ply"),
        "elements_dir": str(destination / "elements"),
        "element_ply_count": len(element_files),
        "topology": assembly["topology"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="Phase 04/05 scan directory or a workspace containing scan_N folders")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        scans = discover_scans(args.workspace)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Error: Could not run complete Phase 06: {error}", file=sys.stderr)
        print(f"Phase 06 total time: {time.perf_counter() - started:.1f}s")
        return 1

    failures = []
    for scan in scans:
        try:
            result = run_phase06(scan)
        except Exception as error:
            failures.append(scan.name)
            print(f"Error: Could not run Phase 06 for {scan.name}: {error}", file=sys.stderr)
            continue
        print(f"{scan.name} complete PLY -> {result['complete_ply']}")
        print(f"{scan.name} element PLYs -> {result['elements_dir']} ({result['element_ply_count']} files)")
        print(f"{scan.name} topology -> {result['topology']}")
    print(f"Phase 06 total time: {time.perf_counter() - started:.1f}s")
    if failures:
        print(f"Phase 06 failed for scans: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
