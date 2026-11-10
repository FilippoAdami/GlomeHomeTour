#!/usr/bin/env python3
"""Run Phase 08's first macro step: generate floor plans and refine instance categories."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from collections import Counter
from pathlib import Path
from typing import TypeVar

from assign_floor_level_metadata import DEFAULT_ALIGNMENT_DIR, build_floor_level_metadata
from check_instance_floor_distance import check_instance_floor_distances
from check_instance_room_consistency import check_instance_room_consistency
from generate_classified_object_floor_slices import generate_classified_object_slices
from generate_floor_level_meshes import generate_floor_level_meshes
from generate_floor_level_projections import generate_floor_level_projections
from generate_floorplan_measurements import generate_floorplan_measurements
from generate_floorplan_instance_overlays import generate_floorplan_instance_overlays


DEFAULT_RESOLUTION = 1024
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _run_timed_step(
    name: str,
    operation: Callable[[], T],
    timings: list[dict],
    macro_started: float,
) -> T:
    started = time.perf_counter()
    LOGGER.info("Starting substep: %s", name)
    try:
        result = operation()
    except Exception:
        elapsed = time.perf_counter() - started
        timings.append({"step": name, "status": "failed", "elapsed_seconds": round(elapsed, 3)})
        LOGGER.exception(
            "Substep %s failed after %.2f s; macro elapsed %.2f s",
            name, elapsed, time.perf_counter() - macro_started,
        )
        raise
    elapsed = time.perf_counter() - started
    timings.append({"step": name, "status": "complete", "elapsed_seconds": round(elapsed, 3)})
    LOGGER.info("Completed substep: %s in %.2f s", name, elapsed)
    return result


def generate_floor_plans(
    alignment_dir: Path = DEFAULT_ALIGNMENT_DIR,
    resolution: int = DEFAULT_RESOLUTION,
) -> dict:
    alignment_dir = Path(alignment_dir).resolve()
    if resolution < 64:
        raise ValueError("Projection resolution must be at least 64 pixels")
    macro_started = time.perf_counter()
    timings = []
    scene_dir = alignment_dir.parent
    artifacts_dir = scene_dir / "artifacts" / "Generate Floor Plans"
    mesh_dir = artifacts_dir / "floor_level_meshes"
    macro_manifest_path = artifacts_dir / "generate_floor_plans_manifest.json"
    _write_json(macro_manifest_path, {
        "schema_version": 1,
        "phase_status": "in_progress",
        "macro_step": "Generate floor plans",
        "macro_step_status": "running",
        "alignment_dir": str(alignment_dir),
        "resolution_px": resolution,
        "timings": timings,
    })

    metadata_path = artifacts_dir / "floor_level_metadata.json"
    _run_timed_step(
        "floor_level_metadata",
        lambda: _write_json(metadata_path, build_floor_level_metadata(alignment_dir)),
        timings,
        macro_started,
    )
    floor_mesh_manifest = _run_timed_step(
        "floor_level_meshes",
        lambda: generate_floor_level_meshes(alignment_dir, metadata_path, mesh_dir),
        timings,
        macro_started,
    )
    projection_manifest = _run_timed_step(
        "floor_level_projections",
        lambda: generate_floor_level_projections(
            mesh_dir,
            artifacts_dir / "floor_level_projections",
            resolution,
        ),
        timings,
        macro_started,
    )
    classified_slice_manifest = _run_timed_step(
        "classified_object_floor_slices",
        lambda: generate_classified_object_slices(
            mesh_dir,
            artifacts_dir / "classified_object_slice_experiment",
            resolution,
        ),
        timings,
        macro_started,
    )
    floor_check_manifest = _run_timed_step(
        "instance_floor_distance_check",
        lambda: check_instance_floor_distances(
            scene_dir,
            artifacts_dir / "instance_floor_distance_check",
            artifacts_dir=artifacts_dir,
        ),
        timings,
        macro_started,
    )
    room_check_manifest = _run_timed_step(
        "room_category_consistency",
        lambda: check_instance_room_consistency(
            scene_dir,
            artifacts_dir / "room_category_consistency",
            artifacts_dir=artifacts_dir,
        ),
        timings,
        macro_started,
    )
    instance_overlay_manifest = _run_timed_step(
        "floorplan_instance_overlays",
        lambda: generate_floorplan_instance_overlays(artifacts_dir, artifacts_dir / "floorplan_instance_overlays"),
        timings,
        macro_started,
    )
    measurement_manifest = _run_timed_step(
        "floorplan_measurements",
        lambda: generate_floorplan_measurements(artifacts_dir, overlay_manifest=instance_overlay_manifest),
        timings,
        macro_started,
    )

    final_associations = room_check_manifest["associations"]
    result = {
        "schema_version": 1,
        "phase_status": "in_progress",
        "macro_step": "Generate floor plans",
        "macro_step_status": "complete",
        "alignment_dir": str(alignment_dir),
        "resolution_px": resolution,
        "floor_level_metadata": str(metadata_path),
        "timings": timings,
        "total_duration_seconds": round(time.perf_counter() - macro_started, 3),
        "floor_level_mesh_manifest": str(mesh_dir / "floor_level_meshes_manifest.json"),
        "floor_level_projection_manifest": str(
            artifacts_dir / "floor_level_projections" / "floor_level_projections_manifest.json"
        ),
        "classified_object_floor_slices_manifest": str(
            artifacts_dir / "classified_object_slice_experiment" / "classified_object_floor_slices_manifest.json"
        ),
        "instance_floor_distance_manifest": str(
            artifacts_dir / "instance_floor_distance_check" / "instance_floor_distance_manifest.json"
        ),
        "room_category_consistency_manifest": str(
            artifacts_dir / "room_category_consistency" / "room_category_consistency_manifest.json"
        ),
        "final_instance_associations": str(
            artifacts_dir / "room_category_consistency" / "associations_after_room_check.json"
        ),
        "final_instance_report_dir": str(
            artifacts_dir / "room_category_consistency" / "instance_reports"
        ),
        "floorplan_instance_overlays_manifest": str(
            artifacts_dir / "floorplan_instance_overlays" / "floorplan_instance_overlays_manifest.json"
        ),
        "floorplan_measurements_manifest": str(
            artifacts_dir / "floorplan_measurements" / "floorplan_measurements_manifest.json"
        ),
        "counts": {
            "floor_levels": len(floor_mesh_manifest["levels"]),
            "floor_plan_images": len(projection_manifest["projections"]),
            "classified_object_slice_assignments": sum(
                len(scan.get("objects", []))
                for level in classified_slice_manifest["levels"]
                for scan in level.get("scans", [])
            ),
            "floor_checked_instances": room_check_manifest["floor_checked_instance_count"],
            "final_instance_reports": room_check_manifest["final_instance_reports_written"],
            "final_assigned_instances": sum(item.get("category_id") is not None for item in final_associations),
            "final_unassigned_instances": sum(item.get("category_id") is None for item in final_associations),
            "floorplan_instance_overlay_views": len(instance_overlay_manifest["floor_levels"]),
            "floorplan_instance_overlay_instances": sum(
                level["displayed_assigned_floor_count"]
                + level["displayed_visible_lower_floor_count"]
                for level in instance_overlay_manifest["floor_levels"]
            ),
            "floorplan_instance_overlay_panels": sum(
                level["phase07_panel_count"] for level in instance_overlay_manifest["floor_levels"]
            ),
            "floorplan_instance_overlay_passages": sum(
                level["door_passage_count"] for level in instance_overlay_manifest["floor_levels"]
            ),
            "floorplan_instance_overlay_symbols": sum(
                level["displayed_assigned_floor_count"]
                + level["displayed_visible_lower_floor_count"]
                + level["phase07_panel_count"]
                for level in instance_overlay_manifest["floor_levels"]
            ),
            "floorplan_measurement_overlays": measurement_manifest["measurement_overlay_count"],
            "floorplan_unique_wall_segments": sum(
                level["wall_segment_count"] for level in measurement_manifest["floor_levels"]
            ),
            "floorplan_wall_segment_measurements": measurement_manifest["wall_segment_annotation_count"],
            "floorplan_between_scan_internal_segments": measurement_manifest[
                "between_scan_segment_count"
            ],
            "final_categories": dict(sorted(Counter(
                item["category"] for item in final_associations if item.get("category") is not None
            ).items())),
            "floor_distance_rejected_instances": floor_check_manifest["rejected_association_count"],
        },
    }
    _write_json(macro_manifest_path, result)
    LOGGER.info("Generate floor plans completed in %.2f s", time.perf_counter() - macro_started)
    return result


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR,
                        help="Phase 07 output directory; its parent is the scene root.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                        help="Top-down PNG resolution (default: 1024 pixels square).")
    args = parser.parse_args()
    result = generate_floor_plans(args.alignment_dir, args.resolution)
    counts = result["counts"]
    print(
        f"{result['macro_step']}: {counts['floor_levels']} floor levels, "
        f"{counts['floor_plan_images']} images, {counts['final_instance_reports']} "
        f"final instance reports, {counts['final_assigned_instances']} assigned and "
        f"{counts['final_unassigned_instances']} unassigned, "
        f"{counts['floorplan_instance_overlay_symbols']} floorplan symbols "
        f"({counts['floorplan_instance_overlay_panels']} Phase 07 panels), "
        f"{counts['floorplan_wall_segment_measurements']} wall-segment annotations across "
        f"{counts['floorplan_measurement_overlays']} measurement overlays "
        f"({counts['floorplan_unique_wall_segments']} unique wall segments, "
        f"{counts['floorplan_between_scan_internal_segments']} between-scan segments forced inside). "
        f"Total: {result['total_duration_seconds']:.2f} s. Phase 08 remains in progress."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
