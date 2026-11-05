#!/usr/bin/env python3
"""Stage 07 — Unified Alignment Pipeline Orchestrator.

Sequentially executes:
  1. Multi-Room Scan Alignment: Registers all room scans into a global coordinate system using
     ARCore path tracking, trajectory-guided door pairing, coaxial portal clearance, and all-plane
     coplanar snapping (flush floor levels and collinear wall facades).
  2. Global Asset Synchronization: Applies each scan's solved SE(3) transformation matrix to its
     object instances, sealed meshes, regularized meshes, and Gaussian Splats.
  3. Object Contour Fitting: Detects filtered structural-contact contours, moves touching groups
     inside each regularized shell, then performs bounded plane and per-wall median pulls.
  4. Outlier Replacement: Applies the locked area/depth/seam/rectangle cutoffs, classifies each
     retained region, removes it from object and sealed meshes, and inserts silver panels.

CLI Usage:
  backend/.venv/bin/python backend/07_alignment/step_alignment.py --workspace backend/current_scene
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scan_loader import REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2, ScanDataset, ScanData
from door_alignment_engine import DoorAlignmentEngine, DoorPairing
from coplanar_planes_engine import CoplanarPlanesEngine
from multi_asset_synchronizer import MultiAssetSynchronizer
from single_scan_synchronizer import apply_transformation_to_single_scan
from object_outlier_regions import create_final_red_objects
from step_replace_red_regions_with_panels import run as replace_final_red_regions

TEMPORARY_PIPELINE_DIR = CURRENT_DIR / "temporary_alternative_pipeline"
if str(TEMPORARY_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(TEMPORARY_PIPELINE_DIR))

from detect_outside_contact_contours import run as detect_object_contours
from fit_contours_inside_regularized_mesh import run as fit_contours_inside_shell
from fit_contours_near_regularized_planes import run as fit_contours_near_planes


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("phase07_alignment")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def _timed_call(logger: logging.Logger, timings: dict[str, float], name: str, operation):
    started = time.perf_counter()
    try:
        return operation()
    finally:
        elapsed = time.perf_counter() - started
        timings[name] = elapsed
        logger.info("Timing %-48s %.2f s", name, elapsed)


def cleanup_temporary_outputs(output_dir: Path) -> None:
    """Remove contour work products and reports after final objects are published."""
    for name in ("outside_contact_contours", "contour_inside_fit", "nearest_plane_fit"):
        shutil.rmtree(output_dir / name, ignore_errors=True)

    stats_dir = output_dir / "stats"
    for name in (
        "aligned_objects_manifest.json",
        "multi_room_alignment_report.json",
        "aligned_scene_all_planes_diagnostic.png",
    ):
        (stats_dir / name).unlink(missing_ok=True)
    if stats_dir.is_dir():
        for report in stats_dir.glob("*_plane_snap_report.json"):
            report.unlink(missing_ok=True)


def run_phase07_alignment(
    workspace: Path,
    output_dir: Optional[Path] = None,
    skip_scans: bool = False,
    coplanar_threshold: float = 0.35,
    keep_temporary_outputs: bool = False,
) -> Dict[str, Any]:
    """Executes the complete, scene-agnostic Phase 07 alignment pipeline."""
    logger = setup_logger()
    total_start = time.perf_counter()
    substep_timings: dict[str, float] = {}

    def timed(name: str, operation):
        return _timed_call(logger, substep_timings, name, operation)

    workspace = Path(workspace).resolve()

    if output_dir is None:
        output_dir = workspace / "multi_room_alignment"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 75)
    logger.info("PHASE 07: UNIFIED MULTI-ROOM & OBJECT ALIGNMENT PIPELINE")
    logger.info("=" * 75)
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Deliverables Output Directory: {output_dir}")

    # =========================================================================
    # STEP 1: Multi-Room Scan Alignment
    # =========================================================================
    step1_start = time.perf_counter()
    dataset = timed("step1.scan_dataset_discovery", lambda: ScanDataset(workspace))
    scans = dataset.scans
    scan_keys = sorted(list(scans.keys()))

    door_pairings: List[DoorPairing] = []
    door_metrics: List[Dict[str, Any]] = []
    coplanar_report: Dict[str, Any] = {}

    if skip_scans:
        logger.info("\n[Step 1/5] Skipping scan alignment (--skip-scans specified). Loading existing transforms...")
        existing_tf_path = output_dir / "stats" / "aligned_scans_transforms.json"
        if not existing_tf_path.is_file():
            existing_tf_path = output_dir / "aligned_scans_transforms.json"
        if not existing_tf_path.is_file():
            existing_tf_path = workspace / "aligned_scans_transforms.json"
        if existing_tf_path.is_file():
            def read_existing_transforms():
                with open(existing_tf_path, "r") as f:
                    tf_data = json.load(f)
                return {
                    sname: np.array(sinfo["transform_matrix"], dtype=np.float64)
                    for sname, sinfo in tf_data.get("scans", {}).items()
                    if sname in scans
                }
            transforms_final = timed("step1.load_existing_transforms", read_existing_transforms)
        else:
            logger.warning("Existing transform manifest not found. Using identity transforms.")
            transforms_final = {sname: np.eye(4, dtype=np.float64) for sname in scan_keys}
    elif len(scans) < 2:
        logger.info(f"\n[Step 1/5] Workspace has {len(scans)} scan(s). Using identity transformation.")
        transforms_final = {sname: np.eye(4, dtype=np.float64) for sname in scan_keys}
        coplanar_report = {"status": "single_scan_identity", "message": "Multi-room alignment trivial for 1 scan."}
    else:
        logger.info(f"\n[Step 1/5] Registering {len(scans)} rooms into global coordinate frame...")

        # 1.1 Door pairing via ARCore transit paths
        logger.info("  [1.1] Trajectory-Guided Doorway Pairing via ARCore transit paths...")
        door_engine = DoorAlignmentEngine(max_overlap_pct=2.0)
        door_pairings = timed(
            "step1.1.door_pairing", lambda: door_engine.pair_doors_via_trajectories(scans)
        )
        logger.info(f"  Discovered {len(door_pairings)} inter-room door correspondence(s).")

        # 1.2 Coaxial portal alignment & non-penetrating wall clearance
        logger.info("  [1.2] Coaxial Portal Registration & Dynamic Wall Clearance...")
        transforms_doors, door_metrics = timed(
            "step1.2.portal_alignment",
            lambda: door_engine.solve_coaxial_alignment(scans, door_pairings),
        )

        # 1.3 All-plane coplanar alignment (horizontal floors & vertical walls)
        logger.info(f"  [1.3] All-Plane Coplanar Alignment (Floor levels & facade collinearity, max_dist={coplanar_threshold*100:.1f}cm, zero overlap)...")
        coplanar_engine = CoplanarPlanesEngine(
            max_distance_m=coplanar_threshold,
            max_angle_deg=5.0,
        )
        all_facets = timed(
            "step1.3.extract_scan_planes",
            lambda: coplanar_engine.extract_all_scan_planes(scans, transforms_doors),
        )
        candidates = timed(
            "step1.4.discover_coplanar_candidates",
            lambda: coplanar_engine.discover_coplanar_candidates(all_facets),
        )
        for c in candidates:
            logger.info(f"    Candidate: {c.description} | Gap: {c.distance_before_cm:.1f} cm")

        transforms_final, coplanar_report = timed(
            "step1.5.optimize_all_planes",
            lambda: coplanar_engine.optimize_all_planes(
                scans, transforms_doors, candidates, door_pairings=door_pairings, all_scan_planes=all_facets
            ),
        )

    if len(scans) > 1:
        overlap_metrics = timed(
            "step1.6.validate_floor_overlaps",
            lambda: CoplanarPlanesEngine._calculate_all_overlaps(scans, transforms_final),
        )
        if any(
            row["overlap_area_m2"] > REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2
            for row in overlap_metrics.values()
        ):
            raise RuntimeError("Final regularized floor footprints overlap; refusing to export Phase 07 alignment.")
        coplanar_report["mesh_overlaps"] = overlap_metrics
        coplanar_report["overlap_basis"] = "exact regularized floor footprints; positive-area overlap is rejected"

    step1_elapsed = time.perf_counter() - step1_start
    logger.info("Completed macro step 1/5 (scan alignment) in %.2f s", step1_elapsed)

    # =========================================================================
    # STEP 2: Global Asset Synchronization & Single-Scan Transformation
    # =========================================================================
    step2_start = time.perf_counter()
    logger.info("\n[Step 2/5] Synchronizing deliverables & transforming individual scan assets...")

    # These are generated directories. Rebuild them so removed source instances or
    # scan assets cannot survive from an earlier run.
    def clear_previous_outputs():
        for name in (
            "objects",
            "scans",
            "outside_contact_contours",
            "contour_inside_fit",
            "nearest_plane_fit",
        ):
            shutil.rmtree(output_dir / name, ignore_errors=True)
    timed("step2.clear_generated_outputs", clear_previous_outputs)

    # 2.1 Multi-asset combined models and manifests
    synchronizer = MultiAssetSynchronizer(output_dir)
    full_report_metrics = {
        "door_pairings": door_metrics,
        "coplanar_alignment": coplanar_report,
    }
    exported_assets = timed(
        "step2.multi_asset_synchronization",
        lambda: synchronizer.synchronize_and_export_all(scans, transforms_final, full_report_metrics),
    )
    substep_timings.update({f"step2.{name}": value for name, value in synchronizer.timings_seconds.items()})

    # 2.2 Synchronize individual single scans (mesh.ply, gaussian_splats, complete.ply, ARCore & COLMAP poses)
    logger.info("\n  [2.2] Transforming single scan PLYs, Gaussian splats, and camera poses (ARCore + COLMAP)...")
    for sname, scan in sorted(scans.items()):
        T = transforms_final[sname]
        res = timed(
            f"step2.single_scan_sync.{sname}",
            lambda: apply_transformation_to_single_scan(
                scan_dir=scan.info.scan_dir,
                transform=T,
                export_dir=output_dir / "scans" / sname,
                gaussian_mesh_scenes=synchronizer.gaussian_overlap_scenes,
                pretransformed_gaussian_splats=synchronizer.transformed_gaussian_splat_paths.get(sname),
            ),
        )
        logger.info(
            f"    Synchronized {sname}: {res['updated_assets_count']} assets aligned "
            f"(is_identity={res['is_identity']})"
        )

    step2_elapsed = time.perf_counter() - step2_start
    logger.info("Completed macro step 2/5 (asset synchronization) in %.2f s", step2_elapsed)

    # =========================================================================
    # STEP 3: Contact Contours & Object Fitting
    # =========================================================================
    step3_start = time.perf_counter()
    logger.info("\n[Step 3/5] Detecting contact contours and fitting aligned objects...")
    contour_report = timed(
        "step3.detect_contact_contours", lambda: detect_object_contours(output_dir, workspace)
    )
    inside_report = timed("step3.fit_inside_shell", lambda: fit_contours_inside_shell(output_dir))
    plane_report = timed(
        "step3.fit_near_regularized_planes",
        lambda: fit_contours_near_planes(output_dir, outside_fraction_limit=0.05),
    )
    published = timed(
        "step3.publish_object_placement_translations",
        lambda: synchronizer.apply_object_placement_translations(inside_report, plane_report),
    )
    exported_assets.update(published)
    step3_elapsed = time.perf_counter() - step3_start
    logger.info("Completed macro step 3/5 (object fitting) in %.2f s", step3_elapsed)

    step4_start = time.perf_counter()
    logger.info("\n[Step 4/5] Detecting and exporting final red outlier regions...")
    red_objects_path, red_summary = timed(
        "step4.detect_and_export_red_regions",
        lambda: create_final_red_objects(scans, transforms_final, output_dir),
    )
    substep_timings.update({
        f"step4.{name}": value for name, value in red_summary.pop("timings_seconds").items()
    })
    exported_assets["aligned_objects_final_red"] = red_objects_path
    logger.info(
        "    Published %d fitted objects; %d contour triangles, %d median pulls; original colors restored.",
        plane_report["object_count"], contour_report["contact_contour_triangle_count"],
        plane_report["objects_raised_to_wall_median"],
    )
    logger.info(
        "    Final red object PLY: %d area/depth patches -> %d contiguous groups -> %d red triangles (%s).",
        red_summary["first_cutoff_regions"], red_summary["contiguous_groups"],
        red_summary["red_triangles"], red_objects_path,
    )
    step4_elapsed = time.perf_counter() - step4_start
    logger.info("Completed macro step 4/5 (outlier detection) in %.2f s", step4_elapsed)

    step5_start = time.perf_counter()
    logger.info("\n[Step 5/5] Classifying retained regions and replacing them with silver panels...")
    replacement_summary = timed(
        "step5.panel_replacement", lambda: replace_final_red_regions(workspace, output_dir)
    )
    substep_timings.update({
        f"step5.{name}": value for name, value in replacement_summary["timings_seconds"].items()
    })
    exported_assets.pop("aligned_objects_final_red", None)
    exported_assets["aligned_objects"] = output_dir / "aligned_objects.ply"
    exported_assets["aligned_sealed_meshes"] = output_dir / "aligned_sealed_meshes.ply"
    exported_assets["aligned_sealed_meshes_glb"] = output_dir / "aligned_sealed_meshes.glb"
    exported_assets["phase07_region_replacements_manifest"] = (
        output_dir / "stats" / "phase07_region_replacements.json")
    exported_assets["phase07_panel_instances_dir"] = output_dir / "objects"
    logger.info("    Replaced %d retained regions; regularized meshes were left untouched.",
                len(replacement_summary["regions"]))

    if not keep_temporary_outputs:
        timed("step5.cleanup_temporary_outputs", lambda: cleanup_temporary_outputs(output_dir))
        exported_assets = {
            "aligned_objects": output_dir / "aligned_objects.ply",
            "objects_dir": output_dir / "objects",
            "aligned_regularized_meshes": output_dir / "aligned_regularized_meshes.ply",
            "aligned_sealed_gaussians": output_dir / "aligned_sealed_gaussians.ply",
            "aligned_sealed_meshes": output_dir / "aligned_sealed_meshes.ply",
            "aligned_sealed_meshes_glb": output_dir / "aligned_sealed_meshes.glb",
            "phase07_region_replacements_manifest": output_dir / "stats" / "phase07_region_replacements.json",
            "phase07_panel_instances_dir": output_dir / "objects",
            "scans_dir": output_dir / "scans",
            "aligned_scans_transforms": output_dir / "stats" / "aligned_scans_transforms.json",
        }
    step5_elapsed = time.perf_counter() - step5_start
    logger.info("Completed macro step 5/5 (panel replacement and cleanup) in %.2f s", step5_elapsed)

    logger.info("Exported Deliverables:")
    for asset_name, path in exported_assets.items():
        logger.info(f"  {asset_name:<35}: {path}")
    total_elapsed = time.perf_counter() - total_start
    macro_timings = {
        "step1_scans_seconds": step1_elapsed,
        "step2_sync_seconds": step2_elapsed,
        "step3_objects_seconds": step3_elapsed,
        "step4_red_outliers_seconds": step4_elapsed,
        "step5_panel_replacement_seconds": step5_elapsed,
    }
    timing_report_path = output_dir / "stats" / "phase07_timing_report.json"
    timing_report_path.parent.mkdir(parents=True, exist_ok=True)
    timing_report_path.write_text(json.dumps({
        "total_seconds": total_elapsed,
        "macro_steps_seconds": macro_timings,
        "substeps_seconds": substep_timings,
    }, indent=2) + "\n")
    exported_assets["phase07_timing_report"] = timing_report_path
    logger.info("=" * 75)
    logger.info("PHASE 07 COMPLETED IN %.2f s", total_elapsed)
    logger.info("Timing report: %s", timing_report_path)
    logger.info("=" * 75)

    return {
        "status": "success",
        "total_elapsed_seconds": total_elapsed,
        "timings": macro_timings,
        "substep_timings_seconds": substep_timings,
        "exported_assets": {k: str(v) for k, v in exported_assets.items()},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Unified Phase 07 Alignment Pipeline (Objects + Multi-Room Scans).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("backend/current_scene"),
        help="Workspace directory containing scan subdirectories (default: backend/current_scene)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for aligned property models (default: <workspace>/multi_room_alignment)",
    )
    parser.add_argument(
        "--skip-scans",
        action="store_true",
        help="Skip multi-room scan alignment and reuse existing transformation manifest",
    )
    parser.add_argument(
        "--coplanar-threshold",
        type=float,
        default=0.35,
        help="Maximum distance in meters to discover candidate coplanar walls/floors (default: 0.35)",
    )

    args = parser.parse_args()
    result = run_phase07_alignment(
        workspace=args.workspace,
        output_dir=args.output_dir,
        skip_scans=args.skip_scans,
        coplanar_threshold=args.coplanar_threshold,
    )
    if result.get("status") == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
