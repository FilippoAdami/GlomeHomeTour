#!/usr/bin/env python3
"""Run Phase 07 with object instances transformed alongside their sealed meshes."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
ALIGNMENT_DIR = CURRENT_DIR.parent
BACKEND_DIR = ALIGNMENT_DIR.parent
sys.path.insert(0, str(ALIGNMENT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

import step_alignment as phase7
from detect_outside_contact_contours import run as detect_outside_contact_contours
from fit_contours_inside_regularized_mesh import run as fit_contours_inside_regularized_mesh
from fit_contours_near_regularized_planes import run as fit_contours_near_regularized_planes
from joint_asset_synchronizer import JointAssetSynchronizer

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=BACKEND_DIR / "current_scene")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_dir = workspace / "multi_room_alignment" / "temporary_alternative_pipeline"

    phase7.MultiAssetSynchronizer = JointAssetSynchronizer
    result = phase7.run_phase07_alignment(
        workspace=workspace, output_dir=output_dir, keep_temporary_outputs=True
    )
    if result.get("status") != "success":
        raise RuntimeError("Temporary Phase 07 pipeline did not complete successfully")

    contour_report = detect_outside_contact_contours(output_dir, workspace)
    contour_fit_report = fit_contours_inside_regularized_mesh(output_dir)
    nearest_plane_report = fit_contours_near_regularized_planes(
        output_dir, outside_fraction_limit=0.05
    )
    report_path = output_dir / "stats" / "aligned_objects_manifest.json"
    manifest = json.loads(report_path.read_text())
    print(json.dumps({
        "status": result["status"],
        "output_dir": str(output_dir),
        "aligned_object_count": manifest["total_objects"],
        "shape_outlier_contour_triangles_removed": contour_report[
            "shape_outlier_contact_contour_triangles_removed"
        ],
        "oversized_contour_triangles_removed": contour_report[
            "oversized_contact_contour_triangles_removed"
        ],
        "isolated_contour_triangles_removed": contour_report[
            "isolated_contact_contour_triangles_removed"
        ],
        "small_isolated_contour_groups_removed": contour_report[
            "small_isolated_group_count"
        ],
        "small_isolated_contour_triangles_removed": contour_report[
            "small_isolated_group_triangles_removed"
        ],
        "touching_instance_pairs": contour_report["touching_instance_pair_count"],
        "touching_contact_groups": contour_report["touching_contact_group_count"],
        "contact_contour_triangle_count": contour_report["contact_contour_triangle_count"],
        "contact_contour_ply": str(output_dir / "outside_contact_contours"
                                    / contour_report["combined_objects_ply"]),
        "objects_moved_inside_regularized_mesh": contour_fit_report["objects_moved"],
        "inside_fit_touching_contact_groups": contour_fit_report[
            "touching_contact_group_count"
        ],
        "maximum_object_translation_m": contour_fit_report["maximum_translation_m"],
        "unresolved_contour_objects": contour_fit_report[
            "objects_with_unresolved_outside_contours"
        ],
        "unresolved_contour_points": contour_fit_report[
            "outside_contour_point_count_after"
        ],
        "objects_without_feasible_translation": contour_fit_report[
            "objects_without_feasible_translation"
        ],
        "inside_fit_ply": str(output_dir / "contour_inside_fit"
                               / contour_fit_report["combined_objects_ply"]),
        "objects_moved_toward_nearest_plane": nearest_plane_report["objects_moved"],
        "nearest_plane_touching_contact_groups": nearest_plane_report[
            "touching_contact_group_count"
        ],
        "maximum_below_floor_fraction": nearest_plane_report[
            "maximum_below_floor_fraction"
        ],
        "floor_correction_groups": nearest_plane_report[
            "floor_correction_group_count"
        ],
        "maximum_floor_correction_m": nearest_plane_report[
            "maximum_floor_correction_m"
        ],
        "maximum_nearest_plane_translation_m": nearest_plane_report[
            "maximum_nearest_plane_translation_m"
        ],
        "average_contour_distance_before_m": nearest_plane_report[
            "average_contour_distance_before_m"
        ],
        "average_contour_distance_after_m": nearest_plane_report[
            "average_contour_distance_after_m"
        ],
        "red_triangles_outside_after_nearest_plane_fit": nearest_plane_report[
            "outside_triangle_count_after"
        ],
        "objects_over_five_percent_outside_after_nearest_plane_fit": nearest_plane_report[
            "objects_over_outside_limit_after"
        ],
        "objects_raised_to_wall_median": nearest_plane_report[
            "objects_raised_to_wall_median"
        ],
        "wall_median_groups": nearest_plane_report["wall_median_group_count"],
        "wall_median_block_reasons": nearest_plane_report["wall_median_block_reasons"],
        "floor_guard_violations_after_nearest_plane_fit": nearest_plane_report[
            "floor_guard_violations_after"
        ],
        "objects_with_unresolved_hard_constraints": nearest_plane_report[
            "objects_with_unresolved_hard_constraints"
        ],
        "nearest_plane_fit_ply": str(output_dir / "nearest_plane_fit"
                                      / nearest_plane_report["combined_objects_ply"]),
        "sealed_mesh": str(output_dir / "aligned_sealed_meshes.ply"),
        "objects_mesh": str(output_dir / "aligned_objects.ply"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
