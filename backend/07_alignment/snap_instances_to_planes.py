#!/usr/bin/env python3
"""Stage 07 — Atomic Single-Instance Plane Snapping Engine.

Evaluates each non-structural object instance singularly against the Stage 06
regularized structural mesh planes (walls, floors, ceilings). If an instance is
within 15 cm of a plane, it is translated along cardinal axes (X, Y, Z) to become
adjacent to that plane, while keeping at least 98% of its surface inside when possible.

CLI Usage:
  backend/.venv/bin/python backend/07_alignment/snap_instances_to_planes.py \
      --alignment-dir backend/current_scene/scan_0/objects_alignment \
      --shell backend/current_scene/scan_0/mesh_elaboration/complete.ply
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fit_contact_groups_inside import (
    surface_meshes,
    as_scene,
    contained_surface_percent,
    publish_directory,
)
from structural_surfaces import load_role_meshes

logger = logging.getLogger("snap_instances_to_planes")


@dataclass
class StructuralPlaneFacet:
    """Represents a planar facet extracted from the regularized structural shell."""
    role: str  # 'floor', 'wall', 'ceiling'
    axis: int  # 0 for X, 1 for Y, 2 for Z
    normal: np.ndarray  # 3D unit normal vector
    coord: float  # Scalar plane coordinate along the normal axis
    spans: List[Tuple[float, float]]  # [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
    area: float  # Area in square meters


def setup_logger() -> logging.Logger:
    log = logging.getLogger("snap_instances_to_planes")
    log.setLevel(logging.INFO)
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        log.addHandler(ch)
    return log


def extract_structural_facets(
    shell: o3d.geometry.TriangleMesh,
    shell_path: Optional[Path] = None,
    min_area: float = 0.35,
) -> List[StructuralPlaneFacet]:
    """Extracts major planar surfaces (floors, walls, ceilings) from regularized shell."""
    role_surfaces = load_role_meshes(shell, shell_path)
    facets: List[StructuralPlaneFacet] = []

    # Process each structural role
    role_configs = [
        ("floor", role_surfaces.get("floor"), [(np.array([0.0, 1.0, 0.0]), 1)]),
        ("ceiling", role_surfaces.get("ceiling"), [(np.array([0.0, -1.0, 0.0]), 1)]),
        (
            "wall",
            role_surfaces.get("wall"),
            [
                (np.array([1.0, 0.0, 0.0]), 0),
                (np.array([-1.0, 0.0, 0.0]), 0),
                (np.array([0.0, 0.0, 1.0]), 2),
                (np.array([0.0, 0.0, -1.0]), 2),
            ],
        ),
    ]

    for role_name, mesh, cardinal_dirs in role_configs:
        if mesh is None or len(mesh.triangles) == 0:
            continue

        mesh.compute_triangle_normals()
        normals = np.asarray(mesh.triangle_normals)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        centroids = vertices[faces].mean(axis=1)
        areas = (
            np.linalg.norm(
                np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
                axis=1,
            )
            / 2.0
        )

        for n_dir, ax in cardinal_dirs:
            dots = np.dot(normals, n_dir)
            mask = (dots > 0.85) & (areas > 1e-6)
            if not np.any(mask):
                continue

            proj = centroids[mask, ax]
            sub_areas = areas[mask]
            sub_pts = centroids[mask]

            order = np.argsort(proj)
            proj_s = proj[order]
            areas_s = sub_areas[order]
            pts_s = sub_pts[order]

            cur_p = [proj_s[0]]
            cur_a = [areas_s[0]]
            cur_pts = [pts_s[0]]

            for p_val, a_val, pt in zip(proj_s[1:], areas_s[1:], pts_s[1:]):
                if p_val - cur_p[-1] < 0.05:
                    cur_p.append(p_val)
                    cur_a.append(a_val)
                    cur_pts.append(pt)
                else:
                    tot_a = float(sum(cur_a))
                    if tot_a >= min_area:
                        arr = np.array(cur_pts)
                        avg_p = float(np.average(cur_p, weights=cur_a))
                        facets.append(
                            StructuralPlaneFacet(
                                role=role_name,
                                axis=ax,
                                normal=n_dir,
                                coord=avg_p,
                                spans=[
                                    (float(arr[:, 0].min()), float(arr[:, 0].max())),
                                    (float(arr[:, 1].min()), float(arr[:, 1].max())),
                                    (float(arr[:, 2].min()), float(arr[:, 2].max())),
                                ],
                                area=tot_a,
                            )
                        )
                    cur_p = [p_val]
                    cur_a = [a_val]
                    cur_pts = [pt]

            tot_a = float(sum(cur_a))
            if tot_a >= min_area:
                arr = np.array(cur_pts)
                avg_p = float(np.average(cur_p, weights=cur_a))
                facets.append(
                    StructuralPlaneFacet(
                        role=role_name,
                        axis=ax,
                        normal=n_dir,
                        coord=avg_p,
                        spans=[
                            (float(arr[:, 0].min()), float(arr[:, 0].max())),
                            (float(arr[:, 1].min()), float(arr[:, 1].max())),
                            (float(arr[:, 2].min()), float(arr[:, 2].max())),
                        ],
                        area=tot_a,
                    )
                )

    return facets


def evaluate_candidate_for_axis(
    v: np.ndarray,
    current_shift: np.ndarray,
    ax: int,
    facets: List[StructuralPlaneFacet],
    max_distance_m: float = 0.15,
    target_gap_m: float = 0.002,
    transverse_margin_m: float = 0.10,
    footprint: Any = None,
    prefer_interior: bool = False,
) -> Optional[Tuple[StructuralPlaneFacet, float, float]]:
    """Finds the best candidate facet along cardinal axis `ax` within max_distance_m.

    Returns:
      (facet, signed_translation, gap_before), or None if no candidate.
    """
    u_vals = v[:, ax] + current_shift[ax]
    # Use 1st and 99th percentiles for robustness against scanner floaters/noise
    if len(u_vals) > 100:
        u_min = float(np.percentile(u_vals, 1.0))
        u_max = float(np.percentile(u_vals, 99.0))
    else:
        u_min = float(u_vals.min())
        u_max = float(u_vals.max())

    best_facet = None
    min_gap = 1e9
    best_translation = 0.0

    t_axes = [i for i in [0, 1, 2] if i != ax]

    for f in facets:
        if f.axis != ax:
            continue

        # Transverse overlap check (ensure the instance actually projects onto the facet)
        overlap = True
        for t_ax in t_axes:
            f_min, f_max = f.spans[t_ax]
            i_vals = v[:, t_ax] + current_shift[t_ax]
            i_min, i_max = float(i_vals.min()), float(i_vals.max())
            if i_max < f_min - transverse_margin_m or i_min > f_max + transverse_margin_m:
                overlap = False
                break
        if not overlap:
            continue

        c = f.coord
        # If the instance starts outside, move it onto the room-interior side of a support plane.
        if prefer_interior and footprint is not None:
            if f.role == "wall":
                center = footprint.centroid
                interior_coord = center.x if ax == 0 else center.y
                interior_direction = float(np.sign(interior_coord - f.coord))
            else:
                interior_direction = 1.0 if f.role == "floor" else -1.0
            if not interior_direction:
                continue
            raw_min, raw_max = float(u_vals.min()), float(u_vals.max())
            gap_before = (raw_min - f.coord if raw_min > f.coord else
                          f.coord - raw_max if raw_max < f.coord else 0.0)
            if gap_before > max_distance_m:
                continue
            interior_edge = raw_min if interior_direction > 0 else raw_max
            target_coord = f.coord + interior_direction * target_gap_m
            requested_translation = target_coord - interior_edge
            translation = float(np.clip(requested_translation, -max_distance_m, max_distance_m))
            if abs(translation) < 1e-4:
                continue
            if gap_before < min_gap:
                min_gap = gap_before
                best_facet = f
                best_translation = translation
            continue

        # Determine whether instance is on the positive or negative side of plane
        if u_min >= c - 0.005:
            gap = u_min - c
            direction = -1.0  # translate in -ax direction towards plane
        elif u_max <= c + 0.005:
            gap = c - u_max
            direction = +1.0  # translate in +ax direction towards plane
        else:
            gap = 0.0
            direction = 0.0

        translation = direction * max(0.0, gap - target_gap_m)
        if target_gap_m < gap <= max_distance_m and gap < min_gap:
            min_gap = gap
            best_facet = f
            best_translation = translation

    if best_facet is not None:
        return best_facet, best_translation, min_gap
    return None


def snap_single_instance(
    mesh: o3d.geometry.TriangleMesh,
    facets: List[StructuralPlaneFacet],
    floor_scene: Any,
    ceiling_scene: Any,
    footprint: Any,
    ray_top: float,
    ray_bottom: float,
    max_distance_m: float = 0.15,
    min_containment_percent: float = 98.0,
    target_gap_m: float = 0.002,
) -> Tuple[np.ndarray, float, float, List[Dict[str, Any]]]:
    """Snap to nearby planes while keeping at least 98% containment when possible.

    Returns:
      (total_shift, baseline_pct, final_pct, snapped_planes_info)
    """
    v = np.asarray(mesh.vertices)
    if len(v) == 0:
        return np.zeros(3), 100.0, 100.0, []

    baseline_pct = contained_surface_percent(
        mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom, translation=np.zeros(3)
    )

    total_shift = np.zeros(3, dtype=np.float64)
    current_pct = baseline_pct
    snapped_planes_info: List[Dict[str, Any]] = []

    # Sequence of axes: Floor (Y=1) first for vertical resting contact, then walls (X=0, Z=2)
    axis_order = [1, 0, 2]

    for ax in axis_order:
        cand = evaluate_candidate_for_axis(
            v=v,
            current_shift=total_shift,
            ax=ax,
            facets=facets,
            max_distance_m=max_distance_m,
            target_gap_m=target_gap_m,
            footprint=footprint,
            prefer_interior=current_pct < min_containment_percent,
        )
        if cand is None:
            continue

        facet, axis_shift, gap = cand
        if abs(axis_shift) < 1e-4:
            continue

        trial_shift = total_shift.copy()
        trial_shift[ax] += axis_shift

        trial_pct = contained_surface_percent(
            mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom, translation=trial_shift
        )

        if current_pct >= min_containment_percent and trial_pct >= min_containment_percent:
            total_shift = trial_shift
            current_pct = trial_pct
            snapped_planes_info.append({
                "role": facet.role,
                "axis": ["X", "Y", "Z"][ax],
                "direction": "+" if axis_shift > 0 else "-",
                "gap_before_cm": round(gap * 100.0, 2),
                "plane_coord": round(facet.coord, 4),
                "shift_cm": round(axis_shift * 100.0, 2),
                "clamped": False,
            })
        elif current_pct >= min_containment_percent:
            # Keep the 2% exterior cap while moving as close to the support plane as possible.
            low, high = 0.0, 1.0
            best_alpha = 0.0
            for _ in range(8):
                mid = (low + high) / 2.0
                sub_shift = total_shift.copy()
                sub_shift[ax] += mid * axis_shift
                sub_pct = contained_surface_percent(
                    mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom, translation=sub_shift
                )
                if sub_pct >= min_containment_percent:
                    best_alpha = mid
                    low = mid
                else:
                    high = mid

            if best_alpha > 0.05:
                clamped_shift = best_alpha * axis_shift
                total_shift[ax] += clamped_shift
                current_pct = contained_surface_percent(
                    mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom,
                    translation=total_shift,
                )
                snapped_planes_info.append({
                    "role": facet.role,
                    "axis": ["X", "Y", "Z"][ax],
                    "direction": "+" if clamped_shift > 0 else "-",
                    "gap_before_cm": round(gap * 100.0, 2),
                    "plane_coord": round(facet.coord, 4),
                    "shift_cm": round(clamped_shift * 100.0, 2),
                    "clamped": True,
                    "clamp_alpha": round(best_alpha, 3),
                })
        else:
            # Starting outside: sample the bounded move and keep the placement with least exterior.
            choices = []
            for alpha in np.linspace(0.1, 1.0, 10):
                sub_shift = total_shift.copy()
                sub_shift[ax] += alpha * axis_shift
                sub_pct = contained_surface_percent(
                    mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom,
                    translation=sub_shift,
                )
                choices.append((sub_pct, alpha, sub_shift))
            best_pct, best_alpha, best_shift = max(choices, key=lambda row: (row[0], row[1]))
            if best_pct > current_pct + 1e-6:
                total_shift = best_shift
                current_pct = best_pct
                inward_shift = best_alpha * axis_shift
                snapped_planes_info.append({
                    "role": facet.role,
                    "axis": ["X", "Y", "Z"][ax],
                    "direction": "+" if inward_shift > 0 else "-",
                    "gap_before_cm": round(gap * 100.0, 2),
                    "plane_coord": round(facet.coord, 4),
                    "shift_cm": round(inward_shift * 100.0, 2),
                    "clamped": bool(best_alpha < 0.999),
                    "clamp_alpha": round(float(best_alpha), 3),
                })

    final_pct = contained_surface_percent(
        mesh, floor_scene, ceiling_scene, footprint, ray_top, ray_bottom, translation=total_shift
    )
    return total_shift, baseline_pct, final_pct, snapped_planes_info


def run_snap_instances_to_planes(
    alignment_dir: Path,
    shell_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_distance_m: float = 0.15,
    min_containment_percent: float = 98.0,
    target_gap_m: float = 0.002,
) -> Dict[str, Any]:
    """Executes single-instance plane snapping across all instances in alignment_dir."""
    log = setup_logger()
    started = time.perf_counter()
    alignment_dir = Path(alignment_dir).resolve()

    manifest_path = alignment_dir / "aligned_instances" / "manifest.json"
    if not manifest_path.is_file():
        # Fallback to alignment_dir / manifest.json if direct
        manifest_path = alignment_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing objects manifest in {alignment_dir}")

    manifest_data = json.loads(manifest_path.read_text())
    instances_dir = manifest_path.parent

    # Resolve shell mesh
    if shell_path is None:
        source_shell_str = manifest_data.get("source_shell")
        if source_shell_str:
            shell_path = Path(source_shell_str)
        else:
            scan_dir = alignment_dir.parent
            shell_path = scan_dir / "mesh_elaboration" / "complete.ply"

    shell_path = Path(shell_path).resolve()
    if not shell_path.is_file():
        raise FileNotFoundError(f"Regularized structural mesh not found at {shell_path}")

    log.info("=" * 70)
    log.info("STAGE 07: ATOMIC SINGLE-INSTANCE PLANE SNAPPING")
    log.info(f"Alignment Directory: {alignment_dir}")
    log.info(f"Structural Shell:   {shell_path}")
    log.info(f"Max Plane Distance: {max_distance_m * 100.0:.1f} cm")
    log.info(f"Min Containment:    {min_containment_percent:.1f}% (max {100.0 - min_containment_percent:.1f}% outside)")
    log.info("=" * 70)

    # Load shell and role meshes
    shell = o3d.io.read_triangle_mesh(str(shell_path))
    if shell.is_empty():
        raise ValueError(f"Structural shell is empty: {shell_path}")

    floor_mesh, ceiling_mesh, wall_mesh, footprint = surface_meshes(shell, shell_path)
    floor_scene, ceiling_scene = as_scene(floor_mesh), as_scene(ceiling_mesh)
    ray_top = float(shell.get_max_bound()[1] + 1.0)
    ray_bottom = float(shell.get_min_bound()[1] - 1.0)

    # Extract all planar facets from regularized mesh
    facets = extract_structural_facets(shell, shell_path, min_area=0.30)
    log.info(f"Extracted {len(facets)} structural planar facets (walls, floors, ceilings).")
    for f in facets:
        ax_name = ["X", "Y", "Z"][f.axis]
        log.debug(f"  Facet {f.role:<8}: axis={ax_name}, coord={f.coord:.3f}, area={f.area:.2f}m2")

    instances = manifest_data.get("instances", [])
    log.info(f"Evaluating {len(instances)} single instance(s) for plane snapping...")

    staging_dir = Path(tempfile.mkdtemp(prefix=".plane_snapped_instances.", dir=alignment_dir))
    updated_instances = []
    instance_reports = []
    snapped_count = 0

    try:
        for idx, item in enumerate(instances):
            rel_p = Path(item["path"])
            src_mesh_p = instances_dir / rel_p
            if not src_mesh_p.is_file():
                log.warning(f"Instance mesh not found: {src_mesh_p}. Skipping.")
                continue

            m = o3d.io.read_triangle_mesh(str(src_mesh_p))
            if m.is_empty() or not len(m.triangles):
                log.warning(f"Instance mesh is empty: {src_mesh_p}. Skipping.")
                continue

            shift, base_pct, final_pct, planes_info = snap_instance_fn = snap_single_instance(
                mesh=m,
                facets=facets,
                floor_scene=floor_scene,
                ceiling_scene=ceiling_scene,
                footprint=footprint,
                ray_top=ray_top,
                ray_bottom=ray_bottom,
                max_distance_m=max_distance_m,
                min_containment_percent=min_containment_percent,
                target_gap_m=target_gap_m,
            )

            shift_norm = float(np.linalg.norm(shift))
            was_snapped = bool(shift_norm > 1e-5)

            if was_snapped:
                snapped_count += 1
                m.translate(shift, relative=True)
                s_cm = np.round(shift * 100.0, 1)
                log.info(
                    f"  [{idx+1}/{len(instances)}] Snapped {rel_p} ({item.get('category')}): "
                    f"shift={s_cm.tolist()}cm, containment={final_pct:.1f}% "
                    f"({len(planes_info)} plane(s))"
                )

            # Write updated instance to staging
            target_p = staging_dir / rel_p
            target_p.parent.mkdir(parents=True, exist_ok=True)
            if not o3d.io.write_triangle_mesh(str(target_p), m, write_ascii=False):
                raise IOError(f"Could not write snapped instance mesh: {target_p}")

            # Update instance manifest entry
            prior_placement = item.get("placement_translation_m", [0.0, 0.0, 0.0])
            total_cumulative_placement = (np.array(prior_placement) + shift).tolist()

            updated_entry = {
                **item,
                "placement_translation_m": total_cumulative_placement,
                "plane_snap_translation_m": shift.tolist(),
                "plane_snap_applied": was_snapped,
                "snapped_planes": planes_info,
                "contained_surface_percent": final_pct,
            }
            updated_instances.append(updated_entry)

            instance_reports.append({
                "path": str(rel_p),
                "category": item.get("category"),
                "class_id": item.get("class_id"),
                "shift_m": shift.tolist(),
                "shift_magnitude_cm": round(shift_norm * 100.0, 2),
                "was_snapped": was_snapped,
                "baseline_contained_surface_percent": round(base_pct, 2),
                "final_contained_surface_percent": round(final_pct, 2),
                "snapped_planes": planes_info,
            })

        # Update and write manifest to staging
        final_manifest = {
            **manifest_data,
            "status": "plane_snapped_non_structural_instances",
            "plane_snap_parameters": {
                "max_distance_m": max_distance_m,
                "min_containment_percent": min_containment_percent,
                "target_gap_m": target_gap_m,
            },
            "instances": updated_instances,
        }
        (staging_dir / "manifest.json").write_text(json.dumps(final_manifest, indent=2) + "\n")

        # Atomically publish to aligned_instances
        dest_instances_dir = alignment_dir / "aligned_instances"
        publish_directory(staging_dir, dest_instances_dir)
        log.info(f"Atomically updated aligned instances directory: {dest_instances_dir}")

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    elapsed = time.perf_counter() - started
    report = {
        "status": "plane_snapped_instances",
        "total_instances": len(instances),
        "snapped_instances_count": snapped_count,
        "max_distance_threshold_m": max_distance_m,
        "minimum_contained_surface_percent": min_containment_percent,
        "target_contact_gap_m": target_gap_m,
        "wall_time_seconds": round(elapsed, 2),
        "instances": instance_reports,
    }

    # Save reports
    report_dest = alignment_dir / "plane_snap_report.json"
    report_dest.write_text(json.dumps(report, indent=2) + "\n")
    fit_sub = alignment_dir / "structural_fit"
    if fit_sub.is_dir():
        (fit_sub / "plane_snap_report.json").write_text(json.dumps(report, indent=2) + "\n")

    log.info(f"Plane snapping completed in {elapsed:.2f}s: {snapped_count}/{len(instances)} instances snapped.")
    log.info(f"Saved plane snapping report to: {report_dest}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment-dir",
        type=Path,
        default=BACKEND_DIR / "current_scene" / "scan_0" / "objects_alignment",
        help="Path to scan objects_alignment directory containing aligned_instances",
    )
    parser.add_argument(
        "--shell",
        type=Path,
        default=None,
        help="Path to Stage 06 complete.ply regularized structural mesh",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (defaults to updating aligned_instances in place)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.15,
        help="Maximum distance threshold in meters to snap to a plane (default: 0.15m = 15cm)",
    )
    parser.add_argument(
        "--min-containment",
        type=float,
        default=98.0,
        help="Minimum surface containment percent required (default: 98.0, up to 2% exterior)",
    )
    parser.add_argument(
        "--target-gap",
        type=float,
        default=0.002,
        help="Target contact gap to the plane in meters (default: 0.002m = 2mm)",
    )

    args = parser.parse_args()
    result = run_snap_instances_to_planes(
        alignment_dir=args.alignment_dir,
        shell_path=args.shell,
        output_dir=args.output_dir,
        max_distance_m=args.max_distance,
        min_containment_percent=args.min_containment,
        target_gap_m=args.target_gap,
    )
    print(
        json.dumps(
            {
                "total_instances": result["total_instances"],
                "snapped_instances": result["snapped_instances_count"],
                "wall_time_seconds": result["wall_time_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
