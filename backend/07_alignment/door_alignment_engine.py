"""Generic inter-room doorway correspondence matcher and coaxial portal alignment engine.

Pairs doorways across scans using mobile ARCore camera transit paths, enforces exact
coaxial portal alignment (zero lateral error along opening tangents), and automatically
discovers physical partition wall thicknesses under non-penetration / bounded overlap constraints.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

try:
    from .scan_loader import (
        REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2,
        ScanData,
        regularized_floor_footprint,
        transform_floor_footprint,
    )
except (ImportError, ValueError):
    from scan_loader import (
        REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2,
        ScanData,
        regularized_floor_footprint,
        transform_floor_footprint,
    )

logger = logging.getLogger("door_alignment")


@dataclass
class DoorPairing:
    """Pairing correspondence between two doorways across adjacent rooms."""
    scan_from: str
    scan_to: str
    door_from: Dict[str, Any]
    door_to: Dict[str, Any]
    pierce_point_from: np.ndarray
    pierce_point_to: np.ndarray
    relative_rotation: np.ndarray       # 3x3 cardinal rotation matrix
    yaw_deg: float
    description: str


class DoorAlignmentEngine:
    """Solves inter-room door correspondences and coaxial portal registrations."""

    def __init__(self, max_overlap_pct: float = 2.0):
        self.max_overlap_pct = max_overlap_pct

    def pair_doors_via_trajectories(
        self,
        scans: Dict[str, ScanData],
        coarse_transforms: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[DoorPairing]:
        """Automatically pairs doorways across adjacent scans using inter-room transit trajectories."""
        pairings: List[DoorPairing] = []
        scan_keys = sorted(list(scans.keys()))
        paired_doors: set[Tuple[str, int]] = set()

        if coarse_transforms is None:
            coarse_transforms = {k: np.eye(4) for k in scan_keys}

        # Check consecutive pairs (scan_0 -> scan_1, scan_1 -> scan_2, ...)
        for i in range(len(scan_keys) - 1):
            s_from_name = scan_keys[i]
            s_to_name = scan_keys[i + 1]

            scan_from = scans[s_from_name]
            scan_to = scans[s_to_name]

            # Available doors excluding already paired instances
            avail_from = [d for d in scan_from.door_instances if (s_from_name, d["instance_idx"]) not in paired_doors]
            avail_to = [d for d in scan_to.door_instances if (s_to_name, d["instance_idx"]) not in paired_doors]

            if not avail_from:
                avail_from = scan_from.door_instances
            if not avail_to:
                avail_to = scan_to.door_instances

            # 1. Determine inter-room transition points
            # In scan_to, transition_poses are poses walked before the active scan loop started
            trans_to = scan_to.transition_poses
            if trans_to is not None and len(trans_to) > 0:
                walk_start_to = trans_to[0]
                walk_end_to = trans_to[-1]
            elif scan_to.trajectory_poses is not None and len(scan_to.trajectory_poses) > 0:
                walk_start_to = scan_to.trajectory_poses[0]
                walk_end_to = scan_to.trajectory_poses[min(30, len(scan_to.trajectory_poses)-1)]
            else:
                walk_start_to = np.zeros(3)
                walk_end_to = np.array([0.0, 0.0, 0.5])

            # In scan_from, active loop end is where scanning concluded
            active_from = scan_from.active_poses
            if active_from is not None and len(active_from) > 0:
                exit_point_from = active_from[-1]
            elif scan_from.trajectory_poses is not None and len(scan_from.trajectory_poses) > 0:
                exit_point_from = scan_from.trajectory_poses[-1]
            else:
                exit_point_from = np.zeros(3)

            # 2. Match traversed doors in scan_from and scan_to
            best_door_from = self._find_closest_door(avail_from, exit_point_from)
            best_door_to = self._find_traversed_door(avail_to, walk_start_to, walk_end_to)

            if best_door_from is None or best_door_to is None:
                # Fallback to largest/dominant doors
                logger.warning(f"Could not cleanly trace trajectory doors between {s_from_name} and {s_to_name}; using geometric fallback.")
                if avail_from:
                    best_door_from = max(avail_from, key=lambda d: d.get("area", 1.0))
                if avail_to:
                    best_door_to = max(avail_to, key=lambda d: d.get("area", 1.0))

            if best_door_from is None or best_door_to is None:
                logger.warning(f"No doors available to pair {s_from_name} and {s_to_name}.")
                continue

            paired_doors.add((s_from_name, best_door_from["instance_idx"]))
            paired_doors.add((s_to_name, best_door_to["instance_idx"]))

            # 3. Compute Cardinal Manhattan Rotation
            # In Manhattan alignment, doorway normal in scan_to must oppose normal in scan_from: R_to * n_to = -n_from
            n_from = best_door_from["normal"]
            n_to = best_door_to["normal"]
            R_rel, yaw_deg = self._solve_cardinal_rotation(n_from, n_to)

            desc = (
                f"{s_from_name} ({best_door_from['description']}) <-> "
                f"{s_to_name} ({best_door_to['description']})"
            )
            logger.info(f"Paired: {desc} | Yaw={yaw_deg:.0f}°")

            pairings.append(
                DoorPairing(
                    scan_from=s_from_name,
                    scan_to=s_to_name,
                    door_from=best_door_from,
                    door_to=best_door_to,
                    pierce_point_from=exit_point_from,
                    pierce_point_to=walk_start_to,
                    relative_rotation=R_rel,
                    yaw_deg=yaw_deg,
                    description=desc,
                )
            )

        return pairings

    def solve_coaxial_alignment(
        self,
        scans: Dict[str, ScanData],
        pairings: List[DoorPairing],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Align door openings while keeping regularized floor footprints disjoint."""
        from scipy.spatial import ConvexHull
        from shapely.affinity import translate
        from shapely.geometry import Polygon

        scan_keys = sorted(list(scans.keys()))
        transforms: Dict[str, np.ndarray] = {scan_keys[0]: np.eye(4, dtype=np.float64)}
        metrics: Dict[str, Any] = {}

        local_floor_polys = {sname: regularized_floor_footprint(scans[sname]) for sname in scan_keys}
        local_sealed_polys = {}
        for sname in scan_keys:
            mesh_path = scans[sname].info.sealed_mesh_path or scans[sname].info.regularized_complete_mesh_path
            mesh = trimesh.load(mesh_path, process=False)
            points = np.asarray(mesh.vertices)[:, [0, 2]]
            local_sealed_polys[sname] = Polygon(points[ConvexHull(points).vertices])

        global_floor_polys = {
            scan_keys[0]: transform_floor_footprint(local_floor_polys[scan_keys[0]], transforms[scan_keys[0]])
        }
        global_sealed_polys = {scan_keys[0]: local_sealed_polys[scan_keys[0]]}

        # Sequentially solve each pair
        for p in pairings:
            s_from = p.scan_from
            s_to = p.scan_to
            T_from = transforms[s_from]

            d_from = p.door_from
            d_to = p.door_to
            R_to_rel = p.relative_rotation

            # Global rotation for s_to
            R_to_glob = T_from[:3, :3] @ R_to_rel

            rotation_only = np.eye(4, dtype=np.float64)
            rotation_only[:3, :3] = R_to_glob
            floor_to_rot = transform_floor_footprint(local_floor_polys[s_to], rotation_only)
            sealed_to_rot = transform_floor_footprint(local_sealed_polys[s_to], rotation_only)
            sealed_area_to = float(sealed_to_rot.area)

            # Door centers in local coordinates
            c_from_local = d_from["center"]
            c_to_local = d_to["center"]

            # Door center in global coordinates for s_from
            c_from_glob = (T_from[:3, :3] @ c_from_local) + T_from[:3, 3]

            # Door normal & tangent in global frame
            norm_glob = T_from[:3, :3] @ d_from["normal"]
            norm_glob /= np.linalg.norm(norm_glob)
            tang_glob = T_from[:3, :3] @ d_from["tangent"]
            tang_glob /= np.linalg.norm(tang_glob)

            # Local rotated center of door in s_to
            c_to_rot = R_to_glob @ c_to_local

            # Coaxial constraint:
            # (R_to_glob * c_to_local + t_to - c_from_glob) . tang_glob = 0.0
            t_to_lat = c_from_glob - c_to_rot
            t_to_lat = t_to_lat - np.dot(t_to_lat, norm_glob) * norm_glob

            # Base translation at zero wall thickness:
            base_t_to = t_to_lat + (np.dot(c_from_glob, norm_glob) - np.dot(c_to_rot, norm_glob)) * norm_glob
            base_t_to[1] = 0.0

            # Determine whether +norm_glob or -norm_glob moves s_to away from s_from
            test_pos = base_t_to + 0.10 * norm_glob
            p_pos = translate(sealed_to_rot, xoff=test_pos[0], yoff=test_pos[2])
            ov_pos = sum(g.intersection(p_pos).area for g in global_sealed_polys.values() if g.intersects(p_pos))

            test_neg = base_t_to - 0.10 * norm_glob
            p_neg = translate(sealed_to_rot, xoff=test_neg[0], yoff=test_neg[2])
            ov_neg = sum(g.intersection(p_neg).area for g in global_sealed_polys.values() if g.intersects(p_neg))

            sign = 1.0 if ov_pos <= ov_neg else -1.0

            # Keep the existing sealed-mesh clearance while requiring disjoint regularized floor footprints.
            best_t_wall = 0.15
            best_t_to = base_t_to + (sign * best_t_wall) * norm_glob
            best_sealed_pct = 0.0
            best_sealed_area = 0.0
            best_floor_area = 0.0
            found_nonoverlapping_placement = False

            thickness_candidates = np.linspace(0.08, 0.45, 75)
            for test_w in thickness_candidates:
                t_test = base_t_to + (sign * test_w) * norm_glob
                t_test[1] = 0.0

                floor_test = translate(floor_to_rot, xoff=t_test[0], yoff=t_test[2])
                sealed_test = translate(sealed_to_rot, xoff=t_test[0], yoff=t_test[2])
                total_floor_overlap = 0.0
                total_sealed_overlap = 0.0
                max_sealed_pct = 0.0
                for g_sname, g_poly in global_floor_polys.items():
                    if g_poly.intersects(floor_test):
                        total_floor_overlap += float(g_poly.intersection(floor_test).area)
                for g_sname, g_poly in global_sealed_polys.items():
                    if g_poly.intersects(sealed_test):
                        inter_area = float(g_poly.intersection(sealed_test).area)
                        min_a = min(float(g_poly.area), sealed_area_to)
                        pct = (inter_area / min_a) * 100.0 if min_a > 0 else 0.0
                        total_sealed_overlap += inter_area
                        max_sealed_pct = max(max_sealed_pct, pct)

                if (
                    total_floor_overlap <= REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2
                    and max_sealed_pct <= self.max_overlap_pct
                ):
                    best_t_wall = float(test_w)
                    best_t_to = t_test
                    best_sealed_pct = max_sealed_pct
                    best_sealed_area = total_sealed_overlap
                    best_floor_area = total_floor_overlap
                    found_nonoverlapping_placement = True
                    break

            if not found_nonoverlapping_placement:
                raise RuntimeError(
                    f"Could not place {s_to} beside {s_from} without overlapping regularized floor footprints "
                    f"and within {self.max_overlap_pct:.1f}% sealed-mesh overlap "
                    "within the 8–45 cm doorway clearance search."
                )

            T_to_final = np.eye(4, dtype=np.float64)
            T_to_final[:3, :3] = R_to_glob
            T_to_final[:3, 3] = best_t_to
            transforms[s_to] = T_to_final

            # Record final global polygon
            global_floor_polys[s_to] = translate(floor_to_rot, xoff=best_t_to[0], yoff=best_t_to[2])
            global_sealed_polys[s_to] = translate(sealed_to_rot, xoff=best_t_to[0], yoff=best_t_to[2])

            # Compute lateral door error residual
            c_to_glob_final = (R_to_glob @ c_to_local) + best_t_to
            lat_err = abs(float(np.dot(c_to_glob_final - c_from_glob, tang_glob)))

            metrics[f"{s_from}_{s_to}"] = {
                "pair_description": p.description,
                "lateral_error_cm": round(lat_err * 100.0, 2),
                "discovered_wall_thickness_cm": round(best_t_wall * 100.0, 2),
                "mesh_overlap_area_m2": round(best_sealed_area, 4),
                "mesh_overlap_percent": round(best_sealed_pct, 2),
                "regularized_floor_overlap_area_m2": round(best_floor_area, 8),
                "overlap_basis": "regularized floors must be disjoint; sealed hull clearance remains capped",
                "status": "COAXIALLY_ALIGNED",
            }
            logger.info(
                f"Solved {s_from} <-> {s_to}: Lateral error = {lat_err*100:.2f} cm | "
                f"Wall thickness = {best_t_wall*100:.1f} cm | Sealed overlap = {best_sealed_pct:.2f}% | "
                f"Regularized floor overlap = {best_floor_area:.8f} m²"
            )

        return transforms, metrics


    @staticmethod
    def _find_closest_door(doors: List[Dict[str, Any]], point: np.ndarray) -> Optional[Dict[str, Any]]:
        """Finds the door whose center is closest to a given trajectory point in Manhattan XZ plane."""
        if not doors:
            return None
        return min(doors, key=lambda d: np.linalg.norm(d["center"][[0, 2]] - point[[0, 2]]))

    @staticmethod
    def _find_traversed_door(doors: List[Dict[str, Any]], start_pt: np.ndarray, end_pt: np.ndarray) -> Optional[Dict[str, Any]]:
        """Finds the door pierced or closest to the transition walk line in Manhattan XZ plane."""
        if not doors:
            return None
        # Closest door to the walk midpoint or start in 2D
        mid_pt = (start_pt + end_pt) / 2.0
        return min(doors, key=lambda d: min(
            np.linalg.norm(d["center"][[0, 2]] - mid_pt[[0, 2]]),
            np.linalg.norm(d["center"][[0, 2]] - start_pt[[0, 2]])
        ))

    @staticmethod
    def _solve_cardinal_rotation(n_from: np.ndarray, n_to: np.ndarray) -> Tuple[np.ndarray, float]:
        """Finds the Manhattan cardinal yaw rotation (0, 90, 180, 270 deg) that satisfies R * n_to = -n_from."""
        target_n_to = -n_from
        best_R = np.eye(3, dtype=np.float64)
        best_yaw = 0.0
        best_dot = -999.0

        for deg in [0.0, 90.0, 180.0, 270.0]:
            rad = np.radians(deg)
            cos_y = np.cos(rad)
            sin_y = np.sin(rad)
            R_candidate = np.array([
                [cos_y, 0.0, sin_y],
                [0.0,   1.0, 0.0],
                [-sin_y, 0.0, cos_y],
            ], dtype=np.float64)

            rotated_n = R_candidate @ n_to
            dot = np.dot(rotated_n, target_n_to)
            if dot > best_dot:
                best_dot = dot
                best_R = R_candidate
                best_yaw = deg

        return best_R, best_yaw

    @staticmethod
    def _get_scan_polygon(scan: ScanData, transform: np.ndarray):
        """Extracts 2D floor boundary polygon in global Manhattan XZ frame."""
        return transform_floor_footprint(regularized_floor_footprint(scan), transform)
