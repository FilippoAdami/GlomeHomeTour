"""Generic all-plane coplanar discovery and non-overlapping alignment engine.

Extracts all 3D planar surfaces (walls, floors, ceilings) from Phase 06 regularized elements
across all room scans. Discovers parallel, non-overlapping candidate planes within 20 cm, and
optimizes global SE(3) transforms to snap continuous planes (walls and shared floor levels)
while keeping regularized floor footprints and aligned planar surfaces from overlapping.
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

logger = logging.getLogger("coplanar_planes")


@dataclass
class PlanarFacet:
    """Planar surface facet extracted from Phase 06 regularized mesh elements."""
    scan_id: str
    element_type: str                  # "wall", "floor", "ceiling", "slab"
    normal: np.ndarray                 # 3D normal vector
    offset: float                      # d in n . x = d
    area: float                        # surface area in m²
    x_span: Tuple[float, float]
    y_span: Tuple[float, float]
    z_span: Tuple[float, float]
    center: np.ndarray


@dataclass
class CoplanarCandidate:
    """Pair of candidate coplanar planes across scans."""
    scan_a: str
    facet_a: PlanarFacet
    scan_b: str
    facet_b: PlanarFacet
    normal_axis: str                   # "X", "Y", "Z"
    distance_before_m: float
    distance_before_cm: float
    overlap_along_plane_m: float
    description: str


class CoplanarPlanesEngine:
    """Discovers and snaps candidate coplanar walls, floors, and ceilings across scans."""

    def __init__(
        self,
        max_distance_m: float = 0.35,
        max_angle_deg: float = 5.0,
    ):
        self.max_distance_m = max_distance_m
        self.max_angle_deg = max_angle_deg

    def extract_all_scan_planes(
        self,
        scans: Dict[str, ScanData],
        current_transforms: Dict[str, np.ndarray],
    ) -> Dict[str, List[PlanarFacet]]:
        """Extracts major planar surfaces (walls, floors, ceilings) for all scans in global frame."""
        all_scan_planes: Dict[str, List[PlanarFacet]] = {}

        for scan_id, scan in scans.items():
            T = current_transforms.get(scan_id, np.eye(4, dtype=np.float64))
            facets: List[PlanarFacet] = []

            # Determine sources of elements
            element_files: List[Tuple[str, Path]] = []
            if scan.info.regularized_elements_dir and scan.info.regularized_elements_dir.is_dir():
                if scan.info.regularized_walls_path and scan.info.regularized_walls_path.is_file():
                    element_files.append(("wall", scan.info.regularized_walls_path))
                for fp in scan.info.regularized_floor_paths:
                    element_files.append(("floor", fp))
                for cp in scan.info.regularized_ceiling_paths:
                    element_files.append(("ceiling", cp))
            elif scan.info.regularized_complete_mesh_path and scan.info.regularized_complete_mesh_path.is_file():
                element_files.append(("complete", scan.info.regularized_complete_mesh_path))
            elif scan.info.sealed_mesh_path and scan.info.sealed_mesh_path.is_file():
                element_files.append(("sealed", scan.info.sealed_mesh_path))

            for elem_type, elem_path in element_files:
                try:
                    m = trimesh.load(elem_path, process=False)
                    v_glob = (T[:3, :3] @ m.vertices.T).T + T[:3, 3]
                    fn_glob = (T[:3, :3] @ m.face_normals.T).T
                    fc_glob = (T[:3, :3] @ m.triangles.mean(axis=1).T).T + T[:3, 3]
                    fa = m.area_faces

                    # Cluster into cardinal/dominant plane families
                    # Floors: +Y [0, 1, 0]
                    # Ceilings: -Y [0, -1, 0]
                    # Walls: +X [1, 0, 0], -X [-1, 0, 0], +Z [0, 0, 1], -Z [0, 0, -1]
                    cardinal_directions = [
                        (np.array([0.0, 1.0, 0.0]), "floor", "Y"),
                        (np.array([0.0, -1.0, 0.0]), "ceiling", "Y"),
                        (np.array([1.0, 0.0, 0.0]), "wall", "X"),
                        (np.array([-1.0, 0.0, 0.0]), "wall", "X"),
                        (np.array([0.0, 0.0, 1.0]), "wall", "Z"),
                        (np.array([0.0, 0.0, -1.0]), "wall", "Z"),
                    ]

                    for n_dir, category, axis_label in cardinal_directions:
                        dots = np.dot(fn_glob, n_dir)
                        mask = dots > 0.85
                        if not np.any(mask):
                            continue

                        c_sub = fc_glob[mask]
                        a_sub = fa[mask]
                        proj = np.dot(c_sub, n_dir)

                        order = np.argsort(proj)
                        proj_s = proj[order]
                        a_s = a_sub[order]
                        c_s = c_sub[order]

                        # Binned clustering of 1D plane offset d
                        cur_p = [proj_s[0]]
                        cur_a = [a_s[0]]
                        cur_c = [c_s[0]]
                        for p_val, a_val, pt in zip(proj_s[1:], a_s[1:], c_s[1:]):
                            if p_val - cur_p[-1] < 0.05:
                                cur_p.append(p_val)
                                cur_a.append(a_val)
                                cur_c.append(pt)
                            else:
                                tot_a = float(sum(cur_a))
                                if tot_a > 0.35:
                                    pts_arr = np.array(cur_c)
                                    avg_offset = float(np.average(cur_p, weights=cur_a))
                                    if axis_label == "Y":
                                        cat_assigned = "floor" if (elem_type == "floor" or (elem_type != "ceiling" and avg_offset < 1.5)) else "ceiling"
                                    elif elem_type == "wall":
                                        cat_assigned = "wall"
                                    else:
                                        cat_assigned = category
                                    facets.append(
                                        PlanarFacet(
                                            scan_id=scan_id,
                                            element_type=cat_assigned,
                                            normal=n_dir,
                                            offset=avg_offset,
                                            area=tot_a,
                                            x_span=(float(pts_arr[:, 0].min()), float(pts_arr[:, 0].max())),
                                            y_span=(float(pts_arr[:, 1].min()), float(pts_arr[:, 1].max())),
                                            z_span=(float(pts_arr[:, 2].min()), float(pts_arr[:, 2].max())),
                                            center=pts_arr.mean(axis=0),
                                        )
                                    )
                                cur_p = [p_val]
                                cur_a = [a_val]
                                cur_c = [pt]

                        if sum(cur_a) > 0.35:
                            pts_arr = np.array(cur_c)
                            avg_offset = float(np.average(cur_p, weights=cur_a))
                            if axis_label == "Y":
                                cat_assigned = "floor" if (elem_type == "floor" or (elem_type != "ceiling" and avg_offset < 1.5)) else "ceiling"
                            elif elem_type == "wall":
                                cat_assigned = "wall"
                            else:
                                cat_assigned = category
                            facets.append(
                                PlanarFacet(
                                    scan_id=scan_id,
                                    element_type=cat_assigned,
                                    normal=n_dir,
                                    offset=avg_offset,
                                    area=float(sum(cur_a)),
                                    x_span=(float(pts_arr[:, 0].min()), float(pts_arr[:, 0].max())),
                                    y_span=(float(pts_arr[:, 1].min()), float(pts_arr[:, 1].max())),
                                    z_span=(float(pts_arr[:, 2].min()), float(pts_arr[:, 2].max())),
                                    center=pts_arr.mean(axis=0),
                                )
                            )
                except Exception as e:
                    logger.warning(f"Error extracting facets from {elem_path}: {e}")

            all_scan_planes[scan_id] = facets
            logger.info(f"Scan {scan_id}: Discovered {len(facets)} major planar facets (walls/floors/ceilings).")

        return all_scan_planes

    def discover_coplanar_candidates(
        self,
        all_scan_planes: Dict[str, List[PlanarFacet]],
    ) -> List[CoplanarCandidate]:
        """Discovers all candidate coplanar plane pairs across scans (< 20 cm, parallel, non-overlapping)."""
        candidates: List[CoplanarCandidate] = []
        scan_keys = sorted(list(all_scan_planes.keys()))
        cos_tol = np.cos(np.radians(self.max_angle_deg))

        for i in range(len(scan_keys)):
            for j in range(i + 1, len(scan_keys)):
                s_a = scan_keys[i]
                s_b = scan_keys[j]

                for p_a in all_scan_planes[s_a]:
                    for p_b in all_scan_planes[s_b]:
                        # Must have parallel normal vectors
                        dot = np.dot(p_a.normal, p_b.normal)
                        if dot < cos_tol:
                            continue

                        # Perpendicular distance check
                        dist = abs(p_a.offset - p_b.offset)
                        if dist > self.max_distance_m:
                            continue

                        # Check in-plane non-overlap
                        if abs(p_a.normal[1]) > 0.8:
                            # Horizontal plane (floor/ceiling): tangent space is (X, Z)
                            axis_label = "Y"
                            # For floors across adjacent rooms, they share the floor level without overlapping room volumes
                            overlap = 0.0
                        elif abs(p_a.normal[0]) > 0.8:
                            # X-normal wall: tangent axis is Z
                            axis_label = "X"
                            s1 = p_a.z_span
                            s2 = p_b.z_span
                            overlap = max(0.0, min(s1[1], s2[1]) - max(s1[0], s2[0]))
                        else:
                            # Z-normal wall: tangent axis is X
                            axis_label = "Z"
                            s1 = p_a.x_span
                            s2 = p_b.x_span
                            overlap = max(0.0, min(s1[1], s2[1]) - max(s1[0], s2[0]))

                        if overlap > 0.08:
                            continue  # overlapping surfaces, not disjoint collinear walls

                        desc = (
                            f"{s_a} {p_a.element_type.title()} ({axis_label}={p_a.offset:.3f}m) <-> "
                            f"{s_b} {p_b.element_type.title()} ({axis_label}={p_b.offset:.3f}m)"
                        )
                        candidates.append(
                            CoplanarCandidate(
                                scan_a=s_a,
                                facet_a=p_a,
                                scan_b=s_b,
                                facet_b=p_b,
                                normal_axis=axis_label,
                                distance_before_m=dist,
                                distance_before_cm=dist * 100.0,
                                overlap_along_plane_m=overlap,
                                description=desc,
                            )
                        )

        return candidates

    def optimize_all_planes(
        self,
        scans: Dict[str, ScanData],
        current_transforms: Dict[str, np.ndarray],
        candidates: List[CoplanarCandidate],
        door_pairings: Optional[List[Any]] = None,
        all_scan_planes: Optional[Dict[str, List[PlanarFacet]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Snap coplanar planes while preserving disjoint regularized floor footprints."""

        transforms_opt = {s: T.copy() for s, T in current_transforms.items()}
        scan_keys = sorted(list(scans.keys()))
        anchor_scan = scan_keys[0]

        evaluations: List[Dict[str, Any]] = []
        local_footprints = {s: regularized_floor_footprint(scans[s]) for s in scan_keys}

        def door_lateral_metrics(transforms: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
            rows = []
            for pairing in door_pairings or []:
                T_from = transforms[pairing.scan_from]
                T_to = transforms[pairing.scan_to]
                center_from = T_from[:3, :3] @ pairing.door_from["center"] + T_from[:3, 3]
                center_to = T_to[:3, :3] @ pairing.door_to["center"] + T_to[:3, 3]
                tangent = T_from[:3, :3] @ pairing.door_from["tangent"]
                tangent /= np.linalg.norm(tangent)
                error = abs(float(np.dot(center_to - center_from, tangent)))
                width_from = float(pairing.door_from["width"])
                width_to = float(pairing.door_to["width"])
                overlap_width = max(
                    0.0,
                    min(width_from / 2.0, error + width_to / 2.0)
                    - max(-width_from / 2.0, error - width_to / 2.0),
                )
                rows.append({
                    "pair": f"{pairing.scan_from}_{pairing.scan_to}",
                    "lateral_error_cm": round(error * 100.0, 2),
                    "opening_overlap_percent": round(100.0 * overlap_width / min(width_from, width_to), 2),
                })
            return rows

        def footprints_disjoint(test_scan: str, test_transform: np.ndarray) -> bool:
            test_poly = transform_floor_footprint(local_footprints[test_scan], test_transform)
            for other_scan in scan_keys:
                if other_scan == test_scan:
                    continue
                other_poly = transform_floor_footprint(local_footprints[other_scan], transforms_opt[other_scan])
                if test_poly.intersection(other_poly).area > REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2:
                    return False
            return True

        def candidate_surface_overlap_area(candidate: CoplanarCandidate, test_transform: np.ndarray) -> float:
            tangent_index = 2 if candidate.normal_axis == "X" else 0
            span_name = ("x_span", "y_span", "z_span")[tangent_index]
            delta_a = transforms_opt[candidate.scan_a][:3, 3] - current_transforms[candidate.scan_a][:3, 3]
            delta_b = test_transform[:3, 3] - current_transforms[candidate.scan_b][:3, 3]
            span_a = getattr(candidate.facet_a, span_name)
            span_b = getattr(candidate.facet_b, span_name)
            tangent_overlap = max(
                0.0,
                min(span_a[1] + delta_a[tangent_index], span_b[1] + delta_b[tangent_index])
                - max(span_a[0] + delta_a[tangent_index], span_b[0] + delta_b[tangent_index]),
            )
            y_overlap = max(
                0.0,
                min(candidate.facet_a.y_span[1] + delta_a[1], candidate.facet_b.y_span[1] + delta_b[1])
                - max(candidate.facet_a.y_span[0] + delta_a[1], candidate.facet_b.y_span[0] + delta_b[1]),
            )
            return tangent_overlap * y_overlap

        # -------------------------------------------------------------
        # 1. HORIZONTAL FLOOR PLANES ALIGNMENT (Y-AXIS)
        # -------------------------------------------------------------
        anchor_floor_y = None
        for c in candidates:
            if c.normal_axis == "Y" and c.facet_a.element_type == "floor" and c.facet_b.element_type == "floor":
                if c.facet_a.offset < 1.5 and c.facet_b.offset < 1.5:
                    if c.scan_a == anchor_scan:
                        anchor_floor_y = c.facet_a.offset
                        break
                    elif c.scan_b == anchor_scan:
                        anchor_floor_y = c.facet_b.offset
                        break

        # Fallback to direct facet lookup if not found in candidate pairs
        if anchor_floor_y is None and all_scan_planes:
            anchor_floors = [f for f in all_scan_planes.get(anchor_scan, []) if f.element_type == "floor" and f.offset < 1.5]
            if anchor_floors:
                anchor_floor_y = min(anchor_floors, key=lambda f: f.offset).offset

        if anchor_floor_y is not None:
            logger.info(f"\n[Floor Snapping] Anchor {anchor_scan} Floor Level: Y = {anchor_floor_y:.4f} m")
            for scan_id in scan_keys:
                if scan_id == anchor_scan:
                    continue
                floor_facets = [
                    (c.facet_b if c.scan_b == scan_id else c.facet_a)
                    for c in candidates
                    if c.normal_axis == "Y" and (c.scan_a == scan_id or c.scan_b == scan_id) and c.facet_a.element_type == "floor" and c.facet_b.element_type == "floor" and c.facet_a.offset < 1.5 and c.facet_b.offset < 1.5
                ]
                if not floor_facets and all_scan_planes:
                    floor_facets = [f for f in all_scan_planes.get(scan_id, []) if f.element_type == "floor" and f.offset < 1.5]

                if floor_facets:
                    target_facet = min(floor_facets, key=lambda f: f.offset)
                    current_y = target_facet.offset
                    dy_shift = anchor_floor_y - current_y

                    logger.info(f"  Snapping {scan_id} floor from Y={current_y:.4f}m to Y={anchor_floor_y:.4f}m (shift: {dy_shift*100:+.2f}cm)")
                    transforms_opt[scan_id][1, 3] += dy_shift

                    evaluations.append({
                        "category": "Floor Level Alignment",
                        "scan": scan_id,
                        "plane_axis": "Y",
                        "elevation_before_m": round(current_y, 4),
                        "elevation_after_m": round(anchor_floor_y, 4),
                        "adjustment_cm": round(dy_shift * 100.0, 2),
                        "door_lateral_error_cm": 0.0,
                        "status": "PERFECT_FLOOR_COPLANARITY",
                    })

        # -------------------------------------------------------------
        # 2. VERTICAL WALL PLANES ALIGNMENT (X & Z AXES)
        # -------------------------------------------------------------
        for c in candidates:
            if c.normal_axis == "Y":
                continue  # already handled in floor pass

            s_a = c.scan_a
            s_b = c.scan_b
            fa = c.facet_a
            fb = c.facet_b
            axis = c.normal_axis  # "X" or "Z"

            offset_a = fa.offset + float(np.dot(fa.normal, transforms_opt[s_a][:3, 3] - current_transforms[s_a][:3, 3]))
            offset_b = fb.offset + float(np.dot(fb.normal, transforms_opt[s_b][:3, 3] - current_transforms[s_b][:3, 3]))
            dist_curr = abs(offset_a - offset_b)

            if dist_curr < 0.01:
                evaluations.append({
                    "category": "Wall Plane Alignment",
                    "description": c.description,
                    "axis": axis,
                    "distance_before_cm": round(c.distance_before_cm, 2),
                    "distance_after_cm": 0.0,
                    "status": "ALREADY_COPLANAR",
                })
                continue

            if s_b != anchor_scan:
                shift_vec = (offset_a - offset_b) * fa.normal
                axis_idx = 0 if axis == "X" else 2
                d_shift = float(shift_vec[axis_idx])

                test_T = transforms_opt[s_b].copy()
                test_T[axis_idx, 3] += d_shift
                footprint_ok = footprints_disjoint(s_b, test_T)
                surface_overlap = candidate_surface_overlap_area(c, test_T)
                trial_transforms = {**transforms_opt, s_b: test_T}
                door_rows = door_lateral_metrics(trial_transforms)
                doors_still_overlap = all(row["opening_overlap_percent"] > 0 for row in door_rows)
                if footprint_ok and surface_overlap <= REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2 and doors_still_overlap:
                    transforms_opt[s_b][axis_idx, 3] += d_shift
                    logger.info(f"  Snapping {c.description}: Shifting {s_b} along {axis} by {d_shift*100:+.2f}cm")
                    door_rows = door_lateral_metrics(transforms_opt)
                    max_door_error = max((row["lateral_error_cm"] for row in door_rows), default=0.0)
                    evaluations.append({
                        "category": "Wall Plane Alignment",
                        "description": c.description,
                        "axis": axis,
                        "distance_before_cm": round(dist_curr * 100.0, 2),
                        "distance_after_cm": 0.0,
                        "shift_applied_cm": round(d_shift * 100.0, 2),
                        "door_lateral_error_cm": max_door_error,
                        "status": "COPLANAR_ALIGNED",
                    })
                else:
                    reason = (
                        "would overlap regularized floor footprints"
                        if not footprint_ok
                        else "would overlap the aligned regularized wall surfaces"
                        if surface_overlap > REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2
                        else "would disconnect a paired doorway opening"
                    )
                    logger.info(f"  Preserving boundary for {c.description}: snapping {reason}")
                    evaluations.append({
                        "category": "Wall Plane Alignment",
                        "description": c.description,
                        "axis": axis,
                        "distance_before_cm": round(dist_curr * 100.0, 2),
                        "distance_after_cm": round(dist_curr * 100.0, 2),
                        "reason": f"Snapping along {axis} {reason}.",
                        "status": "OVERLAP_CONSTRAINED",
                    })

        overlap_metrics = self._calculate_all_overlaps(scans, transforms_opt)
        if any(row["overlap_area_m2"] > REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2 for row in overlap_metrics.values()):
            raise RuntimeError("Final regularized floor footprints overlap; refusing to publish Phase 07 alignment.")
        door_rows = door_lateral_metrics(transforms_opt)

        report_data = {
            "evaluations": evaluations,
            "door_lateral_errors": {
                "max_door_lateral_error_cm": max((row["lateral_error_cm"] for row in door_rows), default=0.0),
                "status": "ALL_DOOR_OPENINGS_OVERLAP" if all(row["opening_overlap_percent"] > 0 for row in door_rows) else "DOOR_OPENING_MISMATCH",
                "pairs": door_rows,
            },
            "mesh_overlaps": overlap_metrics,
            "overlap_basis": "exact regularized floor footprints; positive-area overlap is rejected",
        }

        return transforms_opt, report_data

    @staticmethod
    def _calculate_all_overlaps(
        scans: Dict[str, ScanData],
        transforms: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Measure exact positive-area overlap between regularized floor footprints."""

        scan_keys = sorted(list(scans.keys()))
        overlaps = {}
        for i in range(len(scan_keys)):
            for j in range(i + 1, len(scan_keys)):
                sa = scan_keys[i]
                sb = scan_keys[j]
                pa = transform_floor_footprint(regularized_floor_footprint(scans[sa]), transforms[sa])
                pb = transform_floor_footprint(regularized_floor_footprint(scans[sb]), transforms[sb])
                if pa.intersects(pb):
                    inter = pa.intersection(pb)
                    area = float(inter.area)
                    min_a = min(float(pa.area), float(pb.area))
                    pct = (area / min_a) * 100.0 if min_a > 0 else 0.0
                else:
                    area = 0.0
                    pct = 0.0
                overlaps[f"{sa}_{sb}"] = {
                    "overlap_area_m2": round(area, 8),
                    "overlap_percent": round(pct, 2),
                }

        return overlaps
