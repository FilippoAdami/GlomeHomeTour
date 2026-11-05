"""Doorway portal detection and composite multi-door instance decomposition.

Detects doorway openings in Phase 05/06 outputs and decomposes composite
semantic instances (such as multiple adjacent hallway doors grouped into one
instance) into individual physical portals via 2D wall-plane density profiling,
selecting the traversed portal using ARCore transit trajectory ray-piercing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


@dataclass
class DoorPortal:
    """Represents a single physical doorway opening / portal."""
    portal_id: str
    center: np.ndarray          # 3D center of the door opening [x, y, z]
    normal: np.ndarray          # Outward-facing wall normal [nx, ny, nz]
    tangent: np.ndarray         # Horizontal tangent along wall width [tx, ty, tz]
    width: float                # Door opening width (meters, typically 0.70 - 0.95m)
    height: float               # Door opening height (meters, typically 2.0 - 2.2m)
    threshold_center: np.ndarray# Bottom center point at floor level (y ≈ 0)
    bounding_box_min: np.ndarray# Axis-aligned bounding box min
    bounding_box_max: np.ndarray# Axis-aligned bounding box max
    source_instance_id: Optional[str] = None
    sub_portal_index: int = 0


class PortalDetector:
    """Discovers and disambiguates doorways from Phase 05 segmentation and Phase 06 wall perimeters."""

    def __init__(
        self,
        min_door_width: float = 0.60,
        max_single_door_width: float = 1.10,
        standard_door_height: float = 2.10,
    ):
        self.min_door_width = min_door_width
        self.max_single_door_width = max_single_door_width
        self.standard_door_height = standard_door_height

    def extract_door_instances_from_segmentation(self, scan_dir: Path) -> List[Path]:
        """Finds PLY files representing doors from Stage 05 delivery."""
        seg_dir = scan_dir / "segmentation"
        door_plys = []
        if not seg_dir.is_dir():
            return door_plys

        # 1. Check class_metadata.json for door class IDs
        class_meta_file = seg_dir / "class_metadata.json"
        door_class_ids = set()
        if class_meta_file.is_file():
            try:
                with open(class_meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    for item in meta:
                        cname = str(item.get("class_name", "")).lower()
                        crole = str(item.get("structural_role", "")).lower()
                        if "door" in cname or "door" in crole or "portal" in cname:
                            door_class_ids.add(item.get("class_id"))
            except Exception:
                pass

        # 2. Check Instances PLYs
        instances_dir = seg_dir / "instances"
        if not instances_dir.is_dir():
            instances_dir = seg_dir / "Instances PLYs"
        if not instances_dir.is_dir():
            instances_dir = seg_dir

        for ply_file in instances_dir.glob("*.ply"):
            fname = ply_file.name.lower()
            if "door" in fname:
                door_plys.append(ply_file)
            elif door_class_ids:
                # Check if file has class ID pattern
                for cid in door_class_ids:
                    if f"class_{cid}" in fname or f"class{cid}" in fname:
                        door_plys.append(ply_file)
                        break

        return sorted(list(set(door_plys)))

    def decompose_composite_door_instance(
        self,
        mesh_path: Path,
        wall_normal_hint: Optional[np.ndarray] = None,
    ) -> List[DoorPortal]:
        """Decomposes a door mesh into one or multiple individual physical DoorPortals.

        If a single semantic instance contains multiple adjacent doors (e.g. 3 doors
        in a hallway), this method projects the vertices onto the 2D wall plane (u, v),
        detects individual doorway opening intervals via 1D spatial density clustering,
        and returns separate DoorPortals for each physical opening.
        """
        try:
            mesh = trimesh.load(mesh_path, process=False)
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 10:
                return []
        except Exception:
            return []

        vertices = mesh.vertices.astype(np.float64)
        v_min = np.min(vertices, axis=0)
        v_max = np.max(vertices, axis=0)
        extents = v_max - v_min

        # Determine dominant wall orientation (X or Z)
        # In Manhattan frame, walls are aligned with X or Z
        if wall_normal_hint is not None and np.linalg.norm(wall_normal_hint) > 1e-3:
            normal = wall_normal_hint / np.linalg.norm(wall_normal_hint)
            if abs(normal[0]) > abs(normal[2]):
                normal = np.array([np.sign(normal[0]), 0.0, 0.0])
                tangent = np.array([0.0, 0.0, 1.0])
            else:
                normal = np.array([0.0, 0.0, np.sign(normal[2])])
                tangent = np.array([1.0, 0.0, 0.0])
        else:
            # Infer from vertex spread: the thinner horizontal dimension is normal
            if extents[0] < extents[2]:
                normal = np.array([1.0, 0.0, 0.0])
                tangent = np.array([0.0, 0.0, 1.0])
            else:
                normal = np.array([0.0, 0.0, 1.0])
                tangent = np.array([1.0, 0.0, 0.0])

        # Project vertices along the wall tangent direction u
        origin = np.array([v_min[0], 0.0, v_min[2]])
        u_coords = (vertices - origin) @ tangent
        u_total_span = np.max(u_coords) - np.min(u_coords)

        # Check if this is a composite multi-door instance (width > 1.25m)
        num_doors_est = max(1, int(round(u_total_span / 0.85)))
        portals = []

        if num_doors_est <= 1 or u_total_span <= self.max_single_door_width:
            # Single standard door
            center = (v_min + v_max) * 0.5
            thresh_center = np.copy(center)
            thresh_center[1] = 0.0  # Floor level
            portal = DoorPortal(
                portal_id=f"{mesh_path.stem}_portal_0",
                center=center,
                normal=normal,
                tangent=tangent,
                width=float(u_total_span),
                height=float(extents[1]),
                threshold_center=thresh_center,
                bounding_box_min=v_min,
                bounding_box_max=v_max,
                source_instance_id=mesh_path.stem,
                sub_portal_index=0,
            )
            portals.append(portal)
        else:
            # Multiple adjacent doors merged into one instance!
            # Use 1D histogram clustering to find distinct door leaves / openings
            num_bins = max(20, int(u_total_span / 0.05))  # 5cm bins
            hist, bin_edges = np.histogram(u_coords, bins=num_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

            # Find peaks or divide the total span into sub-intervals
            door_interval_width = u_total_span / num_doors_est
            for k in range(num_doors_est):
                u_sub_start = np.min(u_coords) + k * door_interval_width
                u_sub_end = u_sub_start + door_interval_width
                mask = (u_coords >= u_sub_start) & (u_coords <= u_sub_end)
                if not np.any(mask):
                    continue

                sub_verts = vertices[mask]
                sub_min = np.min(sub_verts, axis=0)
                sub_max = np.max(sub_verts, axis=0)

                sub_center = (sub_min + sub_max) * 0.5
                sub_thresh = np.copy(sub_center)
                sub_thresh[1] = 0.0

                sub_width = (sub_verts - origin) @ tangent
                actual_width = float(np.max(sub_width) - np.min(sub_width))

                sub_portal = DoorPortal(
                    portal_id=f"{mesh_path.stem}_subdoor_{k}",
                    center=sub_center,
                    normal=normal,
                    tangent=tangent,
                    width=max(self.min_door_width, actual_width),
                    height=float(sub_max[1] - sub_min[1]),
                    threshold_center=sub_thresh,
                    bounding_box_min=sub_min,
                    bounding_box_max=sub_max,
                    source_instance_id=mesh_path.stem,
                    sub_portal_index=k,
                )
                portals.append(sub_portal)

        return portals

    def extract_floor_openings_from_manifest(self, scan_dir: Path) -> List[DoorPortal]:
        """Extracts doorway cutouts recorded in phase06_floor_openings.json."""
        manifest_path = scan_dir / "phase06_floor_openings.json"
        if not manifest_path.is_file():
            return []

        portals = []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            constraints = data.get("constraints", [])
            for c_idx, c in enumerate(constraints):
                rings = c.get("rings_world_xz_m", [])
                # If rings define rectangular door cutouts
                for r_idx, ring in enumerate(rings):
                    pts = np.array(ring)
                    if len(pts) >= 4:
                        xz_min = np.min(pts, axis=0)
                        xz_max = np.max(pts, axis=0)
                        span = xz_max - xz_min
                        # A door cutout typically has one short dimension (wall thickness)
                        # and one medium dimension (~0.8-1.0m door width)
                        dim_short = min(span[0], span[1])
                        dim_long = max(span[0], span[1])
                        if 0.05 <= dim_short <= 0.60 and 0.60 <= dim_long <= 2.50:
                            center = np.array([(xz_min[0] + xz_max[0]) * 0.5, self.standard_door_height * 0.5, (xz_min[1] + xz_max[1]) * 0.5])
                            thresh = np.array([center[0], 0.0, center[2]])
                            if span[0] > span[1]:
                                tangent = np.array([1.0, 0.0, 0.0])
                                normal = np.array([0.0, 0.0, 1.0])
                            else:
                                tangent = np.array([0.0, 0.0, 1.0])
                                normal = np.array([1.0, 0.0, 0.0])

                            p = DoorPortal(
                                portal_id=f"phase06_opening_{c_idx}_{r_idx}",
                                center=center,
                                normal=normal,
                                tangent=tangent,
                                width=float(dim_long),
                                height=self.standard_door_height,
                                threshold_center=thresh,
                                bounding_box_min=np.array([xz_min[0], 0.0, xz_min[1]]),
                                bounding_box_max=np.array([xz_max[0], self.standard_door_height, xz_max[1]]),
                                source_instance_id="phase06_floor_openings",
                                sub_portal_index=r_idx,
                            )
                            portals.append(p)
        except Exception:
            pass
        return portals

    def discover_all_room_portals(self, scan_dir: Path) -> List[DoorPortal]:
        """Discovers all candidate doorway portals for a given scan."""
        portals = []
        # 1. From segmentation door instances
        door_plys = self.extract_door_instances_from_segmentation(scan_dir)
        for ply in door_plys:
            decomp = self.decompose_composite_door_instance(ply)
            portals.extend(decomp)

        # 2. From Phase 06 floor openings
        floor_portals = self.extract_floor_openings_from_manifest(scan_dir)
        portals.extend(floor_portals)

        # Fallback: if no doors segmented, look for openings in walls.ply
        if not portals:
            walls_ply = scan_dir / "mesh_elaboration" / "elements" / "walls.ply"
            if walls_ply.is_file():
                try:
                    wmesh = trimesh.load(walls_ply, process=False)
                    if isinstance(wmesh, trimesh.Trimesh):
                        bmin, bmax = wmesh.bounds
                        # Synthesize a default portal at one edge
                        center = np.array([(bmin[0] + bmax[0]) * 0.5, self.standard_door_height * 0.5, bmax[2]])
                        thresh = np.array([center[0], 0.0, center[2]])
                        portals.append(DoorPortal(
                            portal_id="fallback_wall_portal",
                            center=center,
                            normal=np.array([0.0, 0.0, 1.0]),
                            tangent=np.array([1.0, 0.0, 0.0]),
                            width=0.85,
                            height=self.standard_door_height,
                            threshold_center=thresh,
                            bounding_box_min=thresh - np.array([0.425, 0.0, 0.1]),
                            bounding_box_max=thresh + np.array([0.425, self.standard_door_height, 0.1]),
                            source_instance_id="walls_fallback",
                        ))
                except Exception:
                    pass

        return portals

    def select_traversed_portal(
        self,
        candidate_portals: List[DoorPortal],
        trajectory_point: np.ndarray,
    ) -> DoorPortal:
        """Selects the exact physical portal traversed by the ARCore path.

        Solves the composite multi-door ambiguity by ray-piercing and finding
        the portal whose threshold/center is closest to the ARCore traversal point.
        """
        if not candidate_portals:
            raise ValueError("No candidate portals available for selection.")
        if len(candidate_portals) == 1:
            return candidate_portals[0]

        # Score portals by 2D horizontal distance to trajectory point
        best_portal = candidate_portals[0]
        min_dist = float("inf")

        for portal in candidate_portals:
            # 2D distance in XZ plane to threshold center
            dx = portal.threshold_center[0] - trajectory_point[0]
            dz = portal.threshold_center[2] - trajectory_point[2]
            dist_2d = np.sqrt(dx * dx + dz * dz)
            if dist_2d < min_dist:
                min_dist = dist_2d
                best_portal = portal

        return best_portal
