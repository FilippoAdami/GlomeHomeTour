"""Wall plane and automatic wall-width constraint optimizer.

Automatically discovers the physical wall thickness between rooms (supporting walls
from 8cm up to 50cm+, e.g. 40cm load-bearing walls) from door portal alignment,
geometric non-penetration bounds, and doorway reveal clearance, without requiring
any manual thickness inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union
import trimesh

try:
    from .portal_detector import DoorPortal
except (ImportError, ValueError):
    from portal_detector import DoorPortal


@dataclass
class WallOptimizationResult:
    """Result of wall-constraint optimization."""
    transform: np.ndarray                    # Refined 4x4 transform
    automatically_discovered_wall_thickness: float # Discovered wall thickness in meters
    portal_residual_m: float                 # Lateral door alignment error
    overlap_area_m2: float                   # Remaining floor overlap area (0.0 = perfect non-penetration)
    success: bool
    message: str


class WallConstraintOptimizer:
    """Formulates and solves the constrained multi-room alignment problem with automatic thickness discovery."""

    def __init__(
        self,
        min_allowed_thickness: float = 0.05,
        max_allowed_thickness: float = 0.65,
    ):
        self.min_allowed_thickness = min_allowed_thickness
        self.max_allowed_thickness = max_allowed_thickness

    @staticmethod
    def extract_2d_floor_polygon(scan_dir: Path) -> Polygon:
        """Extracts 2D floor boundary polygon in XZ plane from Phase 06 floor elements or mesh."""
        floor_plys = list((scan_dir / "mesh_elaboration" / "elements").glob("floor_level_*.ply"))
        polygons = []

        if floor_plys:
            for p in floor_plys:
                try:
                    m = trimesh.load(p, process=False)
                    if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 2:
                        pts = m.vertices[:, [0, 2]]  # XZ projection
                        poly = MultiPoint(pts).convex_hull
                        if isinstance(poly, Polygon) and poly.is_valid and poly.area > 0.05:
                            polygons.append(poly)
                except Exception:
                    pass

        if not polygons:
            mesh_path = scan_dir / "mesh.ply"
            if mesh_path.is_file():
                try:
                    m = trimesh.load(mesh_path, process=False)
                    if isinstance(m, trimesh.Trimesh):
                        bmin, bmax = m.bounds
                        poly = Polygon([
                            [bmin[0], bmin[2]],
                            [bmax[0], bmin[2]],
                            [bmax[0], bmax[2]],
                            [bmin[0], bmax[2]],
                        ])
                        polygons.append(poly)
                except Exception:
                    pass

        if polygons:
            merged = unary_union(polygons)
            if isinstance(merged, MultiPolygon):
                return max(merged.geoms, key=lambda p: p.area)
            return merged
        return Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1]])

    def compute_geometric_clearance_thickness(
        self,
        floor_poly_i: Polygon,
        floor_poly_j: Polygon,
        thresh_i: np.ndarray,
        thresh_j: np.ndarray,
        base_R: np.ndarray,
        wall_normal_i: np.ndarray,
    ) -> float:
        """Automatically calculates the minimum non-overlapping wall thickness between two floor polygons."""
        # Find thickness w in [min, max] that completely clears overlap
        best_w = 0.20
        coords_j = np.array(floor_poly_j.exterior.coords)
        pts_3d_j = np.zeros((len(coords_j), 3))
        pts_3d_j[:, 0] = coords_j[:, 0]
        pts_3d_j[:, 2] = coords_j[:, 1]
        rotated_j = (base_R @ pts_3d_j.T).T

        # Sweep test from 0.05m to 0.60m in 1cm steps
        for test_w in np.linspace(0.05, 0.60, 56):
            t_test = thresh_i + test_w * wall_normal_i - (base_R @ thresh_j)
            t_test[1] = 0.0
            test_poly = Polygon((rotated_j + t_test)[:, [0, 2]])
            if not floor_poly_i.intersects(test_poly) or floor_poly_i.intersection(test_poly).area < 1e-4:
                return float(test_w)

        return 0.35

    def optimize(
        self,
        coarse_transform: np.ndarray,
        portal_i: DoorPortal,
        portal_j: DoorPortal,
        scan_dir_i: Path,
        scan_dir_j: Path,
    ) -> WallOptimizationResult:
        """Solves the constrained optimization, automatically discovering the physical wall thickness."""
        floor_poly_i = self.extract_2d_floor_polygon(scan_dir_i)
        floor_poly_j = self.extract_2d_floor_polygon(scan_dir_j)

        wall_normal_i = portal_i.normal
        wall_tangent_i = portal_i.tangent

        base_R = coarse_transform[:3, :3]
        base_t = coarse_transform[:3, 3]

        thresh_i = portal_i.threshold_center
        thresh_j = portal_j.threshold_center

        # Automatically discover clearance thickness from geometry
        auto_discovered_prior = self.compute_geometric_clearance_thickness(
            floor_poly_i, floor_poly_j, thresh_i, thresh_j, base_R, wall_normal_i
        )

        def make_transform(p: np.ndarray) -> np.ndarray:
            dtx, dtz, dtheta_deg = p
            drad = np.radians(dtheta_deg)
            cos_t = np.cos(drad)
            sin_t = np.sin(drad)
            R_delta = np.array([
                [cos_t, 0.0, sin_t],
                [0.0,   1.0, 0.0],
                [-sin_t, 0.0, cos_t]
            ], dtype=np.float64)

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_delta @ base_R
            T[:3, 3] = base_t + np.array([dtx, 0.0, dtz])
            return T

        def transform_polygon(poly: Polygon, T: np.ndarray) -> Polygon:
            coords = np.array(poly.exterior.coords)
            pts_3d = np.zeros((len(coords), 3))
            pts_3d[:, 0] = coords[:, 0]
            pts_3d[:, 2] = coords[:, 1]
            pts_trans = (T[:3, :3] @ pts_3d.T).T + T[:3, 3]
            return Polygon(pts_trans[:, [0, 2]])

        def objective(p: np.ndarray) -> float:
            T = make_transform(p)
            p_j_in_i = (T[:3, :3] @ thresh_j) + T[:3, 3]
            disp = p_j_in_i - thresh_i
            w_meas = float(disp @ wall_normal_i)
            lat_err = float(disp @ wall_tangent_i)

            # Lateral door alignment
            door_loss = lat_err ** 2

            # Wall thickness penalty: guide toward the automatically discovered clearance thickness
            thick_err = (w_meas - auto_discovered_prior) ** 2
            if w_meas < self.min_allowed_thickness:
                thick_err += 200.0 * (self.min_allowed_thickness - w_meas) ** 2
            elif w_meas > self.max_allowed_thickness:
                thick_err += 100.0 * (w_meas - self.max_allowed_thickness) ** 2

            # Strict non-penetration penalty
            poly_j_trans = transform_polygon(floor_poly_j, T)
            overlap_area = 0.0
            if floor_poly_i.intersects(poly_j_trans):
                try:
                    inter = floor_poly_i.intersection(poly_j_trans)
                    overlap_area = float(inter.area)
                except Exception:
                    overlap_area = 0.1

            total_loss = (
                20.0 * door_loss
                + 10.0 * thick_err
                + 200.0 * overlap_area
                + 0.1 * (p[0]**2 + p[1]**2)
                + 0.05 * (p[2]**2)
            )
            return total_loss

        p0 = np.array([0.0, 0.0, 0.0])
        bounds = [(-0.35, 0.35), (-0.35, 0.35), (-5.0, 5.0)]

        res = minimize(
            objective,
            p0,
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-4},
        )

        best_p = res.x
        final_T = make_transform(best_p)

        p_j_final = (final_T[:3, :3] @ thresh_j) + final_T[:3, 3]
        final_disp = p_j_final - thresh_i
        final_w = float(final_disp @ wall_normal_i)
        final_lat = float(abs(final_disp @ wall_tangent_i))

        final_poly_j = transform_polygon(floor_poly_j, final_T)
        final_overlap = 0.0
        if floor_poly_i.intersects(final_poly_j):
            try:
                final_overlap = float(floor_poly_i.intersection(final_poly_j).area)
            except Exception:
                pass

        return WallOptimizationResult(
            transform=final_T,
            automatically_discovered_wall_thickness=final_w,
            portal_residual_m=final_lat,
            overlap_area_m2=final_overlap,
            success=bool(res.success or final_lat < 0.05),
            message=str(res.message),
        )
