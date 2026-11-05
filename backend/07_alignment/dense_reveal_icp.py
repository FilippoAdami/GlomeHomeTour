"""Dense doorway reveal and door-jamb Point-to-Plane ICP refinement.

Crops Phase 04 dense surface meshes around deep doorway reveals (accommodating
up to 40-50cm jamb tunnel depths) and performs localized horizontal Point-to-Plane
ICP on jamb faces, head soffits, and floor threshold strips.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh

try:
    from .portal_detector import DoorPortal
except (ImportError, ValueError):
    from portal_detector import DoorPortal


@dataclass
class ICPRefinementResult:
    """Result of dense reveal ICP refinement."""
    transform: np.ndarray      # 4x4 refined transformation matrix
    rmse_residual_m: float     # Root-mean-square error in meters
    fitness_score: float       # Inlier fraction (0.0 to 1.0)
    applied: bool              # Whether ICP refinement was accepted over the prior
    message: str


class DenseRevealICP:
    """Localized planar ICP on Phase 04 doorway reveal surfaces."""

    def __init__(
        self,
        crop_radius: float = 1.80,   # Generous radius covering 40-50cm deep reveals + jamb trim
        max_correspondence_distance: float = 0.08, # 8cm max correspondence distance
        max_iterations: int = 50,
    ):
        self.crop_radius = crop_radius
        self.max_correspondence_distance = max_correspondence_distance
        self.max_iterations = max_iterations

    @staticmethod
    def crop_mesh_around_point(
        mesh_path: Path,
        center_pt: np.ndarray,
        radius: float,
    ) -> Optional[o3d.geometry.PointCloud]:
        """Loads and crops a dense mesh around a center point, returning a PointCloud with normals."""
        if not mesh_path.is_file():
            return None
        try:
            m = trimesh.load(mesh_path, process=False)
            if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0:
                return None

            dists = np.linalg.norm(m.vertices - center_pt, axis=1)
            mask = dists <= radius
            if np.sum(mask) < 50:
                return None

            sub_v = m.vertices[mask]
            sub_n = m.vertex_normals[mask] if len(m.vertex_normals) == len(m.vertices) else None

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sub_v)
            if sub_n is not None and len(sub_n) > 0:
                pcd.normals = o3d.utility.Vector3dVector(sub_n)
            else:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30))
            return pcd
        except Exception:
            return None

    def refine(
        self,
        initial_transform: np.ndarray,
        portal_i: DoorPortal,
        scan_dir_i: Path,
        scan_dir_j: Path,
    ) -> ICPRefinementResult:
        """Runs localized Point-to-Plane ICP on the cropped doorway reveals."""
        mesh_i_path = scan_dir_i / "mesh.ply"
        mesh_j_path = scan_dir_j / "mesh.ply"

        crop_center_i = portal_i.threshold_center
        pcd_i = self.crop_mesh_around_point(mesh_i_path, crop_center_i, self.crop_radius)

        T_inv = np.linalg.inv(initial_transform)
        crop_center_j = (T_inv[:3, :3] @ crop_center_i) + T_inv[:3, 3]
        pcd_j = self.crop_mesh_around_point(mesh_j_path, crop_center_j, self.crop_radius)

        if pcd_i is None or pcd_j is None or len(pcd_i.points) < 50 or len(pcd_j.points) < 50:
            return ICPRefinementResult(
                transform=initial_transform,
                rmse_residual_m=0.0,
                fitness_score=0.0,
                applied=False,
                message="Insufficient cropped dense points at doorway; retained wall-constrained fit.",
            )

        pcd_i_down = pcd_i.voxel_down_sample(voxel_size=0.02)
        pcd_j_down = pcd_j.voxel_down_sample(voxel_size=0.02)

        if not pcd_i_down.has_normals():
            pcd_i_down.estimate_normals()
        if not pcd_j_down.has_normals():
            pcd_j_down.estimate_normals()

        reg_result = o3d.pipelines.registration.registration_icp(
            pcd_j_down,
            pcd_i_down,
            self.max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations),
        )

        T_icp = reg_result.transformation

        R_icp = T_icp[:3, :3]
        up_vec = R_icp @ np.array([0.0, 1.0, 0.0])
        tilt_deg = np.degrees(np.arccos(np.clip(up_vec[1], -1.0, 1.0)))

        if tilt_deg > 2.0 or reg_result.fitness < 0.15:
            return ICPRefinementResult(
                transform=initial_transform,
                rmse_residual_m=float(reg_result.inlier_rmse),
                fitness_score=float(reg_result.fitness),
                applied=False,
                message=f"ICP fitness too low ({reg_result.fitness:.2f}) or tilt excessive ({tilt_deg:.2f}°); retained wall-constrained fit.",
            )

        yaw_rad = np.arctan2(R_icp[0, 2], R_icp[2, 2])
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)
        R_planar = np.array([
            [cos_y, 0.0, sin_y],
            [0.0,   1.0, 0.0],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float64)

        T_final = np.copy(T_icp)
        T_final[:3, :3] = R_planar
        T_final[1, 3] = 0.0  # Preserve floor Y=0 alignment

        return ICPRefinementResult(
            transform=T_final,
            rmse_residual_m=float(reg_result.inlier_rmse),
            fitness_score=float(reg_result.fitness),
            applied=True,
            message="Point-to-plane ICP successfully refined doorway reveal alignment.",
        )
