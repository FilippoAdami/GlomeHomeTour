"""GlomeHomeTour: Manhattan-World Planar RANSAC Snapping (Phase 3.3).

Fits orthogonal planes (90-degree Manhattan constraints) to architectural
surfels (floor, ceiling, walls) and projects noisy surfel positions onto
their mathematical boundary planes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .types import BoundingBox3D, SurfelCloudTorch


@dataclass
class FittedPlane:
    """Mathematical plane equation: n_x*x + n_y*y + n_z*z + d = 0."""
    name: str                           # "floor", "ceiling", "wall_pos_x", "wall_neg_x", etc.
    normal: Tuple[float, float, float]  # Unit normal vector (nx, ny, nz)
    d: float                            # Distance parameter
    inlier_indices: List[int]           # Indices of supporting surfels
    axis: str                           # "X", "Y", "Z"

    def distance_to_point(self, point: Union[Sequence[float], np.ndarray, torch.Tensor]) -> float:
        """Compute signed distance from 3D point to this plane."""
        nx, ny, nz = self.normal
        return float(nx * point[0] + ny * point[1] + nz * point[2] + self.d)

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """Orthogonally project (M, 3) points onto this plane."""
        n = np.array(self.normal, dtype=np.float32)
        # dist = p . n + d
        dists = np.sum(points * n, axis=-1, keepdims=True) + self.d
        return points - dists * n


class ManhattanPlanarRANSAC:
    """Enforces strict orthogonal (Manhattan-World) plane fitting.

    Snaps planes to canonical axes (X, Y, Z) with 0.00-degree angular deviation,
    fitting dominant boundary planes via RANSAC.
    """

    def __init__(
        self,
        distance_threshold: float = 0.05,  # 5 cm RANSAC inlier threshold
        min_inlier_ratio: float = 0.20,
    ):
        self.distance_threshold = distance_threshold
        self.min_inlier_ratio = min_inlier_ratio

    def fit_floor_and_ceiling(
        self, surfels: SurfelCloudTorch
    ) -> Tuple[FittedPlane, FittedPlane]:
        """Fit horizontal planes Y = floor_y and Y = ceiling_y."""
        pos = surfels.positions.cpu().numpy()
        norm = surfels.normals.cpu().numpy()

        # Floor: pointing up (+Y)
        floor_candidates = np.where(norm[:, 1] > 0.75)[0]
        if len(floor_candidates) > 0:
            floor_y = float(np.median(pos[floor_candidates, 1]))
        else:
            floor_y = float(np.min(pos[:, 1]))

        # Normal = [0, 1, 0], d = -floor_y (since 1*y - floor_y = 0)
        floor_inliers = np.where(np.abs(pos[:, 1] - floor_y) <= self.distance_threshold)[0].tolist()
        floor_plane = FittedPlane(
            name="floor",
            normal=(0.0, 1.0, 0.0),
            d=-floor_y,
            inlier_indices=floor_inliers,
            axis="Y",
        )

        # Ceiling: pointing down (-Y)
        ceiling_candidates = np.where(norm[:, 1] < -0.75)[0]
        if len(ceiling_candidates) > 0:
            ceiling_y = float(np.median(pos[ceiling_candidates, 1]))
        else:
            ceiling_y = float(np.max(pos[:, 1]))

        ceiling_inliers = np.where(np.abs(pos[:, 1] - ceiling_y) <= self.distance_threshold)[0].tolist()
        ceiling_plane = FittedPlane(
            name="ceiling",
            normal=(0.0, -1.0, 0.0),
            d=ceiling_y,
            inlier_indices=ceiling_inliers,
            axis="Y",
        )

        return floor_plane, ceiling_plane

    def fit_manhattan_walls(
        self,
        surfels: SurfelCloudTorch,
        wall_surfel_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, FittedPlane]:
        """Fit orthogonal vertical boundary planes aligned with X and Z axes."""
        pos = surfels.positions.cpu().numpy()
        if wall_surfel_indices is not None and len(wall_surfel_indices) > 0:
            pos_walls = pos[wall_surfel_indices]
            idx_map = np.array(wall_surfel_indices)
        else:
            pos_walls = pos
            idx_map = np.arange(len(pos))

        planes: Dict[str, FittedPlane] = {}

        # 1. Min X Wall (Left)
        x_min = float(np.percentile(pos_walls[:, 0], 2))
        inliers_min_x = idx_map[np.abs(pos_walls[:, 0] - x_min) <= self.distance_threshold].tolist()
        planes["wall_neg_x"] = FittedPlane(
            name="wall_neg_x",
            normal=(1.0, 0.0, 0.0),  # Facing inward towards room center
            d=-x_min,
            inlier_indices=inliers_min_x,
            axis="X",
        )

        # 2. Max X Wall (Right)
        x_max = float(np.percentile(pos_walls[:, 0], 98))
        inliers_max_x = idx_map[np.abs(pos_walls[:, 0] - x_max) <= self.distance_threshold].tolist()
        planes["wall_pos_x"] = FittedPlane(
            name="wall_pos_x",
            normal=(-1.0, 0.0, 0.0), # Facing inward
            d=x_max,
            inlier_indices=inliers_max_x,
            axis="X",
        )

        # 3. Min Z Wall (Front)
        z_min = float(np.percentile(pos_walls[:, 2], 2))
        inliers_min_z = idx_map[np.abs(pos_walls[:, 2] - z_min) <= self.distance_threshold].tolist()
        planes["wall_neg_z"] = FittedPlane(
            name="wall_neg_z",
            normal=(0.0, 0.0, 1.0),  # Facing inward
            d=-z_min,
            inlier_indices=inliers_min_z,
            axis="Z",
        )

        # 4. Max Z Wall (Back)
        z_max = float(np.percentile(pos_walls[:, 2], 98))
        inliers_max_z = idx_map[np.abs(pos_walls[:, 2] - z_max) <= self.distance_threshold].tolist()
        planes["wall_pos_z"] = FittedPlane(
            name="wall_pos_z",
            normal=(0.0, 0.0, -1.0), # Facing inward
            d=z_max,
            inlier_indices=inliers_max_z,
            axis="Z",
        )

        return planes

    def snap_surfels_to_planes(
        self,
        surfels: SurfelCloudTorch,
        planes: Dict[str, FittedPlane],
    ) -> SurfelCloudTorch:
        """Project inlier surfels precisely onto their fitted planes to eliminate scan ripple."""
        pos = surfels.positions.clone().cpu().numpy()
        norm = surfels.normals.clone().cpu().numpy()

        for plane in planes.values():
            if not plane.inlier_indices:
                continue
            idx = np.array(plane.inlier_indices)
            sub_pos = pos[idx]
            projected = plane.project_points(sub_pos)
            pos[idx] = projected
            norm[idx] = np.array(plane.normal, dtype=np.float32)

        dev = surfels.device
        return SurfelCloudTorch(
            positions=torch.from_numpy(pos).to(dev),
            normals=torch.from_numpy(norm).to(dev),
            scales_2d=surfels.scales_2d.clone(),
            colors_rgb=surfels.colors_rgb.clone(),
            opacities=surfels.opacities.clone(),
            rotations=surfels.rotations.clone() if surfels.rotations is not None else None,
            roughness=surfels.roughness.clone() if surfels.roughness is not None else None,
            metallic=surfels.metallic.clone() if surfels.metallic is not None else None,
        )
