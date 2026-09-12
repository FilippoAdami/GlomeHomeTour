"""GlomeHomeTour Backend: Planar Mirror Detection & Virtual Reflection Passes.

Identifies planar mirror clusters (high metallic, low roughness) via RANSAC,
computes 3D Householder reflection transformation matrices, and synthesizes
virtual reflection camera view passes for glossy indoor surfaces (mirrors, polished marble).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reconstruction.training.model import Material2DGSModel
from reconstruction.training.rasterizer_interface import Base2DGSRasterizer, GBufferOutput


@dataclass
class MirrorPlane:
    """Represents a detected planar mirror surface in world space.

    Equation: n . x + d = 0, where n is a unit normal vector.
    """
    normal: torch.Tensor          # (3,) unit vector pointing out from mirror
    d: float                      # Offset distance scalar
    inlier_indices: torch.Tensor  # 1D index tensor of surfels on this mirror
    rmse_fit: float               # Fitting RMSE in meters
    area_estimate_m2: float       # Approximate surface area in m^2

    def compute_reflection_matrix_4x4(self) -> torch.Tensor:
        """Construct the 4x4 Householder affine reflection matrix.

        x_refl = (I - 2 * n * n^T) * x - 2 * d * n
        """
        device = self.normal.device
        dtype = self.normal.dtype
        n = self.normal.view(3, 1)

        r_refl = torch.eye(3, device=device, dtype=dtype) - 2.0 * torch.matmul(n, n.T)
        t_refl = -2.0 * self.d * self.normal.view(3, 1)

        mat = torch.eye(4, device=device, dtype=dtype)
        mat[:3, :3] = r_refl
        mat[:3, 3:4] = t_refl
        return mat

    def transform_camera_w2c(self, w2c: torch.Tensor) -> torch.Tensor:
        """Compute virtual camera world-to-camera matrix reflected across this mirror.

        T_w2c_virtual = T_w2c * H_refl
        """
        h_refl = self.compute_reflection_matrix_4x4().to(device=w2c.device, dtype=w2c.dtype)
        if w2c.ndim == 2:
            return torch.matmul(w2c, h_refl)
        return torch.matmul(w2c, h_refl.unsqueeze(0))


class PlanarMirrorDetector:
    """Detects planar mirrors using RANSAC over high metallic and low roughness surfels."""

    def __init__(
        self,
        min_metallic: float = 0.70,
        max_roughness: float = 0.20,
        min_cluster_points: int = 15,
        ransac_distance_threshold: float = 0.03,  # 3 cm plane tolerance
        ransac_max_iterations: int = 200,
        min_inlier_ratio: float = 0.35,
    ) -> None:
        self.min_metallic = min_metallic
        self.max_roughness = max_roughness
        self.min_cluster_points = min_cluster_points
        self.ransac_distance_threshold = ransac_distance_threshold
        self.ransac_max_iterations = ransac_max_iterations
        self.min_inlier_ratio = min_inlier_ratio

    def detect_mirrors(
        self,
        model: Material2DGSModel,
        max_planes: int = 3,
    ) -> List[MirrorPlane]:
        """Discover mirror planes from model surfel parameters.

        Args:
            model: Material2DGSModel.
            max_planes: Maximum number of dominant mirror surfaces to detect.

        Returns:
            List of detected MirrorPlane objects sorted by inlier count descending.
        """
        if model.num_gaussians < self.min_cluster_points:
            return []

        device = model.xyz.device
        dtype = model.xyz.dtype

        # 1. Candidate selection: high metallic, low roughness
        metallic = model.metallic.squeeze(-1).detach()
        roughness = model.roughness.squeeze(-1).detach()
        pos = model.xyz.detach()
        normals = model.normals.detach()

        candidate_mask = (metallic >= self.min_metallic) & (roughness <= self.max_roughness)
        candidate_indices = torch.nonzero(candidate_mask).squeeze(-1)

        if len(candidate_indices) < self.min_cluster_points:
            return []

        pts = pos[candidate_indices]  # (K, 3)
        norms = normals[candidate_indices]  # (K, 3)
        num_candidates = pts.shape[0]

        remaining_mask = torch.ones(num_candidates, dtype=torch.bool, device=device)
        detected_planes: List[MirrorPlane] = []

        for _ in range(max_planes):
            active_indices = torch.nonzero(remaining_mask).squeeze(-1)
            if len(active_indices) < self.min_cluster_points:
                break

            active_pts = pts[active_indices]
            active_norms = norms[active_indices]
            num_active = active_pts.shape[0]

            best_inliers: Optional[torch.Tensor] = None
            best_normal: Optional[torch.Tensor] = None
            best_d: float = 0.0

            # Vectorized RANSAC sampling
            # Randomly draw triplets of points
            num_iters = min(self.ransac_max_iterations, max(20, num_active * 2))
            sample_indices = torch.randint(0, num_active, (num_iters, 3), device=device)

            p1 = active_pts[sample_indices[:, 0]]
            p2 = active_pts[sample_indices[:, 1]]
            p3 = active_pts[sample_indices[:, 2]]

            # Plane normal n = normalize((p2 - p1) x (p3 - p1))
            v12 = p2 - p1
            v13 = p3 - p1
            cand_normals = torch.cross(v12, v13, dim=-1)
            c_len = torch.linalg.norm(cand_normals, dim=-1, keepdim=True)
            valid_normals = c_len.squeeze(-1) > 1e-4

            cand_normals = cand_normals / torch.clamp(c_len, min=1e-6)
            cand_d = -torch.sum(cand_normals * p1, dim=-1)  # (num_iters,)

            # Test inlier support
            for iter_idx in range(num_iters):
                if not valid_normals[iter_idx]:
                    continue

                n_cand = cand_normals[iter_idx]
                d_cand = cand_d[iter_idx].item()

                # Distance of all active points to plane: |n . p + d|
                dists = torch.abs(torch.matmul(active_pts, n_cand) + d_cand)
                # Normal agreement: point normals must be nearly parallel or antiparallel to plane normal
                dot_norms = torch.abs(torch.matmul(active_norms, n_cand))

                inliers = (dists <= self.ransac_distance_threshold) & (dot_norms >= 0.85)
                inlier_count = int(inliers.sum().item())

                if best_inliers is None or inlier_count > int(best_inliers.sum().item()):
                    best_inliers = inliers
                    best_normal = n_cand
                    best_d = d_cand

            if best_inliers is None:
                break

            inlier_ratio = float(best_inliers.sum().item()) / num_active
            if inlier_ratio < self.min_inlier_ratio and int(best_inliers.sum().item()) < self.min_cluster_points:
                break

            # Least-squares plane refinement over inliers
            inlier_pts = active_pts[best_inliers]
            centroid = torch.mean(inlier_pts, dim=0, keepdim=True)
            centered = inlier_pts - centroid
            # SVD: covariance matrix
            cov = torch.matmul(centered.T, centered)
            u, s, vh = torch.linalg.svd(cov)
            # The singular vector corresponding to smallest singular value is the refined plane normal
            refined_n = vh[2]
            refined_n = F.normalize(refined_n, dim=-1, eps=1e-6)

            # Ensure normal points in consistent direction with surfels
            mean_inlier_norm = torch.mean(active_norms[best_inliers], dim=0)
            if torch.dot(refined_n, mean_inlier_norm) < 0:
                refined_n = -refined_n

            refined_d = -torch.dot(refined_n, centroid.squeeze(0)).item()

            # Final inlier metrics
            final_dists = torch.abs(torch.matmul(inlier_pts, refined_n) + refined_d)
            rmse = float(torch.sqrt(torch.mean(final_dists ** 2)).item())

            # Bounding box surface area estimate
            min_bound = torch.amin(inlier_pts, dim=0)
            max_bound = torch.amax(inlier_pts, dim=0)
            dims = max_bound - min_bound
            # Area as product of two largest spatial extents
            sorted_dims = torch.sort(dims).values
            area = float((sorted_dims[1] * sorted_dims[2]).item())

            # Map back to global model indices
            global_inliers = candidate_indices[active_indices[best_inliers]]

            plane = MirrorPlane(
                normal=refined_n,
                d=refined_d,
                inlier_indices=global_inliers,
                rmse_fit=rmse,
                area_estimate_m2=area,
            )
            detected_planes.append(plane)

            # Remove inliers from remaining set
            remaining_mask[active_indices[best_inliers]] = False

        return detected_planes


def render_mirror_reflection_pass(
    rasterizer: Base2DGSRasterizer,
    model: Material2DGSModel,
    w2c: torch.Tensor,
    intrinsics: Any,
    image_size: Tuple[int, int],
    mirror_plane: MirrorPlane,
) -> GBufferOutput:
    """Render the scene from the virtual reflection camera pose of a detected mirror plane.

    Args:
        rasterizer: 2DGS differentiable rasterizer (PyTorch fallback or HIP).
        model: Material2DGSModel.
        w2c: Camera world-to-camera matrix (4, 4).
        intrinsics: Camera intrinsics.
        image_size: (H, W).
        mirror_plane: Detected MirrorPlane.

    Returns:
        GBufferOutput rasterized from the virtual reflected camera viewpoint.
    """
    virtual_w2c = mirror_plane.transform_camera_w2c(w2c)
    return rasterizer(model, extrinsics=virtual_w2c, intrinsics=intrinsics, image_size=image_size)
