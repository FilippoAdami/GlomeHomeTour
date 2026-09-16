"""GlomeHomeTour: GPU-Accelerated Volumetric TSDF Fusion & Multi-Scale Surfel Extraction.

Fuses multi-view depth maps into a continuous 3D Truncated Signed Distance Field (TSDF)
to eliminate multi-layer depth stacking and floating free-space noise by construction,
then extracts a multi-scale hierarchical surfel cloud (coarse on flat planes, fine on corners).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from package_loader import CameraIntrinsics, Keyframe
from initialization import SurfelCloud, build_orthonormal_tangent_frame, SH_C0


class TSDFVolume:
    """GPU-accelerated volumetric TSDF with sub-voxel zero-crossing surfel extraction."""

    def __init__(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        voxel_size: float = 0.015,  # 1.5 cm voxels
        truncation_multiplier: float = 3.0,
        device: Optional[str] = None,
    ):
        self.voxel_size = float(voxel_size)
        self.truncation_dist = float(voxel_size * truncation_multiplier)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Pad bounding box slightly to avoid boundary clipping
        margin = float(voxel_size * 4.0)
        self.bbox_min = np.asarray(bbox_min, dtype=np.float32) - margin
        self.bbox_max = np.asarray(bbox_max, dtype=np.float32) + margin

        extent = self.bbox_max - self.bbox_min
        self.dims = np.ceil(extent / self.voxel_size).astype(np.int32)
        self.nx, self.ny, self.nz = int(self.dims[0]), int(self.dims[1]), int(self.dims[2])

        # Allocate 3D TSDF, weight, and color volumes
        # Initialize TSDF to 1.0 (free space)
        self.tsdf = torch.ones((self.nx, self.ny, self.nz), dtype=torch.float32, device=self.device)
        self.weight = torch.zeros((self.nx, self.ny, self.nz), dtype=torch.float32, device=self.device)
        self.color = torch.zeros((self.nx, self.ny, self.nz, 3), dtype=torch.float32, device=self.device)

        # Precompute 1D coordinate vectors in world space
        self.x_coords = torch.linspace(self.bbox_min[0], self.bbox_max[0], self.nx, device=self.device)
        self.y_coords = torch.linspace(self.bbox_min[1], self.bbox_max[1], self.ny, device=self.device)
        self.z_coords = torch.linspace(self.bbox_min[2], self.bbox_max[2], self.nz, device=self.device)

    def integrate_frame(
        self,
        depth_map: np.ndarray,
        rgb_img: np.ndarray,
        c2w: np.ndarray,  # (4, 4) OpenGL/ARCore camera-to-world
        intrinsics: CameraIntrinsics,
        weight_multiplier: float = 1.0,
        z_near: float = 0.2,
        z_far: float = 15.0,
    ) -> None:
        """Integrate a single keyframe depth and color into the 3D TSDF volume."""
        h, w = depth_map.shape[:2]
        if rgb_img.shape[:2] != (h, w):
            rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)

        depth_tensor = torch.from_numpy(depth_map.astype(np.float32)).to(self.device)
        rgb_tensor = torch.from_numpy(rgb_img.astype(np.float32) / 255.0).to(self.device)

        # Camera extrinsics (OpenGL convention: +X right, +Y up, -Z fwd)
        r_cw = torch.from_numpy(c2w[:3, :3].astype(np.float32)).to(self.device)
        t_cw = torch.from_numpy(c2w[:3, 3].astype(np.float32)).to(self.device)

        fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
        cx, cy = float(intrinsics.cx), float(intrinsics.cy)

        # Slice-by-slice integration along Z to maintain low GPU memory footprint
        chunk_slice_z = 32
        for z_start in range(0, self.nz, chunk_slice_z):
            z_end = min(z_start + chunk_slice_z, self.nz)
            
            # Meshgrid for this sub-volume
            grid_x, grid_y, grid_z = torch.meshgrid(
                self.x_coords,
                self.y_coords,
                self.z_coords[z_start:z_end],
                indexing="ij",
            )
            pts_world = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (Nx, Ny, Nz_chunk, 3)

            # Transform to camera coordinates: P_cam = (P_world - t_cw) @ R_cw
            pts_cam = torch.matmul(pts_world - t_cw, r_cw)

            # In OpenGL camera coordinates, viewing distance along optical axis is -Z
            proj_z = -pts_cam[..., 2]

            # In-front of camera check
            valid_z = (proj_z >= z_near) & (proj_z <= z_far)

            # Project to pixel coordinates: u = x*fx/z + cx, v = -y*fy/z + cy
            u = (fx * (pts_cam[..., 0] / torch.clamp(proj_z, min=1e-4)) + cx).round().long()
            v = (-fy * (pts_cam[..., 1] / torch.clamp(proj_z, min=1e-4)) + cy).round().long()

            valid_uv = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

            if not torch.any(valid_uv):
                continue

            u_valid = u[valid_uv]
            v_valid = v[valid_uv]
            z_valid = proj_z[valid_uv]

            d_obs = depth_tensor[v_valid, u_valid]
            valid_depth = (d_obs > z_near) & (d_obs < z_far) & torch.isfinite(d_obs)

            if not torch.any(valid_depth):
                continue

            # Compute signed distance: positive in free space (in front), negative behind surface
            dist = d_obs - z_valid
            valid_dist = valid_depth & (dist >= -self.truncation_dist)

            if not torch.any(valid_dist):
                continue

            # Normalized truncated signed distance in [-1.0, 1.0]
            tsdf_val = torch.clamp(dist / self.truncation_dist, min=-1.0, max=1.0)

            # Mask indices in this sub-volume
            active_mask = torch.zeros_like(valid_uv)
            # Scatter active valid positions
            indices_valid = torch.nonzero(valid_uv)
            active_indices = indices_valid[valid_dist]

            ix = active_indices[:, 0]
            iy = active_indices[:, 1]
            iz = active_indices[:, 2] + z_start

            # Running weighted TSDF accumulation
            w_old = self.weight[ix, iy, iz]
            w_new = w_old + weight_multiplier
            t_old = self.tsdf[ix, iy, iz]

            t_val = tsdf_val[valid_dist]
            t_updated = (t_old * w_old + t_val * weight_multiplier) / torch.clamp(w_new, min=1e-5)

            self.tsdf[ix, iy, iz] = t_updated
            self.weight[ix, iy, iz] = torch.clamp(w_new, max=50.0)

            # Color accumulation
            rgb_val = rgb_tensor[v_valid[valid_dist], u_valid[valid_dist]]
            c_old = self.color[ix, iy, iz]
            self.color[ix, iy, iz] = (c_old * w_old.unsqueeze(-1) + rgb_val * weight_multiplier) / torch.clamp(w_new.unsqueeze(-1), min=1e-5)

    def extract_multiscale_surfels(
        self,
        min_weight: float = 2.0,
        max_surfels: int = 500_000,
        curvature_low: float = 0.06,
        curvature_high: float = 0.16,
    ) -> SurfelCloud:
        """Extract multi-scale hierarchical surfels at continuous zero crossings."""
        with torch.no_grad():
            tsdf_cpu = self.tsdf.cpu().numpy()
            weight_cpu = self.weight.cpu().numpy()
            color_cpu = self.color.cpu().numpy()

        # 1. Detect zero crossings in 3D: where adjacent voxels change sign and weight is sufficient
        # Check sign transitions along X, Y, and Z axes
        sign_x = (tsdf_cpu[:-1, :, :] * tsdf_cpu[1:, :, :]) <= 0.0
        sign_y = (tsdf_cpu[:, :-1, :] * tsdf_cpu[:, 1:, :]) <= 0.0
        sign_z = (tsdf_cpu[:, :, :-1] * tsdf_cpu[:, :, 1:]) <= 0.0

        # Pad to match volume dimensions
        is_zc_x = np.pad(sign_x, ((0, 1), (0, 0), (0, 0)), mode="constant")
        is_zc_y = np.pad(sign_y, ((0, 0), (0, 1), (0, 0)), mode="constant")
        is_zc_z = np.pad(sign_z, ((0, 0), (0, 0), (0, 1)), mode="constant")

        zero_crossing = (is_zc_x | is_zc_y | is_zc_z) & (weight_cpu >= min_weight) & (np.abs(tsdf_cpu) < 0.85)

        if not np.any(zero_crossing):
            raise RuntimeError("No valid zero crossings found in TSDF volume.")

        zc_indices = np.argwhere(zero_crossing)  # (M, 3)
        ix, iy, iz = zc_indices[:, 0], zc_indices[:, 1], zc_indices[:, 2]

        # 2. Compute spatial gradients \nabla TSDF via central differences to obtain surface normals
        gx = np.zeros_like(tsdf_cpu)
        gy = np.zeros_like(tsdf_cpu)
        gz = np.zeros_like(tsdf_cpu)

        gx[1:-1, :, :] = (tsdf_cpu[2:, :, :] - tsdf_cpu[:-2, :, :]) / (2.0 * self.voxel_size)
        gy[:, 1:-1, :] = (tsdf_cpu[:, 2:, :] - tsdf_cpu[:, :-2, :]) / (2.0 * self.voxel_size)
        gz[:, :, 1:-1] = (tsdf_cpu[:, :, 2:] - tsdf_cpu[:, :, :-2]) / (2.0 * self.voxel_size)

        nx = gx[ix, iy, iz]
        ny = gy[ix, iy, iz]
        nz = gz[ix, iy, iz]
        normals = np.column_stack([nx, ny, nz]).astype(np.float32)

        # Normalize normal vectors
        n_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        valid_norm = (n_len.ravel() > 1e-4)
        normals = normals / np.maximum(n_len, 1e-6)

        ix, iy, iz = ix[valid_norm], iy[valid_norm], iz[valid_norm]
        normals = normals[valid_norm]

        # 3. Sub-voxel position refinement via linear TSDF zero interpolation
        # P_world = bbox_min + idx * voxel_size - (tsdf / norm_grad) * normal
        x_base = self.bbox_min[0] + ix * self.voxel_size
        y_base = self.bbox_min[1] + iy * self.voxel_size
        z_base = self.bbox_min[2] + iz * self.voxel_size
        pos_base = np.column_stack([x_base, y_base, z_base])

        t_val = tsdf_cpu[ix, iy, iz]
        # Shift slightly along normal to find exact 0-crossing
        sub_voxel_shift = np.clip(t_val * self.truncation_dist, -self.voxel_size, self.voxel_size)
        positions = (pos_base - normals * sub_voxel_shift[:, None]).astype(np.float32)

        colors = np.clip(color_cpu[ix, iy, iz], 0.0, 1.0).astype(np.float32)

        # 4. Multi-Scale Curvature & Normal Variance Classification
        # Measure local normal disagreement across neighbors to identify flat vs edge regions
        # Fast spatial hashing for local neighborhood curvature
        cell_size = self.voxel_size * 2.0
        hash_coords = np.floor(positions / cell_size).astype(np.int32)
        _, group_inv = np.unique(hash_coords, axis=0, return_inverse=True)

        # Approximate curvature from normal divergence in local cell
        n_pts = len(positions)
        curvatures = np.zeros(n_pts, dtype=np.float32)
        
        # Vectorized mean normal per cell
        sum_normals = np.zeros((np.max(group_inv) + 1, 3), dtype=np.float32)
        np.add.at(sum_normals, group_inv, normals)
        cell_counts = np.bincount(group_inv)
        mean_normals = sum_normals / np.maximum(cell_counts[:, None], 1)
        mean_norm_len = np.linalg.norm(mean_normals, axis=-1)
        curvatures = 1.0 - mean_norm_len[group_inv]

        # 5. 3-Tier Multi-Scale Stride Sampling
        tier_edge = curvatures >= curvature_high
        tier_obj = (curvatures >= curvature_low) & (curvatures < curvature_high)
        tier_flat = curvatures < curvature_low

        selected_indices = []
        scales_list = []

        # Tier 0: Fine surfels on sharp edges (keep 100% density, stride 1, sigma = 0.9 * v)
        if np.any(tier_edge):
            idx_edge = np.where(tier_edge)[0]
            selected_indices.append(idx_edge)
            scales_list.append(np.full((len(idx_edge), 2), self.voxel_size * 0.75, dtype=np.float32))

        # Tier 1: Medium surfels on curved objects (stride 2 = 3cm, sigma = 1.8 * v)
        if np.any(tier_obj):
            idx_obj = np.where(tier_obj)[0]
            # Spatial voxel decimation at 3cm
            v_obj_coords = np.floor(positions[idx_obj] / (self.voxel_size * 2.0)).astype(np.int32)
            _, u_obj = np.unique(v_obj_coords, axis=0, return_index=True)
            chosen_obj = idx_obj[u_obj]
            selected_indices.append(chosen_obj)
            scales_list.append(np.full((len(chosen_obj), 2), self.voxel_size * 1.5, dtype=np.float32))

        # Tier 2: Broad surfels on flat walls/floors (stride 4 = 6cm, sigma = 3.6 * v)
        if np.any(tier_flat):
            idx_flat = np.where(tier_flat)[0]
            # Spatial voxel decimation at 6cm
            v_flat_coords = np.floor(positions[idx_flat] / (self.voxel_size * 4.0)).astype(np.int32)
            _, u_flat = np.unique(v_flat_coords, axis=0, return_index=True)
            chosen_flat = idx_flat[u_flat]
            selected_indices.append(chosen_flat)
            scales_list.append(np.full((len(chosen_flat), 2), self.voxel_size * 3.0, dtype=np.float32))

        all_idx = np.concatenate(selected_indices)
        all_scales = np.concatenate(scales_list, axis=0)

        final_pos = positions[all_idx]
        final_norm = normals[all_idx]
        final_col = colors[all_idx]

        # Enforce budget cap if needed
        if len(final_pos) > max_surfels:
            perm = np.random.RandomState(42).permutation(len(final_pos))[:max_surfels]
            final_pos = final_pos[perm]
            final_norm = final_norm[perm]
            final_col = final_col[perm]
            all_scales = all_scales[perm]

        tangent_u, tangent_v = build_orthonormal_tangent_frame(final_norm)
        sh_degree_0 = (final_col * SH_C0).astype(np.float32)
        opacities = np.full(len(final_pos), 0.95, dtype=np.float32)

        return SurfelCloud(
            positions=final_pos,
            normals=final_norm,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            scales_2d=all_scales,
            colors_rgb=final_col,
            sh_degree_0=sh_degree_0,
            opacities=opacities,
        )
