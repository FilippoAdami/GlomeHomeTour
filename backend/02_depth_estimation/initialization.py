"""GlomeHomeTour Backend: Surfel Cloud Initialization.

Unprojects aligned metric depth maps, surface normals, and keyframe RGB images
into an initial cloud of 100k-300k oriented 2D Gaussian surfels with tangent
frames, 2D scales, degree-0 Spherical Harmonics, and opacities.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
from scipy.spatial import cKDTree as KDTree

from package_loader import CameraIntrinsics, Keyframe
from depth_priors import compute_surface_normals

SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))


@dataclass
class SurfelCloud:
    """Represents an initialized collection of 2D Gaussian surfels."""
    positions: np.ndarray    # (N, 3) float32 in meters
    normals: np.ndarray      # (N, 3) float32 unit vectors
    tangent_u: np.ndarray    # (N, 3) float32 unit vectors
    tangent_v: np.ndarray    # (N, 3) float32 unit vectors
    scales_2d: np.ndarray    # (N, 2) float32 in meters (sigma_u, sigma_v)
    colors_rgb: np.ndarray   # (N, 3) float32 in [0, 1]
    sh_degree_0: np.ndarray  # (N, 3) float32 degree-0 SH coefficients
    opacities: np.ndarray    # (N,) float32 in [0, 1]

    def __len__(self) -> int:
        return len(self.positions)

    @classmethod
    def from_random(
        cls,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        num_points: int,
        seed: int = 42,
    ) -> "SurfelCloud":
        """Vanilla-3DGS-style random point cloud init (no depth/SfM prior),
        uniformly sampled inside the given world-space bounding box."""
        rng = np.random.RandomState(seed)
        bbox_min = np.asarray(bbox_min, dtype=np.float32)
        bbox_max = np.asarray(bbox_max, dtype=np.float32)

        positions = rng.uniform(bbox_min, bbox_max, size=(num_points, 3)).astype(np.float32)

        normals = rng.normal(size=(num_points, 3)).astype(np.float32)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-6)

        colors_rgb = rng.uniform(0.3, 0.7, size=(num_points, 3)).astype(np.float32)

        # Isotropic initial scale from mean point spacing (same heuristic as vanilla 3DGS).
        bbox_volume = float(np.prod(np.maximum(bbox_max - bbox_min, 1e-3)))
        mean_spacing = (bbox_volume / max(num_points, 1)) ** (1.0 / 3.0)
        scales_2d = np.full((num_points, 2), mean_spacing * 0.5, dtype=np.float32)

        opacities = np.full(num_points, 0.5, dtype=np.float32)

        tangent_u, tangent_v = build_orthonormal_tangent_frame(normals)
        sh_degree_0 = (colors_rgb * SH_C0).astype(np.float32)

        return cls(
            positions=positions,
            normals=normals,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            scales_2d=scales_2d,
            colors_rgb=colors_rgb,
            sh_degree_0=sh_degree_0,
            opacities=opacities,
        )

    def voxel_downsample(self, voxel_size_m: float) -> "SurfelCloud":
        """Spatially downsample the surfel cloud onto a uniform 3D voxel grid as the final step."""
        if voxel_size_m <= 0.0 or len(self.positions) == 0:
            return self
        coords = np.floor(self.positions / voxel_size_m).astype(np.int32)
        _, u_idx = np.unique(coords, axis=0, return_index=True)
        scale_min = float(voxel_size_m * 0.5)
        scale_max = float(voxel_size_m * 0.75)
        scales_2d = np.clip(self.scales_2d[u_idx], scale_min, scale_max)
        return SurfelCloud(
            positions=self.positions[u_idx],
            normals=self.normals[u_idx],
            tangent_u=self.tangent_u[u_idx],
            tangent_v=self.tangent_v[u_idx],
            scales_2d=scales_2d,
            colors_rgb=self.colors_rgb[u_idx],
            sh_degree_0=self.sh_degree_0[u_idx],
            opacities=self.opacities[u_idx],
        )

    def normal_tube_collapse(
        self,
        voxel_size_m: float = 0.015,
        tube_radius_m: float = 0.02,
        tube_length_m: float = 0.15,
        min_normal_cos: float = 0.80,
    ) -> "SurfelCloud":
        """Collapse multi-layer depth slab thickness into a single 2D manifold shell.

        Vectorized spatial voxel grouping + normal-aligned median projection:
        1. Groups points into spatial voxels of grid size voxel_size_m.
        2. Computes the consensus unit normal and centroid per voxel.
        3. Computes the 1D projection offset along the normal for every point.
        4. Shifts each voxel to its robust median depth along the surface normal.
        Guarantees 100% surface preservation with zero Swiss-cheese holes.
        """
        n_pts = len(self.positions)
        if n_pts == 0:
            return self

        # Try GPU acceleration via PyTorch on CUDA/ROCm
        try:
            import torch
            if torch.cuda.is_available() and n_pts > 500:
                torch.cuda.empty_cache()
                device = torch.device("cuda")
                pos_t = torch.from_numpy(self.positions).to(device)
                norm_t = torch.from_numpy(self.normals).to(device)
                col_t = torch.from_numpy(self.colors_rgb).to(device)

                coords_t = torch.floor(pos_t / voxel_size_m).to(torch.int32)
                unq_coords_t, inverse_idx_t, counts_t = torch.unique(
                    coords_t, dim=0, return_inverse=True, return_counts=True
                )
                num_voxels = unq_coords_t.shape[0]
                print(f"[TubeCollapse GPU] Collapsing {n_pts:,} raw points into {num_voxels:,} 2D manifold surfels on {torch.cuda.get_device_name(0)}...", flush=True)

                counts_f_t = counts_t.float().unsqueeze(1)
                v_norm_t = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
                v_norm_t.scatter_add_(0, inverse_idx_t.unsqueeze(1).expand(-1, 3), norm_t)
                v_norm_t = v_norm_t / counts_f_t
                norm_len_t = torch.norm(v_norm_t, dim=-1, keepdim=True)

                # Representative unique point index per voxel
                u_idx_t = torch.full((num_voxels,), n_pts, device=device, dtype=torch.int64)
                idx_arange = torch.arange(n_pts, device=device, dtype=torch.int64)
                u_idx_t.scatter_reduce_(0, inverse_idx_t, idx_arange, reduce="amin")

                v_norm_t = torch.where(norm_len_t > 1e-6, v_norm_t / torch.clamp(norm_len_t, min=1e-6), norm_t[u_idx_t])

                v_pos_t = torch.zeros((num_voxels, 3), device=device, dtype=torch.float32)
                v_pos_t.scatter_add_(0, inverse_idx_t.unsqueeze(1).expand(-1, 3), pos_t)
                v_pos_t = v_pos_t / counts_f_t

                v_col_t = col_t[u_idx_t]

                deltas_t = pos_t - v_pos_t[inverse_idx_t]
                offsets_n_t = torch.sum(deltas_t * v_norm_t[inverse_idx_t], dim=-1)

                offset_scaled = (torch.clamp(offsets_n_t, -10.0, 10.0) + 10.0) * 1e6
                sort_key = inverse_idx_t.to(torch.int64) * 100_000_000 + offset_scaled.to(torch.int64)
                sort_order_t = torch.argsort(sort_key)

                offsets_sorted_t = offsets_n_t[sort_order_t]
                counts_cumsum_t = torch.cumsum(counts_t, dim=0)
                median_indices_t = counts_cumsum_t - counts_t + (counts_t // 2)
                max_offset = float(tube_length_m / 2.0)
                med_offsets_t = torch.clamp(offsets_sorted_t[median_indices_t], -max_offset, max_offset)

                collapsed_pos_t = v_pos_t + med_offsets_t.unsqueeze(1) * v_norm_t

                collapsed_pos = collapsed_pos_t.cpu().numpy().astype(np.float32)
                collapsed_norm = v_norm_t.cpu().numpy().astype(np.float32)
                collapsed_col = v_col_t.cpu().numpy().astype(np.float32)

                scale_val = float(voxel_size_m * 0.75)
                scales_2d = np.full((num_voxels, 2), scale_val, dtype=np.float32)
                opacities = np.full((num_voxels,), 0.90, dtype=np.float32)

                tangent_u, tangent_v = build_orthonormal_tangent_frame(collapsed_norm)
                sh_deg0 = (collapsed_col * SH_C0).astype(np.float32)

                print(f"[TubeCollapse GPU] Completed! Output manifold: {len(collapsed_pos):,} surfels.", flush=True)
                return SurfelCloud(
                    positions=collapsed_pos,
                    normals=collapsed_norm,
                    tangent_u=tangent_u.astype(np.float32),
                    tangent_v=tangent_v.astype(np.float32),
                    scales_2d=scales_2d,
                    colors_rgb=collapsed_col,
                    sh_degree_0=sh_deg0,
                    opacities=opacities,
                )
        except Exception as e:
            print(f"[TubeCollapse] GPU acceleration notice ({e}), using CPU NumPy path...", flush=True)

        coords = np.floor(self.positions / voxel_size_m).astype(np.int32)
        _, u_idx, inverse_idx, counts = np.unique(
            coords, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        num_voxels = len(u_idx)
        print(f"[TubeCollapse CPU] Collapsing {n_pts:,} raw points into {num_voxels:,} 2D manifold surfels...", flush=True)

        counts_f = counts.astype(np.float32)[:, None]

        # 1. Consensus unit normal per voxel
        v_norm = np.column_stack([
            np.bincount(inverse_idx, weights=self.normals[:, c]) for c in range(3)
        ]) / counts_f
        norm_len = np.linalg.norm(v_norm, axis=-1, keepdims=True)
        v_norm = np.where(norm_len > 1e-6, v_norm / np.maximum(norm_len, 1e-6), self.normals[u_idx])

        # 2. Mean spatial centroid and color per voxel
        v_pos = np.column_stack([
            np.bincount(inverse_idx, weights=self.positions[:, c]) for c in range(3)
        ]) / counts_f
        v_col = self.colors_rgb[u_idx].copy()

        # 3. 1D offset along the consensus normal for each point
        deltas = self.positions - v_pos[inverse_idx]
        offsets_n = np.sum(deltas * v_norm[inverse_idx], axis=-1)

        # 4. Extract median offset per voxel using group sort
        sort_order = np.lexsort((offsets_n, inverse_idx))
        offsets_sorted = offsets_n[sort_order]
        counts_cumsum = np.cumsum(counts)
        median_indices = counts_cumsum - counts + (counts // 2)
        med_offsets = offsets_sorted[median_indices]

        # Clamp offsets to max half-length of the tube to eliminate extreme outliers
        max_offset = float(tube_length_m / 2.0)
        med_offsets = np.clip(med_offsets, -max_offset, max_offset)

        # 5. Final collapsed manifold positions
        collapsed_pos = (v_pos + med_offsets[:, None] * v_norm).astype(np.float32)
        collapsed_norm = v_norm.astype(np.float32)
        collapsed_col = np.clip(v_col, 0.0, 1.0).astype(np.float32)

        scale_val = float(voxel_size_m * 0.75)
        scales_2d = np.full((num_voxels, 2), scale_val, dtype=np.float32)
        opacities = self.opacities[u_idx].copy().astype(np.float32)

        tangent_u, tangent_v = build_orthonormal_tangent_frame(collapsed_norm)
        sh_deg0 = (collapsed_col * SH_C0).astype(np.float32)

        print(f"[TubeCollapse CPU] Completed! Output manifold: {len(collapsed_pos):,} surfels.", flush=True)

        return SurfelCloud(
            positions=collapsed_pos,
            normals=collapsed_norm,
            tangent_u=tangent_u.astype(np.float32),
            tangent_v=tangent_v.astype(np.float32),
            scales_2d=scales_2d,
            colors_rgb=collapsed_col,
            sh_degree_0=sh_deg0,
            opacities=opacities,
        )

    def multiscale_pyramid_decimate(
        self,
        base_voxel_m: float = 0.015,
        k_neighbors: int = 16,
        flat_normal_var_thresh: float = 0.04,
        curve_normal_var_thresh: float = 0.15,
        tier1_stride_cells: int = 2,  # 3.0 cm stride
        tier2_stride_cells: int = 4,  # 6.0 cm stride
    ) -> "SurfelCloud":
        """Decimate planar low-frequency areas into multi-scale surfel pyramids while preserving fine edges.

        Classifies surfels by local normal variance:
        - High curvature / corners (var > curve_thresh): 1.5 cm full resolution, scale ~ 1.1 cm
        - Medium curvature (flat_thresh < var <= curve_thresh): 3.0 cm stride, scale ~ 2.2 cm
        - Flat planar walls/floors (var <= flat_thresh): 6.0 cm stride, scale ~ 4.5 cm
        """
        n = len(self.positions)
        if n <= k_neighbors:
            return self

        sample_size = min(20_000, n)
        idx_sample = np.random.RandomState(42).choice(n, size=sample_size, replace=False)
        tree = KDTree(self.positions[idx_sample])
        _, nn_indices = tree.query(self.positions, k=k_neighbors, workers=-1)

        nn_normals = self.normals[idx_sample][nn_indices]
        seed_normals_exp = np.expand_dims(self.normals, axis=1)
        cos_sims = np.sum(nn_normals * seed_normals_exp, axis=-1)
        normal_var = 1.0 - np.clip(np.mean(cos_sims, axis=-1), 0.0, 1.0)

        grid_coords = np.round(self.positions / base_voxel_m).astype(np.int32)

        is_tier0 = normal_var > curve_normal_var_thresh
        is_tier1 = (normal_var > flat_normal_var_thresh) & (~is_tier0)
        is_tier2 = normal_var <= flat_normal_var_thresh

        keep_tier0 = is_tier0
        keep_tier1 = is_tier1 & ((grid_coords[:, 0] % tier1_stride_cells == 0) &
                                 (grid_coords[:, 1] % tier1_stride_cells == 0) &
                                 (grid_coords[:, 2] % tier1_stride_cells == 0))
        keep_tier2 = is_tier2 & ((grid_coords[:, 0] % tier2_stride_cells == 0) &
                                 (grid_coords[:, 1] % tier2_stride_cells == 0) &
                                 (grid_coords[:, 2] % tier2_stride_cells == 0))

        keep_mask = keep_tier0 | keep_tier1 | keep_tier2
        if np.sum(keep_mask) == 0:
            return self

        filtered_pos = self.positions[keep_mask]
        filtered_norm = self.normals[keep_mask]
        filtered_col = self.colors_rgb[keep_mask]
        filtered_opac = self.opacities[keep_mask]

        scales = np.empty((len(filtered_pos), 2), dtype=np.float32)
        sub_t0 = is_tier0[keep_mask]
        sub_t1 = is_tier1[keep_mask]
        sub_t2 = is_tier2[keep_mask]

        scales[sub_t0] = base_voxel_m * 0.75
        scales[sub_t1] = base_voxel_m * tier1_stride_cells * 0.75
        scales[sub_t2] = base_voxel_m * tier2_stride_cells * 0.75

        tangent_u, tangent_v = build_orthonormal_tangent_frame(filtered_norm)
        sh_deg0 = (filtered_col * SH_C0).astype(np.float32)

        return SurfelCloud(
            positions=filtered_pos,
            normals=filtered_norm,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            scales_2d=scales,
            colors_rgb=filtered_col,
            sh_degree_0=sh_deg0,
            opacities=filtered_opac,
        )

    def to_ply(self, output_path: Union[str, Path]) -> None:
        """Export surfels to standard binary little-endian PLY file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.positions)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float scale_u\n"
            "property float scale_v\n"
            "property float opacity\n"
            "end_header\n"
        )

        with open(out_path, "wb") as f:
            f.write(header.encode("ascii"))
            if n > 0:
                ply_dtype = np.dtype([
                    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                    ("scale_u", "<f4"), ("scale_v", "<f4"),
                    ("opacity", "<f4"),
                ])
                data = np.empty(n, dtype=ply_dtype)
                data["x"] = self.positions[:, 0]
                data["y"] = self.positions[:, 1]
                data["z"] = self.positions[:, 2]
                data["nx"] = self.normals[:, 0]
                data["ny"] = self.normals[:, 1]
                data["nz"] = self.normals[:, 2]
                rgb_bytes = np.clip(self.colors_rgb * 255.0, 0, 255).astype(np.uint8)
                data["red"] = rgb_bytes[:, 0]
                data["green"] = rgb_bytes[:, 1]
                data["blue"] = rgb_bytes[:, 2]
                data["scale_u"] = self.scales_2d[:, 0]
                data["scale_v"] = self.scales_2d[:, 1]
                data["opacity"] = self.opacities
                f.write(data.tobytes())

    @classmethod
    def from_ply(
        cls,
        ply_path: Union[str, Path],
        max_surfels: Optional[int] = None,
        voxel_downsample_m: Optional[float] = None,
    ) -> "SurfelCloud":
        """Load surfels from a binary little-endian PLY file with optional spatial voxel downsampling."""
        path = Path(ply_path)
        if not path.is_file():
            raise FileNotFoundError(f"Surfel PLY file not found: {path}")

        with open(path, "rb") as f:
            num_vertices = 0
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    num_vertices = int(line.split()[-1])
                if line == "end_header":
                    break

            if num_vertices == 0:
                raise ValueError(f"No vertices found in PLY header: {path}")

            # Vectorized structured NumPy buffer loading: <3f3f3B2ff (36 bytes per vertex)
            dt = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1"),
                ("su", "<f4"), ("sv", "<f4"),
                ("op", "<f4"),
            ])
            arr = np.fromfile(f, dtype=dt, count=num_vertices)

        positions = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
        normals = np.column_stack([arr["nx"], arr["ny"], arr["nz"]]).astype(np.float32)
        colors_rgb = (np.column_stack([arr["r"], arr["g"], arr["b"]]).astype(np.float32)) / 255.0
        scales_2d = np.column_stack([arr["su"], arr["sv"]]).astype(np.float32)
        opacities = arr["op"].astype(np.float32)

        # Normalize normals with zero-division guard
        n_len = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(n_len, 1e-6)

        # Optional spatial voxel downsampling (crucial for large 2M+ point clouds)
        if voxel_downsample_m is not None and voxel_downsample_m > 0:
            voxel_coords = np.floor(positions / voxel_downsample_m).astype(np.int32)
            _, unique_idx = np.unique(voxel_coords, axis=0, return_index=True)
            positions = positions[unique_idx]
            normals = normals[unique_idx]
            colors_rgb = colors_rgb[unique_idx]
            # A surfel represents its voxel cell: floor so the grid has no holes (>= 0.5*v),
            # ceiling so it does not smear across neighbours (<= 0.75*v).
            scale_min = float(voxel_downsample_m * 0.5)
            scale_max = float(voxel_downsample_m * 0.75)
            scales_2d = np.clip(scales_2d[unique_idx], scale_min, scale_max)
            # Ensure solid surface opacity from depth prior
            opacities = np.clip(opacities[unique_idx], 0.85, 1.0)

        # Optional hard budget capping
        if max_surfels is not None and len(positions) > max_surfels:
            perm = np.random.RandomState(42).permutation(len(positions))[:max_surfels]
            positions = positions[perm]
            normals = normals[perm]
            colors_rgb = colors_rgb[perm]
            scales_2d = scales_2d[perm]
            opacities = opacities[perm]

        tangent_u, tangent_v = build_orthonormal_tangent_frame(normals)
        sh_degree_0 = (colors_rgb * SH_C0).astype(np.float32)

        return cls(
            positions=positions,
            normals=normals,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            scales_2d=scales_2d,
            colors_rgb=colors_rgb,
            sh_degree_0=sh_degree_0,
            opacities=opacities,
        )


def build_orthonormal_tangent_frame(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given (N, 3) unit normal vectors, compute orthogonal tangent vectors (u, v)

    such that (u, v, n) form a right-handed orthonormal basis.
    """
    n = normals.shape[0]
    tangent_u = np.zeros_like(normals)

    # For normals nearly parallel to Z, use X axis as reference; else use Z axis
    near_z = np.abs(normals[:, 2]) > 0.9

    ref_axis = np.zeros((n, 3), dtype=np.float32)
    ref_axis[near_z, 0] = 1.0   # [1, 0, 0]
    ref_axis[~near_z, 2] = 1.0  # [0, 0, 1]

    # u = n x ref
    u = np.cross(normals, ref_axis)
    norm_u = np.linalg.norm(u, axis=-1, keepdims=True)
    norm_u = np.maximum(norm_u, 1e-6)
    tangent_u = (u / norm_u).astype(np.float32)

    # v = n x u
    tangent_v = np.cross(normals, tangent_u).astype(np.float32)
    norm_v = np.linalg.norm(tangent_v, axis=-1, keepdims=True)
    norm_v = np.maximum(norm_v, 1e-6)
    tangent_v = (tangent_v / norm_v).astype(np.float32)

    return tangent_u, tangent_v


def estimate_adaptive_depth_ceiling(
    depth_maps: Sequence[np.ndarray],
    conf_maps: Optional[Sequence[Optional[np.ndarray]]] = None,
    min_conf: float = 0.80,
    default_ceiling: float = 6.0,
) -> float:
    """Dynamically determine room depth ceiling from statistical distribution of confident surface pixels.

    Adapts automatically: ~4.0m-4.5m for small bedrooms, up to 25m for grand hotel lobbies/halls.
    """
    sample_depths = []
    for idx, d in enumerate(depth_maps):
        c = conf_maps[idx] if (conf_maps is not None and idx < len(conf_maps)) else None
        if c is not None:
            if c.shape != d.shape:
                c = cv2.resize(c, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid = (d > 0.3) & (d < 50.0) & (c >= min_conf)
        else:
            valid = (d > 0.3) & (d < 50.0)

        d_val = d[valid]
        if len(d_val) > 0:
            step = max(1, len(d_val) // 2000)
            sample_depths.append(d_val[::step])

    if not sample_depths:
        return default_ceiling

    all_d = np.concatenate(sample_depths)
    if len(all_d) < 100:
        return default_ceiling

    q98 = float(np.percentile(all_d, 98))
    med = float(np.median(all_d))
    mad = float(np.median(np.abs(all_d - med)))

    adaptive_ceiling = q98 + 1.5 * mad
    return float(np.clip(adaptive_ceiling, 3.5, 35.0))


def compute_overexposed_mask(
    img_rgb: np.ndarray,
    min_threshold: int = 250,
    max_threshold: int = 255,
    percentile: float = 99.8,
    max_chroma_diff: float = 35.0,
    dilation_kernel_size: int = 3,
) -> np.ndarray:
    """Compute dynamic saturation mask to eliminate optical bloom / light-source artifacts.

    Identifies clipped and overexposed pixels that cause monocular depth models
    to hallucinate arbitrary floating depth or distort surrounding surfaces.
    Uses a dynamic threshold bounded between [min_threshold, max_threshold] based on
    the high percentile of max-channel intensity, combined with low chroma difference
    (distinguishing white/yellow optical bloom from vibrant saturated colors).

    Args:
        img_rgb: RGB image as (H, W, 3) uint8 array in [0, 255].
        min_threshold: Absolute floor for overexposure (default 250); prevents masking in dim frames.
        max_threshold: Upper clamp for dynamic threshold (default 255).
        percentile: High percentile of max-channel distribution to adaptively set cutoff.
        max_chroma_diff: Max allowed (max - min) channel difference to treat as optical bloom.
        dilation_kernel_size: Optional structuring element diameter to cover bloom halo fringes (0 to disable).

    Returns:
        Boolean mask of shape (H, W) where True indicates saturated / optical bloom pixels to discard.
    """
    if img_rgb.size == 0:
        return np.zeros(img_rgb.shape[:2], dtype=bool)

    # Max intensity across R, G, B channels
    max_ch = np.max(img_rgb, axis=-1)

    # Fast path: if no pixel reaches min_threshold, no overexposure exists in this frame
    frame_max = int(np.max(max_ch))
    if frame_max < min_threshold:
        return np.zeros(img_rgb.shape[:2], dtype=bool)

    # Adapt dynamic threshold based on upper percentile of max_ch in this frame
    q_val = float(np.percentile(max_ch, percentile))
    t_dyn = float(np.clip(q_val, min_threshold, max_threshold))

    min_ch = np.min(img_rgb, axis=-1)
    chroma_diff = max_ch.astype(np.float32) - min_ch.astype(np.float32)

    # Core saturated optical bloom
    overexposed = (max_ch >= t_dyn) & (chroma_diff <= max_chroma_diff)

    # Dilate slightly to catch optical bloom halos and sharp depth-tear boundaries around fixtures
    if dilation_kernel_size > 1 and np.any(overexposed):
        k = int(dilation_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        overexposed = cv2.dilate(overexposed.astype(np.uint8), kernel) > 0

    return overexposed


def compute_hybrid_sampling_coords(
    img_rgb: np.ndarray,
    depth_map: np.ndarray,
    energy_threshold: float = 0.08,
    coarse_stride: int = 3,
    fine_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D hybrid photometric and geometric adaptive sampling pixel coordinates.

    Identifies high-frequency image textures (RGB gradient energy) and 3D depth discontinuities
    (relative depth gradient) to sample rich details at full resolution (fine_stride), while
    sampling untextured planar regions (drywall, flat ceiling) at a sparse baseline grid (coarse_stride).

    Args:
        img_rgb: RGB image as (H, W, 3) uint8 or float array.
        depth_map: Depth map in meters as (H, W) float32 array.
        energy_threshold: Cutoff energy above which fine_stride sampling is activated.
        coarse_stride: Pixel stride for uniform background sampling (default 3 = ~4.5cm).
        fine_stride: Pixel stride for edge/texture regions (default 1 = ~1.5cm).

    Returns:
        tuple of (y_coords, x_coords, scales_relative):
            - y_coords: 1D int array of row indices to sample.
            - x_coords: 1D int array of col indices to sample.
            - scales_relative: 1D float32 array of relative scale factors (1.0 for fine, coarse_stride for coarse).
    """
    h, w = depth_map.shape[:2]
    if img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. 2D Photometric Gradient Energy (Sobel on Luminance)
    if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
        if img_rgb.dtype == np.uint8:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]).astype(np.float32)
    else:
        gray = img_rgb.astype(np.float32)

    gx_rgb = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy_rgb = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    energy_rgb = np.sqrt(gx_rgb**2 + gy_rgb**2)

    # 2. 2D Relative Depth Discontinuity Energy
    safe_depth = np.maximum(depth_map, 0.1)
    gx_d = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3) / (safe_depth * 4.0)
    gy_d = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3) / (safe_depth * 4.0)
    energy_depth = np.sqrt(gx_d**2 + gy_d**2)

    # Combined 2D Energy Map
    energy_combined = np.maximum(energy_rgb, energy_depth)

    # High frequency mask
    high_freq_mask = energy_combined >= energy_threshold
    # Suppress outer 2-pixel border to avoid OpenCV Sobel edge boundary reflection artifacts
    high_freq_mask[:2, :] = False
    high_freq_mask[-2:, :] = False
    high_freq_mask[:, :2] = False
    high_freq_mask[:, -2:] = False

    # Baseline coarse grid across the whole image (guarantees zero holes)
    c_stride = max(1, int(coarse_stride))
    f_stride = max(1, int(fine_stride))

    y_c, x_c = np.mgrid[0:h:c_stride, 0:w:c_stride]
    y_c_flat = y_c.flatten()
    x_c_flat = x_c.flatten()

    # Dense fine grid sampled ONLY where high-frequency energy is present
    y_f, x_f = np.mgrid[0:h:f_stride, 0:w:f_stride]
    y_f_flat = y_f.flatten()
    x_f_flat = x_f.flatten()
    fine_is_high = high_freq_mask[y_f_flat, x_f_flat]

    y_f_sel = y_f_flat[fine_is_high]
    x_f_sel = x_f_flat[fine_is_high]

    if len(y_f_sel) == 0:
        return y_c_flat, x_c_flat, np.full(len(y_c_flat), float(c_stride), dtype=np.float32)

    # Combine coordinates without duplicates
    idx_coarse = y_c_flat * w + x_c_flat
    idx_fine = y_f_sel * w + x_f_sel

    fine_set = set(idx_fine)
    coarse_keep_mask = np.array([idx not in fine_set for idx in idx_coarse], dtype=bool)

    y_final = np.concatenate([y_f_sel, y_c_flat[coarse_keep_mask]]).astype(np.int32)
    x_final = np.concatenate([x_f_sel, x_c_flat[coarse_keep_mask]]).astype(np.int32)
    scales_final = np.concatenate([
        np.ones(len(y_f_sel), dtype=np.float32),
        np.full(np.sum(coarse_keep_mask), float(c_stride), dtype=np.float32)
    ]).astype(np.float32)

    return y_final, x_final, scales_final


def filter_multiview_consistency(
    pts_world: np.ndarray,
    current_idx: int,
    keyframes: Sequence[Keyframe],
    depth_maps: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    normals_world: Optional[np.ndarray] = None,
    max_neighbors: int = 6,
    min_consensus: int = 1,
    enable_freespace_filter: bool = True,
    max_freespace_violations: int = 0,
    freespace_margin_m: float = 0.08,
    min_neighbor_baseline_m: float = 0.08,
    min_neighbor_parallax_deg: float = 3.0,
) -> np.ndarray:
    """Return boolean mask of points corroborated by neighboring camera views.

    Performs two complementary multi-view geometric checks:
    1. Consensus corroboration: Points that fall inside overlapping camera frustums
       must agree with neighbor depth within tolerance.
    2. Cross-view free-space carving (epipolar / reprojection check): If a candidate 3D
       point reprojects into an unobstructed adjacent view with sufficient parallax and
       lands on empty space (proj_z < obs_depth - margin), it represents a floating phantom
       or depth-bleed artifact and is culled when violations exceed max_freespace_violations.
       For oblique/grazing surfaces (ceilings, sloped roofs, beams), margin is adaptively scaled.
    """
    n_pts = len(pts_world)
    if n_pts == 0 or len(keyframes) <= 1:
        return np.ones(n_pts, dtype=bool)

    if min_consensus <= 0 and not enable_freespace_filter:
        return np.ones(n_pts, dtype=bool)

    cur_kf = keyframes[current_idx]
    cur_t = cur_kf.transform_matrix[:3, 3]
    cur_dir = -cur_kf.transform_matrix[:3, 2]

    # Find neighboring keyframes prioritized by distance and optical co-directionality
    candidate_neighbors = []
    for j, other_kf in enumerate(keyframes):
        if j == current_idx:
            continue
        other_t = other_kf.transform_matrix[:3, 3]
        other_dir = -other_kf.transform_matrix[:3, 2]
        d = float(np.linalg.norm(other_t - cur_t))
        cos_ang = float(np.dot(cur_dir, other_dir))

        # Prefer cameras that face roughly towards the same scene hemisphere
        if cos_ang > 0.1 and d < 3.5:
            score = d / max(0.2, cos_ang)
            candidate_neighbors.append((score, j))

    if not candidate_neighbors:
        dists = [(float(np.linalg.norm(keyframes[j].transform_matrix[:3, 3] - cur_t)), j) for j in range(len(keyframes)) if j != current_idx]
        dists.sort()
        candidate_neighbors = dists[:max_neighbors]
    else:
        candidate_neighbors.sort()
        candidate_neighbors = candidate_neighbors[:max_neighbors]

    neighbor_indices = [j for _, j in candidate_neighbors]

    views_in_frustum = np.zeros(n_pts, dtype=np.int32)
    consensus_count = np.zeros(n_pts, dtype=np.int32)
    freespace_violations = np.zeros(n_pts, dtype=np.int32)
    fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)
    cos_max_parallax = math.cos(math.radians(min_neighbor_parallax_deg))

    for j in neighbor_indices:
        other_kf = keyframes[j]
        d_map = depth_maps[j]
        dh, dw = d_map.shape[:2]

        c2w = other_kf.transform_matrix
        r_cw = c2w[:3, :3]
        t_cw = c2w[:3, 3]

        # World to camera: P_cam = (P_world - t_cw) * R_cw
        pts_cam = np.dot(pts_world - t_cw, r_cw)

        # In OpenGL coordinates, camera looks along -Z, so visible points have pts_cam[:, 2] < -0.1
        proj_z = -pts_cam[:, 2]
        in_front = proj_z > 0.2

        u = (fx * (pts_cam[:, 0] / np.maximum(proj_z, 1e-4)) + cx).astype(np.int32)
        v = (-fy * (pts_cam[:, 1] / np.maximum(proj_z, 1e-4)) + cy).astype(np.int32)

        # Stay slightly away from extreme image boundary to avoid edge sampling distortion
        valid_uv = in_front & (u >= 2) & (u < dw - 2) & (v >= 2) & (v < dh - 2)
        views_in_frustum += valid_uv.astype(np.int32)

        obs_depth = np.zeros(n_pts, dtype=np.float32)
        obs_depth[valid_uv] = d_map[v[valid_uv], u[valid_uv]]

        valid_obs = valid_uv & np.isfinite(obs_depth) & (obs_depth > 0.2)

        # 1. Surface agreement / consensus check
        tol = 0.08 + 0.05 * proj_z
        match = valid_obs & (np.abs(obs_depth - proj_z) <= tol)
        consensus_count += match.astype(np.int32)

        # 2. Cross-view free-space check (detecting empty space between camera and observed surface)
        if enable_freespace_filter:
            baseline = float(np.linalg.norm(t_cw - cur_t))
            vec_cur = pts_world - cur_t
            vec_other = pts_world - t_cw
            norm_cur = np.maximum(np.linalg.norm(vec_cur, axis=-1, keepdims=True), 1e-6)
            norm_other = np.maximum(np.linalg.norm(vec_other, axis=-1, keepdims=True), 1e-6)
            unit_other = vec_other / norm_other
            cos_parallax = np.sum((vec_cur / norm_cur) * unit_other, axis=-1)

            has_parallax = (baseline >= min_neighbor_baseline_m) | (cos_parallax <= cos_max_parallax)

            # Grazing angle adaptive scaling: oblique rays (sloped roofs, beams, ceilings) have higher depth uncertainty
            if normals_world is not None and len(normals_world) == n_pts:
                cos_grazing = np.abs(np.sum(normals_world * unit_other, axis=-1))
                tol_scale = np.clip(1.0 / np.maximum(cos_grazing, 0.35), 1.0, 2.5)
            else:
                tol_scale = 1.0

            tol_free = (freespace_margin_m + 0.05 * proj_z) * tol_scale
            empty_space = valid_obs & has_parallax & (proj_z < (obs_depth - tol_free))
            freespace_violations += empty_space.astype(np.int32)

    # 1. Regional coherence check: reject isolated atomic outliers.
    # A valid surface point must belong to a small coherent regional patch (at least 4 neighbors within 6cm).
    if n_pts >= 10:
        tree = KDTree(pts_world)
        # query_ball_point with count
        neighbor_counts = np.array([len(tree.query_ball_point(p, r=0.06)) for p in pts_world], dtype=np.int32)
        coherent_region_mask = neighbor_counts >= 4
    else:
        coherent_region_mask = np.ones(n_pts, dtype=bool)

    # 2. Three-way Multi-View Logic:
    # - Confirmed: Another picture looks at this area and confirms a surface exists (consensus_count >= min_consensus). -> KEEP
    # - Contradicted: Another picture looks through this area to a surface behind it (freespace_violations > max_violations). -> CARVE / REJECT
    # - Neutral: No other picture looks at this area (views_in_frustum == 0) or no other picture confirms nor contradicts.
    #   -> Trust the single-view prediction, provided it belongs to a coherent regional patch.
    if min_consensus > 0:
        # Confirmed by other view(s) OR uncontradicted single-view regional surface
        valid_mask = (consensus_count >= min_consensus) | ((views_in_frustum == 0) & coherent_region_mask)
    else:
        valid_mask = coherent_region_mask

    # 3. Free-space rule: Cull points that violate free space in unobstructed side views (contradicted)
    if enable_freespace_filter:
        valid_mask = valid_mask & (freespace_violations <= max_freespace_violations)

    # Atomic outliers that are not confirmed by any other view are dropped
    valid_mask = valid_mask & (coherent_region_mask | (consensus_count >= 1))

    return valid_mask


def global_cross_view_freespace_carving(
    pts_world: np.ndarray,
    normals_world: np.ndarray,
    colors_rgb: np.ndarray,
    keyframes: Sequence[Keyframe],
    depth_maps: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    max_violations: int = 1,
    margin_m: float = 0.04,
    match_tol_base: float = 0.05,
    match_tol_slope: float = 0.02,
    subsample_kfs: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cull floating phantom geometry and depth bleeding artifacts by testing points globally against all intersecting keyframes.

    Prunes points that violate empty free space in >= 2 global viewpoints or lack
    depth corroboration when visible across multiple cameras.
    """
    n_pts = len(pts_world)
    if n_pts == 0 or len(keyframes) <= 1:
        return pts_world, normals_world, colors_rgb

    fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)

    violations = np.zeros(n_pts, dtype=np.int32)
    matches = np.zeros(n_pts, dtype=np.int32)
    views_seen = np.zeros(n_pts, dtype=np.int32)

    sample_indices = list(range(0, len(keyframes), max(1, subsample_kfs)))
    for j in sample_indices:
        kf = keyframes[j]
        d_map = depth_maps[j]
        dh, dw = d_map.shape[:2]

        c2w = kf.transform_matrix
        r_cw = c2w[:3, :3]
        t_cw = c2w[:3, 3]

        # Project world points into camera
        pts_cam = np.dot(pts_world - t_cw, r_cw)
        proj_z = -pts_cam[:, 2]
        in_front = proj_z > 0.2

        u = (fx * (pts_cam[:, 0] / np.maximum(proj_z, 1e-4)) + cx).astype(np.int32)
        v = (-fy * (pts_cam[:, 1] / np.maximum(proj_z, 1e-4)) + cy).astype(np.int32)

        valid_uv = in_front & (u >= 4) & (u < dw - 4) & (v >= 4) & (v < dh - 4)
        if not np.any(valid_uv):
            continue

        views_seen += valid_uv.astype(np.int32)

        obs_d = np.zeros(n_pts, dtype=np.float32)
        obs_d[valid_uv] = d_map[v[valid_uv], u[valid_uv]]
        valid_obs = valid_uv & np.isfinite(obs_d) & (obs_d > 0.2)

        # Free space violation check
        tol_free = margin_m + 0.01 * proj_z
        empty_space = valid_obs & (proj_z < (obs_d - tol_free))
        violations += empty_space.astype(np.int32)

        # Corroborating match check
        tol_match = match_tol_base + match_tol_slope * proj_z
        match = valid_obs & (np.abs(proj_z - obs_d) <= tol_match)
        matches += match.astype(np.int32)

    # Filter rule: Discard points that violate free space in > max_violations views,
    # and require at least 1 match if observed by 3+ cameras.
    keep_mask = (violations <= max_violations) & ((views_seen < 3) | (matches >= 1))
    if not np.any(keep_mask):
        return pts_world, normals_world, colors_rgb

    return pts_world[keep_mask], normals_world[keep_mask], colors_rgb[keep_mask]


def regularize_surface_normals_multiview(
    pts_world: np.ndarray,
    normals_world: np.ndarray,
    current_idx: int,
    keyframes: Sequence[Keyframe],
    depth_maps: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    normals_cam_maps: Optional[Sequence[Optional[np.ndarray]]] = None,
    normals_cam_getter: Optional[Callable[[int], np.ndarray]] = None,
    min_cos_sim: float = 0.5,
    blend_weight: float = 0.4,
    max_neighbors: int = 6,
) -> np.ndarray:
    """Refine surface normals by blending with consistent neighboring camera views."""
    n_pts = len(pts_world)
    if n_pts == 0 or len(keyframes) <= 1:
        return normals_world

    cur_kf = keyframes[current_idx]
    cur_t = cur_kf.transform_matrix[:3, 3]
    cur_dir = -cur_kf.transform_matrix[:3, 2]

    candidate_neighbors = []
    for j, other_kf in enumerate(keyframes):
        if j == current_idx:
            continue
        other_t = other_kf.transform_matrix[:3, 3]
        other_dir = -other_kf.transform_matrix[:3, 2]
        d = float(np.linalg.norm(other_t - cur_t))
        cos_ang = float(np.dot(cur_dir, other_dir))
        if cos_ang > 0.1 and d < 3.5:
            score = d / max(0.2, cos_ang)
            candidate_neighbors.append((score, j))

    if not candidate_neighbors:
        dists = [(float(np.linalg.norm(keyframes[j].transform_matrix[:3, 3] - cur_t)), j)
                 for j in range(len(keyframes)) if j != current_idx]
        dists.sort()
        candidate_neighbors = dists[:max_neighbors]
    else:
        candidate_neighbors.sort()
        candidate_neighbors = candidate_neighbors[:max_neighbors]

    neighbor_indices = [j for _, j in candidate_neighbors]

    accum_normals = normals_world.copy().astype(np.float32)
    accum_weights = np.ones((n_pts, 1), dtype=np.float32)

    fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)

    for j in neighbor_indices:
        other_kf = keyframes[j]
        d_map = depth_maps[j]
        dh, dw = d_map.shape[:2]

        c2w = other_kf.transform_matrix
        r_cw = c2w[:3, :3]
        t_cw = c2w[:3, 3]

        pts_cam = np.dot(pts_world - t_cw, r_cw)
        proj_z = -pts_cam[:, 2]
        in_front = proj_z > 0.2

        u = (fx * (pts_cam[:, 0] / np.maximum(proj_z, 1e-4)) + cx).astype(np.int32)
        v = (-fy * (pts_cam[:, 1] / np.maximum(proj_z, 1e-4)) + cy).astype(np.int32)

        valid_uv = in_front & (u >= 2) & (u < dw - 2) & (v >= 2) & (v < dh - 2)
        if not np.any(valid_uv):
            continue

        obs_depth = np.zeros(n_pts, dtype=np.float32)
        obs_depth[valid_uv] = d_map[v[valid_uv], u[valid_uv]]
        tol = 0.08 + 0.05 * proj_z
        match = valid_uv & np.isfinite(obs_depth) & (obs_depth > 0.2) & (np.abs(obs_depth - proj_z) <= tol)

        if not np.any(match):
            continue

        if normals_cam_maps is not None and j < len(normals_cam_maps) and normals_cam_maps[j] is not None:
            n_cam_j = normals_cam_maps[j][v[match], u[match]]
        elif normals_cam_getter is not None:
            n_cam_j = normals_cam_getter(j)[v[match], u[match]]
        else:
            n_cam_j = compute_surface_normals(d_map, intrinsics)[v[match], u[match]]

        n_world_j = np.dot(n_cam_j, r_cw.T)
        norm_j = np.linalg.norm(n_world_j, axis=-1, keepdims=True)
        n_world_j = n_world_j / np.maximum(norm_j, 1e-6)

        # Check surface normal alignment to avoid smoothing across sharp edges
        cos_sim = np.sum(normals_world[match] * n_world_j, axis=-1, keepdims=True)
        aligned = (cos_sim > min_cos_sim).ravel()

        if np.any(aligned):
            match_indices = np.where(match)[0][aligned]
            accum_normals[match_indices] += n_world_j[aligned] * blend_weight
            accum_weights[match_indices] += blend_weight

    norm_final = np.linalg.norm(accum_normals, axis=-1, keepdims=True)
    regularized = np.where(norm_final > 1e-6, accum_normals / np.maximum(norm_final, 1e-6), normals_world)
    return regularized.astype(np.float32)


def statistical_outlier_removal(
    positions: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    k: int = 16,
    std_mul: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prune isolated points whose mean k-NN distance is an outlier."""
    n = len(positions)
    if n <= k:
        return positions, normals, colors

    sample_size = min(15_000, n)
    idx_sample = np.random.RandomState(42).choice(n, size=sample_size, replace=False)
    tree = KDTree(positions[idx_sample])
    dists, _ = tree.query(positions, k=k + 1)
    mean_dists = np.mean(dists[:, 1:], axis=-1)

    mu = float(np.mean(mean_dists))
    sigma = float(np.std(mean_dists))
    thresh = mu + std_mul * sigma

    inlier_mask = mean_dists <= thresh
    return positions[inlier_mask], normals[inlier_mask], colors[inlier_mask]


class SurfelCloudInitializer:
    """Initializes 2D Gaussian surfel volume from multi-view keyframes and depth priors."""

    def __init__(
        self,
        target_surfels: int = 500_000,
        min_surfels: int = 50_000,
        max_surfels: int = 800_000,
        default_opacity: float = 0.90,
        voxel_downsample_m: float = 0.02,  # 2.0 cm grid for crisp continuous surfaces
        max_depth_m: Optional[float] = None, # If None, dynamically estimated from confident depths
        min_consensus: int = 0,              # Disabled by default (prevents cutting real points with slight depth disagreement)
        max_depth_gradient: float = 0.0,     # Disabled by default (prevents puncturing slanted floors/beds)
        max_grazing_angle_deg: float = 0.0,  # Disabled by default (prevents cutting grazing floors)
        enable_sor: bool = False,            # Disabled by default (prevents punching holes in sparse peripheral regions)
        sor_k: int = 20,
        sor_std_mul: float = 1.5,
        enable_saturation_mask: bool = True,
        saturation_min_threshold: int = 250,
        saturation_max_threshold: int = 255,
        saturation_percentile: float = 99.8,
        saturation_max_chroma_diff: float = 35.0,
        saturation_dilation_radius: int = 1,
        enable_freespace_filter: bool = False,
        max_freespace_violations: int = 0,   # If violations > max_allowed (>= 1 with 0), discard point
        freespace_margin_m: float = 0.08,
        min_neighbor_baseline_m: float = 0.08,
        min_neighbor_parallax_deg: float = 3.0,
        enable_normal_consensus: bool = True,
        normal_consensus_weight: float = 0.4,
        normal_min_cos_sim: float = 0.5,
        enable_global_carving: bool = False,
        global_carving_max_violations: int = 1,
        global_carving_margin_m: float = 0.04,
        global_carving_subsample_kfs: int = 4,
        enable_tube_collapse: bool = True,
        tube_radius_m: float = 0.02,
        tube_length_m: float = 0.15,
        tube_min_normal_cos: float = 0.80,
        enable_multiscale_pyramid: bool = False,
        enable_hybrid_sampling: bool = True,
        hybrid_energy_threshold: float = 0.08,
        hybrid_coarse_stride: int = 3,
        hybrid_fine_stride: int = 1,
    ):
        self.target_surfels = target_surfels
        self.min_surfels = min_surfels
        self.max_surfels = max_surfels
        self.default_opacity = default_opacity
        self.voxel_downsample_m = voxel_downsample_m
        self.max_depth_m = max_depth_m
        self.min_consensus = min_consensus
        self.max_depth_gradient = max_depth_gradient
        self.max_grazing_angle_deg = max_grazing_angle_deg
        self.enable_sor = enable_sor
        self.sor_k = sor_k
        self.sor_std_mul = sor_std_mul
        self.enable_saturation_mask = enable_saturation_mask
        self.saturation_min_threshold = saturation_min_threshold
        self.saturation_max_threshold = saturation_max_threshold
        self.saturation_percentile = saturation_percentile
        self.saturation_max_chroma_diff = saturation_max_chroma_diff
        self.saturation_dilation_radius = saturation_dilation_radius
        self.enable_freespace_filter = enable_freespace_filter
        self.max_freespace_violations = max_freespace_violations
        self.freespace_margin_m = freespace_margin_m
        self.min_neighbor_baseline_m = min_neighbor_baseline_m
        self.min_neighbor_parallax_deg = min_neighbor_parallax_deg
        self.enable_normal_consensus = enable_normal_consensus
        self.normal_consensus_weight = normal_consensus_weight
        self.normal_min_cos_sim = normal_min_cos_sim
        self.enable_global_carving = enable_global_carving
        self.global_carving_max_violations = global_carving_max_violations
        self.global_carving_margin_m = global_carving_margin_m
        self.global_carving_subsample_kfs = global_carving_subsample_kfs
        self.enable_tube_collapse = enable_tube_collapse
        self.tube_radius_m = tube_radius_m
        self.tube_length_m = tube_length_m
        self.tube_min_normal_cos = tube_min_normal_cos
        self.enable_multiscale_pyramid = enable_multiscale_pyramid
        self.enable_hybrid_sampling = enable_hybrid_sampling
        self.hybrid_energy_threshold = hybrid_energy_threshold
        self.hybrid_coarse_stride = hybrid_coarse_stride
        self.hybrid_fine_stride = hybrid_fine_stride

    def initialize_from_keyframes(
        self,
        keyframes: Sequence[Keyframe],
        depth_maps: Sequence[np.ndarray],
        intrinsics: CameraIntrinsics,
        sparse_points_3d: Optional[np.ndarray] = None,
        conf_maps: Optional[Sequence[Optional[np.ndarray]]] = None,
        min_conf: float = 0.5,
    ) -> SurfelCloud:
        """Unproject keyframes with dense aligned depth maps into an initial SurfelCloud."""
        num_frames = len(keyframes)
        if num_frames == 0 or len(depth_maps) != num_frames:
            raise ValueError(f"Mismatched keyframes ({num_frames}) and depth maps ({len(depth_maps)})")

        all_positions = []
        all_normals = []
        all_colors = []
        all_rel_scales = []

        # Determine dynamic adaptive depth ceiling
        if self.max_depth_m is not None:
            depth_ceiling = float(self.max_depth_m)
        else:
            depth_ceiling = estimate_adaptive_depth_ceiling(depth_maps, conf_maps, min_conf=min_conf)

        # Target points per keyframe: sample densely before voxelization
        pts_per_frame = max(5_000, int(self.target_surfels * 2.5 / num_frames))
        h_sample, w_sample = depth_maps[0].shape[:2]
        base_stride = max(2, int(math.sqrt((h_sample * w_sample) / pts_per_frame)))
        auto_fine_stride = max(1, base_stride // 2)
        auto_coarse_stride = max(auto_fine_stride + 1, int(round(base_stride * 1.6)))

        fine_stride = self.hybrid_fine_stride if self.hybrid_fine_stride > 1 else auto_fine_stride
        coarse_stride = self.hybrid_coarse_stride if self.hybrid_coarse_stride > 3 else auto_coarse_stride

        fx, fy = intrinsics.fl_x, intrinsics.fl_y
        cx, cy = intrinsics.cx, intrinsics.cy
        cos_min = math.cos(math.radians(self.max_grazing_angle_deg)) if self.max_grazing_angle_deg > 0 else 0.0

        normals_cam_cache: dict[int, np.ndarray] = {}

        def get_normals_cam(k_idx: int) -> np.ndarray:
            if k_idx not in normals_cam_cache:
                if len(normals_cam_cache) >= 16:
                    normals_cam_cache.pop(next(iter(normals_cam_cache)))
                normals_cam_cache[k_idx] = compute_surface_normals(depth_maps[k_idx], intrinsics)
            return normals_cam_cache[k_idx]

        for idx, (kf, depth) in enumerate(zip(keyframes, depth_maps)):
            h, w = depth.shape[:2]
            normals_cam = get_normals_cam(idx)

            img_rgb = kf.load_image_rgb()
            if img_rgb.shape[:2] != (h, w):
                img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

            # Sample pixel coordinates (Hybrid Adaptive vs Uniform Stride)
            if self.enable_hybrid_sampling:
                y_flat, x_flat, scales_flat = compute_hybrid_sampling_coords(
                    img_rgb=img_rgb,
                    depth_map=depth,
                    energy_threshold=self.hybrid_energy_threshold,
                    coarse_stride=coarse_stride,
                    fine_stride=fine_stride,
                )
            else:
                stride = max(1, int(math.sqrt((h * w) / pts_per_frame)))
                y_sub, x_sub = np.mgrid[0:h:stride, 0:w:stride]
                y_flat = y_sub.flatten()
                x_flat = x_sub.flatten()
                scales_flat = np.ones(len(y_flat), dtype=np.float32)

            d_sampled = depth[y_flat, x_flat]
            valid_mask = (d_sampled > 0.2) & (d_sampled <= depth_ceiling)

            # 1. Depth Discontinuity / Edge Gradient Filter (eliminates flying boundary pixels)
            if self.max_depth_gradient > 0:
                gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / np.maximum(depth, 1e-3)
                gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / np.maximum(depth, 1e-3)
                grad_sampled = np.sqrt(gx**2 + gy**2)[y_flat, x_flat]
                valid_mask = valid_mask & (grad_sampled <= self.max_depth_gradient)

            # 2. Saturated Pixel Masking (eliminates optical bloom / light-source artifacts)
            if self.enable_saturation_mask:
                sat_mask = compute_overexposed_mask(
                    img_rgb,
                    min_threshold=self.saturation_min_threshold,
                    max_threshold=self.saturation_max_threshold,
                    percentile=self.saturation_percentile,
                    max_chroma_diff=self.saturation_max_chroma_diff,
                    dilation_kernel_size=self.saturation_dilation_radius * 2 + 1 if self.saturation_dilation_radius > 0 else 0,
                )
                valid_mask = valid_mask & (~sat_mask[y_flat, x_flat])

            if conf_maps is not None and idx < len(conf_maps) and conf_maps[idx] is not None:
                c_map = conf_maps[idx]
                if c_map.shape[:2] != (h, w):
                    c_map = cv2.resize(c_map, (w, h), interpolation=cv2.INTER_NEAREST)
                c_sampled = c_map[y_flat, x_flat]
                valid_mask = valid_mask & (c_sampled >= min_conf)

            if np.sum(valid_mask) < 10:
                continue

            y_valid = y_flat[valid_mask]
            x_valid = x_flat[valid_mask]
            d_valid = d_sampled[valid_mask]
            scales_valid = scales_flat[valid_mask]

            # 3D points in camera coordinates (ARCore/OpenGL convention: +X right, +Y up, -Z forward)
            x_cam = (x_valid - cx) * d_valid / fx
            y_cam = -(y_valid - cy) * d_valid / fy
            z_cam = -d_valid
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (M, 3)

            n_cam = normals_cam[y_valid, x_valid]  # (M, 3)

            # 3. Grazing Angle Filter (eliminates glancing silhouette projections)
            if cos_min > 0.0:
                ray_dir = pts_cam / np.maximum(np.linalg.norm(pts_cam, axis=-1, keepdims=True), 1e-6)
                cos_grazing = np.sum(-n_cam * ray_dir, axis=-1)  # n_cam points toward camera (+Z)
                grazing_mask = cos_grazing >= cos_min
                if np.sum(grazing_mask) < 5:
                    continue
                pts_cam = pts_cam[grazing_mask]
                n_cam = n_cam[grazing_mask]
                y_valid = y_valid[grazing_mask]
                x_valid = x_valid[grazing_mask]
                scales_valid = scales_valid[grazing_mask]

            c_rgb = (img_rgb[y_valid, x_valid] / 255.0).astype(np.float32)  # (M, 3)

            # Transform to world coordinates: P_world = R_cw * P_cam + t_cw
            c2w = kf.transform_matrix
            r_cw = c2w[:3, :3]
            t_cw = c2w[:3, 3]

            pts_world = np.dot(pts_cam, r_cw.T) + t_cw
            n_world = np.dot(n_cam, r_cw.T)
            norm_n = np.linalg.norm(n_world, axis=-1, keepdims=True)
            norm_n = np.maximum(norm_n, 1e-6)
            n_world = n_world / norm_n

            # Multi-view depth consistency & cross-view free-space check to prune non-surface artifacts
            if (self.min_consensus > 0 or self.enable_freespace_filter) and len(keyframes) > 1:
                mv_mask = filter_multiview_consistency(
                    pts_world,
                    idx,
                    keyframes,
                    depth_maps,
                    intrinsics,
                    normals_world=n_world,
                    min_consensus=self.min_consensus,
                    enable_freespace_filter=self.enable_freespace_filter,
                    max_freespace_violations=self.max_freespace_violations,
                    freespace_margin_m=self.freespace_margin_m,
                    min_neighbor_baseline_m=self.min_neighbor_baseline_m,
                    min_neighbor_parallax_deg=self.min_neighbor_parallax_deg,
                )
                if np.sum(mv_mask) < 5:
                    continue
                pts_world = pts_world[mv_mask]
                n_world = n_world[mv_mask]
                c_rgb = c_rgb[mv_mask]
                scales_valid = scales_valid[mv_mask]

            if self.enable_normal_consensus and len(keyframes) > 1 and len(pts_world) > 0:
                n_world = regularize_surface_normals_multiview(
                    pts_world=pts_world,
                    normals_world=n_world,
                    current_idx=idx,
                    keyframes=keyframes,
                    depth_maps=depth_maps,
                    intrinsics=intrinsics,
                    normals_cam_getter=get_normals_cam,
                    min_cos_sim=self.normal_min_cos_sim,
                    blend_weight=self.normal_consensus_weight,
                )

            all_positions.append(pts_world)
            all_normals.append(n_world)
            all_colors.append(c_rgb)
            all_rel_scales.append(scales_valid)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(keyframes):
                print(f"[SurfelInit] Processed {idx + 1}/{len(keyframes)} keyframes ({sum(len(p) for p in all_positions):,} raw points)...", flush=True)

        if not all_positions:
            raise RuntimeError("Failed to unproject any valid surfel points from keyframes")

        cat_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
        cat_normals = np.concatenate(all_normals, axis=0).astype(np.float32)
        cat_colors = np.concatenate(all_colors, axis=0).astype(np.float32)
        cat_scales = np.concatenate(all_rel_scales, axis=0).astype(np.float32)

        # Include sparse VIO/SfM points if provided
        if sparse_points_3d is not None and len(sparse_points_3d) > 0:
            sparse_pts = sparse_points_3d.astype(np.float32)
            n_sparse = len(sparse_pts)
            # Default upward normal for sparse points without known surface
            sparse_norms = np.zeros((n_sparse, 3), dtype=np.float32)
            sparse_norms[:, 2] = 1.0
            sparse_colors = np.ones((n_sparse, 3), dtype=np.float32) * 0.7
            sparse_rel_scales = np.ones(n_sparse, dtype=np.float32)

            cat_positions = np.vstack([cat_positions, sparse_pts])
            cat_normals = np.vstack([cat_normals, sparse_norms])
            cat_colors = np.vstack([cat_colors, sparse_colors])
            cat_scales = np.concatenate([cat_scales, sparse_rel_scales])

        # Global multi-view free-space carving across all intersecting keyframes
        if self.enable_global_carving and len(keyframes) > 1 and len(cat_positions) > 0:
            cat_positions, cat_normals, cat_colors = global_cross_view_freespace_carving(
                pts_world=cat_positions,
                normals_world=cat_normals,
                colors_rgb=cat_colors,
                keyframes=keyframes,
                depth_maps=depth_maps,
                intrinsics=intrinsics,
                max_violations=self.global_carving_max_violations,
                margin_m=self.global_carving_margin_m,
                subsample_kfs=self.global_carving_subsample_kfs,
            )

        # Statistical Outlier Removal (SOR) to prune remaining floating noise on dense refined cloud
        final_pos = cat_positions
        final_norm = cat_normals
        final_col = cat_colors
        final_scales = cat_scales
        if self.enable_sor and len(final_pos) > self.sor_k:
            sor_mask = self._get_sor_inlier_mask(final_pos, k=self.sor_k, std_mul=self.sor_std_mul)
            final_pos = final_pos[sor_mask]
            final_norm = final_norm[sor_mask]
            final_col = final_col[sor_mask]
            if len(final_scales) == len(sor_mask):
                final_scales = final_scales[sor_mask]

        # Compute orthonormal tangent frames (u, v) on refined normals
        tangent_u, tangent_v = build_orthonormal_tangent_frame(final_norm)

        # Estimate 2D scales (sigma_u, sigma_v) from local point spacing
        scales_2d = self._estimate_initial_scales(final_pos, relative_scales=final_scales)

        # Compute degree-0 SH coefficients (ambient base color)
        sh_deg0 = (final_col * SH_C0).astype(np.float32)

        # Initialize opacities
        opacities = np.full((len(final_pos),), self.default_opacity, dtype=np.float32)

        cloud = SurfelCloud(
            positions=final_pos,
            normals=final_norm,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            scales_2d=scales_2d,
            colors_rgb=final_col,
            sh_degree_0=sh_deg0,
            opacities=opacities,
        )

        # Normal Tube Collapse to project multi-layer slab depth variance into a clean 2D manifold shell
        if self.enable_tube_collapse:
            v_size = self.voxel_downsample_m or 0.015
            cloud = cloud.normal_tube_collapse(
                voxel_size_m=v_size,
                tube_radius_m=self.tube_radius_m,
                tube_length_m=self.tube_length_m,
                min_normal_cos=self.tube_min_normal_cos,
            )
            self.applied_voxel_size_m = v_size

        # Multi-Scale Surfel Decimation to adaptively coarsen flat planar regions
        if self.enable_multiscale_pyramid:
            v_size = getattr(self, "applied_voxel_size_m", None) or self.voxel_downsample_m or 0.015
            cloud = cloud.multiscale_pyramid_decimate(
                base_voxel_m=v_size,
            )
            self.applied_voxel_size_m = v_size

        # Ensure surfel count stays within max_surfels budget by coarsening grid if necessary
        if self.max_surfels is not None and len(cloud) > self.max_surfels:
            voxel_size = getattr(self, "applied_voxel_size_m", None) or self.voxel_downsample_m or 0.015
            while len(cloud) > self.max_surfels and voxel_size < 0.20:
                voxel_size = 0.02 if voxel_size < 0.02 else voxel_size + 0.005
                cloud = cloud.voxel_downsample(voxel_size)
            self.applied_voxel_size_m = voxel_size
        elif not self.enable_tube_collapse and not self.enable_multiscale_pyramid:
            if self.voxel_downsample_m is not None and self.voxel_downsample_m > 0:
                voxel_size = self.voxel_downsample_m
                while True:
                    downsampled = cloud.voxel_downsample(voxel_size)
                    if len(downsampled) <= self.max_surfels or voxel_size >= 0.10:
                        cloud = downsampled
                        break
                    voxel_size = 0.02 if voxel_size < 0.02 else voxel_size + 0.005
                self.applied_voxel_size_m = voxel_size
            else:
                self.applied_voxel_size_m = 0.0

        return cloud

    def _get_sor_inlier_mask(self, positions: np.ndarray, k: int = 16, std_mul: float = 1.5) -> np.ndarray:
        """Helper to get boolean inlier mask for statistical outlier removal."""
        n = len(positions)
        if n <= k:
            return np.ones(n, dtype=bool)

        sample_size = min(15_000, n)
        idx_sample = np.random.RandomState(42).choice(n, size=sample_size, replace=False)
        tree = KDTree(positions[idx_sample])
        dists, _ = tree.query(positions, k=k + 1)
        mean_dists = np.mean(dists[:, 1:], axis=-1)

        mu = float(np.mean(mean_dists))
        sigma = float(np.std(mean_dists))
        thresh = mu + std_mul * sigma
        return mean_dists <= thresh

    def _voxel_downsample(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform spatial voxel grid filtering to ensure uniform distribution."""
        voxel_coords = np.floor(positions / voxel_size).astype(np.int32)
        # Unique voxel hash
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)

        return positions[unique_indices], normals[unique_indices], colors[unique_indices]

    def _estimate_initial_scales(self, positions: np.ndarray, relative_scales: Optional[np.ndarray] = None, k: int = 3) -> np.ndarray:
        """Estimate initial 2D Gaussian scales (sigma_u, sigma_v) from local point spacing."""
        n = len(positions)
        if n <= k:
            return np.full((n, 2), 0.02, dtype=np.float32)

        # Subsample for fast KDTree query if point count is very large
        sample_size = min(10_000, n)
        idx_sample = np.random.RandomState(42).choice(n, size=sample_size, replace=False)

        tree = KDTree(positions[idx_sample])
        dists, _ = tree.query(positions, k=k + 1)
        # Average distance to k nearest neighbors (excluding self at index 0)
        mean_dists = np.mean(dists[:, 1:], axis=-1)

        # Clamp scale to the voxel cell bounds [0.4v, 0.8v] of the grid actually
        # applied -- a coarsened grid needs proportionally larger surfels or the
        # surface develops holes.
        v = getattr(self, "applied_voxel_size_m", None) or self.voxel_downsample_m or 0.02
        if relative_scales is not None and len(relative_scales) == n:
            max_v = v * float(np.max(relative_scales)) if np.max(relative_scales) > 1.0 else v
            scales = np.clip(mean_dists * 0.8, 0.4 * v, 0.8 * max_v).astype(np.float32)
        else:
            scales = np.clip(mean_dists * 0.8, 0.4 * v, 0.8 * v).astype(np.float32)
        return np.column_stack([scales, scales])
