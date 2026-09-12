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
from scipy.spatial import KDTree

from ingestion.package_loader import CameraIntrinsics, Keyframe
from reconstruction.depth_priors import compute_surface_normals

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

            # Format: 3f (pos), 3f (norm), 3B (color), 2f (scales), 1f (opacity)
            record_format = "<3f3f3B2ff"

            rgb_bytes = np.clip(self.colors_rgb * 255.0, 0, 255).astype(np.uint8)
            records = []
            for i in range(n):
                px, py, pz = self.positions[i]
                nx, ny, nz = self.normals[i]
                r, g, b = rgb_bytes[i]
                su, sv = self.scales_2d[i]
                op = self.opacities[i]
                records.append(struct.pack(record_format, px, py, pz, nx, ny, nz, r, g, b, su, sv, op))

            f.write(b"".join(records))

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


def filter_multiview_consistency(
    pts_world: np.ndarray,
    current_idx: int,
    keyframes: Sequence[Keyframe],
    depth_maps: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    max_neighbors: int = 6,
    min_consensus: int = 1,
) -> np.ndarray:
    """Return boolean mask of points corroborated by neighboring camera views.

    Eliminates non-surface artifacts (such as sky through windows, open doors, and reflections).
    Points observed from a unique angle with no overlapping views in their field of view are preserved,
    while points that fall inside overlapping camera frustums must agree with the neighbor depth.
    """
    n_pts = len(pts_world)
    if n_pts == 0 or min_consensus <= 0 or len(keyframes) <= 1:
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
    fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
    cx, cy = float(intrinsics.cx), float(intrinsics.cy)

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

        valid_uv = in_front & (u >= 0) & (u < dw) & (v >= 0) & (v < dh)
        views_in_frustum += valid_uv.astype(np.int32)

        obs_depth = np.zeros(n_pts, dtype=np.float32)
        obs_depth[valid_uv] = d_map[v[valid_uv], u[valid_uv]]

        tol = 0.10 + 0.07 * proj_z
        match = valid_uv & np.isfinite(obs_depth) & (obs_depth > 0.2) & (np.abs(obs_depth - proj_z) <= tol)
        consensus_count += match.astype(np.int32)

    # If other views have the point in their frustum, require at least min_consensus matches.
    # If no other views have the point in frame (views_in_frustum == 0), keep it (uniquely seen area).
    return (views_in_frustum == 0) | (consensus_count >= min_consensus)


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
        min_consensus: int = 1,              # Multi-view frustum corroboration (>= 1 neighbor match when visible)
        max_depth_gradient: float = 0.0,     # Disabled by default (prevents puncturing slanted floors/beds)
        max_grazing_angle_deg: float = 0.0,  # Disabled by default (prevents cutting grazing floors)
        enable_sor: bool = True,             # Statistical Outlier Removal in 3D Euclidean space
        sor_k: int = 20,
        sor_std_mul: float = 1.5,
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

        # Determine dynamic adaptive depth ceiling
        if self.max_depth_m is not None:
            depth_ceiling = float(self.max_depth_m)
        else:
            depth_ceiling = estimate_adaptive_depth_ceiling(depth_maps, conf_maps, min_conf=min_conf)

        # Target points per keyframe: sample densely before voxelization
        pts_per_frame = max(5_000, int(self.target_surfels * 2.5 / num_frames))

        fx, fy = intrinsics.fl_x, intrinsics.fl_y
        cx, cy = intrinsics.cx, intrinsics.cy
        cos_min = math.cos(math.radians(self.max_grazing_angle_deg)) if self.max_grazing_angle_deg > 0 else 0.0

        for idx, (kf, depth) in enumerate(zip(keyframes, depth_maps)):
            h, w = depth.shape[:2]
            normals_cam = compute_surface_normals(depth, intrinsics)

            img_rgb = kf.load_image_rgb()
            if img_rgb.shape[:2] != (h, w):
                img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

            # Sample stride to extract ~pts_per_frame points
            stride = max(1, int(math.sqrt((h * w) / pts_per_frame)))

            y_sub, x_sub = np.mgrid[0:h:stride, 0:w:stride]
            y_flat = y_sub.flatten()
            x_flat = x_sub.flatten()

            d_sampled = depth[y_flat, x_flat]
            valid_mask = (d_sampled > 0.2) & (d_sampled <= depth_ceiling)

            # 1. Depth Discontinuity / Edge Gradient Filter (eliminates flying boundary pixels)
            if self.max_depth_gradient > 0:
                gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / np.maximum(depth, 1e-3)
                gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / np.maximum(depth, 1e-3)
                grad_sampled = np.sqrt(gx**2 + gy**2)[y_flat, x_flat]
                valid_mask = valid_mask & (grad_sampled <= self.max_depth_gradient)

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

            # 3D points in camera coordinates (ARCore/OpenGL convention: +X right, +Y up, -Z forward)
            x_cam = (x_valid - cx) * d_valid / fx
            y_cam = -(y_valid - cy) * d_valid / fy
            z_cam = -d_valid
            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (M, 3)

            n_cam = normals_cam[y_valid, x_valid]  # (M, 3)

            # 2. Grazing Angle Filter (eliminates glancing silhouette projections)
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

            # Multi-view depth consistency check to prune non-surface artifacts
            if self.min_consensus > 0 and len(keyframes) > 1:
                mv_mask = filter_multiview_consistency(
                    pts_world,
                    idx,
                    keyframes,
                    depth_maps,
                    intrinsics,
                    min_consensus=self.min_consensus,
                )
                if np.sum(mv_mask) < 5:
                    continue
                pts_world = pts_world[mv_mask]
                n_world = n_world[mv_mask]
                c_rgb = c_rgb[mv_mask]

            all_positions.append(pts_world)
            all_normals.append(n_world)
            all_colors.append(c_rgb)

        if not all_positions:
            raise RuntimeError("Failed to unproject any valid surfel points from keyframes")

        cat_positions = np.concatenate(all_positions, axis=0).astype(np.float32)
        cat_normals = np.concatenate(all_normals, axis=0).astype(np.float32)
        cat_colors = np.concatenate(all_colors, axis=0).astype(np.float32)

        # Include sparse VIO/SfM points if provided
        if sparse_points_3d is not None and len(sparse_points_3d) > 0:
            sparse_pts = sparse_points_3d.astype(np.float32)
            n_sparse = len(sparse_pts)
            # Default upward normal for sparse points without known surface
            sparse_norms = np.zeros((n_sparse, 3), dtype=np.float32)
            sparse_norms[:, 2] = 1.0
            sparse_colors = np.ones((n_sparse, 3), dtype=np.float32) * 0.7

            cat_positions = np.vstack([cat_positions, sparse_pts])
            cat_normals = np.vstack([cat_normals, sparse_norms])
            cat_colors = np.vstack([cat_colors, sparse_colors])

        # Voxel grid downsampling
        final_pos, final_norm, final_col = self._voxel_downsample(
            cat_positions, cat_normals, cat_colors, self.voxel_downsample_m
        )

        # Statistical Outlier Removal (SOR) to prune remaining floating noise
        if self.enable_sor and len(final_pos) > self.sor_k:
            final_pos, final_norm, final_col = statistical_outlier_removal(
                final_pos, final_norm, final_col, k=self.sor_k, std_mul=self.sor_std_mul
            )

        # Adjust to target budget
        n_surfels = len(final_pos)
        if n_surfels > self.max_surfels:
            perm = np.random.RandomState(42).permutation(n_surfels)[:self.target_surfels]
            final_pos = final_pos[perm]
            final_norm = final_norm[perm]
            final_col = final_col[perm]

        # Compute orthonormal tangent frames (u, v)
        tangent_u, tangent_v = build_orthonormal_tangent_frame(final_norm)

        # Estimate 2D scales (sigma_u, sigma_v) from local k-NN spacing
        scales_2d = self._estimate_initial_scales(final_pos)

        # Compute degree-0 SH coefficients (ambient base color)
        sh_deg0 = (final_col * SH_C0).astype(np.float32)

        # Initialize opacities
        opacities = np.full((len(final_pos),), self.default_opacity, dtype=np.float32)

        return SurfelCloud(
            positions=final_pos,
            normals=final_norm,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            scales_2d=scales_2d,
            colors_rgb=final_col,
            sh_degree_0=sh_deg0,
            opacities=opacities,
        )

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

    def _estimate_initial_scales(self, positions: np.ndarray, k: int = 3) -> np.ndarray:
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

        # Clamp scale between 0.8 cm and 1.6 cm (matching 2.0 cm voxel cell bounds [0.4v, 0.8v])
        scales = np.clip(mean_dists * 0.8, 0.008, 0.016).astype(np.float32)
        return np.column_stack([scales, scales])
