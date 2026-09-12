"""GlomeHomeTour Backend: Dynamic Spatial & Co-visibility Keyframe Selector.

Selects an optimal, non-redundant subset of anchor keyframes from continuous
trajectory sequences based on spatial baseline distance, angular parallax,
and 3D frustum co-visibility overlap. Automatically scales across environments
of any size (from small studios to large multi-room venues).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .package_loader import CameraIntrinsics, Keyframe


@dataclass
class KeyframeSelectionResult:
    """Summary of dynamic keyframe selection."""
    selected_indices: list[int]
    selected_keyframes: list[Keyframe]
    discarded_indices: list[int]
    total_evaluated: int
    selection_ratio: float
    reasons: dict[str, int] = field(default_factory=dict)


class DynamicKeyframeSelector:
    """Selects non-redundant keyframes with optimal baseline parallax and coverage."""

    def __init__(
        self,
        min_translation_m: float = 0.35,     # 35 cm minimum translation
        min_rotation_deg: float = 18.0,       # 18 degrees minimum rotation
        min_covisibility: float = 0.35,       # If overlap drops below 35%, accept to avoid gaps
        max_covisibility: float = 0.78,       # If overlap with existing keyframe > 78%, discard as redundant
        reference_depth_m: float = 2.0,       # Reference plane depth for frustum projection (meters)
        sample_grid_size: int = 6,            # N x N sample rays for fast frustum intersection
    ):
        self.min_translation_m = min_translation_m
        self.min_rotation_deg = min_rotation_deg
        self.min_covisibility = min_covisibility
        self.max_covisibility = max_covisibility
        self.reference_depth_m = reference_depth_m
        self.sample_grid_size = sample_grid_size

    @classmethod
    def for_2dgs_training(
        cls,
        min_translation_m: float = 0.15,
        min_rotation_deg: float = 8.0,
        min_covisibility: float = 0.45,
        max_covisibility: float = 0.65,
        reference_depth_m: float = 2.0,
        sample_grid_size: int = 6,
    ) -> "DynamicKeyframeSelector":
        """Factory configured specifically for dense 2DGS training keyframe selection."""
        return cls(
            min_translation_m=min_translation_m,
            min_rotation_deg=min_rotation_deg,
            min_covisibility=min_covisibility,
            max_covisibility=max_covisibility,
            reference_depth_m=reference_depth_m,
            sample_grid_size=sample_grid_size,
        )

    def select_keyframes(
        self,
        keyframes: Sequence[Keyframe],
        intrinsics: CameraIntrinsics,
        min_keyframes: Optional[int] = None,
        max_keyframes: Optional[int] = None,
    ) -> KeyframeSelectionResult:
        """Dynamically filter keyframes to an optimal anchor subset."""
        n_total = len(keyframes)
        if n_total == 0:
            return KeyframeSelectionResult([], [], [], 0, 0.0, {})

        if n_total <= 2:
            return KeyframeSelectionResult(
                selected_indices=list(range(n_total)),
                selected_keyframes=list(keyframes),
                discarded_indices=[],
                total_evaluated=n_total,
                selection_ratio=1.0,
                reasons={"initial": n_total},
            )

        # Precompute canonical normalized image coordinates for sample rays
        grid_rays_cam = self._generate_canonical_frustum_rays(intrinsics, self.sample_grid_size)

        selected_indices: list[int] = [0]
        selected_kfs: list[Keyframe] = [keyframes[0]]
        discarded_indices: list[int] = []

        reasons = {
            "initial_frame": 1,
            "sufficient_motion": 0,
            "coverage_gap_prevention": 0,
            "redundant_motion": 0,
            "redundant_covisibility": 0,
        }

        # Track poses of selected keyframes: (R_cw, t_cw)
        selected_poses: list[tuple[np.ndarray, np.ndarray]] = [
            (keyframes[0].transform_matrix[:3, :3], keyframes[0].transform_matrix[:3, 3])
        ]

        for idx in range(1, n_total):
            cand_kf = keyframes[idx]
            cand_c2w = cand_kf.transform_matrix
            cand_r = cand_c2w[:3, :3]
            cand_t = cand_c2w[:3, 3]

            last_r, last_t = selected_poses[-1]

            # 1. Check co-visibility against ALL previously selected keyframes to prune loop revisits
            is_redundant_revisit = False
            for prev_r, prev_t in reversed(selected_poses):
                dist_to_prev = float(np.linalg.norm(cand_t - prev_t))
                # Check frustum overlap against any nearby keyframe within scene reference distance
                if dist_to_prev <= (self.reference_depth_m * 2.0):
                    covis = self._compute_frustum_covisibility(
                        cand_r, cand_t, prev_r, prev_t, intrinsics, grid_rays_cam
                    )
                    if covis > self.max_covisibility:
                        is_redundant_revisit = True
                        break

            if is_redundant_revisit:
                discarded_indices.append(idx)
                reasons["redundant_covisibility"] += 1
                continue

            # 2. Compute motion relative to last accepted keyframe
            trans_dist = float(np.linalg.norm(cand_t - last_t))
            r_rel = np.dot(cand_r, last_r.T)
            trace = float(np.trace(r_rel))
            cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
            rot_deg = float(np.degrees(np.arccos(cos_theta)))

            has_motion = (trans_dist >= self.min_translation_m) or (rot_deg >= self.min_rotation_deg)

            # 3. Compute co-visibility overlap with the last accepted keyframe
            covis_last = self._compute_frustum_covisibility(
                cand_r, cand_t, last_r, last_t, intrinsics, grid_rays_cam
            )

            # Prevent coverage gaps: if overlap with last keyframe drops too low, accept immediately
            if covis_last < self.min_covisibility:
                selected_indices.append(idx)
                selected_kfs.append(cand_kf)
                selected_poses.append((cand_r, cand_t))
                reasons["coverage_gap_prevention"] += 1
                continue

            if not has_motion:
                discarded_indices.append(idx)
                reasons["redundant_motion"] += 1
                continue

            # Frame passed: sufficient motion, non-redundant co-visibility
            selected_indices.append(idx)
            selected_kfs.append(cand_kf)
            selected_poses.append((cand_r, cand_t))
            reasons["sufficient_motion"] += 1

        # Enforce max_keyframes cap if specified
        if max_keyframes is not None and len(selected_indices) > max_keyframes:
            step = len(selected_indices) / max_keyframes
            kept_sub_idx = [int(round(i * step)) for i in range(max_keyframes)]
            kept_sub_idx = sorted(list(set(min(len(selected_indices) - 1, idx) for idx in kept_sub_idx)))
            selected_indices = [selected_indices[i] for i in kept_sub_idx]
            selected_kfs = [keyframes[i] for i in selected_indices]

        # Enforce min_keyframes floor if specified
        if min_keyframes is not None and len(selected_indices) < min_keyframes and len(keyframes) > len(selected_indices):
            missing = min(min_keyframes - len(selected_indices), len(discarded_indices))
            if missing > 0:
                # Add frames with highest translation distance to last accepted
                add_indices = sorted(discarded_indices[:missing])
                combined = sorted(list(set(selected_indices + add_indices)))
                selected_indices = combined
                selected_kfs = [keyframes[i] for i in selected_indices]

        ratio = len(selected_indices) / max(1, n_total)
        return KeyframeSelectionResult(
            selected_indices=selected_indices,
            selected_keyframes=selected_kfs,
            discarded_indices=discarded_indices,
            total_evaluated=n_total,
            selection_ratio=ratio,
            reasons=reasons,
        )

    def _generate_canonical_frustum_rays(
        self,
        intrinsics: CameraIntrinsics,
        grid_size: int,
    ) -> np.ndarray:
        """Generate grid of 3D sample points at reference depth in camera frame (OpenGL: -Z forward)."""
        w, h = float(intrinsics.w), float(intrinsics.h)
        fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
        cx, cy = float(intrinsics.cx), float(intrinsics.cy)

        # Uniform grid over image sensor
        xs = np.linspace(w * 0.1, w * 0.9, grid_size, dtype=np.float32)
        ys = np.linspace(h * 0.1, h * 0.9, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        d = float(self.reference_depth_m)
        # OpenGL camera coordinates (+X right, +Y up, -Z forward)
        x_cam = (grid_x - cx) * d / fx
        y_cam = -(grid_y - cy) * d / fy
        z_cam = np.full_like(x_cam, -d)

        return np.stack([x_cam, y_cam, z_cam], axis=-1)  # (M, 3)

    def _compute_frustum_covisibility(
        self,
        r1: np.ndarray,
        t1: np.ndarray,
        r2: np.ndarray,
        t2: np.ndarray,
        intrinsics: CameraIntrinsics,
        rays_cam: np.ndarray,
    ) -> float:
        """Compute visual overlap ratio between Camera 1 and Camera 2."""
        pts_w = np.dot(rays_cam, r1.T) + t1  # (M, 3)
        pts_c2 = np.dot(pts_w - t2, r2)  # (M, 3)

        # In OpenGL coordinates, camera looks along -Z, so visible points have z_c2 < -0.1
        valid_z = pts_c2[:, 2] < -0.1
        if not np.any(valid_z):
            return 0.0

        pts_valid = pts_c2[valid_z]
        depths = -pts_valid[:, 2]

        fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
        cx, cy = float(intrinsics.cx), float(intrinsics.cy)
        w, h = float(intrinsics.w), float(intrinsics.h)

        u = fx * (pts_valid[:, 0] / depths) + cx
        v = -fy * (pts_valid[:, 1] / depths) + cy

        in_bounds = (u >= 0.0) & (u <= w) & (v >= 0.0) & (v <= h)
        return float(np.sum(in_bounds)) / float(len(rays_cam))
