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

import cv2
import numpy as np

from package_loader import CameraIntrinsics, Keyframe


@dataclass
class KeyframeSelectionResult:
    """Summary of dynamic keyframe selection."""
    selected_indices: list[int]
    selected_keyframes: list[Keyframe]
    discarded_indices: list[int]
    total_evaluated: int
    selection_ratio: float
    reasons: dict[str, int] = field(default_factory=dict)


# OpenGL camera (+X right, +Y up, -Z forward) -> OpenCV (+X right, +Y down, +Z forward).
_GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def estimate_scene_depths(
    keyframes: Sequence[Keyframe],
    intrinsics: CameraIntrinsics,
    rotate_to_portrait: bool = True,
    near_percentile: float = 35.0,
    depth_range_m: tuple[float, float] = (0.4, 8.0),
    smooth_window: int = 5,
) -> np.ndarray:
    """Median-ish distance from each camera to what it is actually looking at.

    How far away the scene is decides how fast overlap is lost: at 3 m a 50 cm
    sidestep barely shifts the view, at 50 cm it replaces it. Without this the
    selector has to assume a fixed distance, and on this capture that assumption
    (2 m, against a real median of ~0.8 m) rated plainly disjoint frame pairs at
    0.85-0.90 overlap.

    Tracks corners into a later frame and triangulates them against the known
    poses -- no reconstruction, no depth network, ~40 ms per frame. Takes a low
    percentile rather than the median because it is the *nearest* surfaces that
    drive content out of frame. Frames that cannot be measured (blank wall, no
    parallax) fall back to the scene median.

    Returns one depth per keyframe, clamped to ``depth_range_m``.
    """
    n = len(keyframes)
    if n == 0:
        return np.zeros(0)

    k_mat = np.array([
        [intrinsics.fl_x, 0.0, intrinsics.cx],
        [0.0, intrinsics.fl_y, intrinsics.cy],
        [0.0, 0.0, 1.0],
    ])

    def world_to_cam(c2w: np.ndarray) -> np.ndarray:
        rot = np.dot(c2w[:3, :3], _GL_TO_CV).T
        return np.hstack([rot, (-np.dot(rot, c2w[:3, 3])).reshape(3, 1)])

    cache: dict[int, np.ndarray] = {}

    def gray(i: int) -> np.ndarray:
        if i not in cache:
            img = cv2.cvtColor(keyframes[i].load_image_rgb(), cv2.COLOR_RGB2GRAY)
            # Poses are in the upright frame and so are the intrinsics; the stored
            # frames are still landscape, so rotate to match before measuring.
            if rotate_to_portrait:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cache[i] = img
        return cache[i]

    depths = np.full(n, np.nan)
    for i in range(n):
        src = gray(i)
        corners = cv2.goodFeaturesToTrack(src, maxCorners=400, qualityLevel=0.01, minDistance=12)
        if corners is None or len(corners) < 20:
            continue
        pose_i = keyframes[i].transform_matrix
        # Grow the baseline until triangulation is actually conditioned -- adjacent
        # frames of a slow walk have near-zero parallax and give garbage depth.
        for gap in (4, 8, 16, 32):
            j = i + gap
            if j >= n:
                break
            pose_j = keyframes[j].transform_matrix
            if float(np.linalg.norm(pose_j[:3, 3] - pose_i[:3, 3])) < 0.04:
                continue
            tracked, status, _ = cv2.calcOpticalFlowPyrLK(
                src, gray(j), corners, None, winSize=(21, 21), maxLevel=3)
            if tracked is None:
                continue
            ok = status.ravel() == 1
            src_pts, dst_pts = corners[ok].reshape(-1, 2), tracked[ok].reshape(-1, 2)
            if len(src_pts) < 20:
                continue
            w2c_i = world_to_cam(pose_i)
            homog = cv2.triangulatePoints(
                np.dot(k_mat, w2c_i), np.dot(k_mat, world_to_cam(pose_j)),
                src_pts.T, dst_pts.T)
            pts = (homog[:3] / homog[3]).T
            z = np.dot(pts, w2c_i[:3, :3].T)[:, 2] + w2c_i[2, 3]
            z = z[(z > depth_range_m[0] * 0.5) & (z < depth_range_m[1] * 1.5)]
            if len(z) >= 25:
                depths[i] = float(np.percentile(z, near_percentile))
                break
        cache.pop(i, None)

    measured = depths[~np.isnan(depths)]
    fallback = float(np.median(measured)) if len(measured) else float(np.mean(depth_range_m))
    filled = np.where(np.isnan(depths), fallback, depths)

    # Short-baseline triangulation is noisy: the same frame measures 0.4 m on one
    # pass and 0.8 m on the next, which flips borderline overlap decisions. Depth
    # along a walked trajectory varies smoothly, so a rolling median drops the
    # spikes without shifting the overall level.
    if smooth_window > 1 and len(filled) >= smooth_window:
        pad = smooth_window // 2
        padded = np.pad(filled, pad, mode="edge")
        filled = np.array([
            float(np.median(padded[i:i + smooth_window])) for i in range(len(filled))
        ])

    return np.clip(filled, *depth_range_m)


class DynamicKeyframeSelector:
    """Selects non-redundant keyframes with optimal baseline parallax and coverage."""

    def __init__(
        self,
        min_translation_m: float = 0.35,     # 35 cm minimum translation
        min_rotation_deg: float = 18.0,       # 18 degrees minimum rotation
        min_covisibility: float = 0.35,       # Overlap floor guaranteed between consecutive keyframes
        max_covisibility: float = 0.78,       # If overlap with the last keyframe > 78%, discard as redundant
        reference_depth_m: float = 2.0,       # Reference plane depth for frustum projection (meters)
        sample_grid_size: int = 10,           # N x N sample rays for fast frustum intersection
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
        min_translation_m: float = 0.08,
        min_rotation_deg: float = 4.0,
        min_covisibility: float = 0.50,
        max_covisibility: float = 0.80,
        reference_depth_m: float = 2.0,
        sample_grid_size: int = 10,
    ) -> "DynamicKeyframeSelector":
        """Factory configured specifically for dense 2DGS training keyframe selection.

        ``max_covisibility`` is the only parameter that materially moves the
        selected count (sweeping translation/rotation over 2-3x changed it by
        <1%). ``min_covisibility`` is the one that matters for correctness: it
        is a floor on the overlap between *consecutive* selected keyframes, so
        erring high costs frames but never coverage.

        These numbers only mean anything when ``scene_depths`` is supplied. They
        are calibrated against measured depth, where overlap decays much faster
        than the old fixed-2 m assumption implied, so they read lower than the
        thresholds they replace while being strictly stricter in practice. On
        Bedroom2 this pair keeps ~48% of stage 2 (median depth 0.8 m).
        """
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
        scene_depths: Optional[np.ndarray] = None,
    ) -> KeyframeSelectionResult:
        """Dynamically filter keyframes to an optimal anchor subset.

        ``scene_depths`` (one measured depth per keyframe, from
        :func:`estimate_scene_depths`) is what makes the overlap numbers mean
        anything. Without it every frame is assumed to be looking at something
        ``reference_depth_m`` away, which silently rates disjoint views as
        heavily overlapping wherever the real scene is nearer than that.

        ``min_keyframes``/``max_keyframes`` are best-effort targets, not hard
        limits: they are met by re-walking the trajectory at a looser/tighter
        overlap threshold, and the walk will not go below ``min_covisibility``
        between neighbours to satisfy a budget. A fast pan has a minimum number
        of frames that keeps it connected, and coverage outranks compactness.
        """
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

        self._ray_cache = {}

        # The overlap -> count mapping is scene dependent (room size, walking
        # speed), so a requested band is hit by re-running the walk with a
        # tighter/looser overlap threshold. Never by subsampling the result:
        # dropping every Nth anchor from a chain built for overlap is exactly
        # what used to leave neighbouring keyframes with no common view at all.
        threshold = self.max_covisibility
        result = self._select_chain(keyframes, intrinsics, scene_depths, threshold)

        floor = self.min_covisibility + 0.05
        while (max_keyframes is not None and len(result.selected_indices) > max_keyframes
               and threshold - 0.03 >= floor):
            threshold -= 0.03
            result = self._select_chain(keyframes, intrinsics, scene_depths, threshold)

        while (min_keyframes is not None and len(result.selected_indices) < min_keyframes
               and threshold + 0.03 <= 0.97):
            threshold += 0.03
            result = self._select_chain(keyframes, intrinsics, scene_depths, threshold)

        return result

    def _select_chain(
        self,
        keyframes: Sequence[Keyframe],
        intrinsics: CameraIntrinsics,
        scene_depths: Optional[np.ndarray],
        max_covisibility: float,
    ) -> KeyframeSelectionResult:
        """Walk the trajectory once, keeping an overlapping chain of anchors.

        The invariant this maintains: every consecutive pair of selected
        keyframes shares at least ``min_covisibility`` mutual frustum overlap,
        unless the input trajectory itself contains a jump that large (a VIO
        relocalisation teleport), which no subset of it could bridge.
        """
        n_total = len(keyframes)
        selected_indices: list[int] = [0]
        discarded_indices: list[int] = []
        reasons = {
            "initial_frame": 1,
            "sufficient_motion": 0,
            "coverage_gap_prevention": 0,
            "redundant_motion": 0,
            "redundant_covisibility": 0,
            "unavoidable_gap": 0,
        }

        def pose(i: int) -> tuple[np.ndarray, np.ndarray]:
            c2w = keyframes[i].transform_matrix
            return c2w[:3, :3], c2w[:3, 3]

        def accept(i: int, reason: str) -> None:
            selected_indices.append(i)
            reasons[reason] += 1

        last = 0
        for idx in range(1, n_total):
            covis = self._mutual_covisibility(idx, last, keyframes, intrinsics, scene_depths)

            # Overlap with the anchor has collapsed. The frame before this one
            # was still above the floor, so anchor on *that* instead of accepting
            # a view the anchor can no longer see -- the old code accepted `idx`
            # here, one frame too late, which is what opened the gaps.
            if covis < self.min_covisibility and idx - 1 > last:
                last = idx - 1
                accept(last, "coverage_gap_prevention")
                covis = self._mutual_covisibility(idx, last, keyframes, intrinsics, scene_depths)

            if covis > max_covisibility:
                discarded_indices.append(idx)
                reasons["redundant_covisibility"] += 1
                continue

            cand_r, cand_t = pose(idx)
            last_r, last_t = pose(last)
            trans_dist = float(np.linalg.norm(cand_t - last_t))
            cos_theta = np.clip((float(np.trace(np.dot(cand_r, last_r.T))) - 1.0) / 2.0, -1.0, 1.0)
            rot_deg = float(np.degrees(np.arccos(cos_theta)))
            has_motion = (trans_dist >= self.min_translation_m) or (rot_deg >= self.min_rotation_deg)

            # Only enforceable while overlap is still healthy; below the floor,
            # dropping the frame would re-open the gap we just avoided.
            if not has_motion and covis >= self.min_covisibility:
                discarded_indices.append(idx)
                reasons["redundant_motion"] += 1
                continue

            if covis < self.min_covisibility:
                reasons["unavoidable_gap"] += 1
            accept(idx, "sufficient_motion")
            last = idx

        ratio = len(selected_indices) / max(1, n_total)
        return KeyframeSelectionResult(
            selected_indices=selected_indices,
            selected_keyframes=[keyframes[i] for i in selected_indices],
            discarded_indices=discarded_indices,
            total_evaluated=n_total,
            selection_ratio=ratio,
            reasons=reasons,
        )

    def _generate_canonical_frustum_rays(
        self,
        intrinsics: CameraIntrinsics,
        grid_size: int,
        depth: Optional[float] = None,
    ) -> np.ndarray:
        """Generate grid of 3D sample points at ``depth`` in camera frame (OpenGL: -Z forward)."""
        w, h = float(intrinsics.w), float(intrinsics.h)
        fx, fy = float(intrinsics.fl_x), float(intrinsics.fl_y)
        cx, cy = float(intrinsics.cx), float(intrinsics.cy)

        # Uniform grid over image sensor
        xs = np.linspace(w * 0.1, w * 0.9, grid_size, dtype=np.float32)
        ys = np.linspace(h * 0.1, h * 0.9, grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        d = float(self.reference_depth_m if depth is None else depth)
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

    def _mutual_covisibility(
        self,
        i: int,
        j: int,
        keyframes: Sequence[Keyframe],
        intrinsics: CameraIntrinsics,
        scene_depths: Optional[np.ndarray] = None,
    ) -> float:
        """Worst-direction frustum overlap between two keyframes.

        ``_compute_frustum_covisibility`` is asymmetric -- it samples one view's
        frustum and asks how much of it the other can see. Approaching a wall
        scores high one way and low the other, so testing a single direction let
        pairs through that share almost nothing in the other. Take the minimum.

        Sampled at the *nearer* of the two frames' measured scene depths: near
        geometry is what leaves the frame first, so it sets the overlap.
        """
        depth = None
        if scene_depths is not None:
            depth = float(min(scene_depths[i], scene_depths[j]))
        rays_cam = self._rays_at(intrinsics, depth)

        a_r, a_t = keyframes[i].transform_matrix[:3, :3], keyframes[i].transform_matrix[:3, 3]
        b_r, b_t = keyframes[j].transform_matrix[:3, :3], keyframes[j].transform_matrix[:3, 3]
        return min(
            self._compute_frustum_covisibility(a_r, a_t, b_r, b_t, intrinsics, rays_cam),
            self._compute_frustum_covisibility(b_r, b_t, a_r, a_t, intrinsics, rays_cam),
        )

    def _rays_at(self, intrinsics: CameraIntrinsics, depth: Optional[float]) -> np.ndarray:
        """Sample rays for a depth, memoised at 1 cm resolution."""
        key = None if depth is None else round(depth, 2)
        cache = getattr(self, "_ray_cache", None)
        if cache is None:
            cache = self._ray_cache = {}
        if key not in cache:
            cache[key] = self._generate_canonical_frustum_rays(
                intrinsics, self.sample_grid_size, key)
        return cache[key]
