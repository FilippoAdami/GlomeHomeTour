"""GlomeHomeTour Backend: Ingestion Quality Gate.

Filters raw keyframes based on Laplacian blur scoring, kinematic baseline
redundancy pruning, and exposure uniformity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .package_loader import Keyframe


@dataclass
class FrameQualityMetrics:
    index: int
    file_path: str
    blur_score: float
    is_blurred: bool
    baseline_dist_m: float
    angular_dist_deg: float
    is_redundant: bool
    mean_luminance: float
    is_exposure_outlier: bool
    accepted: bool
    rejection_reason: Optional[str] = None


@dataclass
class QualityGateResult:
    accepted_indices: list[int]
    discarded_indices: list[int]
    metrics: list[FrameQualityMetrics]
    accepted_keyframes: list[Keyframe]
    discarded_keyframes: list[Keyframe]
    summary: dict[str, int] = field(default_factory=dict)


class QualityGate:
    def __init__(
        self,
        blur_threshold: Optional[float] = None,
        adaptive_blur: bool = True,
        adaptive_blur_factor: float = 0.50,
        min_blur_floor: float = 25.0,
        min_translation_m: float = 0.03,  # 3 cm
        min_rotation_deg: float = 2.0,     # 2 degrees
        dark_threshold: float = 15.0,
        blown_threshold: float = 245.0,
        max_illumination_jump: float = 60.0,
    ):
        self.blur_threshold = blur_threshold
        self.adaptive_blur = adaptive_blur
        self.adaptive_blur_factor = adaptive_blur_factor
        self.min_blur_floor = min_blur_floor
        self.min_translation_m = min_translation_m
        self.min_rotation_deg = min_rotation_deg
        self.dark_threshold = dark_threshold
        self.blown_threshold = blown_threshold
        self.max_illumination_jump = max_illumination_jump

    def compute_blur_score(self, image_rgb: np.ndarray) -> float:
        """Compute variance of the Laplacian as a defocus/motion blur metric."""
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        elif image_rgb.ndim == 2:
            gray = image_rgb
        else:
            raise ValueError(f"Unsupported image shape for blur scoring: {image_rgb.shape}")

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _extract_rotation_and_translation(self, transform_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract 3x3 rotation matrix and 3D translation vector."""
        r = transform_4x4[:3, :3]
        t = transform_4x4[:3, 3]
        return r, t

    def _compute_motion_deltas(
        self,
        r1: np.ndarray,
        t1: np.ndarray,
        r2: np.ndarray,
        t2: np.ndarray,
    ) -> tuple[float, float]:
        """Compute Euclidean translation distance and geodesic rotation difference in degrees."""
        trans_dist = float(np.linalg.norm(t2 - t1))

        # Relative rotation R_rel = R2 * R1^T
        r_rel = np.dot(r2, r1.T)
        # Trace formula: tr(R) = 1 + 2*cos(theta)
        trace = float(np.trace(r_rel))
        cos_theta = (trace - 1.0) / 2.0
        # Clamp to [-1, 1] for numerical stability
        cos_theta = max(-1.0, min(1.0, cos_theta))
        rot_deg = float(math.degrees(math.acos(cos_theta)))

        return trans_dist, rot_deg

    def evaluate(self, keyframes: list[Keyframe]) -> QualityGateResult:
        """Evaluate and filter a sequence of keyframes."""
        metrics: list[FrameQualityMetrics] = []
        accepted_indices: list[int] = []
        discarded_indices: list[int] = []

        last_accepted_r: Optional[np.ndarray] = None
        last_accepted_t: Optional[np.ndarray] = None
        last_luminance: Optional[float] = None

        count_blur = 0
        count_redundant = 0
        count_exposure = 0

        # Pre-compute blur scores to establish adaptive threshold if enabled
        blur_scores: list[float] = []
        images_rgb: list[np.ndarray] = []
        for kf in keyframes:
            img_rgb = kf.load_image_rgb()
            images_rgb.append(img_rgb)
            blur_scores.append(self.compute_blur_score(img_rgb))

        if self.blur_threshold is not None and not self.adaptive_blur:
            effective_blur_threshold = self.blur_threshold
        elif self.adaptive_blur and blur_scores:
            median_blur = float(np.median(blur_scores))
            effective_blur_threshold = max(self.min_blur_floor, self.adaptive_blur_factor * median_blur)
            if self.blur_threshold is not None:
                effective_blur_threshold = min(self.blur_threshold, effective_blur_threshold)
        else:
            effective_blur_threshold = self.min_blur_floor

        for idx, kf in enumerate(keyframes):
            img_rgb = images_rgb[idx]
            blur_score = blur_scores[idx]
            mean_lum = float(np.mean(img_rgb))

            is_blurred = blur_score < effective_blur_threshold

            is_exposure_outlier = (
                mean_lum < self.dark_threshold or
                mean_lum > self.blown_threshold or
                (last_luminance is not None and abs(mean_lum - last_luminance) > self.max_illumination_jump)
            )

            r_curr, t_curr = self._extract_rotation_and_translation(kf.transform_matrix)

            if last_accepted_r is not None and last_accepted_t is not None:
                trans_dist, rot_dist_deg = self._compute_motion_deltas(
                    last_accepted_r, last_accepted_t, r_curr, t_curr
                )
                is_redundant = (
                    trans_dist < self.min_translation_m and
                    rot_dist_deg < self.min_rotation_deg
                )
            else:
                # First frame is never redundant
                trans_dist = 0.0
                rot_dist_deg = 0.0
                is_redundant = False

            # Determine acceptance and primary rejection reason
            rejection_reason = None
            if is_exposure_outlier:
                rejection_reason = "exposure"
                count_exposure += 1
            elif is_blurred:
                rejection_reason = "blur"
                count_blur += 1
            elif is_redundant:
                rejection_reason = "redundancy"
                count_redundant += 1

            accepted = (rejection_reason is None)

            metric = FrameQualityMetrics(
                index=idx,
                file_path=kf.file_path,
                blur_score=blur_score,
                is_blurred=is_blurred,
                baseline_dist_m=trans_dist,
                angular_dist_deg=rot_dist_deg,
                is_redundant=is_redundant,
                mean_luminance=mean_lum,
                is_exposure_outlier=is_exposure_outlier,
                accepted=accepted,
                rejection_reason=rejection_reason,
            )
            metrics.append(metric)

            if accepted:
                accepted_indices.append(idx)
                last_accepted_r = r_curr
                last_accepted_t = t_curr
                last_luminance = mean_lum
            else:
                discarded_indices.append(idx)

        accepted_kfs = [keyframes[i] for i in accepted_indices]
        discarded_kfs = [keyframes[i] for i in discarded_indices]

        summary = {
            "total": len(keyframes),
            "accepted": len(accepted_kfs),
            "rejected_blur": count_blur,
            "rejected_redundancy": count_redundant,
            "rejected_exposure": count_exposure,
        }

        return QualityGateResult(
            accepted_indices=accepted_indices,
            discarded_indices=discarded_indices,
            metrics=metrics,
            accepted_keyframes=accepted_kfs,
            discarded_keyframes=discarded_kfs,
            summary=summary,
        )
