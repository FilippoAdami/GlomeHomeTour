"""GlomeHomeTour Backend: Ingestion Quality Gate.

Stage 1 of the staged filtering pipeline (see ``run_staged_filtering.py``):
per-image quality only -- sharpness, exposure and texture content. Redundancy is
stage 2 (:func:`prune_redundant`) and anchor selection is stage 3
(:class:`~keyframe_selector.DynamicKeyframeSelector`).

Sharpness is scored *relative to the scene*, on two axes that fail in opposite
situations:

  * absolute detail   -- Laplacian variance of the sharpest tiles. Drops on a
    dark or flat frame even when it is perfectly in focus.
  * normalised detail -- Laplacian variance divided by the tile's own intensity
    variance, on the most-textured tiles. Immune to brightness, but it punishes
    sharp frames that happen to be dominated by high-contrast structure.

A frame is only called blurred when it fails *both*, which is what real motion
blur does. Judging on absolute Laplacian variance alone discarded ~50% of a
correctly-captured unevenly-lit bedroom scan (dim wood ceilings read as
"blurry"); this reads the same scan at ~98% accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from package_loader import Keyframe


@dataclass
class FrameQualityMetrics:
    index: int
    file_path: str
    blur_score: float          # absolute detail (sharpest-tile Laplacian variance)
    is_blurred: bool
    mean_luminance: float
    is_exposure_outlier: bool
    accepted: bool
    rejection_reason: Optional[str] = None
    normalized_blur_score: float = 0.0   # detail / local contrast
    relative_sharpness: float = 0.0      # best of the two axes, 1.0 == scene median
    saturated_fraction: float = 0.0
    black_fraction: float = 0.0
    texture_score: float = 0.0           # intensity variance of most-textured tiles


@dataclass
class QualityGateResult:
    accepted_indices: list[int]
    discarded_indices: list[int]
    metrics: list[FrameQualityMetrics]
    accepted_keyframes: list[Keyframe]
    discarded_keyframes: list[Keyframe]
    summary: dict[str, int] = field(default_factory=dict)


@dataclass
class _FrameStats:
    blur_score: float
    normalized_blur: float
    texture: float
    mean_luminance: float
    saturated_fraction: float
    black_fraction: float


def compute_motion_deltas(
    r1: np.ndarray,
    t1: np.ndarray,
    r2: np.ndarray,
    t2: np.ndarray,
) -> tuple[float, float]:
    """Euclidean translation distance and geodesic rotation difference in degrees."""
    trans_dist = float(np.linalg.norm(t2 - t1))

    # Relative rotation R_rel = R2 * R1^T; tr(R) = 1 + 2*cos(theta)
    r_rel = np.dot(r2, r1.T)
    cos_theta = (float(np.trace(r_rel)) - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp for numerical stability
    return trans_dist, float(math.degrees(math.acos(cos_theta)))


def prune_redundant(
    keyframes: Sequence[Keyframe],
    min_translation_m: float = 0.03,
    min_rotation_deg: float = 2.0,
) -> tuple[list[int], list[int]]:
    """Stage 2: drop frames with no usable parallax against the last kept frame.

    Deliberately near-duplicate removal only -- a frame taken from the same spot
    at the same angle adds no triangulation baseline but does add optimisation
    weight to whatever it happens to be pointing at. Real view selection is
    stage 3's job.

    Returns:
        (kept_indices, dropped_indices)
    """
    kept: list[int] = []
    dropped: list[int] = []
    last_r = last_t = None

    for idx, kf in enumerate(keyframes):
        r_curr = kf.transform_matrix[:3, :3]
        t_curr = kf.transform_matrix[:3, 3]
        if last_r is not None:
            trans_dist, rot_deg = compute_motion_deltas(last_r, last_t, r_curr, t_curr)
            if trans_dist < min_translation_m and rot_deg < min_rotation_deg:
                dropped.append(idx)
                continue
        kept.append(idx)
        last_r, last_t = r_curr, t_curr

    return kept, dropped


class QualityGate:
    def __init__(
        self,
        blur_threshold: Optional[float] = None,     # optional absolute detail floor
        relative_blur_threshold: float = 0.35,      # fraction of scene-median sharpness
        max_reject_fraction: float = 0.20,          # safety valve, see _apply_reject_cap
        dark_threshold: float = 12.0,               # mean luminance floor
        blown_threshold: float = 250.0,             # mean luminance ceiling
        max_saturated_fraction: float = 0.30,       # clipped-white pixels
        max_black_fraction: float = 0.50,           # crushed-black pixels
        min_texture: float = 60.0,                  # featureless-frame floor
        work_resolution: int = 960,                 # long edge used for scoring
    ):
        self.blur_threshold = blur_threshold
        self.relative_blur_threshold = relative_blur_threshold
        self.max_reject_fraction = max_reject_fraction
        self.dark_threshold = dark_threshold
        self.blown_threshold = blown_threshold
        self.max_saturated_fraction = max_saturated_fraction
        self.max_black_fraction = max_black_fraction
        self.min_texture = min_texture
        self.work_resolution = work_resolution

    # ------------------------------------------------------------------ scoring

    @staticmethod
    def _to_gray(image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if image_rgb.ndim == 2:
            return image_rgb
        raise ValueError(f"Unsupported image shape for blur scoring: {image_rgb.shape}")

    def _tile_stats(self, gray: np.ndarray, tiles_per_side: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Per-tile (Laplacian variance, intensity variance)."""
        g = gray.astype(np.float32)
        h, w = g.shape
        tile_h, tile_w = h // tiles_per_side, w // tiles_per_side
        if tile_h == 0 or tile_w == 0:
            lap = cv2.Laplacian(g, cv2.CV_32F)
            return np.array([float(lap.var())]), np.array([float(g.var())])

        lap = cv2.Laplacian(g, cv2.CV_32F)
        lap_vars, int_vars = [], []
        for ty in range(tiles_per_side):
            for tx in range(tiles_per_side):
                ys, xs = slice(ty * tile_h, (ty + 1) * tile_h), slice(tx * tile_w, (tx + 1) * tile_w)
                lap_vars.append(float(lap[ys, xs].var()))
                int_vars.append(float(g[ys, xs].var()))
        return np.array(lap_vars), np.array(int_vars)

    def compute_blur_score(self, image_rgb: np.ndarray, tiles_per_side: int = 4) -> float:
        """Absolute detail: mean Laplacian variance of the sharpest quarter of tiles.

        Whole-frame Laplacian variance is dominated by large flat surfaces, so a
        genuinely sharp photo of a mostly-blank room scores as "blurry". Scoring
        only the sharpest tiles avoids that, but the result still scales with
        scene brightness and contrast -- use :meth:`evaluate` for the judgement.
        """
        lap_vars, _ = self._tile_stats(self._to_gray(image_rgb), tiles_per_side)
        lap_vars = np.sort(lap_vars)[::-1]
        return float(lap_vars[: max(1, len(lap_vars) // 4)].mean())

    def _frame_stats(self, image_rgb: np.ndarray, tiles_per_side: int = 4) -> _FrameStats:
        gray = self._to_gray(image_rgb)
        long_edge = max(gray.shape)
        if long_edge > self.work_resolution:
            s = self.work_resolution / long_edge
            gray = cv2.resize(gray, (max(1, int(gray.shape[1] * s)), max(1, int(gray.shape[0] * s))),
                              interpolation=cv2.INTER_AREA)

        lap_vars, int_vars = self._tile_stats(gray, tiles_per_side)
        top = max(1, len(lap_vars) // 4)
        blur_score = float(np.sort(lap_vars)[::-1][:top].mean())

        # Normalised detail on the most-textured tiles: high-frequency energy as a
        # fraction of the structure actually present, so dim frames aren't penalised.
        textured = np.argsort(int_vars)[::-1][:top]
        normalized = float(np.median(lap_vars[textured] / (int_vars[textured] + 1.0)))

        return _FrameStats(
            blur_score=blur_score,
            normalized_blur=normalized,
            texture=float(int_vars[textured].mean()),
            mean_luminance=float(gray.mean()),
            saturated_fraction=float((gray >= 250).mean()),
            black_fraction=float((gray <= 8).mean()),
        )

    # ------------------------------------------------------------------ gate

    def _apply_reject_cap(self, blur_flags: list[bool], relative: np.ndarray) -> None:
        """Re-accept the sharpest blur-rejected frames if the cull is implausibly large.

        A whole scan reading as blurred almost always means the metric, not the
        capture, is off (a very dark or very flat scene compresses both axes).
        Losing the truly worst frames still helps; losing most of the scan makes
        reconstruction impossible, so cap the damage and let the operator see it
        in the summary.
        """
        n = len(blur_flags)
        max_reject = int(n * self.max_reject_fraction)
        rejected = [i for i, f in enumerate(blur_flags) if f]
        if len(rejected) <= max_reject:
            return
        for i in sorted(rejected, key=lambda i: -relative[i])[: len(rejected) - max_reject]:
            blur_flags[i] = False

    def evaluate(self, keyframes: list[Keyframe]) -> QualityGateResult:
        """Evaluate and filter a sequence of keyframes on image quality alone."""
        stats = [self._frame_stats(kf.load_image_rgb()) for kf in keyframes]
        if not stats:
            return QualityGateResult([], [], [], [], [], {"total": 0, "accepted": 0,
                                                          "rejected_blur": 0, "rejected_exposure": 0,
                                                          "rejected_texture": 0})

        # Scene-relative sharpness: 1.0 == median frame of this capture, on the
        # better of the two axes. Medians over accept-plausible frames only would
        # be circular, so use every frame -- blur is a minority failure by design.
        absolute = np.array([s.blur_score for s in stats])
        normalized = np.array([s.normalized_blur for s in stats])
        abs_ref = max(float(np.median(absolute)), 1e-6)
        norm_ref = max(float(np.median(normalized)), 1e-9)
        relative = np.maximum(absolute / abs_ref, normalized / norm_ref)

        blur_flags = [
            bool(relative[i] < self.relative_blur_threshold
                 or (self.blur_threshold is not None and absolute[i] < self.blur_threshold))
            for i in range(len(stats))
        ]
        self._apply_reject_cap(blur_flags, relative)

        metrics: list[FrameQualityMetrics] = []
        accepted_indices: list[int] = []
        discarded_indices: list[int] = []
        count_blur = count_exposure = count_texture = 0

        for idx, kf in enumerate(keyframes):
            st = stats[idx]

            # Exposure: judge clipping, not brightness. Uneven daylight makes mean
            # luminance swing wildly between window-facing and wall-facing frames
            # while both remain perfectly reconstructable.
            is_exposure_outlier = (
                st.saturated_fraction > self.max_saturated_fraction
                or st.black_fraction > self.max_black_fraction
                or st.mean_luminance < self.dark_threshold
                or st.mean_luminance > self.blown_threshold
            )
            is_blurred = blur_flags[idx]
            is_featureless = st.texture < self.min_texture

            # Texture before blur: a blank wall has no detail to lose, so the blur
            # axes read it as motion blur. "Featureless" is the actionable reason.
            rejection_reason = None
            if is_exposure_outlier:
                rejection_reason = "exposure"
                count_exposure += 1
            elif is_featureless:
                rejection_reason = "texture"
                count_texture += 1
            elif is_blurred:
                rejection_reason = "blur"
                count_blur += 1

            accepted = (rejection_reason is None)

            metrics.append(FrameQualityMetrics(
                index=idx,
                file_path=kf.file_path,
                blur_score=st.blur_score,
                is_blurred=is_blurred,
                mean_luminance=st.mean_luminance,
                is_exposure_outlier=is_exposure_outlier,
                accepted=accepted,
                rejection_reason=rejection_reason,
                normalized_blur_score=st.normalized_blur,
                relative_sharpness=float(relative[idx]),
                saturated_fraction=st.saturated_fraction,
                black_fraction=st.black_fraction,
                texture_score=st.texture,
            ))

            (accepted_indices if accepted else discarded_indices).append(idx)

        accepted_kfs = [keyframes[i] for i in accepted_indices]
        discarded_kfs = [keyframes[i] for i in discarded_indices]

        summary = {
            "total": len(keyframes),
            "accepted": len(accepted_kfs),
            "rejected_blur": count_blur,
            "rejected_exposure": count_exposure,
            "rejected_texture": count_texture,
        }

        return QualityGateResult(
            accepted_indices=accepted_indices,
            discarded_indices=discarded_indices,
            metrics=metrics,
            accepted_keyframes=accepted_kfs,
            discarded_keyframes=discarded_kfs,
            summary=summary,
        )
