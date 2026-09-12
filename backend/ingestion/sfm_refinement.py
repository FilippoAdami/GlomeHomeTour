"""GlomeHomeTour Backend: Hybrid VIO+SfM Pose Refinement.

Seeds bundle adjustment with high-frequency VIO trajectory priors and refines
camera poses and lens distortion parameters (k1, k2, p1, p2) via non-linear
least squares on multi-view visual keypoint correspondences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from .package_loader import CameraIntrinsics, Keyframe


@dataclass
class SfMRefinementResult:
    refined_keyframes: list[Keyframe]
    refined_distortion: tuple[float, float, float, float]  # (k1, k2, p1, p2)
    reprojection_error_initial: float
    reprojection_error_final: float
    num_points_3d: int
    num_observations: int


def _project_point(
    point_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    k: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Project a 3D point into pixel coordinates using OpenCV camera model."""
    pts_in = point_3d.reshape(1, 1, 3).astype(np.float64)
    pts_out, _ = cv2.projectPoints(pts_in, rvec, tvec, k, dist_coeffs)
    return pts_out.reshape(2)


class HybridSfMRefiner:
    def __init__(
        self,
        vio_weight: float = 10.0,
        max_features_per_frame: int = 1000,
        min_matches: int = 15,
        huber_loss_scale: float = 1.0,
    ):
        self.vio_weight = vio_weight
        self.max_features_per_frame = max_features_per_frame
        self.min_matches = min_matches
        self.huber_loss_scale = huber_loss_scale
        # Prefer SIFT if available, fallback to ORB
        try:
            self.detector = cv2.SIFT_create(nfeatures=self.max_features_per_frame)
            self.matcher_norm = cv2.NORM_L2
        except Exception:
            self.detector = cv2.ORB_create(nfeatures=self.max_features_per_frame)
            self.matcher_norm = cv2.NORM_HAMMING

    def refine(
        self,
        intrinsics: CameraIntrinsics,
        keyframes: list[Keyframe],
        optimize_distortion: bool = True,
        max_nfev: int = 50,
    ) -> SfMRefinementResult:
        """Execute hybrid VIO+SfM bundle adjustment.

        If visual features or matches are insufficient (e.g. synthetic test fixtures),
        preserves initial VIO poses cleanly.
        """
        num_frames = len(keyframes)
        initial_dist = (intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2)

        if num_frames < 2:
            return SfMRefinementResult(
                refined_keyframes=keyframes,
                refined_distortion=initial_dist,
                reprojection_error_initial=0.0,
                reprojection_error_final=0.0,
                num_points_3d=0,
                num_observations=0,
            )

        # 1. Feature extraction
        keypoints_list: list[list[cv2.KeyPoint]] = []
        descriptors_list: list[Optional[np.ndarray]] = []

        for kf in keyframes:
            img = kf.load_image_rgb()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kps, des = self.detector.detectAndCompute(gray, None)
            keypoints_list.append(kps)
            descriptors_list.append(des)

        # 2. Pairwise matching between adjacent frames & loop candidates
        # Convert camera-to-world (c2w) to world-to-camera (w2c)
        w2c_list = []
        for kf in keyframes:
            c2w = kf.transform_matrix
            w2c = np.linalg.inv(c2w)
            rvec, _ = cv2.Rodrigues(w2c[:3, :3])
            tvec = w2c[:3, 3]
            w2c_list.append((rvec.flatten(), tvec.flatten()))

        # Build feature tracks / observations
        # Track: point_id -> list of (frame_idx, (u, v))
        tracks: list[list[tuple[int, tuple[float, float]]]] = []
        matcher = cv2.BFMatcher(self.matcher_norm, crossCheck=False)

        for i in range(num_frames - 1):
            des1 = descriptors_list[i]
            des2 = descriptors_list[i + 1]
            if des1 is None or des2 is None or len(des1) < self.min_matches or len(des2) < self.min_matches:
                continue

            knn_matches = matcher.knnMatch(des1, des2, k=2)
            good_matches = []
            for m_pair in knn_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_matches:
                for m in good_matches:
                    pt1 = keypoints_list[i][m.queryIdx].pt
                    pt2 = keypoints_list[i + 1][m.trainIdx].pt
                    tracks.append([(i, pt1), (i + 1, pt2)])

        if len(tracks) < 5:
            # Fallback if too few feature tracks found
            return SfMRefinementResult(
                refined_keyframes=keyframes,
                refined_distortion=initial_dist,
                reprojection_error_initial=0.0,
                reprojection_error_final=0.0,
                num_points_3d=0,
                num_observations=0,
            )

        # 3. Triangulate initial 3D points
        k_matrix = np.array([
            [intrinsics.fl_x, 0.0, intrinsics.cx],
            [0.0, intrinsics.fl_y, intrinsics.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.array(initial_dist, dtype=np.float64)

        points_3d = []
        valid_tracks = []

        for track in tracks:
            (f1, pt1), (f2, pt2) = track[0], track[1]
            rvec1, tvec1 = w2c_list[f1]
            rvec2, tvec2 = w2c_list[f2]
            r1, _ = cv2.Rodrigues(rvec1)
            r2, _ = cv2.Rodrigues(rvec2)

            p1 = np.dot(k_matrix, np.hstack((r1, tvec1.reshape(3, 1))))
            p2 = np.dot(k_matrix, np.hstack((r2, tvec2.reshape(3, 1))))

            pt1_h = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
            pt2_h = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)

            p4d = cv2.triangulatePoints(p1, p2, pt1_h, pt2_h)
            p4d /= p4d[3]
            pt_3d = p4d[:3, 0]

            # Check positive depth in both cameras
            z1 = np.dot(r1[2, :], pt_3d) + tvec1[2]
            z2 = np.dot(r2[2, :], pt_3d) + tvec2[2]
            if z1 > 0.1 and z2 > 0.1 and np.linalg.norm(pt_3d) < 50.0:
                points_3d.append(pt_3d)
                valid_tracks.append(track)

        num_points = len(points_3d)
        if num_points < 5:
            return SfMRefinementResult(
                refined_keyframes=keyframes,
                refined_distortion=initial_dist,
                reprojection_error_initial=0.0,
                reprojection_error_final=0.0,
                num_points_3d=0,
                num_observations=0,
            )

        if num_points > 200:
            idx_sub = np.linspace(0, num_points - 1, 200, dtype=int)
            points_3d = [points_3d[i] for i in idx_sub]
            valid_tracks = [valid_tracks[i] for i in idx_sub]
            num_points = len(points_3d)

        # 4. Set up parameters for bundle adjustment
        # Camera parameters: 6 per camera (rvec: 3, tvec: 3)
        # Distortion parameters: 4 (k1, k2, p1, p2) if optimize_distortion else 0
        # 3D points: 3 * num_points
        initial_cam_params = np.hstack([np.hstack(w2c) for w2c in w2c_list])
        initial_points = np.hstack(points_3d)

        initial_params = [initial_cam_params, initial_points]
        if optimize_distortion:
            initial_params.append(dist_coeffs)
        x0 = np.concatenate(initial_params)

        # Store prior camera parameters for VIO prior penalty
        prior_cam_params = initial_cam_params.copy()

        # Compute initial reprojection error
        def compute_residuals(params: np.ndarray) -> np.ndarray:
            cam_params = params[: 6 * num_frames].reshape((num_frames, 6))
            pts_offset = 6 * num_frames
            pts = params[pts_offset: pts_offset + 3 * num_points].reshape((num_points, 3))

            if optimize_distortion:
                d_coeffs = params[pts_offset + 3 * num_points: pts_offset + 3 * num_points + 4]
            else:
                d_coeffs = dist_coeffs

            residuals = []
            # Reprojection errors
            for pt_idx, track in enumerate(valid_tracks):
                x_pt = pts[pt_idx]
                for f_idx, obs_uv in track:
                    rvec = cam_params[f_idx, :3]
                    tvec = cam_params[f_idx, 3:6]
                    proj_uv = _project_point(x_pt, rvec, tvec, k_matrix, d_coeffs)
                    diff = proj_uv - np.array(obs_uv)
                    residuals.append(diff[0])
                    residuals.append(diff[1])

            # VIO Prior regularization penalty
            vio_diff = (cam_params.flatten() - prior_cam_params) * math.sqrt(self.vio_weight)
            residuals.extend(vio_diff.tolist())

            return np.array(residuals, dtype=np.float64)

        res_init = compute_residuals(x0)
        # Separate reprojection residual from prior penalty
        n_obs = sum(len(tr) for tr in valid_tracks)
        error_init = float(np.mean(np.abs(res_init[: 2 * n_obs])))

        # Solve non-linear least squares
        opt_res = least_squares(
            compute_residuals,
            x0,
            loss="huber",
            f_scale=self.huber_loss_scale,
            max_nfev=max_nfev,
            method="trf",
        )

        res_final = opt_res.fun
        error_final = float(np.mean(np.abs(res_final[: 2 * n_obs])))

        # Extract refined parameters
        refined_cam_params = opt_res.x[: 6 * num_frames].reshape((num_frames, 6))
        pts_offset = 6 * num_frames
        if optimize_distortion:
            refined_d = opt_res.x[pts_offset + 3 * num_points: pts_offset + 3 * num_points + 4]
            final_dist = (float(refined_d[0]), float(refined_d[1]), float(refined_d[2]), float(refined_d[3]))
        else:
            final_dist = initial_dist

        # Build refined keyframes
        refined_keyframes = []
        for idx, kf in enumerate(keyframes):
            rvec = refined_cam_params[idx, :3]
            tvec = refined_cam_params[idx, 3:6]
            r, _ = cv2.Rodrigues(rvec)

            # Reconstruct w2c and invert to c2w
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = r
            w2c[:3, 3] = tvec
            refined_c2w = np.linalg.inv(w2c)

            new_kf = Keyframe(
                file_path=kf.file_path,
                timestamp_ns=kf.timestamp_ns,
                fl_x=kf.fl_x,
                fl_y=kf.fl_y,
                cx=kf.cx,
                cy=kf.cy,
                transform_matrix=refined_c2w,
                image_loader=kf.image_loader,
            )
            refined_keyframes.append(new_kf)

        return SfMRefinementResult(
            refined_keyframes=refined_keyframes,
            refined_distortion=final_dist,
            reprojection_error_initial=error_init,
            reprojection_error_final=error_final,
            num_points_3d=num_points,
            num_observations=n_obs,
        )
