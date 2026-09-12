"""GlomeHomeTour Backend: Pose Aligner & Timestamp Synchronizer.

Provides sub-millisecond VIO timestamp interpolation using Quaternion SLERP
and cubic spline/linear translation interpolation, matching keyframe midpoint
timestamps to the continuous VIO trajectory.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
from scipy.interpolate import CubicSpline

from .package_loader import Keyframe, TrajectorySample


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Spherical Linear Interpolation (SLERP) between unit quaternions q0 and q1.

    Args:
        q0: array-like [qx, qy, qz, qw]
        q1: array-like [qx, qy, qz, qw]
        u: interpolation factor in [0.0, 1.0]

    Returns:
        Interpolated unit quaternion [qx, qy, qz, qw] as float64 array.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    # Normalize inputs
    norm0 = np.linalg.norm(q0)
    norm1 = np.linalg.norm(q1)
    if norm0 > 0:
        q0 = q0 / norm0
    if norm1 > 0:
        q1 = q1 / norm1

    dot = float(np.dot(q0, q1))

    # Take the shortest path on the 4D hypersphere (antipodal symmetry)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Clamp dot product to [-1, 1]
    dot = min(1.0, max(-1.0, dot))

    # If quaternions are extremely close, perform normalized linear interpolation (NLERP)
    if dot > 0.9995:
        result = (1.0 - u) * q0 + u * q1
        norm = np.linalg.norm(result)
        return result / norm if norm > 0 else q0

    omega = math.acos(dot)
    sin_omega = math.sin(omega)

    scale0 = math.sin((1.0 - u) * omega) / sin_omega
    scale1 = math.sin(u * omega) / sin_omega

    result = scale0 * q0 + scale1 * q1
    norm = np.linalg.norm(result)
    return result / norm if norm > 0 else q0


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qx, qy, qz, qw] to 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    # Normalize
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n > 0:
        qx /= n
        qy /= n
        qz /= n
        qw /= n

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def matrix_to_quaternion(r: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [qx, qy, qz, qw]."""
    m = np.asarray(r, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q


def opengl_to_opencv(transform_4x4: np.ndarray) -> np.ndarray:
    """Convert camera-to-world matrix from OpenGL (+X right, +Y up, +Z back)
    to OpenCV convention (+X right, +Y down, +Z forward).
    """
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.dot(transform_4x4, flip_yz)


def opencv_to_opengl(transform_4x4: np.ndarray) -> np.ndarray:
    """Convert camera-to-world matrix from OpenCV (+X right, +Y down, +Z forward)
    to OpenGL convention (+X right, +Y up, +Z back).
    """
    flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.dot(transform_4x4, flip_yz)


class PoseAligner:
    """Synchronizes and interpolates camera poses using the high-rate VIO trajectory."""

    def __init__(self, trajectory: Sequence[Union[TrajectorySample, Sequence[float]]]):
        if not trajectory:
            raise ValueError("Trajectory cannot be empty")

        if isinstance(trajectory[0], TrajectorySample):
            self.timestamps = np.array([s.timestamp_ns for s in trajectory], dtype=np.int64)
            self.translations = np.array([[s.tx, s.ty, s.tz] for s in trajectory], dtype=np.float64)
            self.quaternions = np.array([[s.qx, s.qy, s.qz, s.qw] for s in trajectory], dtype=np.float64)
        else:
            arr = np.asarray(trajectory, dtype=np.float64)
            self.timestamps = arr[:, 0].astype(np.int64)
            self.translations = arr[:, 1:4]
            self.quaternions = arr[:, 4:8]

        # Ensure timestamps are strictly increasing
        sort_indices = np.argsort(self.timestamps)
        self.timestamps = self.timestamps[sort_indices]
        self.translations = self.translations[sort_indices]
        self.quaternions = self.quaternions[sort_indices]

        # Setup spline for translations if sufficient samples exist
        if len(self.timestamps) >= 4:
            # Normalize timestamps to seconds to maintain high numerical precision in spline
            t_sec = (self.timestamps - self.timestamps[0]) * 1e-9
            self._spline_translation = CubicSpline(t_sec, self.translations)
        else:
            self._spline_translation = None

    def interpolate_pose(self, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate pose (translation, quaternion, 4x4 matrix) at timestamp_ns.

        Returns:
            (translation (3,), quaternion (4,), transform_matrix (4,4))
        """
        # Clamping for edge cases
        if timestamp_ns <= self.timestamps[0]:
            t = self.translations[0].copy()
            q = self.quaternions[0].copy()
        elif timestamp_ns >= self.timestamps[-1]:
            t = self.translations[-1].copy()
            q = self.quaternions[-1].copy()
        else:
            # Find interval
            idx = int(np.searchsorted(self.timestamps, timestamp_ns)) - 1
            idx = max(0, min(len(self.timestamps) - 2, idx))

            t0, t1 = self.timestamps[idx], self.timestamps[idx + 1]
            dt = t1 - t0
            u = float(timestamp_ns - t0) / float(dt) if dt > 0 else 0.0

            # Translation interpolation
            if self._spline_translation is not None:
                t_sec = (timestamp_ns - self.timestamps[0]) * 1e-9
                t = self._spline_translation(t_sec)
            else:
                t = (1.0 - u) * self.translations[idx] + u * self.translations[idx + 1]

            # Quaternion SLERP
            q0 = self.quaternions[idx]
            q1 = self.quaternions[idx + 1]
            q = quaternion_slerp(q0, q1, u)

        r = quaternion_to_matrix(q)
        mat_4x4 = np.eye(4, dtype=np.float64)
        mat_4x4[:3, :3] = r
        mat_4x4[:3, 3] = t

        return t, q, mat_4x4

    def synchronize_keyframes(self, keyframes: list[Keyframe]) -> list[Keyframe]:
        """Return updated keyframes with transform_matrix precisely synchronized to VIO."""
        synced: list[Keyframe] = []
        for kf in keyframes:
            _, _, mat_4x4 = self.interpolate_pose(kf.timestamp_ns)
            synced_kf = Keyframe(
                file_path=kf.file_path,
                timestamp_ns=kf.timestamp_ns,
                fl_x=kf.fl_x,
                fl_y=kf.fl_y,
                cx=kf.cx,
                cy=kf.cy,
                transform_matrix=mat_4x4,
                image_loader=kf.image_loader,
            )
            synced.append(synced_kf)
        return synced
