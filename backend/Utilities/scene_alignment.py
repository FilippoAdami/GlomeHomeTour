"""Small world-alignment helpers shared by pose and depth stages."""

from __future__ import annotations

import cv2
import numpy as np

GRID_RES_M = 0.03
EDGE_PERCENTILE = 90.0


def horizontal_align_deg(xyz: np.ndarray, up_axis: int = 1) -> float:
    """Estimate the yaw that aligns dominant horizontal edges with the axes."""
    horiz = xyz[:, [i for i in range(3) if i != up_axis]]
    h_lo = horiz.min(axis=0)
    grid_shape = np.ceil((horiz.max(axis=0) - h_lo) / GRID_RES_M).astype(int) + 1
    idx = np.floor((horiz - h_lo) / GRID_RES_M).astype(int)

    density = np.zeros(grid_shape[::-1], dtype=np.float32)
    np.add.at(density, (idx[:, 1], idx[:, 0]), 1.0)
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=1.5)

    gx = cv2.Sobel(density, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(density, cv2.CV_32F, 0, 1, ksize=5)
    mag, ang = np.sqrt(gx ** 2 + gy ** 2), np.arctan2(gy, gx)
    strong = mag > np.percentile(mag, EDGE_PERCENTILE)
    mean_vec = np.sum(mag[strong] * np.exp(1j * 4 * ang[strong]))
    return -float(np.degrees(np.angle(mean_vec)) / 4.0)


def rotate_horizontal(xyz: np.ndarray, angle_deg: float, up_axis: int = 1) -> np.ndarray:
    """Rotate points around the selected up axis, preserving their height."""
    h_idx = [i for i in range(3) if i != up_axis]
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot2d = np.array([[c, -s], [s, c]])
    out = xyz.copy()
    out[:, h_idx] = xyz[:, h_idx] @ rot2d.T
    return out
