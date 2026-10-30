"""Keep non-degenerate faces supported by a dominant structural plane."""
from __future__ import annotations

import numpy as np

from fit_floors_ceilings_planes import fit_robust_plane


FIT_DISTANCE_M = 0.10
KEEP_DISTANCE_M = 0.20
MIN_SUPPORT_FRACTION = 0.50
VERTICAL_NORMAL_Y_MAX = 0.5


def filter_planar_outliers(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return a face mask, retaining vertical faces when they lie near the plane."""
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-12
    keep = valid.copy()
    plane = None
    support = 0.0

    if valid.any():
        points = corners[valid].mean(axis=1)
        weights = double_area[valid] * 0.5
        normals = cross[valid] / double_area[valid, None]
        try:
            normal, offset, support = fit_robust_plane(
                points, weights, FIT_DISTANCE_M, prefer_normal_up=True, face_normals=normals,
            )
        except ValueError:
            support = 0.0
        else:
            plane = (normal, offset)
            if support >= MIN_SUPPORT_FRACTION:
                distances = np.abs(points @ normal + offset)
                keep[valid] = distances <= KEEP_DISTANCE_M

    vertical = valid & (np.abs(cross[:, 1]) / np.maximum(double_area, 1e-12) <= VERTICAL_NORMAL_Y_MAX)
    removed = ~keep
    return keep, {
        "plane_fit_applied": plane is not None and support >= MIN_SUPPORT_FRACTION,
        "plane_support_fraction": float(support),
        "plane_normal": plane[0].tolist() if plane is not None else None,
        "plane_offset": float(plane[1]) if plane is not None else None,
        "triangles_removed": int(removed.sum()),
        "far_vertical_triangles_removed": int((removed & vertical).sum()),
        "other_outlier_triangles_removed": int((removed & valid & ~vertical).sum()),
        "degenerate_triangles_removed": int((~valid).sum()),
    }
