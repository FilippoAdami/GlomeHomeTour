#!/usr/bin/env python3
"""Single-frame geometry diagnostic for the point-cloud curvature bug.

Runs a *true* single-image DA3 inference (length-1 list), unprojects in the camera
frame only (no pose involved -> planarity is pose-independent), and reports how far
a supposedly-flat region deviates from a least-squares plane.

Also A/B-tests the depth semantics (Z-depth vs Euclidean range), since a linear
intrinsics error cannot bend a plane but a wrong depth convention can.

    .venv/bin/python diagnose_single_frame.py --frame 800
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from pipeline_paths import bootstrap
bootstrap()
_da3_src = _backend_dir / "third_party" / "depth_anything_3" / "src"
if _da3_src.is_dir() and str(_da3_src) not in sys.path:
    sys.path.insert(0, str(_da3_src))

from package_loader import CameraIntrinsics, PackageLoader


def portrait_intrinsics(k: CameraIntrinsics) -> CameraIntrinsics:
    """Landscape -> portrait intrinsics for a cv2.ROTATE_90_CLOCKWISE image."""
    return CameraIntrinsics(
        camera_model=k.camera_model,
        fl_x=k.fl_y, fl_y=k.fl_x,
        cx=float(k.h - k.cy), cy=float(k.cx),
        w=k.h, h=k.w,
        camera_angle_x=2.0 * math.atan(0.5 * k.h / k.fl_y),
        k1=k.k1, k2=k.k2, p1=k.p1, p2=k.p2,
    )


def unproject_cam(depth: np.ndarray, k: CameraIntrinsics, euclidean: bool) -> np.ndarray:
    """(H,W) depth -> (H,W,3) OpenCV camera-frame points (+X right, +Y down, +Z fwd)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    ray_x = (u - k.cx) / k.fl_x
    ray_y = (v - k.cy) / k.fl_y
    if euclidean:
        # depth is range along the ray: normalize the ray to unit length first
        inv_len = 1.0 / np.sqrt(ray_x ** 2 + ray_y ** 2 + 1.0)
        ray_x, ray_y, ray_z = ray_x * inv_len, ray_y * inv_len, inv_len
    else:
        ray_z = np.ones_like(ray_x)
    return np.stack([ray_x * depth, ray_y * depth, ray_z * depth], axis=-1)


def plane_rms(points: np.ndarray) -> tuple[float, np.ndarray]:
    """Least-squares plane fit; returns (RMS deviation in meters, unit normal)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    dev = (points - centroid) @ normal  # mean(dev) == 0 by construction
    return float(np.sqrt(np.mean(dev ** 2))), normal


def dominant_plane(points: np.ndarray, tol: float = 0.02, iters: int = 300) -> tuple[float, float]:
    """RANSAC the largest planar surface; returns (inlier RMS in m, inlier fraction).

    A whole-frame least-squares fit is not a curvature metric -- one bed or floor in
    the corner tilts the plane and buries the very bowing this script exists to find.
    """
    rng = np.random.RandomState(0)
    best_inliers = None
    for _ in range(iters):
        trio = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(trio[1] - trio[0], trio[2] - trio[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        inliers = np.abs((points - trio[0]) @ n) < tol
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    rms, _ = plane_rms(points[best_inliers])
    return rms, float(best_inliers.mean())


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n".encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"] = pts
        rec["c"] = rgb
        f.write(rec.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scenes/bedroom_complete.zip")
    ap.add_argument("--frame", type=int, default=800, help="raw frame index in the package")
    ap.add_argument("--crop", type=float, default=0.6, help="central crop fraction used for the plane fit")
    ap.add_argument("--out", default="scenes/diagnostics")
    args = ap.parse_args()

    package = PackageLoader().load(args.scene)
    kf = package.keyframes[args.frame]
    k_p = portrait_intrinsics(package.intrinsics)
    img = cv2.rotate(kf.load_image_rgb(), cv2.ROTATE_90_CLOCKWISE)
    print(f"frame {args.frame}  image {img.shape}  portrait K "
          f"fx={k_p.fl_x:.1f} fy={k_p.fl_y:.1f} cx={k_p.cx:.1f} cy={k_p.cy:.1f}")

    from depth_priors import DepthPriorEstimator
    est = DepthPriorEstimator()
    assert est._use_da3, "DA3 model failed to load"

    import torch
    with torch.inference_mode():
        pred = est._da3_model.inference([img])  # true single-image call
    depth_raw = np.asarray(pred.depth[0], dtype=np.float64)
    print(f"DA3 processed depth shape {depth_raw.shape}  "
          f"median={np.nanmedian(depth_raw):.3f}  min={np.nanmin(depth_raw):.3f} max={np.nanmax(depth_raw):.3f}")
    print(f"pred.intrinsics = {getattr(pred, 'intrinsics', None)}")

    dh, dw = depth_raw.shape
    # Intrinsics matched to DA3's own output grid: exact, no resampling of depth.
    k_da3 = CameraIntrinsics(
        camera_model=k_p.camera_model,
        fl_x=k_p.fl_x * dw / k_p.w, fl_y=k_p.fl_y * dh / k_p.h,
        cx=k_p.cx * dw / k_p.w, cy=k_p.cy * dh / k_p.h,
        w=dw, h=dh, camera_angle_x=k_p.camera_angle_x,
        k1=0.0, k2=0.0, p1=0.0, p2=0.0,
    )

    c0, c1 = (1.0 - args.crop) / 2.0, (1.0 + args.crop) / 2.0
    ys, xs = slice(int(dh * c0), int(dh * c1)), slice(int(dw * c0), int(dw * c1))
    img_small = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)

    # Z-depth vs euclidean range is the one depth-semantics question a single frame can
    # settle: only a wrong convention bends a plane, since any pinhole intrinsics error
    # is a linear map of the camera frame and maps planes to planes.
    for euclidean in (False, True):
        pts = unproject_cam(depth_raw, k_da3, euclidean)
        crop = pts[ys, xs].reshape(-1, 3)
        crop = crop[np.isfinite(crop).all(axis=1)]
        rms, frac = dominant_plane(crop)
        span = float(np.ptp(crop, axis=0).max())
        label = "euclidean-range" if euclidean else "z-depth"
        print(f"[{label:>15}] dominant-plane RMS = {rms * 100:6.2f} cm over {frac:5.1%} of the "
              f"central crop ({span:.2f} m span)")
        write_ply(Path(args.out) / f"frame{args.frame}_{label}.ply",
                  pts.reshape(-1, 3).astype(np.float32), img_small.reshape(-1, 3))

    print(f"PLYs written to {args.out}/")


if __name__ == "__main__":
    main()
