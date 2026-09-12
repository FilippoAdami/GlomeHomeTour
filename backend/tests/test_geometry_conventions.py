"""Geometry-correctness tests for the ARCore <-> DA3 <-> unprojection conventions.

These pin down the two coordinate contracts that silently produced a warped,
oversized point cloud (see backend/project_history.md, 2026-09-10):
  * `Keyframe.transform_matrix` is ARCore/OpenGL camera-to-world, and
    `SurfelCloudInitializer` unprojects with matching OpenGL camera-local rays;
  * DA3 must be handed the OpenCV world-to-camera form of that same pose.
Neither is checkable by eye, and nothing else in the suite exercises geometry.
"""

import numpy as np
import pytest
from PIL import Image

from ingestion.package_loader import CameraIntrinsics, Keyframe
from reconstruction.depth_priors import DepthEstimationError, arcore_c2w_to_da3_w2c
from reconstruction.initialization import SurfelCloudInitializer

W, H = 160, 120
INTRINSICS = CameraIntrinsics(
    camera_model="OPENCV", fl_x=150.0, fl_y=150.0, cx=W / 2.0, cy=H / 2.0,
    w=W, h=H, camera_angle_x=1.0, k1=0.0, k2=0.0, p1=0.0, p2=0.0,
)


def arcore_pose(yaw_deg: float, position: np.ndarray, pitch_deg: float = 0.0) -> np.ndarray:
    """ARCore camera-to-world: +X right, +Y up, camera looks along -Z, world +Y is up.

    Pitch matters: with yaw-only poses a flipped camera-local +Y still lands on a
    vertical wall, so a Y-axis mix-up would go unnoticed.
    """
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    yaw = np.array([[np.cos(a), 0.0, np.sin(a)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(a), 0.0, np.cos(a)]])
    pitch = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(b), -np.sin(b)],
                      [0.0, np.sin(b), np.cos(b)]])
    c2w = np.eye(4)
    c2w[:3, :3] = yaw @ pitch
    c2w[:3, 3] = position
    return c2w


def wall_depth_map(c2w: np.ndarray, wall_z: float = -3.0) -> np.ndarray:
    """Exact Z-depth of the plane {world z == wall_z} as seen from `c2w`."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    # OpenGL camera-local ray, unit component along the -Z view axis
    ray = np.stack([(u - INTRINSICS.cx) / INTRINSICS.fl_x,
                    -(v - INTRINSICS.cy) / INTRINSICS.fl_y,
                    -np.ones_like(u)], axis=-1)
    ray_world = ray @ c2w[:3, :3].T
    # solve origin_z + t * dir_z == wall_z, then depth is t (rays have |z_cam| == 1)
    return (wall_z - c2w[2, 3]) / ray_world[..., 2]


def keyframe(c2w: np.ndarray) -> Keyframe:
    gray = np.full((H, W, 3), 128, dtype=np.uint8)
    return Keyframe(
        file_path="synthetic.jpg", timestamp_ns=0,
        fl_x=INTRINSICS.fl_x, fl_y=INTRINSICS.fl_y, cx=INTRINSICS.cx, cy=INTRINSICS.cy,
        transform_matrix=c2w, image_loader=lambda: Image.fromarray(gray),
    )


def test_da3_extrinsics_roundtrip_matches_unprojection_convention():
    """Projecting with the DA3 w2c must invert the unprojection the initializer does."""
    c2w = arcore_pose(25.0, np.array([0.4, 1.2, 0.7]), pitch_deg=-8.0)
    w2c_cv = arcore_c2w_to_da3_w2c(c2w).astype(np.float64)
    world_pt = np.array([0.15, 0.9, -2.4])

    # Forward: world -> OpenCV camera -> pixel + Z-depth, the way DA3 sees the scene.
    cam_cv = w2c_cv[:3, :3] @ world_pt + w2c_cv[:3, 3]
    assert cam_cv[2] > 0, "point must be in front of the camera in OpenCV convention"
    u = INTRINSICS.fl_x * cam_cv[0] / cam_cv[2] + INTRINSICS.cx
    v = INTRINSICS.fl_y * cam_cv[1] / cam_cv[2] + INTRINSICS.cy
    depth = cam_cv[2]

    # Backward: initialization.py's OpenGL camera-local ray, then the ARCore c2w.
    cam_gl = np.array([(u - INTRINSICS.cx) * depth / INTRINSICS.fl_x,
                       -(v - INTRINSICS.cy) * depth / INTRINSICS.fl_y,
                       -depth])
    assert np.allclose(c2w[:3, :3] @ cam_gl + c2w[:3, 3], world_pt, atol=1e-9)


def test_da3_extrinsics_are_world_to_camera():
    """A c2w handed to DA3 unconverted is the bug; -R^T t is the giveaway."""
    c2w = arcore_pose(40.0, np.array([1.0, 0.5, -2.0]))
    w2c_cv = arcore_c2w_to_da3_w2c(c2w)
    # The camera centre recovered from a w2c is -R^T @ t, and must match the ARCore one.
    centre = -w2c_cv[:3, :3].T @ w2c_cv[:3, 3]
    assert np.allclose(centre, c2w[:3, 3], atol=1e-6)
    assert not np.allclose(w2c_cv[:3, 3], c2w[:3, 3], atol=1e-3), "still camera-to-world"


def test_multiview_flat_wall_unprojects_planar_and_metric():
    """Three views of one wall must fuse into a single plane at the true distance.

    A flipped axis or a c2w/w2c mix-up leaves each view's cloud in a different place,
    which shows up here as centimetres-to-metres of plane RMS.
    """
    poses = [arcore_pose(yaw, np.array([dx, 1.3, 0.0]), pitch_deg=pitch)
             for yaw, dx, pitch in ((0.0, 0.0, 0.0), (-12.0, -0.5, 9.0), (14.0, 0.6, -7.0))]
    keyframes = [keyframe(c2w) for c2w in poses]
    depths = [wall_depth_map(c2w).astype(np.float32) for c2w in poses]

    cloud = SurfelCloudInitializer(
        target_surfels=40_000, voxel_downsample_m=0.01,
        max_depth_m=10.0, min_consensus=0, enable_sor=False,
    ).initialize_from_keyframes(keyframes, depths, INTRINSICS)

    z = cloud.positions[:, 2].astype(np.float64)
    assert len(cloud) > 5_000
    assert abs(z.mean() + 3.0) < 0.01, f"wall at z={z.mean():.3f}, expected -3.0"
    assert np.sqrt(np.mean((z + 3.0) ** 2)) < 0.005, "wall is not planar"


def test_unusable_depth_raises_instead_of_being_fabricated(monkeypatch):
    """The failure that shipped: NaN depth quietly replaced by synthetic geometry."""
    from reconstruction import depth_priors

    est = object.__new__(depth_priors.DepthPriorEstimator)
    est.min_depth, est.max_depth, est.max_invalid_depth_frac = 0.2, 15.0, 0.05
    est._use_da3 = True

    class _NaNModel:
        def inference(self, images, **kwargs):
            nan = np.full((1, H, W), np.nan, dtype=np.float32)
            return type("P", (), {"depth": nan, "conf": None})()

    est._da3_model = _NaNModel()
    with pytest.raises(DepthEstimationError):
        est.estimate_depth_sequence([np.zeros((H, W, 3), dtype=np.uint8)])
