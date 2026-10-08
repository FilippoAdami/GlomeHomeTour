"""Dataset adapters used by the FastGS training entry points."""
from __future__ import annotations
import json
from pathlib import Path
from typing import NamedTuple
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text, read_points3D_binary, read_points3D_text)


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def _world_to_camera(rotation, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation).T
    transform[:3, 3] = translation
    return transform


def _sh0_to_rgb(coefficients):
    return np.asarray(coefficients) * 0.28209479177387814 + 0.5


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud | None
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    if not cam_info:
        return {"translate": np.zeros(3), "radius": 1.0}
    centers = np.asarray([np.linalg.inv(_world_to_camera(camera.R, camera.T))[:3, 3] for camera in cam_info])
    center = centers.mean(axis=0)
    return {"translate": -center, "radius": float(np.linalg.norm(centers - center, axis=1).max() * 1.1)}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cameras = []
    for extrinsic in cam_extrinsics.values():
        intrinsic = cam_intrinsics[extrinsic.camera_id]
        if intrinsic.model == "SIMPLE_PINHOLE":
            focal_x = focal_y = intrinsic.params[0]
        elif intrinsic.model == "PINHOLE":
            focal_x, focal_y = intrinsic.params[:2]
        else:
            raise ValueError(f"unsupported COLMAP camera model: {intrinsic.model}")
        image_path = Path(images_folder) / Path(extrinsic.name).name
        image = Image.open(image_path)
        cameras.append(CameraInfo(intrinsic.id, qvec2rotmat(extrinsic.qvec).T, np.asarray(extrinsic.tvec),
            focal2fov(focal_y, intrinsic.height), focal2fov(focal_x, intrinsic.width), image,
            str(image_path), image_path.stem, intrinsic.width, intrinsic.height))
    return cameras


def fetchPly(path):
    vertices = PlyData.read(path)["vertex"]
    required = ("x", "y", "z", "red", "green", "blue", "nx", "ny", "nz")
    missing = [name for name in required if name not in vertices.data.dtype.names]
    if missing:
        raise ValueError(f"PLY vertex data is missing: {', '.join(missing)}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"])).astype(np.float64) / 255.0
    normals = np.column_stack((vertices["nx"], vertices["ny"], vertices["nz"])).astype(np.float64)
    return BasicPointCloud(points=points, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    points, colors = np.asarray(xyz, dtype=np.float32), np.asarray(rgb)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("xyz and rgb must both have shape (N, 3)")
    vertex = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["nx"] = vertex["ny"] = vertex["nz"] = 0
    vertex["red"], vertex["green"], vertex["blue"] = np.clip(colors, 0, 255).astype(np.uint8).T
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def _read_colmap_model(sparse_dir):
    sparse_dir = Path(sparse_dir)
    try:
        return read_extrinsics_binary(sparse_dir / "images.bin"), read_intrinsics_binary(sparse_dir / "cameras.bin")
    except (OSError, ValueError, UnicodeDecodeError):
        return read_extrinsics_text(sparse_dir / "images.txt"), read_intrinsics_text(sparse_dir / "cameras.txt")


def readColmapSceneInfo(path, images, eval, llffhold=8):
    root, sparse_dir = Path(path), Path(path) / "sparse" / "0"
    extrinsics, intrinsics = _read_colmap_model(sparse_dir)
    camera_infos = sorted(readColmapCameras(extrinsics, intrinsics, root / (images or "images")),
                          key=lambda camera: camera.image_name)
    if eval:
        train_cameras = [camera for index, camera in enumerate(camera_infos) if index % llffhold]
        test_cameras = [camera for index, camera in enumerate(camera_infos) if not index % llffhold]
    else:
        train_cameras, test_cameras = camera_infos, []
    ply_path = sparse_dir / "points3D.ply"
    if not ply_path.exists():
        try:
            xyz, rgb, _ = read_points3D_binary(sparse_dir / "points3D.bin")
        except (OSError, ValueError):
            xyz, rgb, _ = read_points3D_text(sparse_dir / "points3D.txt")
        storePly(ply_path, xyz, rgb)
    try:
        cloud = fetchPly(ply_path)
    except (OSError, ValueError, KeyError):
        cloud = None
    return SceneInfo(cloud, train_cameras, test_cameras, getNerfppNorm(train_cameras), str(ply_path))


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    root = Path(path)
    with open(root / transformsfile, encoding="utf-8") as handle:
        transform_data = json.load(handle)
    fov_x, background, cameras = transform_data["camera_angle_x"], np.ones(3) if white_background else np.zeros(3), []
    for index, frame in enumerate(transform_data["frames"]):
        image_path = root / f"{frame['file_path']}{extension}"
        c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        rgba = np.asarray(Image.open(image_path).convert("RGBA"), dtype=np.float64) / 255.0
        image = Image.fromarray(np.asarray((rgba[:, :, :3] * rgba[:, :, 3:] +
            background * (1 - rgba[:, :, 3:])) * 255, dtype=np.uint8), "RGB")
        fov_y = focal2fov(fov2focal(fov_x, image.width), image.height)
        cameras.append(CameraInfo(index, w2c[:3, :3].T, w2c[:3, 3], fov_y, fov_x, image,
            str(image_path), image_path.stem, image.width, image.height))
    return cameras


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    train_cameras = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    test_cameras = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    if not eval:
        train_cameras.extend(test_cameras)
        test_cameras = []
    ply_path = Path(path) / "points3d.ply"
    if not ply_path.exists():
        points = np.random.random((100_000, 3)) * 2.6 - 1.3
        colors = _sh0_to_rgb(np.random.random((100_000, 3)) / 255.0)
        storePly(ply_path, points, colors * 255)
    try:
        cloud = fetchPly(ply_path)
    except (OSError, ValueError, KeyError):
        cloud = None
    return SceneInfo(cloud, train_cameras, test_cameras, getNerfppNorm(train_cameras), str(ply_path))


sceneLoadTypeCallbacks = {"Colmap": readColmapSceneInfo, "Blender": readNerfSyntheticInfo}
