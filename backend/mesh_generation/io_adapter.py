"""GlomeHomeTour: Mesh Generation I/O Adapters & Loaders.

Handles ingestion of:
- 2DGS radiance field surfel models (`splats.ply` / `.pt`)
- Metric camera calibrations and trajectory poses (`transforms.json`)
- Synthetic and mock scene generation for isolated offline testing
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from .types import (
    BoundingBox3D,
    PBRMaterial,
    SurfelCloudTorch,
)


@dataclass
class CameraIntrinsicsTorch:
    """Camera intrinsics with PyTorch tensor projection utilities."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    k_matrix: torch.Tensor          # (3, 3) float32
    distortion: Optional[torch.Tensor] = None  # (4,) [k1, k2, p1, p2]

    @classmethod
    def create(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        k1: float = 0.0,
        k2: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        device: Union[str, torch.device] = "cpu",
    ) -> CameraIntrinsicsTorch:
        dev = torch.device(device)
        k = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=dev,
        )
        dist = torch.tensor([k1, k2, p1, p2], dtype=torch.float32, device=dev)
        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            k_matrix=k,
            distortion=dist,
        )

    def to(self, device: Union[str, torch.device]) -> CameraIntrinsicsTorch:
        dev = torch.device(device)
        return CameraIntrinsicsTorch(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.width,
            height=self.height,
            k_matrix=self.k_matrix.to(dev),
            distortion=self.distortion.to(dev) if self.distortion is not None else None,
        )


@dataclass
class KeyframePoseTorch:
    """Individual keyframe with camera-to-world (c2w) and world-to-camera (w2c) tensors."""
    file_path: str
    timestamp_ns: int
    c2w: torch.Tensor       # (4, 4) float32
    w2c: torch.Tensor       # (4, 4) float32
    intrinsics: Optional[CameraIntrinsicsTorch] = None

    def to(self, device: Union[str, torch.device]) -> KeyframePoseTorch:
        dev = torch.device(device)
        return KeyframePoseTorch(
            file_path=self.file_path,
            timestamp_ns=self.timestamp_ns,
            c2w=self.c2w.to(dev),
            w2c=self.w2c.to(dev),
            intrinsics=self.intrinsics.to(dev) if self.intrinsics is not None else None,
        )


@dataclass
class TransformsDataset:
    """Complete dataset parsed from transforms.json."""
    camera_model: str
    intrinsics: CameraIntrinsicsTorch
    frames: List[KeyframePoseTorch]

    def to(self, device: Union[str, torch.device]) -> TransformsDataset:
        dev = torch.device(device)
        return TransformsDataset(
            camera_model=self.camera_model,
            intrinsics=self.intrinsics.to(dev),
            frames=[f.to(dev) for f in self.frames],
        )

    def get_c2w_batch(self) -> torch.Tensor:
        """Return stacked tensor of camera-to-world matrices (B, 4, 4)."""
        return torch.stack([f.c2w for f in self.frames], dim=0)

    def get_w2c_batch(self) -> torch.Tensor:
        """Return stacked tensor of world-to-camera matrices (B, 4, 4)."""
        return torch.stack([f.w2c for f in self.frames], dim=0)


def load_splats_ply(
    ply_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> SurfelCloudTorch:
    """Load 2DGS surfel cloud from standard binary or ASCII PLY file.

    Extracts:
    - positions (x, y, z) in meters (+Y up, -Z forward)
    - normals (nx, ny, nz) unit vectors
    - 2D scales (scale_u, scale_v) in meters
    - colors (red, green, blue) normalized to [0, 1]
    - opacities in [0, 1]
    - optional quaternions or PBR attributes if available
    """
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"PLY file does not exist: {path}")

    plydata = PlyData.read(str(path))
    vertex = plydata["vertex"]
    prop_names = set(p.name for p in vertex.properties)

    # 1. Positions
    x = vertex["x"]
    y = vertex["y"]
    z = vertex["z"]
    positions_np = np.stack([x, y, z], axis=-1).astype(np.float32)

    # 2. Normals
    if {"nx", "ny", "nz"}.issubset(prop_names):
        nx = vertex["nx"]
        ny = vertex["ny"]
        nz = vertex["nz"]
        normals_np = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
        norm_mag = np.linalg.norm(normals_np, axis=-1, keepdims=True)
        norm_mag = np.maximum(norm_mag, 1e-6)
        normals_np = normals_np / norm_mag
    else:
        # Default upward normals if absent
        normals_np = np.zeros_like(positions_np)
        normals_np[:, 1] = 1.0

    # 3. Scales
    if {"scale_u", "scale_v"}.issubset(prop_names):
        su = vertex["scale_u"]
        sv = vertex["scale_v"]
        scales_np = np.stack([su, sv], axis=-1).astype(np.float32)
    elif {"scale_0", "scale_1"}.issubset(prop_names):
        s0 = vertex["scale_0"]
        s1 = vertex["scale_1"]
        scales_np = np.stack([s0, s1], axis=-1).astype(np.float32)
    else:
        scales_np = np.full((len(positions_np), 2), 0.03, dtype=np.float32)

    # 4. Colors
    if {"red", "green", "blue"}.issubset(prop_names):
        r = vertex["red"]
        g = vertex["green"]
        b = vertex["blue"]
        colors_np = np.stack([r, g, b], axis=-1).astype(np.float32)
        if colors_np.max() > 1.0 + 1e-4:
            colors_np = colors_np / 255.0
        colors_np = np.clip(colors_np, 0.0, 1.0)
    else:
        colors_np = np.full((len(positions_np), 3), 0.7, dtype=np.float32)

    # 5. Opacities
    if "opacity" in prop_names:
        op_np = np.asarray(vertex["opacity"], dtype=np.float32).reshape(-1, 1)
        # Invert logit if necessary
        if op_np.min() < 0.0 or op_np.max() > 1.0:
            op_np = 1.0 / (1.0 + np.exp(-op_np))
        op_np = np.clip(op_np, 0.0, 1.0)
    else:
        op_np = np.ones((len(positions_np), 1), dtype=np.float32)

    # 6. Rotations (quaternions) if present
    rot_np = None
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(prop_names):
        r0 = vertex["rot_0"]
        r1 = vertex["rot_1"]
        r2 = vertex["rot_2"]
        r3 = vertex["rot_3"]
        rot_np = np.stack([r1, r2, r3, r0], axis=-1).astype(np.float32) # (qx, qy, qz, qw)
    elif {"qx", "qy", "qz", "qw"}.issubset(prop_names):
        rot_np = np.stack([vertex["qx"], vertex["qy"], vertex["qz"], vertex["qw"]], axis=-1).astype(np.float32)

    # 7. Material roughness & metallic if present
    roughness_np = None
    if "roughness" in prop_names:
        roughness_np = np.asarray(vertex["roughness"], dtype=np.float32).reshape(-1, 1)

    metallic_np = None
    if "metallic" in prop_names:
        metallic_np = np.asarray(vertex["metallic"], dtype=np.float32).reshape(-1, 1)

    dev = torch.device(device)
    return SurfelCloudTorch(
        positions=torch.from_numpy(positions_np).to(dev),
        normals=torch.from_numpy(normals_np).to(dev),
        scales_2d=torch.from_numpy(scales_np).to(dev),
        colors_rgb=torch.from_numpy(colors_np).to(dev),
        opacities=torch.from_numpy(op_np).to(dev),
        rotations=torch.from_numpy(rot_np).to(dev) if rot_np is not None else None,
        roughness=torch.from_numpy(roughness_np).to(dev) if roughness_np is not None else None,
        metallic=torch.from_numpy(metallic_np).to(dev) if metallic_np is not None else None,
    )


def save_splats_ply(
    surfels: SurfelCloudTorch,
    output_path: Union[str, Path],
) -> None:
    """Save surfels to binary little-endian PLY file."""
    surfels.to_ply(output_path)


def load_transforms_json(
    json_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> TransformsDataset:
    """Load and parse transforms.json into PyTorch camera models and keyframe poses."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"transforms.json does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    camera_model = data.get("camera_model", "OPENCV")
    fl_x = float(data["fl_x"])
    fl_y = float(data["fl_y"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    w = int(data["w"])
    h = int(data["h"])

    k1 = float(data.get("k1", 0.0))
    k2 = float(data.get("k2", 0.0))
    p1 = float(data.get("p1", 0.0))
    p2 = float(data.get("p2", 0.0))

    intrinsics = CameraIntrinsicsTorch.create(
        fx=fl_x, fy=fl_y, cx=cx, cy=cy, width=w, height=h,
        k1=k1, k2=k2, p1=p1, p2=p2, device=device
    )

    frames: List[KeyframePoseTorch] = []
    dev = torch.device(device)

    for item in data.get("frames", []):
        file_path = item.get("file_path", "")
        ts = int(item.get("timestamp_ns", 0))
        c2w_mat = np.array(item["transform_matrix"], dtype=np.float32)

        if c2w_mat.shape != (4, 4):
            raise ValueError(f"Expected (4, 4) transform_matrix, got {c2w_mat.shape}")

        w2c_mat = np.linalg.inv(c2w_mat)

        c2w_tensor = torch.from_numpy(c2w_mat).to(dev)
        w2c_tensor = torch.from_numpy(w2c_mat).to(dev)

        # Per-frame intrinsics override if available
        frame_intrinsics = None
        if "fl_x" in item and "fl_y" in item:
            frame_intrinsics = CameraIntrinsicsTorch.create(
                fx=float(item["fl_x"]),
                fy=float(item["fl_y"]),
                cx=float(item.get("cx", cx)),
                cy=float(item.get("cy", cy)),
                width=w,
                height=h,
                k1=k1, k2=k2, p1=p1, p2=p2,
                device=device,
            )

        frames.append(
            KeyframePoseTorch(
                file_path=file_path,
                timestamp_ns=ts,
                c2w=c2w_tensor,
                w2c=w2c_tensor,
                intrinsics=frame_intrinsics,
            )
        )

    return TransformsDataset(
        camera_model=camera_model,
        intrinsics=intrinsics,
        frames=frames,
    )


def create_mock_surfel_cloud(
    num_surfels: int = 2400,
    room_dimensions: Tuple[float, float, float] = (5.0, 2.8, 4.0),
    num_furniture_objects: int = 2,
    device: Union[str, torch.device] = "cpu",
) -> SurfelCloudTorch:
    """Generate synthetic, realistic room surfel cloud for offline testing and verification.

    Generates:
    - Floor plane (Y = 0.0, normal = [0, 1, 0])
    - Ceiling plane (Y = height, normal = [0, -1, 0])
    - 4 walls (normals facing inward)
    - 1-2 interior furniture boxes (normals facing outward)
    """
    w, h, d = room_dimensions
    half_w, half_d = w * 0.5, d * 0.5

    positions = []
    normals = []
    colors = []
    scales = []

    # Floor (Y = 0)
    n_floor = num_surfels // 4
    fx = np.random.uniform(-half_w, half_w, n_floor).astype(np.float32)
    fz = np.random.uniform(-half_d, half_d, n_floor).astype(np.float32)
    fy = np.zeros(n_floor, dtype=np.float32)
    positions.append(np.stack([fx, fy, fz], axis=-1))
    normals.append(np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n_floor, 1)))
    colors.append(np.tile(np.array([0.75, 0.65, 0.50], dtype=np.float32), (n_floor, 1))) # Wood floor
    scales.append(np.full((n_floor, 2), 0.04, dtype=np.float32))

    # Ceiling (Y = h)
    n_ceil = num_surfels // 6
    cx = np.random.uniform(-half_w, half_w, n_ceil).astype(np.float32)
    cz = np.random.uniform(-half_d, half_d, n_ceil).astype(np.float32)
    cy = np.full(n_ceil, h, dtype=np.float32)
    positions.append(np.stack([cx, cy, cz], axis=-1))
    normals.append(np.tile(np.array([0.0, -1.0, 0.0], dtype=np.float32), (n_ceil, 1)))
    colors.append(np.tile(np.array([0.92, 0.92, 0.92], dtype=np.float32), (n_ceil, 1))) # White ceiling
    scales.append(np.full((n_ceil, 2), 0.04, dtype=np.float32))

    # 4 Walls
    n_per_wall = num_surfels // 8
    # Wall -X (left)
    wy = np.random.uniform(0.0, h, n_per_wall).astype(np.float32)
    wz = np.random.uniform(-half_d, half_d, n_per_wall).astype(np.float32)
    wx = np.full(n_per_wall, -half_w, dtype=np.float32)
    positions.append(np.stack([wx, wy, wz], axis=-1))
    normals.append(np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_per_wall, 1)))
    colors.append(np.tile(np.array([0.85, 0.85, 0.80], dtype=np.float32), (n_per_wall, 1)))
    scales.append(np.full((n_per_wall, 2), 0.035, dtype=np.float32))

    # Wall +X (right)
    wy = np.random.uniform(0.0, h, n_per_wall).astype(np.float32)
    wz = np.random.uniform(-half_d, half_d, n_per_wall).astype(np.float32)
    wx = np.full(n_per_wall, half_w, dtype=np.float32)
    positions.append(np.stack([wx, wy, wz], axis=-1))
    normals.append(np.tile(np.array([-1.0, 0.0, 0.0], dtype=np.float32), (n_per_wall, 1)))
    colors.append(np.tile(np.array([0.85, 0.85, 0.80], dtype=np.float32), (n_per_wall, 1)))
    scales.append(np.full((n_per_wall, 2), 0.035, dtype=np.float32))

    # Wall -Z (front)
    wx = np.random.uniform(-half_w, half_w, n_per_wall).astype(np.float32)
    wy = np.random.uniform(0.0, h, n_per_wall).astype(np.float32)
    wz = np.full(n_per_wall, -half_d, dtype=np.float32)
    positions.append(np.stack([wx, wy, wz], axis=-1))
    normals.append(np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_per_wall, 1)))
    colors.append(np.tile(np.array([0.80, 0.82, 0.85], dtype=np.float32), (n_per_wall, 1)))
    scales.append(np.full((n_per_wall, 2), 0.035, dtype=np.float32))

    # Wall +Z (back)
    wx = np.random.uniform(-half_w, half_w, n_per_wall).astype(np.float32)
    wy = np.random.uniform(0.0, h, n_per_wall).astype(np.float32)
    wz = np.full(n_per_wall, half_d, dtype=np.float32)
    positions.append(np.stack([wx, wy, wz], axis=-1))
    normals.append(np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (n_per_wall, 1)))
    colors.append(np.tile(np.array([0.80, 0.82, 0.85], dtype=np.float32), (n_per_wall, 1)))
    scales.append(np.full((n_per_wall, 2), 0.035, dtype=np.float32))

    # Furniture object: Sofa (box centered at x=0.5, z=-0.5, y in [0, 0.85])
    if num_furniture_objects >= 1:
        n_furn = num_surfels // 6
        bx = np.random.uniform(-0.6, 0.6, n_furn).astype(np.float32) + 0.5
        by = np.random.uniform(0.0, 0.85, n_furn).astype(np.float32)
        bz = np.random.uniform(-0.4, 0.4, n_furn).astype(np.float32) - 0.5
        positions.append(np.stack([bx, by, bz], axis=-1))
        # Outward pointing normals
        fn = np.random.randn(n_furn, 3).astype(np.float32)
        fn[:, 1] = np.abs(fn[:, 1])
        fn /= np.linalg.norm(fn, axis=-1, keepdims=True)
        normals.append(fn)
        colors.append(np.tile(np.array([0.25, 0.35, 0.65], dtype=np.float32), (n_furn, 1))) # Blue sofa
        scales.append(np.full((n_furn, 2), 0.025, dtype=np.float32))

    all_pos = np.concatenate(positions, axis=0)
    all_norm = np.concatenate(normals, axis=0)
    all_col = np.concatenate(colors, axis=0)
    all_sc = np.concatenate(scales, axis=0)
    all_op = np.ones((len(all_pos), 1), dtype=np.float32)

    dev = torch.device(device)
    return SurfelCloudTorch(
        positions=torch.from_numpy(all_pos).to(dev),
        normals=torch.from_numpy(all_norm).to(dev),
        scales_2d=torch.from_numpy(all_sc).to(dev),
        colors_rgb=torch.from_numpy(all_col).to(dev),
        opacities=torch.from_numpy(all_op).to(dev),
    )


def create_mock_transforms(
    num_frames: int = 8,
    room_dimensions: Tuple[float, float, float] = (5.0, 2.8, 4.0),
    image_size: Tuple[int, int] = (1080, 1920),
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Create valid transforms.json structure conforming to shared/schemas/transforms.schema.json."""
    w, h = image_size
    fl_x = float(w * 0.85)
    fl_y = float(w * 0.85)
    cx = float(w * 0.5)
    cy = float(h * 0.5)
    cam_angle_x = 2.0 * math.atan(w / (2.0 * fl_x))

    frames = []
    radius = min(room_dimensions[0], room_dimensions[2]) * 0.35
    eye_height = 1.6

    for i in range(num_frames):
        theta = 2.0 * math.pi * i / max(1, num_frames)
        tx = float(radius * math.cos(theta))
        ty = float(eye_height)
        tz = float(radius * math.sin(theta))

        # Camera looks towards center origin [0, eye_height, 0]
        # In OpenGL convention (+Y up, -Z forward, +X right):
        # Forward vector is pointing towards origin: dir = center - pos = [-tx, 0, -tz]
        fwd = np.array([-tx, 0.0, -tz], dtype=np.float32)
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-5:
            fwd = fwd / fwd_norm
        else:
            fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        # OpenGL: -Z is forward, so cam_z = -fwd
        cam_z = -fwd
        cam_x = np.cross(up, cam_z)
        cam_x /= np.maximum(np.linalg.norm(cam_x), 1e-5)
        cam_y = np.cross(cam_z, cam_x)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = cam_x
        c2w[:3, 1] = cam_y
        c2w[:3, 2] = cam_z
        c2w[:3, 3] = [tx, ty, tz]

        frames.append({
            "file_path": f"images/frame_{i:04d}.png",
            "timestamp_ns": int(1000000000 + i * 33333333),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "transform_matrix": c2w.tolist(),
        })

    data = {
        "schema_version": "1.0.0",
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_angle_x": cam_angle_x,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": frames,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return data
