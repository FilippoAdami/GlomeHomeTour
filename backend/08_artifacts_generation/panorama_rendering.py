"""CPU mesh and gsplat equirectangular panorama rendering.

The aligned Phase 07 Gaussian PLY moves its geometry but retains source-frame SH.
``load_gaussians`` consequently restores the PLY geometry to that source frame and
maps each requested aligned-world camera into it before rasterization.
"""
from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WIDTH = 4096
_CUBE_FOV_DEG = 92.0
_RAY_BATCH_ROWS = 32


def equirectangular_directions(width: int, height: int | None = None) -> np.ndarray:
    """Unit world directions; centre column faces +X and increasing x faces +Z."""
    height = height or width // 2
    longitude = (np.arange(width, dtype=np.float32) + 0.5) / width * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.cos(lon), np.sin(lat), cos_lat * np.sin(lon)), axis=-1).astype(np.float32)


def cube_face_axes() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return right, down, forward axes for an OpenCV (+Z forward) cube camera."""
    return {
        "+x": (np.array([0, 0, 1]), np.array([0, -1, 0]), np.array([1, 0, 0])),
        "-x": (np.array([0, 0, -1]), np.array([0, -1, 0]), np.array([-1, 0, 0])),
        "+z": (np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, 1])),
        "-z": (np.array([1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, -1])),
        "+y": (np.array([1, 0, 0]), np.array([0, 0, -1]), np.array([0, 1, 0])),
        "-y": (np.array([1, 0, 0]), np.array([0, 0, 1]), np.array([0, -1, 0])),
    }


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left.T
    rw, rx, ry, rz = right.T
    return np.column_stack((lw * rw - lx * rx - ly * ry - lz * rz,
                            lw * rx + lx * rw + ly * rz - lz * ry,
                            lw * ry - lx * rz + ly * rw + lz * rx,
                            lw * rz + lx * ry - ly * rx + lz * rw))


def _rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


@dataclass
class GaussianPanoramaModel:
    splats: dict
    sh_degree: int
    world_to_source: np.ndarray
    device: str


def inspect_panel_materials(alignment_dir: Path, scan_id: str) -> dict[str, Any]:
    """Inspect the Phase 07 GLB and panel manifest for one sealed scan.

    Current Phase 07 exports carry RGBA vertex colours only.  Returning no
    material data keeps mesh rendering faithful when there is no measured PBR
    material, illumination, or environment to support optical shading.
    """
    alignment_dir = Path(alignment_dir)
    glb_path = alignment_dir / "aligned_sealed_meshes.glb"
    manifest_path = alignment_dir / "stats" / "phase07_region_replacements.json"
    panel_count = 0
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        panel_count = sum(len(region.get("sealed_panels", [])) for region in manifest.get("regions", [])
                          if region.get("scan_id") == scan_id)
    report: dict[str, Any] = {
        "scan_id": scan_id,
        "glb_path": str(glb_path),
        "panel_manifest_path": str(manifest_path),
        "sealed_panel_count": panel_count,
        "optical_effects": "unavailable: Phase 07 GLB has no material properties",
    }
    if not glb_path.is_file():
        report["status"] = "glb_missing"
        return report

    digest = hashlib.sha256()
    with glb_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    report["glb_sha256"] = digest.hexdigest()

    import trimesh

    scene = trimesh.load(glb_path, force="scene", process=False)
    geometry_name = f"sealed_{scan_id}"
    mesh = scene.geometry.get(geometry_name)
    if mesh is None:
        report["status"] = "scan_geometry_missing"
        report["available_geometries"] = sorted(scene.geometry)
        return report
    report["geometry_name"] = geometry_name
    report["geometry_faces"] = int(len(mesh.faces))
    material = getattr(mesh.visual, "material", None)
    if material is None:
        report["status"] = "vertex_colors_only"
        return report
    # Phase 07 currently cannot reach this branch.  Do not substitute visual
    # effects for unmodelled lighting, reflection, or refraction.
    report["status"] = "unsupported_material_type"
    report["material_type"] = type(material).__name__
    return report


def load_gaussians(ply_path: Path, scan_transform: np.ndarray, device: str = "cuda") -> GaussianPanoramaModel:
    """Load the filtered, aligned Phase 07 PLY for gsplat rendering.

    ``scan_transform`` maps source to aligned world.  It is inverted for both
    Gaussian geometry and cameras because PLY SH coefficients were not rotated
    by Phase 07 and therefore still describe source-frame directions.
    """
    import torch
    from plyfile import PlyData

    transform = np.asarray(scan_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("scan_transform must be a finite 4x4 source-to-world transform")
    if not np.allclose(transform[3], (0., 0., 0., 1.), atol=1e-6):
        raise ValueError("scan_transform must be an affine rigid transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5):
        raise ValueError("scan_transform rotation must be proper and orthonormal")
    vertex = PlyData.read(str(ply_path))["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", *(f"scale_{i}" for i in range(3)), *(f"rot_{i}" for i in range(4)),
                *(f"f_dc_{i}" for i in range(3))}
    missing = required - names
    if missing:
        raise ValueError(f"Gaussian PLY lacks fields: {sorted(missing)}")
    rest_names = sorted((name for name in names if name.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if len(rest_names) % 3:
        raise ValueError("Gaussian PLY f_rest fields are not RGB triples")
    coefficient_count = 1 + len(rest_names) // 3
    sh_degree = math.isqrt(coefficient_count) - 1
    if (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError(f"Unsupported SH coefficient count: {coefficient_count}")

    inverse = np.ascontiguousarray(np.linalg.inv(transform), dtype=np.float32)
    xyz_world = np.column_stack([vertex[n] for n in ("x", "y", "z")]).astype(np.float32)
    quats = np.column_stack([vertex[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    scales = np.column_stack([vertex[f"scale_{i}"] for i in range(3)]).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    if not (np.isfinite(xyz_world).all() and np.isfinite(quats).all() and np.isfinite(scales).all() and np.isfinite(opacity).all()):
        raise ValueError("Gaussian PLY contains non-finite geometry or appearance values")
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Gaussian PLY contains zero-length quaternions")
    means = np.ascontiguousarray(xyz_world @ inverse[:3, :3].T + inverse[:3, 3], dtype=np.float32)
    inverse_q = _rotation_quaternion(inverse[:3, :3])[None]
    quats = _quaternion_multiply(np.broadcast_to(inverse_q, quats.shape), quats).astype(np.float32)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    dc = np.column_stack([vertex[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)[:, None]
    if rest_names:
        flat = np.column_stack([vertex[name] for name in rest_names]).astype(np.float32)
        rest = flat.reshape(len(vertex), 3, -1).transpose(0, 2, 1)
        colors = np.ascontiguousarray(np.concatenate((dc, rest), axis=1), dtype=np.float32)
    else:
        colors = np.ascontiguousarray(dc, dtype=np.float32)
    if not np.isfinite(colors).all():
        raise ValueError("Gaussian PLY contains non-finite spherical-harmonic coefficients")
    splats = {
        "means": torch.as_tensor(means, dtype=torch.float32, device=device),
        "quats": torch.as_tensor(np.ascontiguousarray(quats), dtype=torch.float32, device=device),
        "scales": torch.exp(torch.as_tensor(np.ascontiguousarray(scales), dtype=torch.float32, device=device)),
        "opacities": torch.sigmoid(torch.as_tensor(np.ascontiguousarray(opacity), dtype=torch.float32, device=device)),
        "colors": torch.as_tensor(colors, dtype=torch.float32, device=device),
    }
    return GaussianPanoramaModel(splats, sh_degree, inverse, device)


def _camera_in_source(model: GaussianPanoramaModel, center_world: np.ndarray, axes_world: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    inverse = model.world_to_source
    rotation = inverse[:3, :3]
    center_world = np.asarray(center_world, dtype=np.float32)
    if center_world.shape != (3,) or not np.isfinite(center_world).all():
        raise ValueError("center_world must be a finite XYZ point")
    center = center_world @ rotation.T + inverse[:3, 3]
    right, down, forward = (rotation @ axis for axis in axes_world)
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = np.stack((right, down, forward))
    viewmat[:3, 3] = -viewmat[:3, :3] @ center
    return viewmat, center


def _cube_to_equirectangular(cube: dict[str, np.ndarray], width: int, *, uint8: bool = True) -> np.ndarray:
    import cv2
    height, size = width // 2, next(iter(cube.values())).shape[0]
    directions = equirectangular_directions(width, height).reshape(-1, 3)
    axes = cube_face_axes()
    first = next(iter(cube.values()))
    channels = 1 if first.ndim == 2 else first.shape[2]
    result = np.zeros((len(directions), channels), dtype=np.float32)
    f = size / (2.0 * math.tan(math.radians(_CUBE_FOV_DEG) / 2.0))
    for name, (right, down, forward) in axes.items():
        depth = directions @ forward
        mask = (depth > 0) & (np.abs(directions @ forward) >= np.abs(directions @ right)) & (np.abs(directions @ forward) >= np.abs(directions @ down))
        if not mask.any():
            continue
        points = directions[mask]
        mx = (points @ right / depth[mask] * f + (size - 1) / 2.0).astype(np.float32)
        my = (points @ down / depth[mask] * f + (size - 1) / 2.0).astype(np.float32)
        indices = np.flatnonzero(mask)
        # OpenCV limits map dimensions to signed 16-bit values.
        for start in range(0, len(indices), 16384):
            part = indices[start:start + 16384]
            sampled = cv2.remap(cube[name], mx[start:start + len(part)].reshape(-1, 1),
                                my[start:start + len(part)].reshape(-1, 1), cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            result[part] = np.asarray(sampled, dtype=np.float32).reshape(-1, channels)
    result = result.reshape(height, width, channels)
    return np.clip(np.rint(result * 255.0), 0, 255).astype(np.uint8) if uint8 else result


def render_gaussian_panorama(model: GaussianPanoramaModel, center_world: np.ndarray, width: int = DEFAULT_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Render six overlapping pinhole faces and stitch an RGB/alpha panorama."""
    import torch
    from gsplat import rasterization

    if width < 64 or width % 2:
        raise ValueError("width must be an even integer of at least 64")
    size = width // 4
    focal = size / (2.0 * math.tan(math.radians(_CUBE_FOV_DEG) / 2.0))
    intrinsic = torch.tensor([[focal, 0., (size - 1) / 2.], [0., focal, (size - 1) / 2.], [0., 0., 1.]], device=model.device)[None]
    named_axes = list(cube_face_axes().items())
    viewmats = np.stack([_camera_in_source(model, center_world, axes)[0] for _, axes in named_axes])
    viewmats = torch.as_tensor(viewmats, device=model.device)
    intrinsics = intrinsic.expand(len(named_axes), -1, -1).contiguous()
    with torch.no_grad():
        images, alphas, _ = rasterization(**model.splats, viewmats=viewmats, Ks=intrinsics,
                                          width=size, height=size, sh_degree=model.sh_degree,
                                          tile_size=16, packed=False, render_mode="RGB")
    image_faces = images.clamp(0, 1).cpu().numpy()
    alpha_faces = alphas.cpu().numpy()
    rendered = {name: image_faces[i] for i, (name, _) in enumerate(named_axes)}
    alpha_faces = {name: alpha_faces[i] for i, (name, _) in enumerate(named_axes)}
    rgb = _cube_to_equirectangular(rendered, width)
    alpha = _cube_to_equirectangular(alpha_faces, width, uint8=False)[..., 0].astype(np.float32)
    return rgb, alpha


def render_mesh_panorama(scene, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray,
                         center_world: np.ndarray, width: int = DEFAULT_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Raycast a vertex-coloured mesh directly into an equirectangular panorama."""
    import open3d as o3d

    if width < 64 or width % 2:
        raise ValueError("width must be an even integer of at least 64")
    vertices, faces, colors = np.asarray(vertices), np.asarray(faces), np.asarray(colors)
    if colors.shape[0] != vertices.shape[0] or colors.shape[1] < 3:
        raise ValueError("colors must have one RGB value per vertex")
    height = width // 2
    directions = equirectangular_directions(width, height)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    center = np.asarray(center_world, dtype=np.float32)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_world must be a finite XYZ point")
    for start in range(0, height, _RAY_BATCH_ROWS):
        stop = min(start + _RAY_BATCH_ROWS, height)
        ray_count = (stop - start) * width
        rays = np.empty((ray_count, 6), dtype=np.float32)
        rays[:, :3] = center
        rays[:, 3:] = directions[start:stop].reshape(-1, 3)
        hits = scene.cast_rays(o3d.core.Tensor(rays))
        ids = hits["primitive_ids"].numpy().reshape(-1)
        mask = ids < len(faces)
        if not mask.any():
            continue
        uv = hits["primitive_uvs"].numpy().reshape(-1, 2)[mask]
        weights = np.column_stack((1.0 - uv.sum(axis=1), uv))
        rgb = np.einsum("ij,ijk->ik", weights, colors[faces[ids[mask]], :3].astype(np.float32))
        output = image[start:stop].reshape(-1, 3)
        output[mask] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        valid[start:stop].reshape(-1)[mask] = True
    return image, valid
