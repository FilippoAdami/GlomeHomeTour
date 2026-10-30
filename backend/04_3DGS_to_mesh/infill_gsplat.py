#!/usr/bin/env python3
"""Generate squeezed 3D Gaussians from infilled mesh regions and seal 3DGS checkpoints."""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

SH_C0 = 0.28209479177387814


def normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """Computes gsplat quaternions [w, x, y, z] aligning the local +Z axis to given unit normals.

    Uses a right-handed orthonormal basis [t1, t2, n] with det(R) = +1.
    """
    normals = np.asarray(normals, dtype=np.float64)
    single = (normals.ndim == 1)
    if single:
        normals = normals[None, :]

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    n = np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), np.array([[0.0, 0.0, 1.0]]))

    # Choose helper vector that is not parallel to n
    helper = np.where(np.abs(n[:, 2:3]) < 0.999, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(helper, n)
    t1_lens = np.linalg.norm(t1, axis=1, keepdims=True)
    t1 = t1 / np.maximum(t1_lens, 1e-12)
    t2 = np.cross(n, t1)

    # Rotation matrix with columns [t1, t2, n]
    R = np.stack([t1, t2, n], axis=-1)
    quats_xyzw = Rotation.from_matrix(R).as_quat()  # [x, y, z, w]
    quats_wxyz = np.column_stack([quats_xyzw[:, 3], quats_xyzw[:, :3]])  # [w, x, y, z]

    return quats_wxyz[0] if single else quats_wxyz


def sample_squeezed_gaussians_from_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_colors: np.ndarray,
    max_gaussians: int | None = None,
    target_disk_radius_m: float = 0.035,
    overlap_factor: float = 1.4,
    normal_thickness_m: float = 0.002,
    opacity: float = 0.99,
    sh_degree: int = 3,
    sample_density_m: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Generates squeezed planar 3D Gaussians covering infilled triangles.

    Each Gaussian is oriented with its shortest axis aligned with the triangle normal,
    tangent scales spanning the local triangle surface, and color directly matching the triangle.
    """
    k_sh = (sh_degree + 1) ** 2 - 1
    empty_result = {
        "means": torch.empty((0, 3), dtype=torch.float32),
        "scales": torch.empty((0, 3), dtype=torch.float32),
        "quats": torch.empty((0, 4), dtype=torch.float32),
        "opacities": torch.empty((0,), dtype=torch.float32),
        "sh0": torch.empty((0, 1, 3), dtype=torch.float32),
        "shN": torch.empty((0, k_sh, 3), dtype=torch.float32),
    }

    if len(triangles) == 0 or len(vertices) == 0:
        return empty_result

    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-12

    if not np.any(valid):
        return empty_result

    p0 = p0[valid]
    p1 = p1[valid]
    p2 = p2[valid]
    normals = cross[valid] / lengths[valid, None]
    areas = 0.5 * lengths[valid]
    total_area = float(np.sum(areas))

    if total_area <= 1e-12:
        return empty_result

    # Process and sanitize colors
    colors = np.asarray(triangle_colors, dtype=np.float32)
    if colors.ndim == 1 and len(colors) == 3:
        colors = np.tile(colors, (len(triangles), 1))
    colors = colors[valid]
    if colors.size > 0:
        if colors.max() > 1.5:  # Handle 0..255 scale
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    # Determine target number of Gaussians
    if sample_density_m is not None and sample_density_m > 0:
        target_sample_area = (math.sqrt(3.0) / 4.0) * (sample_density_m ** 2)
        n_target = max(1, int(round(total_area / target_sample_area)))
    else:
        target_sample_area = math.pi * (target_disk_radius_m ** 2)
        n_target = max(1, int(round(total_area / target_sample_area)))

    if max_gaussians is not None and max_gaussians > 0:
        n_target = min(n_target, max_gaussians)

    if len(p0) <= 10 and max_gaussians is None:
        n_target = max(n_target, len(p0))

    n_target = max(1, n_target)

    # Vectorized stratified sampling across cumulative areas
    cum_areas = np.cumsum(areas)
    rng = np.random.default_rng(42)
    offsets = rng.uniform(0.1, 0.9, size=n_target)
    sample_u = (np.arange(n_target) + offsets) / n_target * total_area
    tri_idx = np.searchsorted(cum_areas, sample_u)
    tri_idx = np.clip(tri_idx, 0, len(areas) - 1)

    r1 = rng.uniform(0.0, 1.0, size=n_target)
    r2 = rng.uniform(0.0, 1.0, size=n_target)
    sqrt_r1 = np.sqrt(r1)
    u_b = 1.0 - sqrt_r1
    v_b = sqrt_r1 * (1.0 - r2)
    w_b = sqrt_r1 * r2

    sample_means = u_b[:, None] * p0[tri_idx] + v_b[:, None] * p1[tri_idx] + w_b[:, None] * p2[tri_idx]
    sample_normals = normals[tri_idx]
    sample_colors = colors[tri_idx]

    # Calculate tangent scale: broad surface coverage with overlap factor
    area_per_sample = total_area / n_target
    s_tangent = max(5.0 * normal_thickness_m, float(overlap_factor * math.sqrt(area_per_sample / math.pi)))

    sample_scales = np.zeros((n_target, 3), dtype=np.float32)
    sample_scales[:, 0] = s_tangent
    sample_scales[:, 1] = s_tangent
    sample_scales[:, 2] = normal_thickness_m

    sample_quats = normal_to_quaternion(sample_normals)

    log_scales = np.log(np.maximum(sample_scales, 1e-6))
    logit_opacity = float(np.log(np.clip(opacity, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(opacity, 1e-4, 1.0 - 1e-4))))
    sample_opacities = np.full((n_target,), logit_opacity, dtype=np.float32)

    sh0_np = ((sample_colors - 0.5) / SH_C0)[:, None, :]  # (N, 1, 3)
    shN_np = np.zeros((n_target, k_sh, 3), dtype=np.float32)

    return {
        "means": torch.from_numpy(sample_means.astype(np.float32)),
        "scales": torch.from_numpy(log_scales.astype(np.float32)),
        "quats": torch.from_numpy(sample_quats.astype(np.float32)),
        "opacities": torch.from_numpy(sample_opacities),
        "sh0": torch.from_numpy(sh0_np.astype(np.float32)),
        "shN": torch.from_numpy(shN_np),
    }


def export_splats_ply(path: Path | str, splats: Dict[str, torch.Tensor | np.ndarray]) -> None:
    """Exports 3D Gaussian Splats to standard binary PLY format."""
    path = Path(path)
    state = {
        name: (val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val))
        for name, val in splats.items()
    }
    n = len(state["means"])
    fields = [(name, "f4") for name in ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2")]
    fields += [(f"f_rest_{i}", "f4") for i in range(45)]
    fields += [("opacity", "f4")] + [(f"scale_{i}", "f4") for i in range(3)] + [(f"rot_{i}", "f4") for i in range(4)]
    out = np.empty(n, dtype=fields)
    for i, axis in enumerate("xyz"):
        out[axis] = state["means"][:, i]
    out["nx"], out["ny"], out["nz"] = 0, 0, 0
    dc = state["sh0"][:, 0]
    rest = state["shN"].transpose(0, 2, 1).reshape(n, -1)
    for i in range(3):
        out[f"f_dc_{i}"] = dc[:, i]
    for i in range(min(45, rest.shape[1])):
        out[f"f_rest_{i}"] = rest[:, i]
    if rest.shape[1] < 45:
        for i in range(rest.shape[1], 45):
            out[f"f_rest_{i}"] = 0.0
    out["opacity"] = state["opacities"].ravel()
    for i in range(3):
        out[f"scale_{i}"] = state["scales"][:, i]
    for i in range(4):
        out[f"rot_{i}"] = state["quats"][:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".ply") as file:
        temporary = Path(file.name)
    try:
        PlyData([PlyElement.describe(out, "vertex")], text=False).write(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_checkpoint(path: Path | str, payload: dict) -> None:
    """Atomically saves torch checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def seal_gsplat_model(
    gsplat_model_path: Path | str,
    sealed_mesh: o3d.geometry.TriangleMesh,
    clean_num_triangles: int,
    infill_colors: np.ndarray | None = None,
    max_added_gaussians: int | None = None,
    max_gsplat_ratio: float = 0.10,
    target_disk_radius_m: float = 0.035,
    overlap_factor: float = 1.4,
    normal_thickness_m: float = 0.002,
    sample_density_m: float | None = None,
    coverage_deficit_threshold_m: float | None = 0.06,
    output_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """Write a sealed derivative; the stage 03 checkpoint and PLY remain read-only."""
    t0 = time.perf_counter()
    source_model = Path(gsplat_model_path).resolve()
    checkpoint_path = source_model / "checkpoint.pt"
    model_path = Path(output_dir).resolve() if output_dir is not None else source_model
    sealed_checkpoint = model_path / "checkpoint_sealed.pt"
    sealed_ply = model_path / "gaussian_splat_sealed.ply"
    if (sealed_checkpoint.resolve() == checkpoint_path.resolve()
            or sealed_ply.resolve() == (source_model / "gaussian_splat.ply").resolve()):
        raise ValueError("Sealed model files must differ from the read-only source files")
    output_paths = {"checkpoint_path": str(sealed_checkpoint), "ply_path": str(sealed_ply)}

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"3DGS checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "gsplat-stage03-v1":
        raise ValueError(f"Unsupported gsplat checkpoint: {checkpoint_path}")
    if checkpoint.get("sealed"):
        splats = checkpoint["splats"]
        atomic_save_checkpoint(sealed_checkpoint, checkpoint)
        export_splats_ply(sealed_ply, splats)
        total = len(splats["means"])
        return {
            **output_paths,
            "original_gaussians": int(checkpoint.get("original_gaussians", total)),
            "infill_triangles": 0,
            "deficit_triangles": 0,
            "total_target_triangles": 0,
            "infill_gaussians_added": int(checkpoint.get("infill_gaussians_added", 0)),
            "total_sealed_gaussians": total,
            "time_seconds": round(time.perf_counter() - t0, 3),
            "already_sealed": True,
        }

    splats = checkpoint["splats"]
    orig_gaussian_count = len(splats["means"])
    # Short training runs can use fewer SH bands than the checkpoint allocates.
    sh_degree = math.isqrt(splats["shN"].shape[1] + 1) - 1

    triangles = np.asarray(sealed_mesh.triangles)
    vertices = np.asarray(sealed_mesh.vertices)
    has_colors = sealed_mesh.has_vertex_colors()
    vertex_colors = np.asarray(sealed_mesh.vertex_colors) if has_colors else np.full((len(vertices), 3), 0.7)

    num_sealed_triangles = len(triangles)
    num_infill_triangles = max(0, num_sealed_triangles - clean_num_triangles)
    infill_tris = triangles[clean_num_triangles:] if num_infill_triangles > 0 else np.empty((0, 3), dtype=np.int32)

    # Triangle colors derived directly from the mesh / infill_colors
    if num_infill_triangles > 0:
        if infill_colors is not None and len(infill_colors) == num_infill_triangles:
            tri_colors = np.asarray(infill_colors, dtype=np.float64)
        elif hasattr(sealed_mesh, "infill_face_colors") and getattr(sealed_mesh, "infill_face_colors", None) is not None and len(sealed_mesh.infill_face_colors) == num_infill_triangles:
            tri_colors = np.asarray(sealed_mesh.infill_face_colors, dtype=np.float64)
        else:
            try:
                from fill_mesh_holes import derive_infilled_face_textures
                derived = derive_infilled_face_textures(sealed_mesh, clean_num_triangles)
            except Exception:
                derived = None
            if derived is not None and len(derived) == num_infill_triangles:
                tri_colors = derived
            else:
                tri_colors = (
                    vertex_colors[infill_tris[:, 0]]
                    + vertex_colors[infill_tris[:, 1]]
                    + vertex_colors[infill_tris[:, 2]]
                ) / 3.0
    else:
        tri_colors = np.empty((0, 3), dtype=np.float64)

    # Detect TSDF triangles that suffer from a Gaussian coverage deficit (e.g. dark doors, unobserved patches)
    deficit_tris = np.empty((0, 3), dtype=np.int32)
    deficit_colors = np.empty((0, 3), dtype=np.float64)

    if (
        coverage_deficit_threshold_m is not None
        and coverage_deficit_threshold_m > 0
        and clean_num_triangles > 0
        and orig_gaussian_count > 0
    ):
        tsdf_tris = triangles[:clean_num_triangles]
        tsdf_p0 = vertices[tsdf_tris[:, 0]]
        tsdf_p1 = vertices[tsdf_tris[:, 1]]
        tsdf_p2 = vertices[tsdf_tris[:, 2]]
        tsdf_centroids = (tsdf_p0 + tsdf_p1 + tsdf_p2) / 3.0

        splat_means_np = (
            splats["means"].detach().cpu().numpy()
            if isinstance(splats["means"], torch.Tensor)
            else np.asarray(splats["means"])
        )
        tree = cKDTree(splat_means_np)
        dists, _ = tree.query(
            tsdf_centroids,
            distance_upper_bound=coverage_deficit_threshold_m,
            workers=-1,
        )
        is_deficit = np.isinf(dists) | (dists >= coverage_deficit_threshold_m * 0.999)
        deficit_tris = tsdf_tris[is_deficit]
        if len(deficit_tris) > 0:
            deficit_colors = (
                vertex_colors[deficit_tris[:, 0]]
                + vertex_colors[deficit_tris[:, 1]]
                + vertex_colors[deficit_tris[:, 2]]
            ) / 3.0
            print(
                f"[GSplat Sealing] Detected {len(deficit_tris):,} TSDF coverage deficit triangles "
                f"(nearest splat > {coverage_deficit_threshold_m*100:.1f}cm)"
            )

    # Combine infilled boundary triangles and coverage-deficit triangles
    if len(infill_tris) > 0 and len(deficit_tris) > 0:
        target_tris = np.concatenate([infill_tris, deficit_tris], axis=0)
        target_colors = np.concatenate([tri_colors, deficit_colors], axis=0)
    elif len(infill_tris) > 0:
        target_tris = infill_tris
        target_colors = tri_colors
    elif len(deficit_tris) > 0:
        target_tris = deficit_tris
        target_colors = deficit_colors
    else:
        target_tris = np.empty((0, 3), dtype=np.int32)
        target_colors = np.empty((0, 3), dtype=np.float64)

    if len(target_tris) == 0:
        updated_checkpoint = dict(checkpoint, sealed=True,
                                  original_gaussians=orig_gaussian_count, infill_gaussians_added=0)
        atomic_save_checkpoint(sealed_checkpoint, updated_checkpoint)
        export_splats_ply(sealed_ply, splats)
        return {
            **output_paths,
            "original_gaussians": orig_gaussian_count,
            "infill_triangles": 0,
            "deficit_triangles": 0,
            "total_target_triangles": 0,
            "infill_gaussians_added": 0,
            "total_sealed_gaussians": orig_gaussian_count,
            "time_seconds": round(time.perf_counter() - t0, 3),
        }

    if target_colors.size > 0:
        if target_colors.max() > 1.5:  # Handle 0..255 scale
            target_colors = target_colors / 255.0
        target_colors = np.clip(target_colors, 0.0, 1.0)

    # Round the addition budget, retaining a one-Gaussian minimum for tiny models.
    max_allowed = max(1, int(round(max_gsplat_ratio * orig_gaussian_count)))
    if max_added_gaussians is not None and max_added_gaussians > 0:
        budget = min(max_added_gaussians, max_allowed)
    else:
        budget = max_allowed

    # Generate squeezed Gaussians from target triangles
    new_splats = sample_squeezed_gaussians_from_triangles(
        vertices=vertices,
        triangles=target_tris,
        triangle_colors=target_colors,
        max_gaussians=budget,
        target_disk_radius_m=target_disk_radius_m,
        overlap_factor=overlap_factor,
        normal_thickness_m=normal_thickness_m,
        sh_degree=sh_degree,
        sample_density_m=sample_density_m,
    )
    num_added = len(new_splats["means"])
    assert num_added <= max_allowed, f"Added Gaussians ({num_added}) exceeds {max_gsplat_ratio*100}% limit ({max_allowed})"

    # Ensure opacities shapes match (1D vs 2D)
    orig_opacities = splats["opacities"]
    new_opacities = new_splats["opacities"]
    if orig_opacities.ndim == 2 and new_opacities.ndim == 1:
        new_opacities = new_opacities[:, None]
    elif orig_opacities.ndim == 1 and new_opacities.ndim == 2:
        new_opacities = new_opacities.squeeze(-1)

    updated_splats = {
        "means": torch.cat([splats["means"], new_splats["means"]], dim=0),
        "scales": torch.cat([splats["scales"], new_splats["scales"]], dim=0),
        "quats": torch.cat([splats["quats"], new_splats["quats"]], dim=0),
        "opacities": torch.cat([orig_opacities, new_opacities], dim=0),
        "sh0": torch.cat([splats["sh0"], new_splats["sh0"]], dim=0),
        "shN": torch.cat([splats["shN"], new_splats["shN"]], dim=0),
    }

    # Atomically save updated checkpoint
    updated_checkpoint = dict(checkpoint)
    updated_checkpoint["splats"] = updated_splats
    updated_checkpoint["sealed"] = True
    updated_checkpoint["original_gaussians"] = orig_gaussian_count
    updated_checkpoint["infill_gaussians_added"] = num_added
    atomic_save_checkpoint(sealed_checkpoint, updated_checkpoint)

    export_splats_ply(sealed_ply, updated_splats)

    t_elapsed = round(time.perf_counter() - t0, 3)
    stats = {
        **output_paths,
        "original_gaussians": orig_gaussian_count,
        "infill_triangles": num_infill_triangles,
        "deficit_triangles": len(deficit_tris),
        "total_target_triangles": len(target_tris),
        "infill_gaussians_added": num_added,
        "total_sealed_gaussians": len(updated_splats["means"]),
        "time_seconds": t_elapsed,
    }
    print(f"[GSplat Sealing] Injected {num_added:,} squeezed Gaussians from {len(target_tris):,} triangles ({num_infill_triangles:,} infilled + {len(deficit_tris):,} deficit) into {sealed_checkpoint.name} in {t_elapsed:.3f}s (Total: {len(updated_splats['means']):,})")
    return stats
