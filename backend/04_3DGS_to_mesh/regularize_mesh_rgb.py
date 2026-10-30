#!/usr/bin/env python3
"""Sharpen mesh vertex RGB while suppressing small detail and respecting geometric creases."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from fill_mesh_holes import export_ply_with_face_attributes, read_ply_face_flags, read_ply_face_property


def regularize_mesh_rgb(mesh_path: Path | str, strength: float = 1.4, detail_threshold: float = 0.004):
    mesh_path = Path(mesh_path).resolve()
    normal_power = 12
    cleanup_threshold = 0.045
    cleanup_dispersion_weight = 2.5
    cleanup_max_blend = 0.65
    luminance_gain = 1.06
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if not mesh.has_vertex_colors():
        raise ValueError(f"Mesh has no vertex RGB colors: {mesh_path}")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors).astype(np.float32)
    if not len(triangles) or not np.isfinite(colors).all():
        raise ValueError(f"Mesh has empty geometry or non-finite RGB values: {mesh_path}")

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals).astype(np.float32)
    started = time.perf_counter()

    # Suppress isolated color outliers before sharpening, while keeping busy neighborhoods intact.
    neighbor_weight = np.zeros(len(vertices), dtype=np.float64)
    neighbor_color = np.zeros((len(vertices), 3), dtype=np.float64)
    neighbor_color_sq = np.zeros(len(vertices), dtype=np.float64)
    for start in range(0, len(triangles), 500_000):
        faces = triangles[start:start + 500_000]
        points = vertices[faces]
        face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
        a = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
        b = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
        src, dst = np.concatenate((a, b)), np.concatenate((b, a))
        edge_normals = np.tile(face_normals, (6, 1))
        alignment = np.minimum(
            np.einsum("ij,ij->i", edge_normals, normals[src]),
            np.einsum("ij,ij->i", edge_normals, normals[dst]),
        )
        weights = np.clip(alignment, 0.0, 1.0) ** normal_power
        neighbor_weight += np.bincount(src, weights=weights, minlength=len(vertices))
        for channel in range(3):
            neighbor_color[:, channel] += np.bincount(
                src, weights=weights * colors[dst, channel], minlength=len(vertices)
            )
        neighbor_color_sq += np.bincount(
            src, weights=weights * np.square(colors[dst]).sum(axis=1), minlength=len(vertices)
        )

    neighbor_mean = neighbor_color / np.maximum(neighbor_weight[:, None], 1e-12)
    neighbor_dispersion = np.sqrt(np.maximum(
        neighbor_color_sq / np.maximum(neighbor_weight, 1e-12)
        - np.square(neighbor_mean).sum(axis=1), 0.0
    ))
    residual = np.linalg.norm(colors - neighbor_mean, axis=1)
    blend = np.clip(
        (residual - (cleanup_threshold + cleanup_dispersion_weight * neighbor_dispersion)) / 0.12,
        0.0, cleanup_max_blend,
    )
    cleaned = np.clip(colors * (1.0 - blend[:, None]) + neighbor_mean * blend[:, None], 0.0, 1.0)
    cleanup_delta = np.abs(cleaned - colors)

    neighbor_weight = np.zeros(len(vertices), dtype=np.float64)
    neighbor_color = np.zeros((len(vertices), 3), dtype=np.float64)

    # Each face contributes its three edges. Closed input gives every edge two equal contributions.
    for start in range(0, len(triangles), 500_000):
        faces = triangles[start:start + 500_000]
        points = vertices[faces]
        face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
        a = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
        b = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
        src = np.concatenate((a, b))
        dst = np.concatenate((b, a))
        edge_normals = np.tile(face_normals, (6, 1))
        alignment = np.minimum(
            np.einsum("ij,ij->i", edge_normals, normals[src]),
            np.einsum("ij,ij->i", edge_normals, normals[dst]),
        )
        weights = np.clip(alignment, 0.0, 1.0) ** normal_power
        neighbor_weight += np.bincount(src, weights=weights, minlength=len(vertices))
        for channel in range(3):
            neighbor_color[:, channel] += np.bincount(
                src, weights=weights * cleaned[dst, channel], minlength=len(vertices)
            )

    smooth = (neighbor_color + cleaned) / (neighbor_weight[:, None] + 1.0)
    detail = cleaned - smooth
    detail_size = np.linalg.norm(detail, axis=1)
    # Soft-threshold near one RGB step to suppress integration jitter without a hard edge cutoff.
    detail_gain = np.maximum(detail_size - detail_threshold, 0.0) / np.maximum(detail_size, 1e-12)
    sharpened = np.clip(cleaned + strength * detail * detail_gain[:, None], 0.0, 1.0)
    if not np.isfinite(sharpened).all():
        raise RuntimeError("RGB regularization produced non-finite values")

    luminance = sharpened @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance_pivot = float(np.median(luminance))
    adjusted_luminance = np.clip(
        luminance_pivot + luminance_gain * (luminance - luminance_pivot), 0.0, 1.0
    )
    luminance_scale = adjusted_luminance / np.maximum(luminance, 1e-6)
    enhanced = np.clip(sharpened * luminance_scale[:, None], 0.0, 1.0)
    luma_delta = np.abs(enhanced - sharpened)

    hole_fill_flags = read_ply_face_flags(mesh_path)
    if hole_fill_flags is not None and len(hole_fill_flags) != len(triangles):
        raise ValueError(f"PLY face metadata does not match mesh topology: {mesh_path}")
    face_attributes = {}
    if hole_fill_flags is not None:
        face_attributes["is_hole_fill"] = ("uchar", hole_fill_flags)
    fill_r = read_ply_face_property(mesh_path, "fill_r")
    fill_g = read_ply_face_property(mesh_path, "fill_g")
    fill_b = read_ply_face_property(mesh_path, "fill_b")
    if fill_r is not None and fill_g is not None and fill_b is not None:
        if len(fill_r) == len(triangles):
            face_attributes["fill_r"] = ("uchar", fill_r)
            face_attributes["fill_g"] = ("uchar", fill_g)
            face_attributes["fill_b"] = ("uchar", fill_b)
    if not face_attributes:
        face_attributes = None
    temporary = mesh_path.with_name(f"{mesh_path.stem}.rgb.tmp{mesh_path.suffix}")
    if not export_ply_with_face_attributes(temporary, vertices, triangles, enhanced, face_attributes):
        raise RuntimeError(f"Could not write regularized mesh: {temporary}")

    # Read back the export before replacing the sealed source; this checks topology and cap labels.
    written = o3d.io.read_triangle_mesh(str(temporary), enable_post_processing=False)
    if (not np.array_equal(np.asarray(written.vertices), vertices)
            or not np.array_equal(np.asarray(written.triangles), triangles)):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("RGB export changed mesh geometry or topology")
    written_flags = read_ply_face_flags(temporary)
    if hole_fill_flags is not None and not np.array_equal(written_flags, hole_fill_flags):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("RGB export changed sealed-cap face labels")
    os.replace(temporary, mesh_path)

    delta = np.abs(enhanced - colors)
    report = {
        "mesh_path": str(mesh_path),
        "vertices": len(vertices),
        "triangles": len(triangles),
        "hole_fill_faces": int(np.count_nonzero(hole_fill_flags)) if hole_fill_flags is not None else None,
        "method": "geometry-aware speckle cleanup, thresholded RGB unsharp mask, and gentle luminance contrast",
        "strength": strength,
        "detail_threshold": detail_threshold,
        "normal_power": normal_power,
        "speckle_cleanup": {
            "order": "before sharpening",
            "residual_threshold": cleanup_threshold,
            "neighbor_dispersion_weight": cleanup_dispersion_weight,
            "max_blend": cleanup_max_blend,
            "affected_vertex_fraction": float(np.mean(blend > 0.0)),
            "mean_absolute_rgb_change": float(cleanup_delta.mean()),
            "p95_absolute_rgb_change": float(np.percentile(cleanup_delta, 95)),
        },
        "luminance_contrast": {
            "gain": luminance_gain,
            "median_luminance_pivot": luminance_pivot,
            "mean_absolute_rgb_change": float(luma_delta.mean()),
            "newly_saturated_channel_fraction": float(np.mean((enhanced >= 1.0) & (sharpened < 1.0))),
        },
        "mean_absolute_rgb_change": float(delta.mean()),
        "p95_absolute_rgb_change": float(np.percentile(delta, 95)),
        "changed_vertex_fraction": float(np.mean(np.any(delta > 1.0 / 255.0, axis=1))),
        "clipped_channel_fraction": float(np.mean((enhanced == 0.0) | (enhanced == 1.0))),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return report
