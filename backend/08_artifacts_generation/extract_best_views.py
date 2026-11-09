#!/usr/bin/env python3
"""Extract compact acquired-photo galleries with Phase 07 aligned COLMAP poses."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch

from best_views_visibility import SurfaceVisibility
from generate_floor_level_meshes import (
    DEFAULT_ALIGNMENT_DIR, _canonical_triangles, _matched_source_faces, _read_mesh,
)


LOGGER = logging.getLogger("extract_best_views")
LOW_PRIORITY_CATEGORIES = frozenset({"window", "door", "stairs", "rug"})
LOW_PRIORITY_WEIGHT = 0.25
SAMPLE_COUNT = 50_000
RENDER_LONG_EDGE = 512
DEFAULT_BATCH_SIZE = 16
UTILITY_TARGET = 0.80
MIN_RELATIVE_GAIN = 0.10
MIN_LAYOUT_GAIN = 0.10
MIN_LAPLACIAN_VARIANCE = 4.0
MIN_SHARPNESS_RATIO = 0.80
MAX_SINGLE_SURFACE_FRACTION = 0.75
MIN_RECOGNIZED_OBJECT_ADEQUACY = 0.5
MAX_VISIBLE_SURFACE_OVERLAP = 0.50


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _rigid(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Phase 07 scan transform must be a finite rigid 4x4 matrix")
    return matrix


def _world_to_camera(image) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3] = image.cam_from_world().matrix()
    return result


def check_reprojections(reconstruction) -> dict:
    """Reject internally inconsistent poses before trusting image/geometry coverage."""
    image_ids = sorted(reconstruction.reg_image_ids())
    pixels, normalized, behind, checked = [], [], 0, 0
    for index in np.linspace(0, len(image_ids) - 1, min(20, len(image_ids)), dtype=int):
        image = reconstruction.images[image_ids[index]]
        calibration = reconstruction.cameras[image.camera_id]
        view = _world_to_camera(image)
        observations = [p for p in image.points2D if p.has_point3D() and p.point3D_id in reconstruction.points3D]
        if not observations:
            continue
        diagonal = math.hypot(calibration.width, calibration.height)
        for point_index in np.linspace(0, len(observations) - 1, min(32, len(observations)), dtype=int):
            point = observations[point_index]
            camera = view[:3, :3] @ reconstruction.points3D[point.point3D_id].xyz + view[:3, 3]
            checked += 1
            if camera[2] <= 0:
                behind += 1
                continue
            projected = calibration.calibration_matrix() @ camera
            error = float(np.linalg.norm(projected[:2] / projected[2] - point.xy))
            if not np.isfinite(error):
                raise ValueError("COLMAP observation contains nonfinite geometry or image coordinates")
            pixels.append(error)
            normalized.append(error / diagonal)
    median = float(np.median(pixels)) if pixels else None
    report = {"checked_observations": checked, "behind_camera_count": behind,
              "median_error_pixels": median}
    if checked >= 32 and ((normalized and np.median(normalized) > 0.01) or behind / checked > 0.05):
        median_text = f"{median:.1f}" if median is not None else "unavailable"
        raise ValueError(f"COLMAP reprojection check failed: median {median_text} pixels, "
                         f"{behind}/{checked} observed points behind cameras. "
                         "Repair the upstream COLMAP model; extract_best_views preserves its poses.")
    report["status"] = "passed" if checked >= 32 else "insufficient_observations"
    return report


def load_cameras(scan_dir: Path, alignment_dir: Path, transform) -> tuple[list[dict], dict]:
    import pycolmap

    transform = _rigid(transform)
    source_dir = scan_dir / ".raw_unaligned" / "sparse" / "0"
    aligned_dir = alignment_dir / "scans" / scan_dir.name / "sparse" / "0"
    for path in (source_dir, aligned_dir):
        if not any((path / f"images.{suffix}").is_file() for suffix in ("bin", "txt")):
            raise FileNotFoundError(f"Missing COLMAP model for pose lineage: {path}")
    source = pycolmap.Reconstruction(str(source_dir))
    aligned = pycolmap.Reconstruction(str(aligned_dir))
    source_ids = set(source.reg_image_ids())
    if source_ids != set(aligned.reg_image_ids()):
        raise ValueError("Source and Phase 07 aligned registered image IDs differ")
    image_root = (scan_dir / "images").resolve()
    cameras = []
    inverse_transform = np.linalg.inv(transform)
    for image_id in sorted(source_ids):
        original, image = source.images[image_id], aligned.images[image_id]
        calibration, original_calibration = aligned.cameras[image.camera_id], source.cameras[original.camera_id]
        if (image.name != original.name or image.camera_id != original.camera_id
                or calibration.model_name != original_calibration.model_name
                or (calibration.width, calibration.height) != (original_calibration.width, original_calibration.height)
                or not np.array_equal(calibration.params, original_calibration.params)):
            raise ValueError(f"Image identity or calibration changed during alignment: {image_id}")
        expected = _world_to_camera(original) @ inverse_transform
        view = _world_to_camera(image)
        if not np.allclose(view, expected, rtol=1e-8, atol=1e-6):
            raise ValueError(f"Stale or doubly transformed Phase 07 camera: {image.name}")
        _rigid(view)
        params = np.asarray(calibration.params)
        model = calibration.model_name
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            distortion = params[3:]
        elif model in {"PINHOLE", "OPENCV", "FULL_OPENCV"}:
            distortion = params[4:]
        else:
            raise ValueError(f"Unsupported camera model {model}; use the pipeline's undistorted pinhole inputs")
        k = calibration.calibration_matrix()
        if (np.any(distortion != 0) or not np.isfinite(params).all()
                or min(k[0, 0], k[1, 1], calibration.width, calibration.height) <= 0):
            raise ValueError(f"Invalid or distorted calibration for {image.name}")
        relative = Path(image.name)
        path = (image_root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(image_root):
            raise ValueError(f"COLMAP image path escapes images directory: {image.name}")
        cameras.append({
            "image_id": int(image_id), "camera_id": int(image.camera_id), "image_name": image.name,
            "source_image": str(path), "width": int(calibration.width), "height": int(calibration.height),
            "camera_model": model, "camera_params": params.tolist(), "intrinsics": k.tolist(),
            "qvec_wxyz": np.roll(image.cam_from_world().rotation.quat, 1).tolist(),
            "tvec": image.cam_from_world().translation.tolist(), "world_to_camera": view.tolist(),
            "camera_to_world": np.linalg.inv(view).tolist(),
        })
    if not cameras:
        raise ValueError(f"No registered images in {aligned_dir}")
    reprojections = check_reprojections(aligned)
    fingerprints = {}
    for name, root in (("source", source_dir), ("aligned", aligned_dir)):
        fingerprints[name] = {p.name: _hash(p) for p in sorted(root.iterdir())
                              if p.name in {"images.bin", "images.txt", "cameras.bin", "cameras.txt"}}
    return cameras, {"checked_images": len(cameras), "source_model": str(source_dir),
                     "aligned_model": str(aligned_dir), "model_hashes": fingerprints,
                     "reprojection_check": reprojections}


def _quality(camera: dict) -> dict:
    try:
        with Image.open(camera["source_image"]) as source:
            if source.size != (camera["width"], camera["height"]):
                return {"status": "excluded", "reason": "image_calibration_dimension_mismatch"}
            source.draft("RGB", (512, 512))
            source.thumbnail((512, 512))
            rgb = np.asarray(source.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mean, std = float(gray.mean()), float(gray.std())
        if std < 1 and (mean < 2 or mean > 253):
            return {"status": "excluded", "reason": "empty_extreme_luminance_frame"}
        laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        return {"status": "usable", "sharpness": laplacian / (std ** 2 + 4),
                "laplacian_variance": laplacian, "luminance_mean": mean, "luminance_std": std,
                "black_fraction": float((gray < 4).mean()), "white_fraction": float((gray > 251).mean())}
    except (OSError, ValueError) as error:
        return {"status": "excluded", "reason": f"image_decode_failed: {error}"}


def image_quality(cameras: list[dict]) -> tuple[list[dict], list[dict]]:
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        records = list(pool.map(_quality, cameras))
    usable, excluded = [], []
    sharpness = [record["sharpness"] for record in records if record["status"] == "usable"]
    median = float(np.median(sharpness)) if sharpness else 0.0
    blur_floor = max(0.022, median * MIN_SHARPNESS_RATIO)
    for camera, quality in zip(cameras, records):
        if quality["status"] != "usable":
            excluded.append({"image_id": camera["image_id"], "image_name": camera["image_name"], **quality})
            continue
        if (quality["laplacian_variance"] < MIN_LAPLACIAN_VARIANCE
                or quality["sharpness"] < blur_floor):
            excluded.append({"image_id": camera["image_id"], "image_name": camera["image_name"],
                             **quality, "status": "excluded", "reason": "soft_focus_or_blur",
                             "blur_floor": blur_floor})
            continue
        exposure = 1 - 0.20 * min(1, quality["black_fraction"] + quality["white_fraction"])
        quality["blur_floor"] = blur_floor
        quality["sharpness_ratio"] = quality["sharpness"] / max(median, 1e-8)
        quality["factor"] = float(min(1, quality["sharpness_ratio"]) ** 2 * exposure)
        usable.append({**camera, "quality": quality})
    if not usable:
        raise ValueError("No usable registered acquired photographs")
    return usable, excluded


def camera_pose_quality(cameras, up_axis, floor_points=None, metric_scale=False):
    """Prefer level, modestly pitched, human-height camera poses."""
    up = np.asarray(up_axis, dtype=np.float64)
    up /= np.linalg.norm(up)
    horizontal_x = np.cross(up, [1., 0., 0.])
    if np.linalg.norm(horizontal_x) < 1e-6:
        horizontal_x = np.cross(up, [0., 0., 1.])
    horizontal_x /= np.linalg.norm(horizontal_x)
    horizontal_z = np.cross(up, horizontal_x)
    floor_tree = None
    if metric_scale and floor_points is not None and len(floor_points) >= 32:
        floor_points = np.asarray(floor_points, dtype=np.float64)
        floor_tree = cKDTree(floor_points @ np.stack((horizontal_x, horizontal_z)).T)
    accepted, excluded = [], []
    for camera in cameras:
        c2w = np.asarray(camera["camera_to_world"], dtype=np.float64)
        forward = c2w[:3, 2]
        screen_up = -c2w[:3, 1]
        level_up = up - forward * np.dot(up, forward)
        level_up /= max(np.linalg.norm(level_up), 1e-8)
        roll = float(np.degrees(np.arctan2(np.dot(np.cross(level_up, screen_up), forward),
                                           np.dot(level_up, screen_up))))
        pitch = float(np.degrees(np.arctan2(np.dot(forward, up),
                                            np.linalg.norm(forward - up * np.dot(forward, up)))))
        roll_factor = float(np.clip(1 - 0.8 * max(abs(roll) - 5, 0) / 20, 0.15, 1))
        pitch_over = max(-20 - pitch, pitch - 10, 0)
        pitch_factor = float(np.clip(1 - 0.8 * pitch_over / 25, 0.15, 1))
        details = {"roll_degrees": roll, "pitch_degrees": pitch,
                   "roll_factor": roll_factor, "pitch_factor": pitch_factor}
        reason = None
        if abs(roll) > 35:
            reason = "camera_roll_exceeds_35_degrees"
        elif pitch < -45 or pitch > 35:
            reason = "camera_pitch_exceeds_listing_range"
        elif floor_tree is not None:
            center = c2w[:3, 3]
            distances, indices = floor_tree.query(center @ np.stack((horizontal_x, horizontal_z)).T,
                                                  k=min(16, len(floor_points)), workers=-1)
            heights = np.atleast_1d(np.dot(center, up) - floor_points[np.atleast_1d(indices)] @ up)
            distances = np.atleast_1d(distances)
            plausible = (heights >= 0.35) & (heights <= 2.5) & (distances <= 0.75)
            if plausible.any():
                height = float(heights[plausible][np.argmin(np.abs(heights[plausible] - 1.45))])
                details["height_above_nearby_floor_m"] = height
                details["floor_support_distance"] = float(distances[plausible].min())
                if height < 1.20 or height > 2.25:
                    reason = "camera_height_outside_listing_range"
                details["height_factor"] = float(np.exp(-0.5 * ((height - 1.45) / 0.45) ** 2))
            else:
                details["height_status"] = "no_nearby_supported_floor"
                details["height_factor"] = 1.0
        else:
            details["height_status"] = "metric_scale_or_segmented_floor_unavailable"
        if reason:
            excluded.append({"image_id": camera["image_id"], "image_name": camera["image_name"],
                             "status": "excluded", "reason": reason, "pose": details})
            continue
        quality = camera["quality"]
        quality["pose"] = details
        quality["factor"] *= roll_factor * pitch_factor * details.get("height_factor", 1.0)
        accepted.append(camera)
    if not accepted:
        raise ValueError("No photographs pass the focus and camera-pose quality gates")
    return accepted, excluded


def sample_surface(vertices, faces, count=SAMPLE_COUNT) -> dict:
    if count < 1:
        raise ValueError("Surface sample count must be positive")
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    areas = np.zeros(len(faces), dtype=np.float64)
    for start in range(0, len(faces), 250_000):
        triangle = vertices[faces[start:start + 250_000]]
        areas[start:start + len(triangle)] = np.linalg.norm(
            np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1,
        ) * 0.5
    areas[areas <= diagonal * diagonal * 1e-14] = 0
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("No nondegenerate acquisition surface")
    rng = np.random.default_rng(0)
    face_ids = np.searchsorted(np.cumsum(areas), (np.arange(count) + rng.random(count)) * total / count)
    triangles = vertices[faces[face_ids]]
    u, v = np.sqrt(rng.random(count)), rng.random(count)
    points = (1 - u[:, None]) * triangles[:, 0] + (u * (1 - v))[:, None] * triangles[:, 1] + (u * v)[:, None] * triangles[:, 2]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return {"positions": points.astype(np.float32), "normals": normals.astype(np.float32),
            "areas": np.full(count, total / count, dtype=np.float32), "face_ids": face_ids,
            "surface_area": total, "diagonal": diagonal}


def source_labels(scan_dir, vertices, faces, samples):
    path = scan_dir / "segmentation" / "mesh_segmented.ply"
    roles = np.full(len(samples["face_ids"]), "unknown", dtype=object)
    classes = np.full(len(roles), -1, dtype=np.int32)
    metadata_path = scan_dir / "segmentation" / "segmentation.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if not path.exists():
        return roles, classes, metadata
    ply = PlyData.read(path, mmap="r", known_list_len={"face": {"vertex_indices": 3}})
    if len(ply["face"]) != len(faces):
        raise ValueError("Stage 05 face order/count differs from acquisition mesh")
    check_ids = np.unique(samples["face_ids"])[::max(1, len(samples["face_ids"]) // 512)]
    seg_vertices = np.column_stack([ply["vertex"][name] for name in ("x", "y", "z")])
    seg_faces = ply["face"]["vertex_indices"][check_ids]
    if not np.allclose(_canonical_triangles(seg_vertices[seg_faces]),
                       _canonical_triangles(vertices[faces[check_ids]]), rtol=0, atol=samples["diagonal"] * 1e-6):
        raise ValueError("Stage 05 source triangle identities do not match acquisition geometry")
    classes = np.asarray(ply["face"]["class_id"][samples["face_ids"]], dtype=np.int32)
    for class_id, item in metadata.get("classes", {}).items():
        roles[classes == int(class_id)] = item.get("structural_role", "unknown")
    return roles, classes, metadata


def layout_weights(samples, roles):
    points, normals, areas = samples["positions"], samples["normals"], samples["areas"]
    cells = np.floor((points - points.min(0)) / (samples["diagonal"] / 8)).astype(np.int32)
    axis = np.argmax(np.abs(normals), axis=1)
    orientation = axis * 2 + (np.take_along_axis(normals, axis[:, None], 1)[:, 0] >= 0)
    # Furniture needs one descriptive aspect; structural surfaces retain their sides.
    orientation[roles == "object"] = 0
    role_ids = np.unique(roles, return_inverse=True)[1]
    _, group = np.unique(np.column_stack((cells, orientation, role_ids)), axis=0, return_inverse=True)
    group_area = np.bincount(group, weights=areas)
    weights = areas / np.sqrt(group_area[group])
    floor, ceiling = roles == "floor", roles == "ceiling"
    other = ~(floor | ceiling)
    weights /= weights.sum()
    if other.any():
        floor_budget, ceiling_budget = min(float(weights[floor].sum()), 0.20), min(float(weights[ceiling].sum()), 0.05)
        for mask, budget in ((floor, floor_budget), (ceiling, ceiling_budget), (other, 1 - floor_budget - ceiling_budget)):
            if mask.any():
                weights[mask] *= budget / weights[mask].sum()
    return weights.astype(np.float32), group


def semantic_targets(scan_dir, alignment_dir, vertices, faces, samples, classes):
    segmentation = scan_dir / "segmentation"
    artifacts_dir = alignment_dir.parent / "artifacts"
    final_path = artifacts_dir / "Generate Floor Plans" / "room_category_consistency" / "associations_after_room_check.json"
    if not final_path.is_file():
        final_path = artifacts_dir / "room_category_consistency" / "associations_after_room_check.json"
    feature_ids = np.full(len(classes), -1, dtype=np.int64)
    targets, warnings = [], []
    unique_faces, inverse = np.unique(samples["face_ids"], return_inverse=True)
    triangles = vertices[faces[unique_faces]]
    tree = cKDTree(triangles.mean(1))
    seen = set()
    if final_path.exists():
        for row in json.loads(final_path.read_text()).get("associations", []):
            if row["scan_id"] != scan_dir.name or row.get("category_id") is None:
                continue
            aligned_path = row.get("floor_measurement_metadata", {}).get("aligned_mesh")
            if not aligned_path:
                continue
            try:
                relative = Path(aligned_path).resolve().relative_to((alignment_dir / "objects" / scan_dir.name).resolve())
            except ValueError:
                warnings.append(f"Instance path outside aligned scan objects: {aligned_path}")
                continue
            source = segmentation / "3d_masks" / relative
            identity = (str(source), row["category"])
            if identity in seen:
                continue
            seen.add(identity)
            if not source.exists():
                warnings.append(f"Source semantic instance missing: {source}")
                continue
            matched, _, _ = _matched_source_faces(faces[unique_faces], [source], np.eye(4), tree, triangles)
            mask = matched[inverse]
            if np.any(mask & (feature_ids >= 0)):
                raise ValueError(f"Overlapping source semantic targets: {source}")
            feature_ids[mask] = len(targets)
            targets.append({"category": row["category"].lower(), "source_class_id": row["source_class_id"],
                            "instance_id": row["instance_id"], "source_instance": str(source),
                            "sample_count": int(mask.sum()), "association_source": str(final_path)})
    else:
        warnings.append("Final macro-step-1 associations unavailable; using geometry and accepted door/window labels")
    # Macro step 1 omits doors/windows from its floor-height check. Use only
    # uniformly accepted Stage 05 class labels for these lower-priority features.
    association_path = segmentation / "non_structural_similarity" / "valid_instance_associations.json"
    if association_path.exists():
        grouped = defaultdict(list)
        for record in json.loads(association_path.read_text()).get("instances", []):
            grouped[int(record["source_class_id"])].append(record)
        for class_id, records in sorted(grouped.items()):
            accepted = [r for r in records if r.get("status") == "accepted"]
            labels = {int(r.get("instance_text_class_id") or r.get("text_class_id") or -1) for r in accepted}
            if len(accepted) != len(records) or len(labels) != 1 or not labels <= {1, 2}:
                continue
            mask = (classes == class_id) & (feature_ids < 0)
            if not mask.any():
                continue
            category = "door" if next(iter(labels)) == 1 else "window"
            feature_ids[mask] = len(targets)
            targets.append({"category": category, "source_class_id": class_id, "sample_count": int(mask.sum()),
                            "association_source": "phase05_uniform_accepted_class"})
    for record in targets:
        record["category_priority"] = LOW_PRIORITY_WEIGHT if record["category"] in LOW_PRIORITY_CATEGORIES else 1.0
    return feature_ids, targets, warnings


def _scaled_cameras(cameras, origin, long_edge):
    groups = defaultdict(list)
    for index, camera in enumerate(cameras):
        factor = long_edge / max(camera["width"], camera["height"])
        width, height = max(1, round(camera["width"] * factor)), max(1, round(camera["height"] * factor))
        calibration = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
        calibration[0] *= width / camera["width"]
        calibration[1] *= height / camera["height"]
        view = np.asarray(camera["world_to_camera"], dtype=np.float64).copy()
        view[:3, 3] += view[:3, :3] @ origin
        groups[(width, height)].append((index, view.astype(np.float32), calibration))
    return groups


def framing_quality(pixels, geometry, roles, views, diagonal):
    """Prefer explanatory views over floor/ceiling shots and foreground obstruction."""
    device = pixels.device
    total = pixels.sum(1).clamp_min(1e-8)
    floor = torch.as_tensor(roles == "floor", device=device)
    ceiling = torch.as_tensor(roles == "ceiling", device=device)
    wall = torch.as_tensor(roles == "wall", device=device)
    objects = torch.as_tensor(roles == "object", device=device)
    unknown = torch.as_tensor(roles == "unknown", device=device)
    clutter = torch.as_tensor(np.isin(roles, ["object", "unknown"]), device=device)
    floor_fraction = pixels[:, floor].sum(1) / total
    ceiling_fraction = pixels[:, ceiling].sum(1) / total
    wall_fraction = pixels[:, wall].sum(1) / total
    object_fraction = pixels[:, objects].sum(1) / total
    unknown_fraction = pixels[:, unknown].sum(1) / total
    nearby = (geometry["camera_depth"] < diagonal * 0.12) & clutter[None]
    near_fraction = (pixels * nearby).sum(1) / total
    rotation = torch.as_tensor(views, device=device)[:, :3, :3]
    pitch = torch.rad2deg(torch.asin(rotation[:, 2, 1].clamp(-1, 1)))
    dominant_fraction = torch.stack((floor_fraction, ceiling_fraction, wall_fraction,
                                     object_fraction, unknown_fraction)).amax(0)
    known_dominant_fraction = torch.stack((floor_fraction, ceiling_fraction, wall_fraction,
                                           object_fraction)).amax(0)
    context_factor = (1 - 0.75 * ((dominant_fraction - 0.65) / 0.35).clamp(0, 1)).clamp_min(0.25)
    factors = torch.stack((1 - 0.85 * ((floor_fraction - 0.45) / 0.45).clamp(0, 1),
                           1 - 0.85 * ((ceiling_fraction - 0.35) / 0.50).clamp(0, 1),
                           context_factor))
    excluded = ((floor_fraction > 0.90) | (ceiling_fraction > 0.85) | (near_fraction > 0.40)
                | (known_dominant_fraction > MAX_SINGLE_SURFACE_FRACTION))
    return factors.amin(0), {"floor_pixel_fraction": floor_fraction, "ceiling_pixel_fraction": ceiling_fraction,
                            "wall_pixel_fraction": wall_fraction, "object_pixel_fraction": object_fraction,
                            "unknown_pixel_fraction": unknown_fraction,
                            "dominant_surface_fraction": dominant_fraction, "context_factor": context_factor,
                            "known_dominant_surface_fraction": known_dominant_fraction,
                            "near_foreground_fraction": near_fraction, "pitch_degrees": pitch,
                            "excluded_for_unhelpful_framing": excluded}


@torch.inference_mode()
def measure_views(renderer, cameras, samples, origin, feature_ids, targets, roles, batch_size=DEFAULT_BATCH_SIZE, long_edge=RENDER_LONG_EDGE):
    count, feature_count = len(cameras), len(targets)
    device = renderer.device
    visibility = torch.empty((count, len(feature_ids)), device=device)
    support = torch.zeros((count, feature_count), device=device)
    footprint = torch.zeros_like(support)
    completeness = torch.zeros_like(support)
    ids = torch.as_tensor(feature_ids, device=device)
    valid = ids >= 0
    areas = torch.as_tensor(samples["areas"], device=device)
    effective_batches = []
    for (width, height), group in _scaled_cameras(cameras, origin, long_edge).items():
        views, intrinsics = np.stack([v for _, v, _ in group]), np.stack([k for _, _, k in group])
        start, current_batch = 0, batch_size
        while start < len(group):
            stop = min(len(group), start + current_batch)
            try:
                scores, pixels, geometry = renderer.evaluate(views[start:stop], intrinsics[start:stop], width, height)
                framing, framing_stats = framing_quality(pixels, geometry, roles, views[start:stop], samples["diagonal"])
                indices = torch.tensor([group[j][0] for j in range(start, stop)], device=device)
                visibility[indices] = scores
                if feature_count and bool(valid.any()):
                    labels = ids[valid].unsqueeze(0).expand(stop - start, -1)
                    support[indices] = torch.zeros((stop - start, feature_count), device=device).scatter_add_(
                        1, labels, scores[:, valid] * areas[valid])
                    footprint[indices] = torch.zeros((stop - start, feature_count), device=device).scatter_add_(
                        1, labels, pixels[:, valid] / (width * height))
                    in_frame_area = torch.zeros((stop - start, feature_count), device=device).scatter_add_(
                        1, labels, geometry["in_frame"][:, valid].float() * areas[valid])
                    total_area = torch.zeros(feature_count, device=device).scatter_add_(0, ids[valid], areas[valid])
                    completeness[indices] = in_frame_area / total_area.clamp_min(1e-12)
                for local, index in enumerate(indices.cpu().tolist()):
                    quality = cameras[index]["quality"]
                    quality["photometric_factor"] = quality["factor"]
                    quality["framing_factor"] = float(framing[local])
                    quality["factor"] *= quality["framing_factor"]
                    quality["framing"] = {name: float(values[local]) for name, values in framing_stats.items()}
                    if bool(framing_stats["excluded_for_unhelpful_framing"][local]):
                        quality["status"] = "excluded"
                        quality["reason"] = "single_surface_or_floor_ceiling_or_near_foreground_dominated"
                effective_batches.append(stop - start)
                del scores, pixels, geometry
                start = stop
            except torch.OutOfMemoryError:
                if current_batch == 1:
                    raise
                torch.cuda.empty_cache()
                current_batch = max(1, current_batch // 2)
                LOGGER.warning("Reducing visibility batch to %d after GPU out-of-memory", current_batch)
    valid_candidates = torch.as_tensor([c["quality"].get("status") != "excluded" for c in cameras], device=device)
    support[~valid_candidates] = 0
    footprint[~valid_candidates] = 0
    completeness[~valid_candidates] = 0
    quality = torch.tensor([c["quality"]["factor"] for c in cameras], device=device)
    visibility *= quality[:, None]
    if feature_count:
        best_support = support.amax(0)
        support = (support / (0.60 * best_support).clamp_min(1e-12)).clamp(0, 1)
        support *= (footprint / 0.01).clamp(0, 1) * (completeness / 0.90).clamp(0, 1).square() * quality[:, None]
    return visibility, support, {"backend": "gpu_projection_cpu_bvh" if device.type == "cuda" else "cpu_open3d",
                                 "projection_check": renderer.projection_check,
                                 "cpu_raycast_seconds": renderer.cpu_raycast_seconds,
                                 "ray_count": renderer.ray_count, "batch_sizes": sorted(set(effective_batches)),
                                 "scoring_long_edge": long_edge}


@torch.inference_mode()
def utility_matrix(visibility, features, weights, targets, regions):
    weight = torch.as_tensor(weights, device=visibility.device, dtype=visibility.dtype)
    region = torch.as_tensor(regions, device=visibility.device, dtype=torch.long)
    support = torch.zeros((len(visibility), int(region.max()) + 1), device=visibility.device, dtype=visibility.dtype).scatter_add_(
        1, region[None].expand(len(visibility), -1), visibility * weight)
    best_support = support.amax(0)
    denominator = best_support.sum()
    if float(denominator) <= 1e-8:
        raise ValueError("No meaningful visible layout support; check acquisition geometry and poses")
    # A useful view of a region is enough; its unseen back sides need no extra photo.
    layout = (support / (0.60 * best_support).clamp_min(1e-12)).clamp(0, 1) * best_support / denominator
    feature_weight = np.zeros(len(targets), dtype=np.float32)
    maxima = features.amax(0).cpu().numpy() if len(targets) else np.zeros(0)
    available = [j for j, value in enumerate(maxima) if value >= 0.5]
    counts = Counter(targets[j]["category"] for j in available)
    for j, target in enumerate(targets):
        target["best_adequacy"] = float(maxima[j])
        target["attainable"] = j in available
        if j in available:
            feature_weight[j] = target["category_priority"] / counts[target["category"]]
    if not feature_weight.any():
        return layout, layout, {"layout_share": 1.0, "feature_share": 0.0}
    feature_weight = torch.as_tensor(feature_weight, device=visibility.device)
    semantic = features * feature_weight / (features.amax(0) * feature_weight).sum()
    return torch.cat((0.75 * layout, 0.25 * semantic), 1), layout, {"layout_share": 0.75, "feature_share": 0.25}


def object_visible_candidates(cameras, features, targets):
    required = [index for index, item in enumerate(targets)
                if item["category"].lower() not in {"window", "door"}]
    if not required:
        return list(range(len(cameras))), []
    object_support = features[:, required]
    visible = object_support >= MIN_RECOGNIZED_OBJECT_ADEQUACY
    categories = [targets[index]["category"].lower() for index in required]
    rows = visible.cpu().tolist()
    adequacies = object_support.amax(1).cpu().tolist()
    candidates = []
    for index, (camera, row) in enumerate(zip(cameras, rows)):
        shown = sorted({category for category, is_visible in zip(categories, row) if is_visible})
        camera["quality"]["visible_recognized_categories"] = shown
        camera["quality"]["recognized_object_adequacy"] = adequacies[index]
        if shown:
            candidates.append(index)
    if not candidates:
        raise ValueError("No acquired photograph adequately shows a recognized categorized object")
    return candidates, sorted(set(categories))


@torch.inference_mode()
def select_views(matrix, layout, image_ids=None, target=UTILITY_TARGET, relative_gain=MIN_RELATIVE_GAIN,
                 visible_support=None):
    """Greedy coverage with no photo cap; suppress supplements redundant with selected views."""
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] == 0:
        raise ValueError("Selection needs a nonempty utility matrix")
    if (not bool(torch.isfinite(matrix).all()) or bool((matrix < 0).any())
            or not 0 < target <= 1 or not 0 < relative_gain <= 1):
        raise ValueError("Selection utilities and thresholds must be finite and positive")
    image_ids = list(range(len(matrix))) if image_ids is None else image_ids
    if len(image_ids) != len(matrix) or len(layout) != len(matrix):
        raise ValueError("Camera identities and layout scores must match utility rows")
    if visible_support is not None:
        if visible_support.ndim != 2 or visible_support.shape[0] != len(matrix) or visible_support.shape[1] == 0:
            raise ValueError("Visible-surface support must match the candidate rows and contain samples")
        visible_support = visible_support.to(device=matrix.device, dtype=torch.bool)
        support_counts = visible_support.sum(1).clamp_min(1)
    layout_matrix = layout if layout.ndim == 2 else None
    layout_scores = layout.sum(1) if layout_matrix is not None else layout
    eligible = matrix.sum(1) > 1e-8
    if not bool(eligible.any()):
        raise ValueError("No photographs pass the listing quality and framing filters")

    def best(scores):
        maximum = float(scores.max())
        tied = torch.where(scores >= maximum - 1e-7)[0].cpu().tolist()
        return min(tied, key=lambda index: image_ids[index])
    overview_scores = layout_scores.clone()
    overview_scores[~eligible] = -1
    overview = best(overview_scores)
    selected = [overview]
    covered = matrix[overview].clone()
    threshold = relative_gain * float(matrix.sum(1).max())
    layout_threshold = MIN_LAYOUT_GAIN * float(layout_scores.max())
    covered_layout = layout_matrix[overview].clone() if layout_matrix is not None else None
    if threshold <= 1e-8:
        raise ValueError("No useful acquired view")
    reason = "candidates_exhausted"
    overlap_blocked = torch.zeros(len(matrix), dtype=torch.bool, device=matrix.device)
    while len(selected) < len(matrix):
        if float(covered.sum()) >= target:
            reason = "weighted_utility_target"
            break
        gains = (matrix - covered).clamp_min(0).sum(1)
        gains[selected] = -1
        gains[~eligible] = -1
        if layout_matrix is not None:
            layout_gains = (layout_matrix - covered_layout).clamp_min(0).sum(1)
            gains[layout_gains < layout_threshold] = -1
        if visible_support is not None:
            selected_mask = torch.zeros(len(matrix), dtype=torch.bool, device=matrix.device)
            selected_mask[selected] = True
            for prior in selected:
                shared = (visible_support & visible_support[prior]).sum(1)
                overlap = shared / torch.minimum(support_counts, support_counts[prior])
                blocked = ((overlap >= MAX_VISIBLE_SURFACE_OVERLAP) & eligible & ~selected_mask
                           & (gains >= 0))
                overlap_blocked |= blocked
                gains[blocked] = -1
        index = best(gains)
        if float(gains[index]) < threshold:
            reason = ("excessive_visible_surface_overlap" if overlap_blocked.any()
                      else "insufficient_new_layout_context") if float(gains.max()) < 0 else "low_marginal_gain"
            break
        selected.append(index)
        covered = torch.maximum(covered, matrix[index])
        if layout_matrix is not None:
            covered_layout = torch.maximum(covered_layout, layout_matrix[index])
    for index in selected[:0:-1]:
        remaining = [j for j in selected if j != index]
        without = matrix[remaining].amax(0)
        if float(covered.sum() - without.sum()) < threshold:
            selected.remove(index)
            covered = without
    contributions = {}
    for index in selected:
        remaining = [j for j in selected if j != index]
        contributions[index] = float(covered.sum() - (matrix[remaining].amax(0).sum() if remaining else 0))
    supplements = sorted(selected[1:], key=lambda j: (-contributions[j], image_ids[j]))
    pair_overlap = 0.0
    if visible_support is not None:
        for index, left in enumerate(selected):
            for right in selected[index + 1:]:
                shared = float((visible_support[left] & visible_support[right]).sum())
                denominator = float(torch.minimum(support_counts[left], support_counts[right]))
                pair_overlap = max(pair_overlap, shared / denominator)
    return [overview, *supplements], {"stopping_reason": reason, "weighted_utility": float(covered.sum()),
                                     "marginal_gain_threshold": threshold, "relative_gain": relative_gain,
                                     "minimum_layout_gain": layout_threshold,
                                     "minimum_layout_gain_ratio": MIN_LAYOUT_GAIN,
                                     "utility_target": target,
                                     "maximum_visible_surface_overlap": (MAX_VISIBLE_SURFACE_OVERLAP
                                                                          if visible_support is not None else None),
                                     "maximum_selected_visible_surface_overlap": (pair_overlap
                                                                                  if visible_support is not None else None),
                                     "overlap_blocked_candidates": int(overlap_blocked.sum()),
                                     "unique_contributions": contributions}


def _contact_sheet(cameras, path):
    columns = min(4, len(cameras))
    sheet = Image.new("RGB", (columns * 256, math.ceil(len(cameras) / columns) * 344), "white")
    draw = ImageDraw.Draw(sheet)
    for index, camera in enumerate(cameras):
        x, y = index % columns * 256, index // columns * 344
        with Image.open(camera["source_image"]) as source:
            image = source.convert("RGB")
            image.thumbnail((248, 306))
            sheet.paste(image, (x + (256 - image.width) // 2, y + 4))
        draw.text((x + 5, y + 313), f"{index + 1}: {camera['image_name']}", fill="black")
    sheet.save(path, quality=88)


def _camera_plot(cameras, selected, points, path):
    canvas = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(canvas)
    centers = np.array([np.asarray(c["camera_to_world"])[:3, 3] for c in cameras])
    cloud = np.concatenate((points[:, [0, 2]], centers[:, [0, 2]]))
    low, high = cloud.min(0), cloud.max(0)
    scale = 780 / max(float((high - low).max()), 1e-6)
    def pixel(point):
        xy = (np.asarray(point)[[0, 2]] - low) * scale
        return float(60 + xy[0]), float(840 - xy[1])
    for point in points[::max(1, len(points) // 25_000)]:
        draw.point(pixel(point), fill="#dddddd")
    for center in centers:
        x, y = pixel(center)
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#888888")
    for rank, index in enumerate(selected, 1):
        pose = np.asarray(cameras[index]["camera_to_world"])
        center = pose[:3, 3]
        direction = pose[:3, 2]
        end = center + direction * (35 / scale)
        start_px, end_px = pixel(center), pixel(end)
        draw.line((start_px, end_px), fill="#d02b21", width=3)
        x, y = start_px
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#d02b21")
        draw.text((x + 6, y + 2), str(rank), fill="black")
    draw.text((30, 20), "Phase 07 aligned world: X right, Z up on plot; +Y is scene up", fill="black")
    draw.text((30, 42), f"X [{low[0]:.3f}, {high[0]:.3f}], Z [{low[1]:.3f}, {high[1]:.3f}] (scene units)", fill="black")
    canvas.save(path)


def _publish(staged: Path, target: Path):
    previous = target.with_name(f".{target.name}.previous-{uuid.uuid4().hex}")
    if target.exists():
        marker = target / "selected_images.json"
        if target.is_symlink() or not marker.is_file():
            raise ValueError(f"Refusing to replace an unrelated output directory: {target}")
        if json.loads(marker.read_text()).get("macro_step") != "extract_best_views":
            raise ValueError(f"Refusing to replace an unrelated gallery: {target}")
        target.rename(previous)
    try:
        staged.rename(target)
    except BaseException:
        if previous.exists():
            previous.rename(target)
        raise
    if previous.exists():
        shutil.rmtree(previous)


def extract_scan(scan_dir, alignment_dir, output_dir, transform, device="cuda", batch_size=DEFAULT_BATCH_SIZE,
                 sample_count=SAMPLE_COUNT, long_edge=RENDER_LONG_EDGE):
    started = time.perf_counter()
    timings = {}
    transform = _rigid(transform)
    cameras, lineage = load_cameras(scan_dir, alignment_dir, transform)
    mesh_path = scan_dir / ".raw_unaligned" / "mesh.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Acquisition mesh missing: {mesh_path}; Phase 07 must preserve its source mesh")
    vertices, faces, _ = _read_mesh(mesh_path)
    timings["load_inputs"] = time.perf_counter() - started
    tick = time.perf_counter()
    samples = sample_surface(vertices, faces, sample_count)
    roles, classes, _ = source_labels(scan_dir, vertices, faces, samples)
    weights, regions = layout_weights(samples, roles)
    feature_ids, targets, warnings = semantic_targets(scan_dir, alignment_dir, vertices, faces, samples, classes)
    samples["positions"] = samples["positions"].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    samples["normals"] = (samples["normals"] @ transform[:3, :3].T).astype(np.float32)
    vertices = vertices.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    origin = (vertices.min(0) + vertices.max(0)) * 0.5
    timings["relevance_and_semantics"] = time.perf_counter() - tick
    tick = time.perf_counter()
    cameras, excluded = image_quality(cameras)
    timings["image_quality"] = time.perf_counter() - tick
    up_axis = transform[:3, :3] @ np.array([0., 1., 0.])
    floor_points = samples["positions"][roles == "floor"].astype(np.float64)
    metric_path = scan_dir / "metric_scale.json"
    metric_scale = (metric_path.is_file()
                    and json.loads(metric_path.read_text()).get("output_coordinate_units") == "meters")
    cameras, pose_excluded = camera_pose_quality(cameras, up_axis, floor_points, metric_scale)
    excluded.extend(pose_excluded)
    tick = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    renderer = SurfaceVisibility((vertices - origin).astype(np.float32), faces,
                                 samples["positions"] - origin, samples["normals"], samples["areas"], device)
    visibility, features, rendering = measure_views(renderer, cameras, samples, origin, feature_ids, targets, roles,
                                                     batch_size, long_edge)
    if renderer.device.type == "cuda":
        torch.cuda.synchronize()
    timings["visibility"] = time.perf_counter() - tick
    framing_excluded = [{"image_id": camera["image_id"], "image_name": camera["image_name"],
                         "status": "excluded", "reason": camera["quality"]["reason"],
                         "quality": camera["quality"]}
                        for camera in cameras if camera["quality"]["status"] == "excluded"]
    excluded.extend(framing_excluded)
    keep = [index for index, camera in enumerate(cameras) if camera["quality"]["status"] != "excluded"]
    cameras = [cameras[index] for index in keep]
    visibility, features = visibility[keep], features[keep]
    if not cameras:
        raise ValueError("No photographs pass the listing framing filters")
    tick = time.perf_counter()
    matrix, layout, utility = utility_matrix(visibility, features, weights, targets, regions)
    candidate_indices, required_categories = object_visible_candidates(cameras, features, targets)
    candidate_rows = torch.as_tensor(candidate_indices, device=visibility.device)
    selected_local, selection = select_views(
        matrix[candidate_rows], layout[candidate_rows], [cameras[index]["image_id"] for index in candidate_indices],
        visible_support=visibility[candidate_rows],
    )
    selected = [candidate_indices[index] for index in selected_local]
    selection["unique_contributions"] = {
        candidate_indices[index]: value for index, value in selection["unique_contributions"].items()
    }
    selected_cameras = [cameras[j] for j in selected]
    surface_coverage = float((visibility[selected].amax(0) * torch.as_tensor(weights, device=visibility.device)).sum()
                            / (visibility.amax(0) * torch.as_tensor(weights, device=visibility.device)).sum())
    layout_coverage = float(matrix[selected, :int(regions.max()) + 1].amax(0).sum()) / utility["layout_share"]
    for j, record in enumerate(targets):
        record["selected_adequacy"] = float(features[selected, j].max())
    candidate_scores = matrix.sum(1).cpu().tolist()
    timings["selection"] = time.perf_counter() - tick
    tick = time.perf_counter()
    target = output_dir / scan_dir.name
    staged = Path(tempfile.mkdtemp(prefix=f".{scan_dir.name}.", dir=output_dir))
    try:
        records = []
        for rank, index in enumerate(selected, 1):
            camera = cameras[index]
            destination = staged / "images" / camera["image_name"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(camera["source_image"], destination)
            digest = _hash(Path(camera["source_image"]))
            if _hash(destination) != digest:
                raise OSError(f"Copied image hash differs: {camera['image_name']}")
            records.append({**camera, "gallery_rank": rank, "image": f"images/{camera['image_name']}",
                            "sha256": digest, "unique_utility": selection["unique_contributions"][index]})
        _json(staged / "selected_images.json", {
            "schema_version": 1, "macro_step": "extract_best_views", "scan_id": scan_dir.name,
            "coordinate_frame": "phase07_aligned_world", "world_units": "phase07_scene_units",
            "camera_axes": "COLMAP: right, down, forward", "quaternion_order": "qw,qx,qy,qz",
            "scan_transform_matrix": transform.tolist(), "pose_lineage": lineage,
            "acquisition_mesh": str(mesh_path), "acquisition_mesh_sha256": _hash(mesh_path), "images": records,
        })
        _contact_sheet(selected_cameras, staged / "contact_sheet.jpg")
        _camera_plot(cameras, selected, samples["positions"], staged / "camera_coverage.png")
        timings["export_and_diagnostics"] = time.perf_counter() - tick
        report = {
            "schema_version": 1, "macro_step": "extract_best_views", "scan_id": scan_dir.name,
            "status": "complete", "selected_count": len(selected), "candidate_count": len(cameras),
            "excluded_images": excluded, "warnings": warnings, "sample_count": sample_count,
            "layout_coverage": layout_coverage, "surface_weighted_coverage": surface_coverage, "utility_mix": utility,
            "selection": {key: value for key, value in selection.items() if key != "unique_contributions"},
            "object_visibility_constraint": {
                "enforced": bool(required_categories), "minimum_adequacy": MIN_RECOGNIZED_OBJECT_ADEQUACY,
                "required_categories": required_categories, "eligible_candidate_count": len(candidate_indices),
                "selected_visible_categories": [cameras[index]["quality"].get("visible_recognized_categories", [])
                                                for index in selected],
            },
            "semantic_targets": targets, "lower_priority_categories": sorted(LOW_PRIORITY_CATEGORIES),
            "lower_priority_weight": LOW_PRIORITY_WEIGHT, "rendering": rendering,
            "candidates": [{"image_id": c["image_id"], "image_name": c["image_name"], "quality": c["quality"],
                            "single_view_utility": candidate_scores[j], "selected": j in selected}
                           for j, c in enumerate(cameras)],
            "timings_seconds": timings, "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        }
        _json(staged / "selection_report.json", report)
        _publish(staged, target)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return {"scan_id": scan_dir.name, "status": "complete", "selected_count": len(selected),
            "candidate_count": len(cameras), "weighted_utility": selection["weighted_utility"],
            "stopping_reason": selection["stopping_reason"], "elapsed_seconds": time.perf_counter() - started,
            "selected_images": str(target / "selected_images.json"), "report": str(target / "selection_report.json")}


def extract_best_views(alignment_dir=DEFAULT_ALIGNMENT_DIR, output_dir=None, device="cuda", batch_size=DEFAULT_BATCH_SIZE):
    alignment_dir = Path(alignment_dir).resolve()
    scene_dir = alignment_dir.parent
    output_dir = Path(output_dir or scene_dir / "artifacts" / "Extract Best Views").resolve()
    if (scene_dir.is_relative_to(output_dir) or output_dir.is_relative_to(alignment_dir)
            or any(output_dir.is_relative_to(p) for p in scene_dir.glob("scan_*"))):
        raise ValueError("Output must be separate from scan inputs and Phase 07 assets")
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise ValueError("Use device cpu/cuda and a positive batch size")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; use --device cpu for the explicit Open3D reference backend")
    path = alignment_dir / "stats" / "aligned_scans_transforms.json"
    transforms = json.loads(path.read_text())["scans"]
    if not transforms:
        raise ValueError("Phase 07 transform manifest contains no scans")
    for scan_id in transforms:
        if Path(scan_id).name != scan_id or not scan_id.startswith("scan_"):
            raise ValueError(f"Invalid scan identity: {scan_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    LOGGER.info("Starting extract_best_views for %d scans on %s", len(transforms), device)
    results = []
    for scan_id, record in sorted(transforms.items()):
        LOGGER.info("Starting extract_best_views for %s", scan_id)
        try:
            result = extract_scan(scene_dir / scan_id, alignment_dir, output_dir,
                                  record["transform_matrix"], device, batch_size)
            LOGGER.info("%s: %d/%d photos; utility %.3f; %.2f s", scan_id, result["selected_count"],
                        result["candidate_count"], result["weighted_utility"], result["elapsed_seconds"])
        except Exception as error:
            LOGGER.exception("extract_best_views failed for %s", scan_id)
            result = {"scan_id": scan_id, "status": "failed", "error": str(error),
                      "previous_output_preserved": (output_dir / scan_id / "selected_images.json").is_file()}
        results.append(result)
    elapsed = time.perf_counter() - started
    manifest = {"schema_version": 1, "macro_step": "extract_best_views", "phase_status": "in_progress",
                "macro_step_status": "complete" if all(r["status"] == "complete" for r in results) else "failed",
                "alignment_dir": str(alignment_dir), "source_transforms_sha256": _hash(path),
                "device": device, "elapsed_seconds": elapsed, "scans": results}
    temporary = output_dir / ".extract_best_views_manifest.json.tmp"
    _json(temporary, manifest)
    temporary.replace(output_dir / "extract_best_views_manifest.json")
    completed_scans = sum(result["status"] == "complete" for result in results)
    LOGGER.info("Finished extract_best_views with status %s in %.2f s (%d/%d scans complete)",
                manifest["macro_step_status"], time.perf_counter() - started,
                completed_scans, len(results))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-dir", type=Path, default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cuda also selects the ROCm GPU")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = extract_best_views(args.alignment_dir, args.output_dir, args.device, args.batch_size)
    return 0 if result["macro_step_status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
