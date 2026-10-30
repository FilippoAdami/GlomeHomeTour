#!/usr/bin/env python3
"""Refine and axis-align every scene described by ingestion results."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import jsonschema

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap, subprocess_env

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.metric_scale import estimate_metric_scale
from colmap_diagnostics import parse_cameras_txt, parse_points3D_txt, parse_images_txt
from densify_pointcloud import remove_outliers, voxel_dedup
from export_keyframes import prune_model as prune_model_colmap
from scene.colmap_loader import qvec2rotmat, rotmat2qvec
from scene.dataset_readers import storePly
from Utilities.scene_alignment import horizontal_align_deg

RESULT_SCHEMA = _backend_dir.parent / "mobile_app" / "shared" / "schemas" / "ingestion_results.schema.json"
TRANSFORMS_SCHEMA = _backend_dir.parent / "mobile_app" / "shared" / "schemas" / "transforms.schema.json"
RESULT_FILENAME = "ingestion_results.json"
VOXEL_SIZE = 0.01
OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO = 20, 2.0
MIN_FEATURE_PRUNE = 30
MIN_SIFT_FEATURES = 200
LEGACY_OUTPUTS = (
    "colmap_model",
    "01_poses_refinment", "colmap_discarded_images", "colmap_non_keyframes",
    "colmap_diagnostics", "sparse_manual", "sparse_triangulated", "sparse_held",
    "images_2", "images_4", "images_8", "images_all", "images_keyframes",
    "images_zoomgroups", "distorted", "camera_path.obj", "transforms_keyframes.json",
    "scene_extent.json", "scene_size.txt", "scene_extent_side.png", "scene_extent_topdown.png",
)


def discover_scenes(workspace: Path) -> list[tuple[Path, Path]]:
    """Return (scene directory, result manifest) pairs from the ingestion layout."""
    root_result = workspace / RESULT_FILENAME
    scan_dirs = sorted((p for p in workspace.iterdir()
                        if p.is_dir() and p.name.startswith("scan_")
                        and p.name.removeprefix("scan_").isdigit()),
                       key=lambda p: int(p.name.removeprefix("scan_"))) if workspace.exists() else []
    scan_results = [(scan, scan / RESULT_FILENAME) for scan in scan_dirs
                    if (scan / RESULT_FILENAME).is_file()]
    if scan_dirs:
        if root_result.exists() or len(scan_results) != len(scan_dirs):
            raise ValueError("Expected either one root ingestion_results.json or one in every scan_N folder")
        return scan_results
    if root_result.is_file():
        return [(workspace, root_result)]
    raise FileNotFoundError(f"No {RESULT_FILENAME} found in {workspace} or its scan_N folders")


def resolve_result_path(scene_dir: Path, value: str | None, field: str,
                        *, required: bool = False) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"ingestion result is missing {field}")
        return None
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError(f"{field} must be a relative path within its scan folder")
    candidate = (scene_dir / value).resolve()
    if not candidate.is_relative_to(scene_dir.resolve()):
        raise ValueError(f"{field} escapes its scan folder: {value}")
    if not candidate.exists():
        raise FileNotFoundError(f"{field} does not exist: {candidate}")
    if field == "images_path" and not candidate.is_dir():
        raise ValueError(f"{field} must point to a directory: {candidate}")
    if field == "transforms_path" and not candidate.is_file():
        raise ValueError(f"{field} must point to a file: {candidate}")
    if field == "colmap_model_path" and not candidate.is_dir():
        raise ValueError(f"{field} must point to a model directory: {candidate}")
    return candidate


def load_ingestion_result(result_path: Path) -> tuple[dict, Path, Path | None]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    try:
        schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(result, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Invalid ingestion result {result_path}: {exc.message}") from exc
    if result.get("schema_version") != "1.0.0" or result.get("status") != "ready":
        raise ValueError(f"Unsupported or incomplete ingestion result: {result_path}")
    if result.get("source_format") not in {"arcore", "arcore_app_certified", "colmap"}:
        raise ValueError(f"Unsupported source_format in {result_path}: {result.get('source_format')!r}")
    if result.get("poses_format") not in {"arcore", "colmap"}:
        raise ValueError(f"Unsupported poses_format in {result_path}: {result.get('poses_format')!r}")
    if result["source_format"] == "colmap" and result["poses_format"] != "colmap":
        raise ValueError(f"COLMAP source must retain COLMAP poses in {result_path}")
    scene_dir = result_path.parent
    images_dir = resolve_result_path(scene_dir, result.get("images_path"), "images_path", required=True)
    transforms_path = resolve_result_path(scene_dir, result.get("transforms_path"), "transforms_path")
    model_path = resolve_result_path(scene_dir, result.get("colmap_model_path"), "colmap_model_path")
    if result["source_format"] in {"arcore", "arcore_app_certified"} and transforms_path is None:
        raise ValueError(f"ARCore-origin result has no transforms file: {result_path}")
    if result["poses_format"] == "colmap" and model_path is None:
        raise ValueError(f"COLMAP ingestion result has no model: {result_path}")
    result["_scene_dir"] = scene_dir
    result["_images_dir"] = images_dir
    result["_transforms_path"] = transforms_path
    result["_model_path"] = model_path
    return result, images_dir, transforms_path


def count_registered_features(sparse_dir: Path) -> dict[str, int]:
    """Count triangulated 3D observations per image in a COLMAP text model."""
    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        return {}
    counts = {}
    lines = [line for line in images_txt.read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split(maxsplit=9)
        i += 1
        if len(parts) != 10:
            continue
        name = Path(parts[9]).name
        obs = lines[i].split() if i < len(lines) else []
        i += 1
        counts[name] = sum(int(obs[k]) != -1 for k in range(2, len(obs), 3))
    return counts


def scale_colmap_model(sparse_dir: Path, scale: float) -> None:
    """Convert COLMAP point positions and camera translations into metric units."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Metric scale must be finite and positive, got {scale}")

    points_path = sparse_dir / "points3D.txt"
    points = []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            points.append(line)
            continue
        fields = line.split()
        fields[1:4] = [f"{float(value) * scale:.12g}" for value in fields[1:4]]
        points.append(" ".join(fields))
    points_path.write_text("\n".join(points) + "\n", encoding="utf-8")

    images_path = sparse_dir / "images.txt"
    lines = images_path.read_text(encoding="utf-8").splitlines()
    scaled, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            scaled.append(line)
            i += 1
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10 or i + 1 >= len(lines):
            raise ValueError(f"Malformed COLMAP images.txt record: {line}")
        fields[5:8] = [f"{float(value) * scale:.12g}" for value in fields[5:8]]
        scaled.extend((" ".join(fields), lines[i + 1]))
        i += 2
    images_path.write_text("\n".join(scaled) + "\n", encoding="utf-8")


def _run_colmap(command: list[str]) -> None:
    env = subprocess_env()
    env["QT_QPA_PLATFORM"] = "xcb"
    subprocess.run(command, check=True, env=env)


def discard_low_sift_images(database: Path) -> set[str]:
    """Remove images with too few extracted SIFT keypoints before matching."""
    if not database.exists():
        return set()
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "images" not in tables:
            return set()
        weak = connection.execute(
            """SELECT images.image_id, images.name
               FROM images LEFT JOIN keypoints USING (image_id)
               WHERE COALESCE(keypoints.rows, 0) < ?""",
            (MIN_SIFT_FEATURES,),
        ).fetchall()
        for image_id, _ in weak:
            connection.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
            connection.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
    return {Path(name).name for _, name in weak}


def run_image_only_colmap(images_dir: Path, files: list[Path],
                          work_dir: Path, matcher: str = "spatial",
                          frame_positions: dict[str, np.ndarray] | None = None) -> tuple[Path, set[str]]:
    """Build image-only COLMAP components using spatial, exhaustive, or sequential matching."""
    database = work_dir / "colmap_database.db"
    image_list = work_dir / "image_list.txt"
    image_list.write_text("\n".join(str(path.relative_to(images_dir)) for path in files) + "\n",
                          encoding="utf-8")
    # 2DGS supports pinhole cameras; ingestion supplies undistorted images.
    _run_colmap([
        "colmap", "feature_extractor", "--database_path", str(database),
        "--image_path", str(images_dir), "--image_list_path", str(image_list),
        "--ImageReader.camera_model", "SIMPLE_PINHOLE",
        "--ImageReader.single_camera", "1", "--FeatureExtraction.use_gpu", "1",
    ])
    low_sift = discard_low_sift_images(database)
    if len(low_sift) == len(files):
        raise RuntimeError(f"SIFT prefilter rejected every image in {images_dir}")

    if matcher == "spatial" and frame_positions:
        # Spatial Candidate Pairing:
        # 1. Coarse proximity lookup using ARCore translations as candidate pair filter (R <= 3.5m)
        sigma_m = 0.05
        covariance = np.diag([sigma_m ** 2] * 3)
        gravity = np.full(3, np.nan)
        with sqlite3.connect(database) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pose_priors (
                    corr_data_id INTEGER PRIMARY KEY,
                    corr_sensor_id INTEGER NOT NULL,
                    corr_sensor_type INTEGER NOT NULL,
                    position BLOB NOT NULL,
                    position_covariance BLOB,
                    gravity BLOB,
                    coordinate_system INTEGER NOT NULL
                )
            """)
            name_to_ids = {name: (image_id, camera_id) for image_id, camera_id, name
                           in conn.execute("SELECT image_id, camera_id, name FROM images")}
            rows = []
            for name, pos in frame_positions.items():
                if name in name_to_ids:
                    img_id, cam_id = name_to_ids[name]
                    rows.append((img_id, cam_id, 0,
                                 pos.astype("<f8").tobytes(),
                                 covariance.astype("<f8").tobytes(),
                                 gravity.astype("<f8").tobytes(),
                                 1))
            conn.execute("DELETE FROM pose_priors")
            conn.executemany("""
                INSERT INTO pose_priors (corr_data_id, corr_sensor_id, corr_sensor_type,
                                         position, position_covariance, gravity,
                                         coordinate_system)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

        # Step A: Sequential matching along capture timeline (overlap 20)
        _run_colmap([
            "colmap", "sequential_matcher", "--database_path", str(database),
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.guided_matching", "1",
            "--SequentialMatching.overlap", "20",
            "--SequentialMatching.loop_detection", "0"
        ])

        # Step B: Spatial proximity candidate matching (3.5m radius, up to 80 neighbors)
        _run_colmap([
            "colmap", "spatial_matcher", "--database_path", str(database),
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.guided_matching", "1",
            "--SpatialMatching.max_distance", "3.5",
            "--SpatialMatching.ignore_z", "0",
            "--SpatialMatching.max_num_neighbors", "80"
        ])

        # CRITICAL SAFETY: Purge pose_priors table so colmap mapper runs 100% pure visual!
        with sqlite3.connect(database) as conn:
            conn.execute("DELETE FROM pose_priors")
            conn.commit()

    elif matcher == "exhaustive" or (matcher == "spatial" and not frame_positions):
        _run_colmap([
            "colmap", "exhaustive_matcher", "--database_path", str(database),
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.guided_matching", "1"
        ])
    elif matcher == "sequential":
        _run_colmap([
            "colmap", "sequential_matcher", "--database_path", str(database),
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.guided_matching", "1",
            "--SequentialMatching.overlap", "20",
            "--SequentialMatching.loop_detection", "0"
        ])
    else:
        raise ValueError(f"Unsupported matcher: {matcher}")

    mapper_output = work_dir / "mapper_models"
    # COLMAP requires mapper output_path to exist before it starts.
    mapper_output.mkdir(parents=True, exist_ok=True)
    _run_colmap([
        "colmap", "mapper", "--database_path", str(database),
        "--image_path", str(images_dir), "--output_path", str(mapper_output),
        "--Mapper.abs_pose_min_num_inliers", "20",
        "--Mapper.init_min_tri_angle", "12",
        "--Mapper.tri_ignore_two_view_tracks", "0",
        "--Mapper.max_reg_trials", "5",
        "--Mapper.ba_global_frames_ratio", "1.25",
        "--Mapper.ba_global_points_ratio", "1.25"
    ])
    text_models = work_dir / "text_models"
    candidates = []
    model_dirs = (sorted(path for path in mapper_output.iterdir() if path.is_dir())
                  if mapper_output.is_dir() else [])
    for model_dir in model_dirs:
        text_model = text_models / model_dir.name
        write_model_text(model_dir, text_model)
        registered_count = len(parse_images_txt(text_model / "images.txt"))
        if registered_count:
            candidates.append((registered_count, model_dir.name, text_model))
    if not candidates:
        raise RuntimeError(f"COLMAP did not reconstruct any images from {images_dir}")
    registered_count, _, largest_model = max(candidates, key=lambda item: (item[0], item[1]))
    return largest_model, low_sift


def write_model_text(model_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    _run_colmap(["colmap", "model_converter", "--input_path", str(model_path),
                 "--output_path", str(output_path), "--output_type", "TXT"])


def prune_model(sparse_dir: Path, names: set[str]) -> None:
    if names:
        prune_model_colmap(str(sparse_dir), names)


def prune_low_feature_frames(sparse_dir: Path, keep: set[str]) -> set[str]:
    """Drop frames that fall below the observation threshold as tracks are pruned."""
    dropped = set()
    while True:
        counts = count_registered_features(sparse_dir)
        low = {name for name in keep if counts.get(Path(name).name, 0) < MIN_FEATURE_PRUNE}
        if not low:
            return dropped
        prune_model(sparse_dir, low)
        keep.difference_update(low)
        dropped.update(low)


def _is_refined_result(result_path: Path) -> bool:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return (result.get("poses_format") == "colmap"
            and result.get("colmap_model_path") == "sparse/0")


def _scene_outputs_current(scene_dir: Path, result_path: Path) -> bool:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    transforms = result.get("transforms_path")
    if transforms:
        transform_path = (scene_dir / transforms).resolve()
        if not transform_path.is_relative_to(scene_dir.resolve()):
            raise ValueError(f"transforms_path escapes its scan folder: {transforms}")
        if not transform_path.is_file():
            return False
    try:
        metric_scale = json.loads((scene_dir / "metric_scale.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(metric_scale, dict):
        return False
    available = metric_scale.get("available")
    applied = metric_scale.get("scale_applied_to_colmap_geometry")
    expected_units = "meters" if available else "COLMAP units"
    if (not isinstance(available, bool) or applied is not available
            or metric_scale.get("output_coordinate_units") != expected_units):
        return False
    return (_is_refined_result(result_path)
            and not any((scene_dir / name).exists() for name in LEGACY_OUTPUTS)
            and (scene_dir / "images").is_dir()
            and (scene_dir / "sparse" / "0" / "images.txt").is_file()
            and (scene_dir / "sparse" / "0" / "points3D.ply").is_file()
            and (scene_dir / "metric_scale.json").is_file())


def _invalidate_downstream_stats(workspace: Path, scene_dirs: list[Path]) -> None:
    stale_steps = {
        "depth", "depth_pose_validation", "da3_depth_inference",
        "guided_depth_filtering", "surface_normal_polishing",
        "tsdf_volumetric_fusion", "surfel_cloud_initialization",
        "plane_regularization_and_lattice_infill", "train", "mesh",
    }
    stats_paths = {workspace / "pipeline_stats.json"}
    stats_paths.update(scene / "pipeline_stats.json" for scene in scene_dirs)
    for stats_path in stats_paths:
        if not stats_path.is_file():
            continue
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        changed = any(step in stats for step in stale_steps)
        if changed:
            for step in stale_steps:
                stats.pop(step, None)
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def clean_point_cloud(sparse_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write the cleaned PLY initialization cloud and return its points/colors."""
    points = parse_points3D_txt(sparse_dir / "points3D.txt")
    if not points:
        raise RuntimeError(f"{sparse_dir / 'points3D.txt'} has no points")
    xyz = np.asarray([point["xyz"] for point in points.values()], dtype=np.float64)
    rgb = np.asarray([point["rgb"] for point in points.values()], dtype=np.float64)
    zeros = np.zeros_like(xyz)
    xyz, rgb, zeros = voxel_dedup(xyz, rgb, zeros, VOXEL_SIZE)
    xyz, rgb, _ = remove_outliers(xyz, rgb, zeros, OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO)
    if not len(xyz):
        raise RuntimeError(f"Point-cloud cleanup removed every point from {sparse_dir}")
    storePly(str(sparse_dir / "points3D.ply"), xyz, rgb)
    return xyz, rgb


def _rotation_to_up(up: np.ndarray) -> np.ndarray:
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    target = np.array([0.0, 1.0, 0.0])
    cross = np.cross(up, target)
    cosine = float(np.clip(np.dot(up, target), -1.0, 1.0))
    if cosine < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    skew = np.array([[0.0, -cross[2], cross[1]],
                     [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def _alignment_transform(xyz: np.ndarray, sparse_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    images = parse_images_txt(sparse_dir / "images.txt")
    ups = np.asarray([image["R_c2w"] @ [0.0, -1.0, 0.0]
                      for image in images.values()])
    mean_up = ups.mean(axis=0)
    agreement = np.linalg.norm(mean_up)
    if not np.isfinite(agreement) or agreement < 0.5:
        raise ValueError(f"COLMAP camera poses do not provide a consistent up direction in {sparse_dir}")
    up_rotation = _rotation_to_up(mean_up)
    xyz_up = xyz @ up_rotation.T
    angle_deg = horizontal_align_deg(xyz_up, up_axis=1)
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    yaw_rotation = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    rotation = yaw_rotation @ up_rotation
    translation = np.array([0.0, -float((xyz @ rotation.T)[:, 1].min()), 0.0])
    return rotation, translation, angle_deg


def transform_colmap_model(sparse_dir: Path, rotation: np.ndarray,
                           translation: np.ndarray, clean_xyz: np.ndarray,
                           clean_rgb: np.ndarray) -> None:
    """Apply a world rigid transform to all COLMAP points and camera poses."""
    points_path = sparse_dir / "points3D.txt"
    transformed = []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            transformed.append(line)
            continue
        fields = line.split()
        xyz = rotation @ np.asarray(fields[1:4], dtype=np.float64) + translation
        fields[1:4] = [f"{value:.12g}" for value in xyz]
        transformed.append(" ".join(fields))
    points_path.write_text("\n".join(transformed) + "\n", encoding="utf-8")

    image_path = sparse_dir / "images.txt"
    lines = image_path.read_text(encoding="utf-8").splitlines()
    transformed, i = [], 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            transformed.append(line)
            i += 1
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10 or i + 1 >= len(lines):
            raise ValueError(f"Malformed COLMAP images.txt record: {line}")
        old_rotation = qvec2rotmat(np.asarray(fields[1:5], dtype=np.float64))
        new_rotation = old_rotation @ rotation.T
        new_translation = np.asarray(fields[5:8], dtype=np.float64) - new_rotation @ translation
        new_qvec = rotmat2qvec(new_rotation)
        fields[1:8] = [f"{value:.12g}" for value in (*new_qvec, *new_translation)]
        transformed.extend((" ".join(fields), lines[i + 1]))
        i += 2
    image_path.write_text("\n".join(transformed) + "\n", encoding="utf-8")

    storePly(str(sparse_dir / "points3D.ply"),
             clean_xyz @ rotation.T + translation, clean_rgb)
    for binary in sparse_dir.glob("*.bin"):
        binary.unlink()
    _run_colmap(["colmap", "model_converter", "--input_path", str(sparse_dir),
                 "--output_path", str(sparse_dir), "--output_type", "BIN"])


def write_colmap_transforms(sparse_dir: Path, source_transforms_path: Path,
                            kept_names: set[str], output_dir: Path) -> Path:
    """Write schema-compatible camera metadata from the final COLMAP model."""
    source = json.loads(source_transforms_path.read_text(encoding="utf-8"))
    source_frames = {Path(frame["file_path"]).name: frame for frame in source["frames"]}
    if len(source_frames) != len(source["frames"]):
        raise ValueError("ARCore transforms contain duplicate image basenames")
    parsed_images = parse_images_txt(sparse_dir / "images.txt")
    images = {Path(name).name: image for name, image in parsed_images.items()}
    cameras = parse_cameras_txt(sparse_dir / "cameras.txt")
    camera_ids = {images[name]["camera_id"] for name in kept_names if name in images}
    if len(camera_ids) != 1:
        raise ValueError(f"Expected one shared COLMAP camera, found {len(camera_ids)}")
    camera = cameras[camera_ids.pop()]
    params = camera["params"]
    if camera["model"] == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif camera["model"] == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        raise ValueError(f"Unsupported output camera model {camera['model']}; expected a pinhole model")

    flip = np.diag([1.0, -1.0, -1.0])
    frames = []
    for name in source_frames:
        if name not in kept_names:
            continue
        image = images.get(name)
        if image is None:
            raise ValueError(f"Kept image {name} is missing from the final COLMAP model")
        c2w = np.eye(4)
        c2w[:3, :3] = image["R_c2w"] @ flip
        c2w[:3, 3] = image["t_c2w"]
        source_frame = source_frames[name]
        frame = {
            "file_path": source_frame["file_path"],
            "timestamp_ns": source_frame["timestamp_ns"],
            "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
            "transform_matrix": c2w.tolist(),
        }
        if "compass_heading_deg" in source_frame:
            frame["compass_heading_deg"] = source_frame["compass_heading_deg"]
        frames.append(frame)
    if len(frames) != len(kept_names):
        raise RuntimeError("Could not create COLMAP transforms for every retained image")
    width, height = camera["width"], camera["height"]
    output = {
        "schema_version": "1.0.0", "camera_model": "OPENCV",
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
        "w": width, "h": height,
        "camera_angle_x": 2.0 * float(np.arctan(width / (2.0 * fx))),
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": frames,
    }
    jsonschema.validate(instance=output,
                        schema=json.loads(TRANSFORMS_SCHEMA.read_text(encoding="utf-8")))
    transforms_out = output_dir / "transforms.json"
    transforms_out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return transforms_out


def calibrate_metric_scale(scene_dir: Path, source_format: str, sparse_dir: Path,
                           transforms_path: Path | None, work_dir: Path) -> dict:
    if source_format not in {"arcore", "arcore_app_certified"}:
        return {"available": False,
                "reason": "metric calibration requires an ARCore-origin scan",
                "source_format": source_format}
    if transforms_path is None:
        return {"available": False, "reason": "missing ARCore frame metadata",
                "source_format": source_format}
    registered_names = {Path(name).name for name in parse_images_txt(sparse_dir / "images.txt")}
    work_dir.mkdir(parents=True, exist_ok=True)
    transforms = write_colmap_transforms(sparse_dir, transforms_path, registered_names, work_dir)
    return estimate_metric_scale(scene_dir, transforms)


def _image_files(images_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    files = sorted(path for path in images_dir.rglob("*")
                   if path.is_file() and path.suffix.casefold() in extensions)
    basenames = [path.name for path in files]
    if len(set(basenames)) != len(basenames):
        raise ValueError(f"Images in {images_dir} must have unique basenames for the COLMAP model")
    return files


def _stage_images(images_dir: Path, files: list[Path], kept_names: set[str], target: Path) -> None:
    target.mkdir(parents=True)
    for source in files:
        if source.name not in kept_names:
            continue
        destination = target / source.relative_to(images_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def _commit_scene(scene_dir: Path, images_dir: Path, staged_images: Path,
                  staged_sparse: Path, transforms_path: Path | None,
                  staged_transforms: Path | None, result_path: Path,
                  staged_result: Path, staged_metric_scale: Path,
                  scratch: Path) -> None:
    """Swap staged outputs into place and restore originals if publication fails."""
    backup = scratch / "backup"
    backup.mkdir(exist_ok=True)
    targets = [(images_dir, staged_images, backup / "images"),
               (scene_dir / "sparse", staged_sparse, backup / "sparse")]
    if transforms_path is not None and staged_transforms is not None:
        targets.append((transforms_path, staged_transforms, backup / "transforms.json"))
    targets.append((result_path, staged_result, backup / "ingestion_results.json"))
    targets.append((scene_dir / "metric_scale.json", staged_metric_scale,
                    backup / "metric_scale.json"))
    moved_old, installed = [], []
    try:
        for target, _, saved in targets:
            if target.exists():
                target.rename(saved)
                moved_old.append((saved, target))
        for target, staged, _ in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(target)
            installed.append(target)
    except Exception:
        for target in reversed(installed):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        for saved, target in reversed(moved_old):
            if saved.exists():
                saved.rename(target)
        raise


def _remove_legacy_outputs(scene_dir: Path) -> None:
    for name in LEGACY_OUTPUTS:
        path = scene_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def process_scene(scene_dir: Path, result_path: Path, scratch_root: Path,
                  matcher: str, ctx: StepContext) -> dict:
    result, images_dir, transforms_path = load_ingestion_result(result_path)
    source_format = result["source_format"]
    is_arcore = source_format in {"arcore", "arcore_app_certified"}
    files = _image_files(images_dir)
    if not files:
        raise RuntimeError(f"No images found in {images_dir}")
    input_image_count = len(files)

    scratch = Path(tempfile.mkdtemp(prefix=f"{scene_dir.name}-", dir=scratch_root))
    try:
        work = scratch / "work"
        work.mkdir()
        if is_arcore:
            if transforms_path is None:
                raise ValueError(f"ARCore input has no frame metadata in {scene_dir}")
            transform_data = json.loads(transforms_path.read_text(encoding="utf-8"))
            source_frames = transform_data["frames"]
            all_names = {Path(frame["file_path"]).name for frame in source_frames}
            if all_names != {path.name for path in files}:
                missing = all_names - {path.name for path in files}
                extra = {path.name for path in files} - all_names
                raise ValueError(f"ARCore image/transform mismatch in {scene_dir}: missing={len(missing)}, extra={len(extra)}")
            frame_positions = {
                Path(fr["file_path"]).name: np.array(fr["transform_matrix"], dtype=float)[:3, 3]
                for fr in source_frames if "transform_matrix" in fr
            }
            source_sparse, low_sift = run_image_only_colmap(
                images_dir, files, work, matcher, frame_positions=frame_positions)
            files = [path for path in files if path.name not in low_sift]
        else:
            source_sparse = result["_model_path"]
            low_sift = set()

        final_sparse = work / "final_sparse" / "0"
        write_model_text(source_sparse, final_sparse)
        feature_counts = count_registered_features(final_sparse)
        image_names = {path.name for path in files}
        low_feature = {path.name for path in files
                       if feature_counts.get(path.name, 0) < MIN_FEATURE_PRUNE}
        keep = image_names - low_feature
        if not keep:
            raise RuntimeError(f"Feature filtering removed every image in {scene_dir}")
        model_names = set(parse_images_txt(final_sparse / "images.txt"))
        dropped_model_names = {name for name in model_names
                               if Path(name).name in low_feature or Path(name).name not in keep}
        prune_model(final_sparse, {Path(name).name for name in dropped_model_names})
        low_after_pruning = prune_low_feature_frames(final_sparse, keep)
        dropped = (low_sift | low_feature | {Path(name).name for name in dropped_model_names}
                   | low_after_pruning)

        if not keep:
            raise RuntimeError(f"No usable images remain in {scene_dir}")

        metric_scale = calibrate_metric_scale(
            scene_dir, source_format, final_sparse, transforms_path,
            work / "metric_calibration")
        metric_scale["scale_applied_to_colmap_geometry"] = bool(metric_scale["available"])
        metric_scale["output_coordinate_units"] = "meters" if metric_scale["available"] else "COLMAP units"
        if metric_scale["available"]:
            scale_colmap_model(final_sparse, metric_scale["meters_per_colmap_unit"])
        staged_metric_scale = work / "metric_scale.json"
        staged_metric_scale.write_text(json.dumps(metric_scale, indent=2), encoding="utf-8")
        if metric_scale["available"]:
            ctx.note(f"{scene_dir.name}: metric scale {metric_scale['meters_per_colmap_unit']:.5f} m/COLMAP unit; "
                     f"applied to sparse points and poses; "
                     f"{len(metric_scale['steps_rejected'])} inconsistent early transition(s)")
        else:
            ctx.note(f"{scene_dir.name}: metric scale unavailable: {metric_scale['reason']}")

        xyz, rgb = clean_point_cloud(final_sparse)
        rotation, translation, angle_deg = _alignment_transform(xyz, final_sparse)
        transform_colmap_model(final_sparse, rotation, translation, xyz, rgb)

        staged_transforms = None
        if is_arcore:
            staged_transforms = write_colmap_transforms(
                final_sparse, transforms_path, keep, work)
        staged_result = work / "ingestion_results.json"
        output_result = {key: value for key, value in result.items() if not key.startswith("_")}
        output_result.update({"poses_format": "colmap", "colmap_model_path": "sparse/0"})
        if is_arcore:
            output_result["transforms_path"] = str(transforms_path.resolve().relative_to(scene_dir.resolve()))
            warning = "Image-only COLMAP poses; ARCore poses and priors were not used."
            if warning not in output_result["warnings"]:
                output_result["warnings"].append(warning)
        else:
            output_result["metadata_path"] = "sparse/0"
        jsonschema.validate(instance=output_result,
                            schema=json.loads(RESULT_SCHEMA.read_text(encoding="utf-8")))
        staged_result.write_text(json.dumps(output_result, indent=2), encoding="utf-8")
        staged_images = work / "images"
        _stage_images(images_dir, files, keep, staged_images)
        staged_sparse = work / "published_sparse"
        final_sparse.parent.rename(staged_sparse)
        _commit_scene(scene_dir, images_dir, staged_images, staged_sparse,
                      transforms_path if is_arcore else None, staged_transforms,
                      result_path, staged_result, staged_metric_scale, scratch)

        _remove_legacy_outputs(scene_dir)
        if low_sift:
            ctx.note(f"{scene_dir.name}: SIFT prefilter discarded {len(low_sift)} image(s) "
                     f"with fewer than {MIN_SIFT_FEATURES} keypoints")
        ctx.note(f"{scene_dir.name}: {'image-only COLMAP' if is_arcore else 'provided COLMAP'}; "
                 f"kept {len(keep)}/{input_image_count} images; "
                 f"coordinates in {metric_scale['output_coordinate_units']}; "
                 f"wall alignment {angle_deg:.2f} deg; minimum cloud height shifted to zero")
        return {"scene": scene_dir.name, "source_format": source_format,
                "poses_format": "colmap", "pose_method": "image_only_colmap" if is_arcore else "provided_colmap",
                "input_images": input_image_count, "retained_images": len(keep),
                "coordinate_units": metric_scale["output_coordinate_units"],
                "discarded_images": len(dropped), "alignment_deg": angle_deg,
                "vertical_shift": float(translation[1])}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=str(_backend_dir / "current_scene"))
    parser.add_argument("--matcher", choices=["spatial", "exhaustive", "sequential"], default="spatial")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    scenes = discover_scenes(workspace)
    outputs = []
    for scene, result_path in scenes:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        outputs.extend((result_path, scene / "sparse" / "0" / "images.txt",
                        scene / "sparse" / "0" / "points3D.ply", scene / "images",
                        scene / "metric_scale.json"))
        transforms = result.get("transforms_path")
        if transforms:
            transform_path = (scene / transforms).resolve()
            if not transform_path.is_relative_to(scene.resolve()):
                raise ValueError(f"transforms_path escapes its scan folder: {transforms}")
            outputs.append(transform_path)
    scenes_to_process = [(scene, result_path) for scene, result_path in scenes
                         if args.force or not _scene_outputs_current(scene, result_path)]
    if not args.force and not scenes_to_process and is_done(workspace, "colmap", outputs):
        print("[colmap] already done, skipping (use --force to re-run)")
        return 0

    if scenes_to_process:
        _invalidate_downstream_stats(workspace, [scene for scene, _ in scenes_to_process])

    per_scene = []
    total_images = 0
    with tempfile.TemporaryDirectory(prefix=".colmap-step-", dir=workspace) as scratch_root:
        with StepContext("colmap", workspace, artifacts_dir=Path(scratch_root) / "logs") as ctx:
            for scene_dir, result_path in scenes:
                if not args.force and (scene_dir, result_path) not in scenes_to_process:
                    ctx.note(f"{scene_dir.name}: outputs already exist, skipping")
                    continue
                summary = process_scene(scene_dir, result_path, Path(scratch_root), args.matcher, ctx)
                per_scene.append(summary)
                total_images += summary["input_images"]
            ctx.metric("scenes", per_scene)
            ctx.metric("frames_processed", total_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
