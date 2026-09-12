#!/usr/bin/env python3
"""GlomeHomeTour Backend: Fast Standalone Quality Gate Image Staging Tool.

Runs QualityGate filtering on raw package frames and stages all accepted
high-quality frames into GS_input/images and transforms.json.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import jsonschema
import numpy as np
from PIL import Image

_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from ingestion.package_loader import CameraIntrinsics, PackageLoader
from ingestion.pose_aligner import PoseAligner
from ingestion.quality_gate import QualityGate

HFOV_DEG = 55.4

def main():
    scene_path = _backend_dir / "scenes" / "bedroom_complete.zip"
    out_dir = _backend_dir / "scenes" / "bedroom_complete_depth_results"
    gs_input_dir = out_dir / "GS_input"
    gs_images_dir = gs_input_dir / "images"

    gs_images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GLOME HOME TOUR: STANDALONE QUALITY GATE STAGING")
    print("=" * 70)
    print(f"Scene Archive:      {scene_path}")
    print(f"Output GS Input:    {gs_input_dir}")
    print("=" * 70)

    print("\n[Step 1] Loading package and executing QualityGate filtering...")
    loader = PackageLoader()
    package = loader.load(scene_path)
    raw_count = len(package.keyframes)

    # Configure QualityGate: adaptive blur factor 0.60, min floor 30.0
    gate = QualityGate(adaptive_blur=True, adaptive_blur_factor=0.60, min_blur_floor=30.0)
    gate_result = gate.evaluate(package.keyframes)
    aligner = PoseAligner(package.trajectory)
    synced = aligner.synchronize_keyframes(gate_result.accepted_keyframes)

    acc_count = len(synced)
    disc_count = raw_count - acc_count

    print(f"Total Raw Captures in ZIP:   {raw_count}")
    print(f"Rejected Blurry/Outliers:   {disc_count}")
    print(f"Accepted High-Quality:       {acc_count}")

    # Convert to upright portrait orientation
    R_roll = np.array([
        [ 0.0, -1.0,  0.0,  0.0],
        [ 1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ], dtype=np.float64)

    intrinsics_raw = package.intrinsics
    hl, wl = intrinsics_raw.h, intrinsics_raw.w
    intrinsics_portrait = CameraIntrinsics(
        camera_model=intrinsics_raw.camera_model,
        fl_x=intrinsics_raw.fl_y,
        fl_y=intrinsics_raw.fl_x,
        cx=float(hl - intrinsics_raw.cy),
        cy=float(intrinsics_raw.cx),
        w=hl,
        h=wl,
        camera_angle_x=math.radians(HFOV_DEG),
        k1=intrinsics_raw.k1,
        k2=intrinsics_raw.k2,
        p1=intrinsics_raw.p1,
        p2=intrinsics_raw.p2,
    )

    print(f"\n[Step 2] Staging {acc_count} quality-filtered JPEG images in {gs_images_dir}...")
    
    # Remove old images in directory to ensure clean state
    for old_file in gs_images_dir.glob("*.jpg"):
        old_file.unlink()

    frames_json = []
    for i, raw_kf in enumerate(synced):
        rel_img_path = f"images/frame_{i:05d}.jpg"
        abs_img_path = gs_input_dir / rel_img_path

        img_raw = raw_kf.load_image_rgb()
        img_up = cv2.rotate(img_raw, cv2.ROTATE_90_CLOCKWISE)
        img_bgr = cv2.cvtColor(img_up, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(abs_img_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        c2w_upright = np.dot(raw_kf.transform_matrix, R_roll)

        frames_json.append({
            "file_path": rel_img_path,
            "timestamp_ns": int(raw_kf.timestamp_ns),
            "fl_x": float(intrinsics_portrait.fl_x),
            "fl_y": float(intrinsics_portrait.fl_y),
            "cx": float(intrinsics_portrait.cx),
            "cy": float(intrinsics_portrait.cy),
            "transform_matrix": c2w_upright.tolist(),
        })

    transforms_data = {
        "schema_version": "1.0.0",
        "camera_model": "OPENCV",
        "fl_x": float(intrinsics_portrait.fl_x),
        "fl_y": float(intrinsics_portrait.fl_y),
        "cx": float(intrinsics_portrait.cx),
        "cy": float(intrinsics_portrait.cy),
        "w": int(intrinsics_portrait.w),
        "h": int(intrinsics_portrait.h),
        "camera_angle_x": float(intrinsics_portrait.camera_angle_x),
        "k1": float(intrinsics_portrait.k1),
        "k2": float(intrinsics_portrait.k2),
        "p1": float(intrinsics_portrait.p1),
        "p2": float(intrinsics_portrait.p2),
        "frames": frames_json,
    }

    transforms_path = gs_input_dir / "transforms.json"
    with open(transforms_path, "w", encoding="utf-8") as f:
        json.dump(transforms_data, f, indent=2)

    # Validate against schema
    schema_path = _backend_dir.parent / "shared" / "schemas" / "transforms.schema.json"
    if schema_path.exists():
        with open(schema_path) as sf:
            schema = json.load(sf)
            jsonschema.validate(instance=transforms_data, schema=schema)
            print("Schema Validation: transforms.json is 100% Draft 2020-12 compliant!")

    print(f"\n[Success] Staged {acc_count} quality-filtered images and updated transforms.json!")

if __name__ == "__main__":
    main()
