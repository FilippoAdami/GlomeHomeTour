#!/usr/bin/env python3
"""
Match and retrieve the closest real captured video frames from the 364 original images
matching the optimal mesh-derived camera poses, and generate side-by-side comparison images.

Outputs:
  1. Matched real images:
     <scene_root>/artifacts/real_frame_matches/<scan_id>/real/class_XXX_real.jpg
  2. Side-by-side comparison images (Mesh vs. Best Real Video Frame):
     <scene_root>/artifacts/real_frame_matches/<scan_id>/comparisons/class_XXX_comparison.png
  3. Real frame matching manifest:
     <scene_root>/artifacts/real_frame_matches/<scan_id>/real/non_structural_frames_real_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
if str(_backend_dir / "03_FastGS_DNSplatter") not in sys.path:
    sys.path.insert(0, str(_backend_dir / "03_FastGS_DNSplatter"))

from Utilities.pipeline_paths import bootstrap
bootstrap()

from gsplat_train import load_cameras

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("match_real_captured_frames")


def find_closest_real_camera(
    ideal_eye: np.ndarray,
    ideal_target: np.ndarray,
    cameras: List[Dict[str, Any]],
    weight_angle: float = 1.20,
) -> Dict[str, Any]:
    """
    Find the closest real camera pose to the given ideal eye and target.
    Metric: Cost = ||c_real - c_ideal|| + weight_angle * angle_diff_radians.
    """
    ideal_dir = ideal_target - ideal_eye
    dir_norm = float(np.linalg.norm(ideal_dir))
    if dir_norm > 1e-6:
        ideal_dir /= dir_norm
    else:
        ideal_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    best_match = None
    min_cost = float("inf")

    for cam in cameras:
        vm = cam["viewmat"]
        R = vm[:3, :3]
        t = vm[:3, 3]
        c_real = -R.T @ t
        d_real = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        pos_dist = float(np.linalg.norm(c_real - ideal_eye))
        cos_sim = float(np.clip(np.dot(d_real, ideal_dir), -1.0, 1.0))
        angle_rad = float(np.arccos(cos_sim))
        angle_deg = float(np.degrees(angle_rad))

        cost = pos_dist + weight_angle * angle_rad

        if cost < min_cost:
            min_cost = cost
            best_match = {
                "image_path": str(cam["image_path"]),
                "image_name": Path(cam["image_path"]).name,
                "cost": round(cost, 4),
                "position_distance_m": round(pos_dist, 3),
                "angle_diff_deg": round(angle_deg, 2),
                "real_camera_center": [round(float(v), 4) for v in c_real],
                "real_camera_dir": [round(float(v), 4) for v in d_real],
            }

    if best_match is None:
        raise ValueError("Could not find any camera match among candidates.")

    return best_match


def create_side_by_side_image(
    mesh_img_path: Path,
    real_img_path: Path,
    output_path: Path,
    title_mesh: str,
    title_real: str,
    target_height: int = 900,
) -> Path:
    """
    Create a clean side-by-side comparison image:
    [Ideal Camera View (Mesh)] | [Best Real Video Frame]
    with informative title headers.
    """
    img_mesh = Image.open(mesh_img_path).convert("RGB")
    img_real = Image.open(real_img_path).convert("RGB")

    # Resize both images to target_height preserving aspect ratio
    w_m = int(round(img_mesh.width * (target_height / float(img_mesh.height))))
    img_mesh_resized = img_mesh.resize((w_m, target_height), Image.Resampling.LANCZOS)

    w_r = int(round(img_real.width * (target_height / float(img_real.height))))
    img_real_resized = img_real.resize((w_r, target_height), Image.Resampling.LANCZOS)

    header_h = 56
    padding = 16
    divider_w = 12

    total_w = w_m + w_r + divider_w + padding * 2
    total_h = target_height + header_h + padding * 2

    canvas = Image.new("RGB", (total_w, total_h), (22, 24, 28))
    draw = ImageDraw.Draw(canvas)

    # Load font
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

    # Draw header titles
    draw.text((padding + 8, padding + 14), title_mesh, fill=(240, 240, 240), font=font)
    draw.text((padding + w_m + divider_w + 8, padding + 14), title_real, fill=(240, 240, 240), font=font)

    # Paste images
    y_offset = padding + header_h
    canvas.paste(img_mesh_resized, (padding, y_offset))
    canvas.paste(img_real_resized, (padding + w_m + divider_w, y_offset))

    # Divider line
    div_x = padding + w_m + divider_w // 2
    draw.line([(div_x, y_offset), (div_x, y_offset + target_height)], fill=(50, 55, 65), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    return output_path


def run_real_frame_matching(
    workspace: Path,
    manifest_path: Optional[Path] = None,
    output_real_dir: Optional[Path] = None,
    output_sbs_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Match the closest real captured video frame for each class in the mesh manifest
    and generate side-by-side comparison images.
    """
    if manifest_path is None:
        manifest_path = (
            workspace
            / "segmentation"
            / "non_structural_category_frames"
            / "mesh"
            / "non_structural_frames_manifest.json"
        )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        mesh_manifest = json.load(f)

    extracted_classes = mesh_manifest.get("extracted_classes", {})
    if not extracted_classes:
        raise ValueError(f"Manifest at {manifest_path} contains 0 extracted classes.")

    scene_root = workspace.parent if workspace.name.startswith("scan_") else workspace
    scan_artifact_dir = scene_root / "artifacts" / "real_frame_matches" / workspace.name
    if output_real_dir is None:
        output_real_dir = scan_artifact_dir / "real"
    output_real_dir.mkdir(parents=True, exist_ok=True)

    if output_sbs_dir is None:
        output_sbs_dir = scan_artifact_dir / "comparisons"
    output_sbs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading real video cameras from the active COLMAP pose set...")
    aligned_sparse = workspace / "phase04_alignment" / "sparse" / "0"
    try:
        poses_in_place = json.loads((workspace / "mesh.json").read_text()).get(
            "mesh_alignment", {}).get("poses_written_in_place", False)
    except (OSError, ValueError):
        poses_in_place = False
    sparse_dir = (aligned_sparse if not poses_in_place
                  and (workspace / "phase04_alignment" / "alignment.json").is_file()
                  and (aligned_sparse / "images.txt").is_file() else None)
    cameras = load_cameras(workspace, sparse_dir=sparse_dir)
    logger.info(f"Loaded {len(cameras)} real video camera poses.")

    matched_manifest: Dict[str, Any] = {
        "source_mesh_manifest": str(manifest_path),
        "total_video_cameras": len(cameras),
        "classes_processed": len(extracted_classes),
        "matched_classes": {},
    }

    t0 = time.time()
    for cid_str, meta in extracted_classes.items():
        cid = int(cid_str)
        eye = np.array(meta["camera_eye"], dtype=np.float32)
        target = np.array(meta["camera_target"], dtype=np.float32)

        match_info = find_closest_real_camera(eye, target, cameras)

        # Copy/save matched real frame
        src_real_img = Path(match_info["image_path"])
        dst_real_img = output_real_dir / f"class_{cid:03d}_real.jpg"
        with Image.open(src_real_img) as img:
            img.save(dst_real_img, quality=95)

        # Generate side-by-side comparison
        mesh_img_path = Path(meta["output_image_path"])
        sbs_path = output_sbs_dir / f"class_{cid:03d}_comparison.png"

        purity = meta.get("purity_ratio", 1.0)
        title_mesh = f"Class {cid:03d} Ideal View (Mesh | Purity {purity:.2f})"
        title_real = f"Best Real Video Frame ({match_info['image_name']} | d={match_info['position_distance_m']:.2f}m, angle={match_info['angle_diff_deg']:.1f}°)"

        create_side_by_side_image(
            mesh_img_path=mesh_img_path,
            real_img_path=dst_real_img,
            output_path=sbs_path,
            title_mesh=title_mesh,
            title_real=title_real,
        )

        role = meta.get("role", "object")
        logger.info(
            f"  Class {cid:3d} ({role:<20}): Matched {match_info['image_name']} "
            f"(pos_dist={match_info['position_distance_m']:.2f}m, ang_diff={match_info['angle_diff_deg']:.1f}°) -> {sbs_path.name}"
        )

        matched_record = {
            "class_id": cid,
            "role": role,
            "mesh_image_path": str(mesh_img_path),
            "real_image_path": str(dst_real_img),
            "side_by_side_path": str(sbs_path),
            "ideal_camera_eye": meta["camera_eye"],
            "ideal_camera_target": meta["camera_target"],
            "purity_ratio": purity,
            "matched_frame_name": match_info["image_name"],
            "position_distance_m": match_info["position_distance_m"],
            "angle_diff_deg": match_info["angle_diff_deg"],
            "real_camera_center": match_info["real_camera_center"],
            "real_camera_dir": match_info["real_camera_dir"],
        }
        matched_manifest["matched_classes"][cid_str] = matched_record

    elapsed = time.time() - t0
    logger.info(
        f"Real frame matching & side-by-side generation complete in {elapsed:.2f}s: "
        f"{len(extracted_classes)}/{len(extracted_classes)} classes processed."
    )

    out_manifest_path = output_real_dir / "non_structural_frames_real_manifest.json"
    with open(out_manifest_path, "w", encoding="utf-8") as f:
        json.dump(matched_manifest, f, indent=2)
    logger.info(f"Saved real matching manifest to {out_manifest_path}")

    return matched_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Match closest real video frames and generate side-by-side comparison images."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_backend_dir / "current_scene/scan_0",
        help="Scan directory containing sparse/0, segmentation/, and mesh.ply.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to non_structural_frames_manifest.json (default: auto-detected).",
    )
    parser.add_argument(
        "--output-real-dir",
        type=Path,
        default=None,
        help="Output directory for matched real images.",
    )
    parser.add_argument(
        "--output-sbs-dir",
        type=Path,
        default=None,
        help="Output directory for side-by-side images.",
    )
    args = parser.parse_args()

    run_real_frame_matching(
        workspace=args.workspace,
        manifest_path=args.manifest,
        output_real_dir=args.output_real_dir,
        output_sbs_dir=args.output_sbs_dir,
    )


if __name__ == "__main__":
    main()
