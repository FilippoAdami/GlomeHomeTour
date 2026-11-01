#!/usr/bin/env python3
"""
Render photorealistic in-context non-structural frames from the 3D Gaussian Splat model.

Uses the exact same set of camera poses, intrinsics, and rectangular crop parameters
derived from the mesh extraction step (non_structural_frames_manifest.json) without removing
occluding classes.

Saves rendered frames to a dedicated comparison folder:
  <workspace>/segmentation/non_structural_category_frames/gs/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from Utilities.pipeline_paths import bootstrap
bootstrap()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("extract_non_structural_frames_gs")


def resolve_gs_checkpoint_path(workspace: Path, explicit_path: Optional[Path] = None) -> Path:
    """Auto-detect the appropriate Gaussian Splat checkpoint.pt for the workspace."""
    if explicit_path is not None and explicit_path.is_file():
        return explicit_path

    # 1. Try reading the source model path recorded during mesh extraction
    mesh_json_path = workspace / "mesh.json"
    if mesh_json_path.is_file():
        try:
            with open(mesh_json_path, "r", encoding="utf-8") as f:
                mesh_info = json.load(f)
            sealed_checkpoint = mesh_info.get("gsplat_sealing", {}).get("checkpoint_path")
            if sealed_checkpoint:
                candidate = Path(sealed_checkpoint)
                if candidate.is_file():
                    return candidate
            src_ply = mesh_info.get("source_model_path")
            if src_ply:
                # Stage 03: .../gaussian_splat/gaussian_splat.ply
                # checkpoint is at: .../gaussian_splat/checkpoint.pt
                ply_path = Path(src_ply)
                candidate = ply_path.parent / "checkpoint.pt"
                if candidate.is_file():
                    return candidate
                # Previous layout stored the PLY under point_cloud/iteration_N.
                candidate2 = ply_path.parents[2] / "checkpoint.pt"
                if candidate2.is_file():
                    return candidate2
        except Exception as e:
            logger.debug(f"Could not parse mesh.json: {e}")

    # 2. Stage 03 stores model artifacts in a scan-local directory.
    model_dir = workspace / "gaussian_splat"
    if (model_dir / "checkpoint.pt").is_file():
        return model_dir / "checkpoint.pt"

    # Previous merged layout exported directly into the scan workspace.
    if (workspace / "checkpoint.pt").is_file():
        return workspace / "checkpoint.pt"

    # Pipeline workspaces may contain one scan folder rather than scan inputs directly.
    scan_checkpoints = sorted(path / "gaussian_splat" / "checkpoint.pt" for path in workspace.glob("scan_*")
                              if (path / "gaussian_splat" / "checkpoint.pt").is_file())
    if len(scan_checkpoints) == 1:
        return resolve_gs_checkpoint_path(scan_checkpoints[0].parent.parent)
    if len(scan_checkpoints) > 1:
        raise ValueError(f"Multiple scan checkpoints found under {workspace}; pass --checkpoint explicitly")

    raise FileNotFoundError(f"Could not find a Gaussian Splat checkpoint.pt in {workspace}")


def load_gaussian_splats(
    checkpoint_path: Path,
    device: str = "cuda",
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Load splat parameters from a checkpoint file.
    Returns (splats_dict, sh_degree).
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA/ROCm GPU is required for gsplat rasterization.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_splats = checkpoint["splats"]

    sh_degree = int(checkpoint.get("sh_degree", 3))

    splats = {
        "means": raw_splats["means"].to(device),
        "quats": F.normalize(raw_splats["quats"].to(device), dim=-1),
        "scales": torch.exp(raw_splats["scales"].to(device)),
        "opacities": torch.sigmoid(raw_splats["opacities"].to(device)),
    }

    if "sh0" in raw_splats and "shN" in raw_splats:
        splats["colors"] = torch.cat(
            [raw_splats["sh0"].to(device), raw_splats["shN"].to(device)], dim=1
        )
    elif "colors" in raw_splats:
        splats["colors"] = raw_splats["colors"].to(device)
    else:
        raise ValueError("Checkpoint does not contain spherical harmonics or colors.")

    return splats, sh_degree


def compute_w2c_and_intrinsics(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fov_deg: float,
    render_res: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OpenCV World-to-Camera (W2C) 4x4 matrix and 3x3 Intrinsic matrix K
    matching Open3D OffscreenRenderer and the camera pose parameters.
    """
    f_vec = target - eye
    f_norm = float(np.linalg.norm(f_vec))
    if f_norm > 1e-6:
        f_vec /= f_norm
    else:
        f_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    r_vec = np.cross(f_vec, up)
    r_norm = float(np.linalg.norm(r_vec))
    if r_norm > 1e-6:
        r_vec /= r_norm
    else:
        r_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    u_vec = np.cross(r_vec, f_vec)
    u_norm = float(np.linalg.norm(u_vec))
    if u_norm > 1e-6:
        u_vec /= u_norm

    # OpenCV Camera Convention:
    # X_c = right (r_vec)
    # Y_c = down (-u_vec)
    # Z_c = forward (f_vec)
    R = np.stack([r_vec, -u_vec, f_vec], axis=0).astype(np.float32)  # [3, 3]
    t = (-R @ eye).astype(np.float32)

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    fov_rad = np.radians(fov_deg)
    fx = float((render_res / 2.0) / np.tan(fov_rad / 2.0))
    fy = fx
    cx = float(render_res / 2.0)
    cy = float(render_res / 2.0)

    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    return viewmat, K


def render_gs_frame(
    splats: Dict[str, torch.Tensor],
    viewmat: np.ndarray,
    K: np.ndarray,
    render_res: int,
    sh_degree: int = 3,
    tile_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """
    Render an RGB image from Gaussian Splats at render_res x render_res.
    Returns uint8 RGB numpy array of shape [H, W, 3].
    """
    from gsplat import rasterization

    viewmats = torch.as_tensor(viewmat, dtype=torch.float32, device=device)[None]
    Ks = torch.as_tensor(K, dtype=torch.float32, device=device)[None]

    with torch.no_grad():
        rendered, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=splats["scales"],
            opacities=splats["opacities"],
            colors=splats["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=render_res,
            height=render_res,
            sh_degree=sh_degree,
            tile_size=tile_size,
            packed=False,
            render_mode="RGB",
        )

    img_np = (rendered[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    return img_np


# Helper functions imported from mesh frame extractor
from extract_non_structural_frames import pad_crop_to_square, validate_rendered_frame


def run_gs_extraction(
    workspace: Path,
    manifest_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    discarded_dir: Optional[Path] = None,
    device: str = "cuda",
    tile_size: int = 16,
    enable_quality_check: bool = True,
    min_unique_colors: int = 8,
    min_dynamic_range: float = 15.0,
    min_color_std: float = 3.5,
    min_laplacian_var: float = 0.05,
    min_fg_pixels: int = 50,
    min_autocorr: float = 0.35,
    min_p95_grad: float = 1.25,
    min_edge_density: float = 1.0,
) -> Dict[str, Any]:
    """
    Extract Gaussian Splat frames for all classes documented in the mesh frames manifest.
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
        raise FileNotFoundError(
            f"Mesh non-structural frames manifest not found at: {manifest_path}. "
            "Please run extract_non_structural_frames.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        mesh_manifest = json.load(f)

    extracted_classes = mesh_manifest.get("extracted_classes", {})
    if not extracted_classes:
        raise ValueError(f"Manifest at {manifest_path} contains 0 extracted classes.")

    ckpt_file = resolve_gs_checkpoint_path(workspace, checkpoint_path)
    logger.info(f"Using Gaussian Splat checkpoint: {ckpt_file}")

    if output_dir is None:
        output_dir = workspace / "segmentation" / "non_structural_category_frames" / "gs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if discarded_dir is None:
        discarded_dir = output_dir.parent / (output_dir.name + "_discarded")

    logger.info(f"Loading Gaussian Splats to {device}...")
    splats, sh_degree = load_gaussian_splats(ckpt_file, device=device)
    num_splats = len(splats["means"])
    logger.info(f"Loaded {num_splats:,} Gaussians (SH degree {sh_degree})")

    padding_color = mesh_manifest.get("padding_color", "black")
    min_resolution = mesh_manifest.get("min_resolution", None)

    instances_dict = mesh_manifest.get("extracted_instances")
    is_instance_mode = bool(instances_dict)
    items_to_render = list(instances_dict.items()) if is_instance_mode else list(extracted_classes.items())

    gs_manifest: Dict[str, Any] = {
        "source_checkpoint": str(ckpt_file),
        "source_mesh_manifest": str(manifest_path),
        "num_gaussians": num_splats,
        "sh_degree": sh_degree,
        "classes_processed": len(extracted_classes),
        "total_instances_extracted": 0,
        "padding_color": padding_color,
        "extracted_classes": {},
        "extracted_instances": {},
        "discarded_instances": {},
    }

    t0 = time.time()
    logger.info(f"Rendering {len(items_to_render)} Gaussian Splat frames...")

    for key_str, meta in items_to_render:
        cid = int(meta["class_id"]) if "class_id" in meta else int(key_str)
        inst_id = int(meta.get("instance_id", 0))
        eye = np.array(meta["camera_eye"], dtype=np.float32)
        target = np.array(meta["camera_target"], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        fov_deg = float(meta["camera_fov_deg"])
        render_res = int(meta.get("raw_render_res", 1536))
        crop_coords = meta.get("crop_coords")  # [cu0, cv0, cu1, cv1]

        viewmat, K = compute_w2c_and_intrinsics(eye, target, up, fov_deg, render_res)

        raw_rgb = render_gs_frame(
            splats=splats,
            viewmat=viewmat,
            K=K,
            render_res=render_res,
            sh_degree=sh_degree,
            tile_size=tile_size,
            device=device,
        )

        if crop_coords is not None:
            cu0, cv0, cu1, cv1 = crop_coords
            crop_arr = raw_rgb[cv0:cv1, cu0:cu1]
        else:
            crop_arr = raw_rgb

        square_png = pad_crop_to_square(
            crop_arr,
            min_side=min_resolution,
            padding_color=padding_color,
        )

        png_filename = meta.get("output_image_file", f"class_{cid:03d}_inst_{inst_id:03d}.png" if is_instance_mode else f"class_{cid:03d}.png")
        out_png_path = output_dir / png_filename
        role = meta.get("role", "unknown")
        crop_h, crop_w = crop_arr.shape[:2]

        is_valid = True
        reason = "valid"
        metrics: Dict[str, Any] = {}
        if enable_quality_check:
            is_valid, reason, metrics = validate_rendered_frame(
                square_png,
                min_unique_colors=min_unique_colors,
                min_dynamic_range=min_dynamic_range,
                min_color_std=min_color_std,
                min_laplacian_var=min_laplacian_var,
                min_fg_pixels=min_fg_pixels,
                min_autocorr=min_autocorr,
                min_p95_grad=min_p95_grad,
                min_edge_density=min_edge_density,
            )
            # For verified objects from mesh manifest, accept authentic renders with strong contrast/dynamic range
            if not is_valid and reason.startswith("no_recognizable_features: flat_or_blurry_surface"):
                if (
                    metrics.get("unique_colors", 0) >= 15
                    and metrics.get("gray_std", 0) >= 10.0
                    and metrics.get("gray_ptp", 0) >= 50.0
                ):
                    is_valid = True
                    reason = "valid (contrast_preserved)"

        if not is_valid:
            discarded_dir.mkdir(parents=True, exist_ok=True)
            discard_path = discarded_dir / png_filename
            square_png.save(discard_path)
            if out_png_path.exists():
                try:
                    out_png_path.unlink()
                except OSError:
                    pass
            discard_record = dict(meta)
            discard_record["output_image_file"] = png_filename
            discard_record["output_image_path"] = str(discard_path)
            discard_record["discard_reason"] = reason
            discard_record["discard_metrics"] = metrics
            gs_manifest["discarded_instances"][key_str] = discard_record
            logger.info(
                f"  Class {cid:3d} (Inst {inst_id:2d}, {role:<20}): [DISCARDED GS] {png_filename}: {reason} -> saved to {discard_path}"
            )
            continue

        square_png.save(out_png_path)
        logger.info(
            f"  Class {cid:3d} (Inst {inst_id:2d}, {role:<20}): GS render -> {png_filename} "
            f"[crop {crop_w}x{crop_h} -> square {square_png.size[0]}x{square_png.size[1]} PNG]"
        )

        gs_meta = dict(meta)
        gs_meta["output_image_file"] = png_filename
        gs_meta["output_image_path"] = str(out_png_path)
        if is_instance_mode:
            gs_manifest["extracted_instances"][key_str] = gs_meta
        else:
            gs_manifest["extracted_classes"][key_str] = gs_meta

    if is_instance_mode:
        # Group instances into extracted_classes for compatibility
        for key_str, gs_meta in gs_manifest["extracted_instances"].items():
            cid_str = str(gs_meta["class_id"])
            inst_id_str = str(gs_meta["instance_id"])
            if cid_str not in gs_manifest["extracted_classes"]:
                gs_manifest["extracted_classes"][cid_str] = dict(gs_meta)
                gs_manifest["extracted_classes"][cid_str]["instances"] = {}
            gs_manifest["extracted_classes"][cid_str]["instances"][inst_id_str] = gs_meta
        for cid_str, class_entry in gs_manifest["extracted_classes"].items():
            class_entry["num_instances"] = len(class_entry.get("instances", {}))

    gs_manifest["total_instances_extracted"] = len(gs_manifest.get("extracted_instances", {}))
    elapsed = time.time() - t0
    logger.info(
        f"Gaussian Splat frame extraction complete in {elapsed:.2f}s: "
        f"{gs_manifest['total_instances_extracted']} valid frames rendered, "
        f"{len(gs_manifest['discarded_instances'])} frames discarded."
    )

    gs_manifest_path = output_dir / "non_structural_frames_gs_manifest.json"
    with open(gs_manifest_path, "w", encoding="utf-8") as f:
        json.dump(gs_manifest, f, indent=2)
    logger.info(f"Saved GS manifest to {gs_manifest_path}")

    if gs_manifest.get("discarded_instances"):
        discard_manifest_path = discarded_dir / "discarded_manifest.json"
        with open(discard_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_discarded": len(gs_manifest["discarded_instances"]),
                "discarded_instances": gs_manifest["discarded_instances"],
            }, f, indent=2)
        logger.info(f"Saved GS discarded manifest to {discard_manifest_path}")

    return gs_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve photorealistic non-structural frames from the Gaussian Splat model using mesh camera poses."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=_backend_dir / "current_scene/scan_0",
        help="Scan workspace containing checkpoint.pt and segmentation/.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to non_structural_frames_manifest.json (default: auto-detected in workspace).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to GS checkpoint.pt (default: auto-detected from the scan workspace or mesh metadata).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for GS frames (default: <workspace>/segmentation/non_structural_category_frames/gs).",
    )
    parser.add_argument(
        "--discarded-dir",
        type=Path,
        default=None,
        help="Directory to save discarded GS images (default: <output_dir>_discarded).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=16,
        help="gsplat rasterization tile size (default: 16).",
    )
    parser.add_argument(
        "--enable-quality-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable quality filter to discard featureless/single-color/noise images (default: True).",
    )
    parser.add_argument(
        "--min-unique-colors",
        type=int,
        default=8,
        help="Minimum unique colors required in foreground (default: 8).",
    )
    parser.add_argument(
        "--min-dynamic-range",
        type=float,
        default=15.0,
        help="Minimum dynamic range in foreground (default: 15.0).",
    )
    parser.add_argument(
        "--min-color-std",
        type=float,
        default=3.5,
        help="Minimum color std in foreground (default: 3.5).",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=0.05,
        help="Minimum Laplacian variance (default: 0.05).",
    )
    parser.add_argument(
        "--min-fg-pixels",
        type=int,
        default=50,
        help="Minimum foreground pixels required (default: 50).",
    )
    parser.add_argument(
        "--min-autocorr",
        type=float,
        default=0.35,
        help="Minimum lag-1 autocorrelation (default: 0.35).",
    )
    parser.add_argument(
        "--min-p95-grad",
        type=float,
        default=1.25,
        help="Minimum 95th percentile gradient (default: 1.25).",
    )
    parser.add_argument(
        "--min-edge-density",
        type=float,
        default=1.0,
        help="Minimum percent of edge pixels (default: 1.0).",
    )
    args = parser.parse_args()

    run_gs_extraction(
        workspace=args.workspace,
        manifest_path=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        discarded_dir=args.discarded_dir,
        tile_size=args.tile_size,
        enable_quality_check=args.enable_quality_check,
        min_unique_colors=args.min_unique_colors,
        min_dynamic_range=args.min_dynamic_range,
        min_color_std=args.min_color_std,
        min_laplacian_var=args.min_laplacian_var,
        min_fg_pixels=args.min_fg_pixels,
        min_autocorr=args.min_autocorr,
        min_p95_grad=args.min_p95_grad,
        min_edge_density=args.min_edge_density,
    )


if __name__ == "__main__":
    main()
