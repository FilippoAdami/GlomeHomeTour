#!/usr/bin/env python3
"""GlomeHomeTour Backend: staged frame filtering.

Three stages, each writing a self-contained folder (``images/`` + a
``transforms.json`` covering exactly those images) so every step can be
inspected or fed to the reconstruction engine on its own:

  1. ``01_quality/``   -- per-image quality gate (blur / exposure / texture).
                          Expected to keep ~80-90% of a decent capture.
  2. ``02_parallax/``  -- drops near-duplicate viewpoints that carry no
                          triangulation baseline. Expected to drop a few percent.
  3. ``03_keyframes/`` -- minimal anchor set for reconstruction: as few views as
                          possible while keeping frustum coverage.

Usage:
    python run_staged_filtering.py Bedroom2 [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import jsonschema
import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap
bootstrap()

from package_loader import CameraIntrinsics, Keyframe, PackageLoader
from keyframe_selector import DynamicKeyframeSelector, estimate_scene_depths
from pose_aligner import PoseAligner
from quality_gate import QualityGate, prune_redundant

# ARCore captures landscape sensor frames for a phone held upright; everything
# downstream (depth priors, 2DGS) works in the upright portrait frame.
R_ROLL = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def to_portrait_intrinsics(raw: CameraIntrinsics) -> CameraIntrinsics:
    """Rotate landscape intrinsics 90 deg clockwise into the upright frame."""
    return CameraIntrinsics(
        camera_model=raw.camera_model,
        fl_x=raw.fl_y,
        fl_y=raw.fl_x,
        cx=float(raw.h - raw.cy),
        cy=float(raw.cx),
        w=raw.h,
        h=raw.w,
        # Derived, not a hardcoded FOV constant: the two staging scripts this
        # replaces disagreed (55.4 vs 40.8 deg) and at most one could be right.
        camera_angle_x=2.0 * math.atan(raw.h / (2.0 * raw.fl_y)),
        k1=raw.k1,
        k2=raw.k2,
        p1=raw.p1,
        p2=raw.p2,
    )


def write_stage(
    out_dir: Path,
    keyframes: list[Keyframe],
    intrinsics: CameraIntrinsics,
    write_images: bool = True,
) -> Path:
    """Write ``images/`` + ``transforms.json`` describing exactly ``keyframes``."""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for stale in images_dir.glob("*.jpg"):
        stale.unlink()

    frames = []
    for i, kf in enumerate(keyframes):
        rel_path = f"images/frame_{i:05d}.jpg"
        if write_images:
            img = cv2.rotate(kf.load_image_rgb(), cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(str(out_dir / rel_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frames.append({
            "file_path": rel_path,
            "timestamp_ns": int(kf.timestamp_ns),
            "fl_x": float(intrinsics.fl_x),
            "fl_y": float(intrinsics.fl_y),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "transform_matrix": kf.transform_matrix.tolist(),
        })

    transforms = {
        "schema_version": "1.0.0",
        "camera_model": "OPENCV",
        "fl_x": float(intrinsics.fl_x),
        "fl_y": float(intrinsics.fl_y),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "w": int(intrinsics.w),
        "h": int(intrinsics.h),
        "camera_angle_x": float(intrinsics.camera_angle_x),
        "k1": float(intrinsics.k1),
        "k2": float(intrinsics.k2),
        "p1": float(intrinsics.p1),
        "p2": float(intrinsics.p2),
        "frames": frames,
    }

    path = out_dir / "transforms.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transforms, f, indent=2)

    schema_path = _backend_dir.parent / "shared" / "schemas" / "transforms.schema.json"
    if schema_path.exists():
        with open(schema_path) as sf:
            jsonschema.validate(instance=transforms, schema=json.load(sf))

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="scene name under backend/scenes (dir or .zip)")
    parser.add_argument("--out", default=None, help="output dir (default: scenes/<scene>_staged)")
    parser.add_argument("--blur-thresh", type=float, default=None,
                        help="absolute sharpness floor on top of the scene-relative one")
    parser.add_argument("--min-translation", type=float, default=0.03,
                        help="stage 2: metres of baseline required between kept frames")
    parser.add_argument("--min-rotation", type=float, default=2.0,
                        help="stage 2: degrees of rotation required between kept frames")
    parser.add_argument("--min-keyframes", type=int, default=None,
                        help="stage 3: floor on the anchor set size")
    parser.add_argument("--max-keyframes", type=int, default=None,
                        help="stage 3: ceiling on the anchor set size")
    parser.add_argument("--no-images", action="store_true",
                        help="write transforms.json only (fast dry run)")
    args = parser.parse_args()

    scenes = _backend_dir / "scenes"
    source = scenes / args.scene
    if not source.exists():
        source = scenes / f"{args.scene}.zip"
    # Captures unpack as <scene>/<scan_timestamp>/, so accept either level.
    if source.is_dir() and not (source / "transforms.json").exists():
        scans = sorted(p for p in source.iterdir() if (p / "transforms.json").exists())
        if not scans:
            sys.exit(f"No capture package found under {source}")
        source = scans[-1]

    out_root = Path(args.out) if args.out else scenes / f"{args.scene}_staged"

    package = PackageLoader().load(source)
    intrinsics = to_portrait_intrinsics(package.intrinsics)
    total = len(package.keyframes)
    print(f"Loaded {total} raw frames from {source}")

    # Stage 1: image quality.
    gate = QualityGate(blur_threshold=args.blur_thresh)
    result = gate.evaluate(package.keyframes)
    q_kfs = PoseAligner(package.trajectory).synchronize_keyframes(result.accepted_keyframes)
    # Roll the poses into the upright frame here rather than at write time: stage 3
    # measures frustum overlap, and doing that with portrait intrinsics against an
    # un-rolled landscape pose transposes the frustum and mismeasures the overlap.
    q_kfs = [replace(kf, transform_matrix=np.dot(kf.transform_matrix, R_ROLL)) for kf in q_kfs]
    print(f"[1/3] quality   {len(q_kfs):5d}/{total} kept ({100*len(q_kfs)/max(1,total):.1f}%)  "
          f"blur={result.summary['rejected_blur']} "
          f"exposure={result.summary['rejected_exposure']} "
          f"texture={result.summary['rejected_texture']}")
    write_stage(out_root / "01_quality", q_kfs, intrinsics, not args.no_images)

    # Stage 2: near-duplicate viewpoints.
    kept, dropped = prune_redundant(q_kfs, args.min_translation, args.min_rotation)
    p_kfs = [q_kfs[i] for i in kept]
    print(f"[2/3] parallax  {len(p_kfs):5d}/{len(q_kfs)} kept "
          f"({100*len(p_kfs)/max(1,len(q_kfs)):.1f}%)  dropped={len(dropped)}")
    write_stage(out_root / "02_parallax", p_kfs, intrinsics, not args.no_images)

    # Stage 3: anchor set. Overlap thresholds are scene-dependent, so the band is
    # enforced explicitly: back-fill under 15% of stage 2, tighten over 50%. The
    # band is wide on purpose -- coverage beats compactness here, and the selector
    # only ever hits it by re-walking at a different overlap threshold, never by
    # subsampling a chain it just built.
    # Measure how far away each frame's subject actually is. Overlap between two
    # views is meaningless without it: the same 50 cm sidestep keeps most of a
    # 3 m wall in frame and loses a 0.5 m one entirely.
    scene_depths = estimate_scene_depths(p_kfs, intrinsics)
    print(f"      scene depth: median {np.median(scene_depths):.2f} m, "
          f"p10 {np.percentile(scene_depths, 10):.2f} m, p90 {np.percentile(scene_depths, 90):.2f} m")

    selector = DynamicKeyframeSelector.for_2dgs_training()
    selection = selector.select_keyframes(
        p_kfs, intrinsics,
        scene_depths=scene_depths,
        min_keyframes=args.min_keyframes if args.min_keyframes is not None else int(0.15 * len(p_kfs)),
        max_keyframes=args.max_keyframes if args.max_keyframes is not None else int(0.50 * len(p_kfs)),
    )
    k_kfs = selection.selected_keyframes
    print(f"[3/3] keyframes {len(k_kfs):5d}/{len(p_kfs)} kept "
          f"({100*len(k_kfs)/max(1,len(p_kfs)):.1f}%)  {selection.reasons}")
    write_stage(out_root / "03_keyframes", k_kfs, intrinsics, not args.no_images)

    print(f"\nStaged under {out_root}")


if __name__ == "__main__":
    main()
