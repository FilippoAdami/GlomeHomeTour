#!/usr/bin/env python3
"""Step 1b -- rotate landscape sensor frames into the upright portrait frame.

In:  ``<workspace>/images/`` + ``transforms.json`` (post quality-gate, still
     the raw landscape sensor frames).
Out: same ``images/`` + ``transforms.json``, rotated 90 deg clockwise in
     place -- every later step (COLMAP, depth, 2DGS) works in this frame.

ARCore captures landscape sensor frames for a phone held upright; nothing
downstream should have to think about that. Runs right after the quality gate
(``step_filter_quality.py``) rather than before it, so blur/exposure/texture
scoring happens on the frames the gate's thresholds were actually tuned
against (``00_ingestion/project_history.md``), and every later step -- COLMAP,
depth priors, 2DGS -- inherits one upright, self-consistent frame.

    python 00_ingestion/step_rotate_upright.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.scene_io import load_scene, write_scene

DEFAULT_WORKSPACE = _backend_dir / "current_scene"

# Rotating the image 90 deg clockwise means camera-local +X (image right) becomes
# old -Y (image up) and +Y (image up) becomes old +X (image right); rolling the
# pose by the same 90 deg keeps pose and pixels describing the same frame.
R_ROLL = np.array([
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def to_portrait_intrinsics(fl_x: float, fl_y: float, cx: float, cy: float,
                            w: int, h: int) -> dict:
    """Rotate landscape intrinsics 90 deg clockwise into the upright frame."""
    return {
        "fl_x": fl_y, "fl_y": fl_x,
        "cx": float(h - cy), "cy": float(cx),
        "w": h, "h": w,
    }


def rotate_upright(workspace: Path, ctx: StepContext) -> None:
    scene = load_scene(workspace)
    header = scene.header

    if header["w"] < header["h"]:
        ctx.note(f"Already portrait ({header['w']}x{header['h']}), nothing to do")
        ctx.metric("already_portrait", True)
        return

    raw_w, raw_h = header["w"], header["h"]
    new_frames = []
    with ctx.timer("rotate"):
        for frame in scene.frames:
            path = workspace / frame["file_path"]
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Could not read {path}")
            cv2.imwrite(str(path), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            new_pose = np.array(frame["transform_matrix"], dtype=np.float64) @ R_ROLL
            new_intr = to_portrait_intrinsics(
                frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"], raw_w, raw_h)
            new_frames.append({
                **frame,
                "fl_x": new_intr["fl_x"], "fl_y": new_intr["fl_y"],
                "cx": new_intr["cx"], "cy": new_intr["cy"],
                "transform_matrix": new_pose.tolist(),
            })

        header_intr = to_portrait_intrinsics(
            header["fl_x"], header["fl_y"], header["cx"], header["cy"], raw_w, raw_h)
        new_header = {
            **header,
            "fl_x": header_intr["fl_x"], "fl_y": header_intr["fl_y"],
            "cx": header_intr["cx"], "cy": header_intr["cy"],
            "w": header_intr["w"], "h": header_intr["h"],
            # Derived from the new width/fl_x, not a hardcoded FOV constant.
            "camera_angle_x": 2.0 * math.atan(header_intr["w"] / (2.0 * header_intr["fl_x"])),
        }
        write_scene(workspace, new_header, new_frames)

    ctx.metric("frames_rotated", len(new_frames))
    ctx.metric("from_wh", [raw_w, raw_h])
    ctx.metric("to_wh", [header_intr["w"], header_intr["h"]])
    ctx.note(f"Rotated {len(new_frames)} frame(s) 90 deg CW: "
             f"{raw_w}x{raw_h} -> {header_intr['w']}x{header_intr['h']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    if not args.force and is_done(workspace, "rotate_upright", [workspace / "transforms.json"]):
        print("[rotate_upright] already done, skipping (use --force to re-run)")
        return 0

    with StepContext("rotate_upright", workspace) as ctx:
        rotate_upright(workspace, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
