#!/usr/bin/env python3
"""Step 0 -- unpack a capture into the workspace as a scene folder.

In:  path to a ``.zip`` (or an already-extracted directory).
Out: ``<workspace>/images/`` + ``<workspace>/transforms.json``.

Handles the three real layouts by locating ``transforms.json`` and treating its
parent as the scene root: flat, a single wrapper directory (the shape the mobile
app writes, ``scan_20260913_133535/``), or an already-extracted directory.

Frames are copied through verbatim -- no rotation, no renumbering. The capture
is internally consistent as it stands (landscape images against landscape
``w``/``h`` and ``fl_x``/``fl_y``), and every later step reads orientation from
this one manifest, so rolling to portrait here would buy nothing and would break
the basename match against COLMAP's image names.

    python Utilities/step_extract.py scenes/Bedroom2.zip [--workspace DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from Utilities.pipeline_step import StepContext, is_done
from Utilities.scene_io import load_scene, write_scene

DEFAULT_WORKSPACE = _backend_dir / "current_scene"


def _find_scene_root(directory: Path) -> Path:
    """The directory holding ``transforms.json``, at most one level down."""
    if (directory / "transforms.json").is_file():
        return directory
    candidates = sorted(p.parent for p in directory.glob("*/transforms.json"))
    if not candidates:
        raise FileNotFoundError(f"No transforms.json in {directory} or its immediate subdirectories")
    if len(candidates) > 1:
        raise RuntimeError(
            f"{len(candidates)} scene roots under {directory} ({', '.join(c.name for c in candidates)}); "
            "point --source at the one you want")
    return candidates[0]


def extract(source: Path, workspace: Path, ctx: StepContext) -> None:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Capture not found: {source}")

    ctx.metric("source", str(source))
    if source.is_file():
        ctx.metric("archive_size_mb", round(source.stat().st_size / 1e6, 1))
        ctx.note(f"Archive: {source} ({source.stat().st_size / 1e6:.1f} MB)")

    staging = workspace / "_extract_tmp"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        if source.is_file():
            if not zipfile.is_zipfile(source):
                raise ValueError(f"{source} is not a zip archive")
            with ctx.timer("unzip"):
                with zipfile.ZipFile(source) as z:
                    z.extractall(staging)
            scene_root = _find_scene_root(staging)
        else:
            scene_root = _find_scene_root(source)
        ctx.note(f"Scene root: {scene_root}")
        ctx.metric("scene_root", str(scene_root))

        scene = load_scene(scene_root)
        ctx.note(f"Schema validation: OK ({len(scene)} frames)")

        images_dir = workspace / "images"
        shutil.rmtree(images_dir, ignore_errors=True)
        images_dir.mkdir(parents=True)
        with ctx.timer("copy_images"):
            for frame in scene.frames:
                src = scene_root / frame["file_path"]
                if not src.is_file():
                    raise FileNotFoundError(f"transforms.json names a missing image: {src}")
                shutil.copy2(src, workspace / frame["file_path"])
        write_scene(workspace, scene.header, scene.frames)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    h = scene.header
    ctx.metric("frames", len(scene))
    ctx.metric("resolution", [h["w"], h["h"]])
    ctx.metric("intrinsics", {k: h[k] for k in ("fl_x", "fl_y", "cx", "cy", "camera_angle_x")})
    ctx.metric("distortion", {k: h[k] for k in ("k1", "k2", "p1", "p2")})
    ctx.metric("schema_valid", True)

    ctx.note(f"Frames:     {len(scene)}")
    ctx.note(f"Resolution: {h['w']}x{h['h']}")
    ctx.note(f"Intrinsics: fl_x={h['fl_x']:.2f} fl_y={h['fl_y']:.2f} "
             f"cx={h['cx']:.2f} cy={h['cy']:.2f}")
    ctx.note(f"Distortion: k1={h['k1']} k2={h['k2']} p1={h['p1']} p2={h['p2']}")

    if any(float(h[k]) != 0.0 for k in ("k1", "k2", "p1", "p2")):
        # Step 2's converter refuses non-zero distortion by design: it emits a
        # PINHOLE model, and the 2DGS loader accepts nothing else. Say so here,
        # where the operator can still undistort, rather than 20 minutes later.
        raise ValueError(
            "Capture declares non-zero lens distortion. convert_transforms_to_colmap.py "
            "will refuse it (PINHOLE output would be wrong). Undistort the images and "
            "zero k1/k2/p1/p2 before running the pipeline.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Capture .zip or already-extracted directory")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--force", action="store_true", help="Re-run even if already complete")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    outputs = [workspace / "transforms.json", workspace / "images"]
    if not args.force and is_done(workspace, "extract", outputs):
        print("[extract] already done, skipping (use --force to re-run)")
        return 0

    with StepContext("extract", workspace) as ctx:
        extract(Path(args.source), workspace, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
