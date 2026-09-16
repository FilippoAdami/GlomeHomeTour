#!/usr/bin/env python3
"""Render step 4's surfel cloud to ordinary PNGs, seen from the real cameras.

The PLY does carry per-vertex ``red``/``green``/``blue``. Blender imports them
as a colour attribute but draws unlit grey vertices until you wire that
attribute into a material, which is why the cloud looks colourless there.
Rather than fight that, this projects the cloud through the COLMAP poses and
writes images you can just look at.

    python 02_depth_estimation/preview_cloud.py [--workspace DIR] [--views N]

Out: ``<workspace>/depth/preview/<image_name>.png``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

_backend_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_backend_dir / "03_2DGS_training"))
from scene.colmap_loader import (  # noqa: E402
    qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary)

DEFAULT_WORKSPACE = _backend_dir / "current_scene"


def intrinsics(cam):
    """(fx, fy, cx, cy) for the COLMAP models this pipeline actually emits."""
    p = cam.params
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        return p[0], p[0], p[1], p[2]
    if cam.model in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        return p[0], p[1], p[2], p[3]
    raise ValueError(f"unhandled camera model {cam.model}")


def render(xyz, rgb, R, t, fx, fy, cx, cy, W, H, down=2):
    """Painter's-algorithm splat: far points first, near ones overwrite.

    Renders at 1/``down`` scale so there is roughly a point per pixel, and
    splats each point over a 2x2 block to close the gaps. Do *not* close them
    by supersampling and averaging down instead -- with sparse coverage that
    averages every pixel against its black neighbours and the whole frame comes
    out near-black.
    """
    W, H = W // down, H // down
    fx, fy, cx, cy = fx / down, fy / down, cx / down, cy / down

    # numpy/CPU on purpose -- the (N,3)@(3,3) hazard in backend/CLAUDE.md is a
    # torch/gfx1200 BLAS bug, it does not apply here.
    cam = xyz @ R.T + t
    z = cam[:, 2]
    keep = z > 1e-3
    cam, z, col = cam[keep], z[keep], rgb[keep]

    u = np.floor(fx * cam[:, 0] / z + cx).astype(np.int64)
    v = np.floor(fy * cam[:, 1] / z + cy).astype(np.int64)

    # All four offsets go into one array so the depth sort covers them together;
    # splatting them in four separate passes would let a far point's later
    # offset overwrite a near point's earlier one.
    u = np.concatenate([u, u + 1, u, u + 1])
    v = np.concatenate([v, v, v + 1, v + 1])
    z = np.tile(z, 4)
    col = np.tile(col, (4, 1))

    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, col = u[ok], v[ok], z[ok], col[ok]

    order = np.argsort(-z)
    img = np.zeros((H, W, 3), np.uint8)
    img[v[order], u[order]] = col[order]
    return Image.fromarray(img)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--ply", default=None, help="defaults to <workspace>/depth/points3D_depth.ply")
    ap.add_argument("--views", type=int, default=8, help="how many cameras to render")
    args = ap.parse_args(argv)

    ws = Path(args.workspace).resolve()
    ply_path = Path(args.ply) if args.ply else ws / "depth" / "points3D_depth.ply"
    out_dir = ws / "depth" / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    v = PlyData.read(ply_path).elements[0]
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)
    print(f"{len(xyz):,} points from {ply_path.name}")

    extr = read_extrinsics_binary(ws / "sparse" / "0" / "images.bin")
    intr = read_intrinsics_binary(ws / "sparse" / "0" / "cameras.bin")
    images = sorted(extr.values(), key=lambda im: im.name)
    step = max(1, len(images) // args.views)

    for im in images[::step][:args.views]:
        cam = intr[im.camera_id]
        fx, fy, cx, cy = intrinsics(cam)
        img = render(xyz, rgb, qvec2rotmat(im.qvec), im.tvec,
                     fx, fy, cx, cy, cam.width, cam.height)
        covered = float((np.asarray(img).sum(2) > 0).mean())
        assert covered > 0.01, f"{im.name} rendered essentially empty ({covered:.3%})"
        img.save(out_dir / f"{Path(im.name).stem}.png")
        print(f"  {im.name}: {covered:6.1%} of frame covered")

    print(f"\n-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
