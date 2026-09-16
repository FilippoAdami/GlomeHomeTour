#!/usr/bin/env python3
"""Substep 4a -- COLMAP poses/intrinsics -> the arrays DA3 consumes.

DA3 must be fed the COLMAP-refined poses from ``sparse/0/``, never the raw
ARCore poses in ``transforms.json``: the refined ones are the corrected ones,
and feeding ARCore poses here would reintroduce exactly the drift step 2 just
removed.

**No conversion math is required, and applying any is the bug.** COLMAP stores
``(qvec, tvec)`` as an OpenCV *world-to-camera* pose, which is precisely what
DA3 wants -- ``convert_transforms_to_colmap.py`` builds them as
``c2w = transform_matrix @ OPENGL_TO_OPENCV; w2c = inv(c2w)``, the same
composition ``arcore_c2w_to_da3_w2c()`` used to perform on raw ARCore poses.
Passing these through that helper would apply a second inversion and a second
axis flip, producing plausible-but-wrong depth. Since the pipeline now always
runs the COLMAP pass, ``DepthPriorEstimator`` no longer converts at all and
expects world-to-camera directly (see ``depth_priors.py``).

Writes ``depth/poses_da3.npz`` (``w2c`` (N,4,4), ``K`` (N,3,3), ``names`` (N,))
so the result is inspectable and step 4 resumes without re-reading the model.

    python 02_depth_estimation/colmap_poses_to_da3.py [--workspace DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_backend_dir = Path(__file__).resolve().parents[1]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from Utilities.pipeline_paths import bootstrap

bootstrap()

from scene.colmap_loader import qvec2rotmat, read_extrinsics_text, read_intrinsics_text

DEFAULT_WORKSPACE = _backend_dir / "current_scene"


def build_da3_poses(
    sparse_dir: Path,
    names: list[str] | None = None,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(w2c (N,4,4), K (N,3,3), names)`` ordered by image name.

    ``names`` restricts and orders the output; when omitted every registered
    image is used, sorted by name. Ordering is load-bearing: COLMAP's images.txt
    is keyed by image id, not capture order, and a mismatch silently pairs each
    frame with another frame's pose.

    ``scale`` rescales the intrinsics for inference at reduced resolution --
    the COLMAP model always keeps full-resolution ``fx, fy, cx, cy``.
    """
    extrinsics = read_extrinsics_text(sparse_dir / "images.txt")
    intrinsics = read_intrinsics_text(sparse_dir / "cameras.txt")

    by_name = {Path(img.name).name: img for img in extrinsics.values()}
    if names is None:
        names = sorted(by_name)
    else:
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(
                f"{len(missing)} image(s) not in the COLMAP model, e.g. {missing[:5]}. "
                "sparse/0/ and images/ must match exactly -- run step 2/3's prune first.")

    w2c = np.zeros((len(names), 4, 4), dtype=np.float32)
    k_mats = np.zeros((len(names), 3, 3), dtype=np.float32)
    for i, name in enumerate(names):
        img = by_name[name]
        w2c[i] = np.eye(4, dtype=np.float32)
        w2c[i, :3, :3] = qvec2rotmat(img.qvec)
        w2c[i, :3, 3] = img.tvec

        cam = intrinsics[img.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        k_mats[i] = np.array([[fx * scale, 0.0, cx * scale],
                              [0.0, fy * scale, cy * scale],
                              [0.0, 0.0, 1.0]], dtype=np.float32)

    return w2c, k_mats, list(names)


def validate_poses(w2c: np.ndarray, expected_centres: np.ndarray | None = None,
                   tol_m: float = 0.5) -> dict[str, float]:
    """Assert the poses are proper rigid world-to-camera transforms.

    A double-applied conversion shows up here as a mirrored determinant or as
    camera centres that do not sit near the ARCore ones, rather than as
    plausible-but-wrong depth 20 minutes later.
    """
    rot = w2c[:, :3, :3].astype(np.float64)
    dets = np.linalg.det(rot)
    if not np.allclose(dets, 1.0, atol=1e-3):
        raise ValueError(f"non-rotation in w2c: det range [{dets.min():.4f}, {dets.max():.4f}], expected +1")

    ortho = np.abs(rot @ rot.transpose(0, 2, 1) - np.eye(3)).max()
    if ortho > 1e-3:
        raise ValueError(f"w2c rotations not orthonormal (max |RR^T - I| = {ortho:.2e})")

    stats = {"det_min": float(dets.min()), "det_max": float(dets.max()), "max_ortho_err": float(ortho)}

    if expected_centres is not None:
        centres = -np.einsum("nij,nj->ni", rot.transpose(0, 2, 1), w2c[:, :3, 3].astype(np.float64))
        offset = np.linalg.norm(centres - expected_centres, axis=1)
        stats["centre_offset_median_m"] = float(np.median(offset))
        stats["centre_offset_max_m"] = float(offset.max())
        if np.median(offset) > tol_m:
            raise ValueError(
                f"recovered camera centres sit {np.median(offset):.2f} m (median) from the "
                f"transforms.json positions, over the {tol_m} m refinement tolerance. "
                "That is a mirrored or doubly-inverted pose, not bundle adjustment.")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    args = parser.parse_args(argv)

    from Utilities.scene_io import load_scene

    workspace = Path(args.workspace)
    scene = load_scene(workspace)
    w2c, k_mats, names = build_da3_poses(workspace / "sparse" / "0", scene.names)

    by_name = {Path(f["file_path"]).name: f for f in scene.frames}
    centres = np.array([np.array(by_name[n]["transform_matrix"])[:3, 3] for n in names])
    stats = validate_poses(w2c, centres)

    out = workspace / "depth" / "poses_da3.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, w2c=w2c, K=k_mats, names=np.array(names))
    print(f"Wrote {len(names)} world-to-camera poses to {out}")
    print(f"  det in [{stats['det_min']:.6f}, {stats['det_max']:.6f}], "
          f"orthonormality err {stats['max_ortho_err']:.2e}")
    print(f"  camera centres vs transforms.json: median {stats['centre_offset_median_m'] * 100:.1f} cm, "
          f"max {stats['centre_offset_max_m'] * 100:.1f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
