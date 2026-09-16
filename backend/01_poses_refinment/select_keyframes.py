#
# Thins a transforms.json capture down to a subset of frames with enough
# baseline (translation) and viewpoint (rotation) change between consecutive
# kept frames to give COLMAP good parallax for triangulation. Copies images/
# to images_keyframes/ and writes a matching transforms_keyframes.json,
# dropping the minimum number of frames needed to keep every kept frame at
# least min_translation apart or min_angle_deg apart from the previous one.
#

import os
import sys
import json
import shutil
import argparse
import numpy as np


def rotation_angle_deg(R_a, R_b):
    # batched: R_a is (N,3,3), R_b is (3,3)
    R_rel = np.einsum("nij,jk->nik", R_a.transpose(0, 2, 1), R_b)
    cos_theta = np.clip((np.trace(R_rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def select_keyframes(frames, min_translation, min_angle_deg):
    # A candidate is redundant only if it's within both min_translation AND
    # min_angle_deg of some *already kept* frame -- checked against all of
    # them, not just the previous one, so loop-closures in the capture path
    # (walking back near an earlier position) don't sneak in near-duplicates.
    kept = [frames[0]]
    kept_pos = [np.array(frames[0]["transform_matrix"])[:3, 3]]
    kept_rot = [np.array(frames[0]["transform_matrix"])[:3, :3]]
    for frame in frames[1:]:
        c2w = np.array(frame["transform_matrix"])
        pos, rot = c2w[:3, 3], c2w[:3, :3]
        dists = np.linalg.norm(np.array(kept_pos) - pos, axis=1)
        angles = rotation_angle_deg(np.array(kept_rot), rot)
        redundant = np.any((dists < min_translation) & (angles < min_angle_deg))
        if not redundant:
            kept.append(frame)
            kept_pos.append(pos)
            kept_rot.append(rot)
    return kept


def main():
    parser = argparse.ArgumentParser(description="Select a subset of frames with enough pose separation for good COLMAP triangulation")
    parser.add_argument("-s", "--source_path", required=True, help="Folder containing transforms.json and an images/ subfolder")
    parser.add_argument("--min_translation", type=float, default=None,
                         help="Minimum camera position change (scene units) between kept frames. Default: 2.5x the median frame-to-frame translation")
    parser.add_argument("--min_angle_deg", type=float, default=5.0,
                         help="Minimum camera rotation change (degrees) between kept frames")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    transforms_path = os.path.join(source_path, "transforms.json")
    if not os.path.exists(transforms_path):
        sys.exit(f"No transforms.json found in {source_path}")

    with open(transforms_path) as f:
        transforms = json.load(f)
    frames = transforms["frames"]

    min_translation = args.min_translation
    if min_translation is None:
        positions = np.array([f["transform_matrix"] for f in frames])[:, :3, 3]
        consecutive_dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        min_translation = 2.5 * np.median(consecutive_dists)

    kept = select_keyframes(frames, min_translation, args.min_angle_deg)

    images_dir = os.path.join(source_path, "images")
    out_images_dir = os.path.join(source_path, "images_keyframes")
    if os.path.exists(out_images_dir):
        shutil.rmtree(out_images_dir)
    shutil.copytree(images_dir, out_images_dir)

    kept_names = {os.path.basename(f["file_path"]) for f in kept}
    removed = 0
    for name in os.listdir(out_images_dir):
        if name not in kept_names:
            os.remove(os.path.join(out_images_dir, name))
            removed += 1

    out_transforms = {**transforms, "frames": kept}
    out_transforms_path = os.path.join(source_path, "transforms_keyframes.json")
    with open(out_transforms_path, "w") as f:
        json.dump(out_transforms, f, indent=2)

    print(f"min_translation={min_translation:.4f}, min_angle_deg={args.min_angle_deg}")
    print(f"Kept {len(kept)}/{len(frames)} frames, removed {removed} images from {out_images_dir}")
    print(f"Wrote {out_transforms_path}")


if __name__ == "__main__":
    main()
