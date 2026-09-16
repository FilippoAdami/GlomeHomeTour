#
# Orchestrates a full run on a transforms.json-based scene: selects a
# well-separated subset of frames, runs COLMAP triangulation on that subset,
# builds downscaled image subfolders, then trains multi-stage from low to
# full resolution (each stage resumes from the previous stage's checkpoint).
#

import os
import sys
import subprocess
from argparse import ArgumentParser

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# (subfolder suffix, downscale factor, cumulative iteration count to reach by
# the end of this stage -- train.py's --iterations is a target, not a duration)
STAGES = [("_8", 8, 7_000), ("_4", 4, 14_000), ("_2", 2, 21_000), ("", 1, 30_000)]

IMAGES_BASE = "images_keyframes"
TRANSFORMS_BASE = "transforms_keyframes.json"


def run(cmd):
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def make_downscaled_images(source_path, factor):
    dst_dir = os.path.join(source_path, f"{IMAGES_BASE}_{factor}")
    if os.path.exists(dst_dir):
        return
    os.makedirs(dst_dir)
    src_dir = os.path.join(source_path, IMAGES_BASE)
    for name in os.listdir(src_dir):
        with Image.open(os.path.join(src_dir, name)) as img:
            img.resize((img.width // factor, img.height // factor), Image.LANCZOS).save(os.path.join(dst_dir, name))


def main():
    parser = ArgumentParser(description="Full keyframe-selection + COLMAP + multi-stage low-to-high-res training pipeline")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", default=None, help="defaults to output/<source folder name>")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    transforms_path = os.path.join(source_path, "transforms.json")
    if not os.path.exists(transforms_path):
        sys.exit(f"No transforms.json found in {source_path}")

    keyframes_images_dir = os.path.join(source_path, IMAGES_BASE)
    keyframes_transforms_path = os.path.join(source_path, TRANSFORMS_BASE)
    if os.path.exists(keyframes_images_dir) and os.path.exists(keyframes_transforms_path):
        print(f"Found existing {IMAGES_BASE}/, skipping keyframe selection.")
    else:
        run([sys.executable, os.path.join(SCRIPT_DIR, "select_keyframes.py"), "-s", source_path])

    sparse_dir = os.path.join(source_path, "sparse", "0")
    if os.path.exists(os.path.join(sparse_dir, "points3D.bin")):
        print(f"Found existing sparse model at {sparse_dir}, skipping COLMAP.")
    else:
        run([sys.executable, os.path.join(SCRIPT_DIR, "convert_transforms_to_colmap.py"),
             "-s", source_path, "--matcher", args.matcher,
             "--images", IMAGES_BASE, "--transforms", TRANSFORMS_BASE])

    model_path = args.model_path or os.path.join(SCRIPT_DIR, "output", os.path.basename(source_path.rstrip("/")))

    checkpoint = None
    for suffix, factor, iterations in STAGES:
        if factor > 1:
            print(f"\nPreparing {IMAGES_BASE}_{factor}/ ...")
            make_downscaled_images(source_path, factor)

        images_arg = f"{IMAGES_BASE}{suffix}"
        print(f"\n=== Training stage: {images_arg} (factor {factor}, {iterations} iters) ===")
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train.py"),
               "-s", source_path, "-m", model_path, "-i", images_arg,
               "--iterations", str(iterations),
               "--checkpoint_iterations", str(iterations)]
        if checkpoint:
            cmd += ["--start_checkpoint", checkpoint]
        run(cmd)
        checkpoint = os.path.join(model_path, f"chkpnt{iterations}.pth")

    print(f"\nDone. Final model at {model_path}")


if __name__ == "__main__":
    main()
