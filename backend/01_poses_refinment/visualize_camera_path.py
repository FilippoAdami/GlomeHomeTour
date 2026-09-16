#
# Exports the camera path as a viewable .obj: camera centers connected in
# capture order, so a triangulation gone wrong (poses jumping around, wrong
# scale, path folding onto itself) is visible at a glance. Writes two paths
# when both are available -- the original transforms.json poses (what you
# captured) and the poses COLMAP actually registered after triangulation
# (what it recovered) -- plus the triangulated sparse point cloud, so you can
# compare "intended path" vs "recovered path" vs "resulting geometry" in one
# viewer (MeshLab, Blender, CloudCompare all read .obj vertex colors + lines).
#

import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import (read_extrinsics_binary, read_extrinsics_text,
                                  read_points3D_binary, read_points3D_text, qvec2rotmat)


def write_obj(path, point_clouds, paths):
    """point_clouds: list of (xyz Nx3, rgb Nx3 uint8). paths: list of (xyz Nx3, rgb 1x3 uint8, name)."""
    with open(path, "w") as f:
        vidx = 0
        for xyz, rgb in point_clouds:
            for p, c in zip(xyz, rgb):
                f.write(f"v {p[0]} {p[1]} {p[2]} {c[0]/255} {c[1]/255} {c[2]/255}\n")
            vidx += len(xyz)
        for xyz, rgb, name in paths:
            f.write(f"o {name}\n")
            start = vidx + 1
            for p in xyz:
                f.write(f"v {p[0]} {p[1]} {p[2]} {rgb[0]/255} {rgb[1]/255} {rgb[2]/255}\n")
            vidx += len(xyz)
            f.write("l " + " ".join(str(i) for i in range(start, vidx + 1)) + "\n")


def load_transforms_path(transforms_path):
    with open(transforms_path) as f:
        frames = json.load(f)["frames"]
    frames = sorted(frames, key=lambda fr: os.path.basename(fr["file_path"]))
    return np.array([np.array(fr["transform_matrix"])[:3, 3] for fr in frames])


def load_colmap_path(sparse_dir):
    try:
        images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    except FileNotFoundError:
        images = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
    images = sorted(images.values(), key=lambda im: im.name)
    centers = []
    for im in images:
        R = qvec2rotmat(im.qvec)
        centers.append(-R.T @ im.tvec)  # camera center in world = -R^T t (w2c convention)
    return np.array(centers)


def load_colmap_points(sparse_dir):
    try:
        xyz, rgb, _ = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))
    except FileNotFoundError:
        xyz, rgb, _ = read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))
    return xyz, rgb.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Export the capture path + COLMAP-recovered path + sparse points as a viewable .obj")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--transforms", default="transforms.json", help="Ground-truth pose file to draw as the intended path (default: full transforms.json)")
    parser.add_argument("-o", "--output", default=None, help="Output .obj path (default: <source_path>/camera_path.obj)")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    output = args.output or os.path.join(source_path, "camera_path.obj")

    point_clouds, paths = [], []

    transforms_path = os.path.join(source_path, args.transforms)
    if os.path.exists(transforms_path):
        centers = load_transforms_path(transforms_path)
        paths.append((centers, (0, 255, 0), "intended_path_transforms_json"))  # green
        print(f"Intended path: {len(centers)} poses from {transforms_path}")

    sparse_dir = os.path.join(source_path, "sparse", "0")
    if os.path.exists(sparse_dir):
        centers = load_colmap_path(sparse_dir)
        paths.append((centers, (255, 0, 0), "recovered_path_colmap"))  # red
        print(f"COLMAP-recovered path: {len(centers)} registered images in {sparse_dir}")

        xyz, rgb = load_colmap_points(sparse_dir)
        point_clouds.append((xyz, rgb))
        print(f"COLMAP sparse point cloud: {len(xyz)} points")

    if not paths:
        sys.exit(f"Found neither {transforms_path} nor {sparse_dir}, nothing to export.")

    write_obj(output, point_clouds, paths)
    print(f"\nWrote {output}")
    print("Open in MeshLab/Blender/CloudCompare: green line = intended capture path, "
          "red line = what COLMAP actually recovered, grey points = triangulated cloud. "
          "A red path that jumps around, folds on itself, or drifts in scale vs the green "
          "one points at bad triangulation rather than a training problem.")


if __name__ == "__main__":
    main()
