#!/usr/bin/env python3
"""Group aligned instance meshes into rigid blocks when their surfaces touch."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "current_scene/scan_0/objects_alignment"


def contact_samples(mesh, max_face_centers=20_000):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centers = vertices[faces].mean(axis=1)
    if len(centers) > max_face_centers:
        centers = centers[np.linspace(0, len(centers) - 1, max_face_centers, dtype=np.int64)]
    return np.concatenate((vertices, centers))


def build_groups(output_dir, contact_distance=0.02):
    output_dir = Path(output_dir)
    report_path = output_dir / "rigid_baseline_report.json"
    report = json.loads(report_path.read_text())
    instance_dir = output_dir / "rigid_baseline_instances"
    items, trees, bounds = [], [], []
    for item in report["object_instances"]:
        path = instance_dir / item["path"]
        mesh = o3d.io.read_triangle_mesh(str(path))
        vertices = np.asarray(mesh.vertices)
        if not len(vertices):
            continue
        items.append(item)
        trees.append(cKDTree(contact_samples(mesh)))
        bounds.append((vertices.min(axis=0), vertices.max(axis=0)))

    parent = list(range(len(items)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = []
    for i in range(len(items)):
        min_a, max_a = bounds[i]
        for j in range(i + 1, len(items)):
            min_b, max_b = bounds[j]
            if np.any(max_a + contact_distance < min_b) or np.any(max_b + contact_distance < min_a):
                continue
            distance_ba = trees[i].query(trees[j].data, k=1, workers=-1)[0]
            distance_ab = trees[j].query(trees[i].data, k=1, workers=-1)[0]
            minimum = min(float(distance_ba.min()), float(distance_ab.min()))
            if minimum <= contact_distance:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
                pairs.append({"instance_a": items[i]["path"], "instance_b": items[j]["path"],
                              "minimum_surface_sample_distance_m": minimum,
                              "supporting_sample_pairs": int((distance_ba <= contact_distance).sum()
                                                               + (distance_ab <= contact_distance).sum())})

    groups_by_root = {}
    for index, item in enumerate(items):
        groups_by_root.setdefault(find(index), []).append(item)
    groups = sorted(groups_by_root.values(), key=lambda group: min(item["path"] for item in group))
    group_dir = output_dir / "contact_groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    for stale in group_dir.glob("group_*.ply"):
        stale.unlink()

    manifest_groups = []
    for group_id, members in enumerate(groups):
        merged = o3d.geometry.TriangleMesh()
        triangle_count = 0
        for item in members:
            mesh = o3d.io.read_triangle_mesh(str(instance_dir / item["path"]))
            merged += mesh
            triangle_count += len(mesh.triangles)
        filename = f"group_{group_id:03d}.ply"
        if not o3d.io.write_triangle_mesh(str(group_dir / filename), merged, write_ascii=False):
            raise IOError(f"Could not write contact group mesh: {group_dir / filename}")
        manifest_groups.append({"group_id": group_id, "path": f"contact_groups/{filename}",
                                "instances": [item["path"] for item in members],
                                "class_ids": sorted({int(item["class_id"]) for item in members}),
                                "instance_count": len(members), "triangle_count": triangle_count})

    manifest = {"status": "contact_groups_candidate", "source_report": str(report_path),
                "contact_distance_m": contact_distance,
                "method": "AABB-pruned nearest distance between mesh vertices and sampled triangle centers",
                "limitation": "Approximate surface contact test; increase sampling or use triangle intersection testing if review reveals missed contacts.",
                "instance_count": len(items), "contact_pair_count": len(pairs),
                "group_count": len(manifest_groups), "contact_pairs": pairs, "groups": manifest_groups}
    (output_dir / "contact_groups.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contact-distance", type=float, default=0.02,
                        help="Maximum sampled surface distance to consider instances touching, in metres")
    args = parser.parse_args()
    if args.contact_distance <= 0:
        parser.error("--contact-distance must be positive")
    manifest = build_groups(args.output_dir, args.contact_distance)
    print(json.dumps({"instances": manifest["instance_count"], "contact_pairs": manifest["contact_pair_count"],
                      "groups": manifest["group_count"], "output": str(args.output_dir / "contact_groups.json")}, indent=2))


if __name__ == "__main__":
    main()
