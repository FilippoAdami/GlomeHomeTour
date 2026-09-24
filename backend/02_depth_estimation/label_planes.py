"""Colour every big vertical/horizontal plane in a surfel cloud, one colour per plane.

Diagnostic only -- geometry is never moved. Writes a recoloured copy plus a JSON index that
names each plane's colour in plain English, so you can open the cloud, spot a colour, and look
up what the detector thought it was.

    python 02_depth_estimation/label_planes.py <cloud.ply> [--min-area 4.0]

Out: <cloud>_labelled.ply  +  <cloud>_labelled.json

The JSON also carries `angle_matrix_deg` (dihedral angle between every pair of planes) and
`candidate_merges`: the pairs that `regularize_planes.py` would treat as one surface -- parallel
within `--merge-angle`, separated by less than `--merge-k` x the smaller plane's size, capped at
`--max-gap`. Two 20x20 cm patches 30 cm apart are not close; two 3x3 m walls 30 cm apart are.
This script only measures; `regularize_planes.py` is what actually merges and snaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from regularize_planes import classify, detect_up, extract_planes

# distinct enough to tell apart as sparse points on a dark background, and nameable
PALETTE = [
    ("red", (220, 30, 30)), ("green", (40, 180, 60)), ("blue", (40, 90, 230)),
    ("orange", (255, 140, 0)), ("purple", (150, 60, 200)), ("cyan", (0, 200, 210)),
    ("yellow", (240, 220, 40)), ("magenta", (230, 60, 180)), ("lime", (160, 230, 50)),
    ("teal", (0, 130, 130)), ("pink", (255, 150, 180)), ("brown", (140, 80, 40)),
    ("navy", (20, 40, 120)), ("olive", (120, 120, 30)), ("maroon", (130, 20, 50)),
    ("gold", (200, 165, 20)), ("turquoise", (60, 220, 170)), ("lavender", (190, 170, 255)),
    ("crimson", (200, 20, 90)), ("forest green", (20, 100, 40)), ("salmon", (250, 130, 110)),
    ("slate blue", (100, 110, 190)), ("mustard", (190, 160, 60)), ("violet", (170, 40, 230)),
]


def label(src: Path, args) -> dict:
    ply = PlyData.read(str(src))
    vert = ply["vertex"].data.copy()
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float64)
    normals = np.stack([vert["nx"], vert["ny"], vert["nz"]], axis=1).astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    up = detect_up(normals)
    eps = args.cluster_eps
    if eps is None:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[:: max(1, len(xyz) // 50000)]))
        eps = 6.0 * float(np.median(np.asarray(pcd.compute_nearest_neighbor_distance())))

    planes = extract_planes(xyz, normals, args.dist_thr, args.min_inliers, args.min_extent,
                            args.normal_tol, args.max_planes, eps)
    classify(planes, up, args.wall_tol, args.floor_tol)

    # extract_planes already measures area by occupied 10 cm cells, not bounding box
    kept = [p for p in planes if p["kind"] != "free" and p["area"] >= args.min_area]
    kept.sort(key=lambda p: -p["area"])

    if args.grey_background:
        for c in ("red", "green", "blue"):
            vert[c] = (vert[c] * 0.25 + 45).astype(vert[c].dtype)

    out = dict(source=str(src), up_axis=[round(float(v), 5) for v in up],
               min_area_m2=args.min_area, planes=[])
    for i, p in enumerate(kept):
        name, rgb = PALETTE[i % len(PALETTE)]
        if i >= len(PALETTE):
            name = f"{name} (repeat {i // len(PALETTE) + 1})"
        for c, val in zip(("red", "green", "blue"), rgb):
            vert[c][p["idx"]] = val
        n = p["normal"] * np.sign(np.dot(p["normal"], up) or 1.0)
        out["planes"].append(dict(
            number=i, colour=name, rgb=list(rgb), kind=p["kind"],
            area_m2=round(p["area"], 2), inliers=int(len(p["idx"])),
            extent_m=[round(float(e), 2) for e in p["extent"]],
            centroid=[round(float(c), 3) for c in p["centroid"]],
            normal=[round(float(v), 4) for v in n],
            height_along_up_m=round(float(np.dot(p["centroid"], up)), 3),
            tilt_from_horizontal_deg=round(p["tilt_from_horizontal"], 2),
            tilt_from_vertical_deg=round(p["tilt_from_vertical"], 2),
        ))

    out["candidate_merges"] = candidate_merges(kept, args)
    out["angle_matrix_deg"] = angle_matrix(kept)
    dst = src.with_name(src.stem + "_labelled.ply")
    PlyData([PlyElement.describe(vert, "vertex")], text=False).write(str(dst))
    out["output"] = str(dst)
    return out


def angle_matrix(planes):
    """Dihedral angle between every pair, folded to [0, 90]: 0 = parallel, 90 = square."""
    n = np.array([p["normal"] for p in planes])
    m = np.degrees(np.arccos(np.clip(np.abs(n @ n.T), 0, 1)))
    np.fill_diagonal(m, 0.0)
    return [[round(float(v), 2) for v in row] for row in m]


def candidate_merges(planes, args):
    """Pairs a size-aware rule would call one surface. Gap is the perpendicular separation;
    the threshold scales with sqrt(area) of the *smaller* plane, so big walls tolerate a big
    gap and small patches do not."""
    pairs = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            a, b = planes[i], planes[j]
            if a["kind"] != b["kind"]:
                continue
            angle = np.degrees(np.arccos(np.clip(abs(np.dot(a["normal"], b["normal"])), 0, 1)))
            if angle > args.merge_angle:
                continue
            mean_n = a["normal"] * np.sign(np.dot(a["normal"], b["normal"]) or 1.0) + b["normal"]
            mean_n /= np.linalg.norm(mean_n)
            gap = abs(float(np.dot(a["centroid"] - b["centroid"], mean_n)))
            thr = min(args.merge_k * np.sqrt(min(a["area"], b["area"])), args.max_gap)
            pairs.append(dict(planes=[i, j], kind=a["kind"], angle_deg=round(float(angle), 2),
                              gap_m=round(gap, 3), threshold_m=round(float(thr), 3),
                              would_merge=bool(gap <= thr)))
    pairs.sort(key=lambda p: (not p["would_merge"], p["gap_m"]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--min-area", type=float, default=4.0, help="colour planes bigger than, m2")
    ap.add_argument("--wall-tol", type=float, default=35.0, help="counts as vertical below, deg")
    ap.add_argument("--floor-tol", type=float, default=20.0, help="counts as horizontal below, deg")
    ap.add_argument("--dist-thr", type=float, default=0.02)
    ap.add_argument("--min-inliers", type=int, default=1500)
    ap.add_argument("--min-extent", type=float, default=0.5)
    ap.add_argument("--normal-tol", type=float, default=20.0)
    ap.add_argument("--max-planes", type=int, default=40)
    ap.add_argument("--cluster-eps", type=float)
    ap.add_argument("--merge-angle", type=float, default=10.0, help="merge candidates within, deg")
    ap.add_argument("--merge-k", type=float, default=0.5, help="gap budget per sqrt(area)")
    ap.add_argument("--max-gap", type=float, default=0.35, help="absolute cap on the merge gap, m")
    ap.add_argument("--no-grey-background", dest="grey_background", action="store_false")
    args = ap.parse_args()

    rep = label(args.source, args)
    js = args.source.with_name(args.source.stem + "_labelled.json")
    js.write_text(json.dumps(rep, indent=2))

    print(f"up {rep['up_axis']}   planes >= {args.min_area} m2: {len(rep['planes'])}")
    for p in rep["planes"]:
        print(f"  #{p['number']:2d} {p['colour']:<14} {p['kind']:<10} {p['area_m2']:6.2f} m2  "
              f"{p['inliers']:6d} pts  h={p['height_along_up_m']:6.2f} m  "
              f"tilt {min(p['tilt_from_horizontal_deg'], p['tilt_from_vertical_deg']):5.2f} deg")
    mat = rep["angle_matrix_deg"]
    print("\nangle between planes (deg, 0 = parallel, 90 = square):")
    print("      " + " ".join(f"{i:>6d}" for i in range(len(mat))))
    for i, row in enumerate(mat):
        print(f"  #{i:2d} " + " ".join(f"{v:6.2f}" for v in row))

    merges = [m for m in rep["candidate_merges"] if m["would_merge"]]
    print(f"\n{len(merges)} merge candidates (of {len(rep['candidate_merges'])} parallel pairs):")
    for m in merges:
        print(f"  #{m['planes'][0]:2d} + #{m['planes'][1]:2d}  {m['kind']:<10} "
              f"{m['angle_deg']:5.2f} deg apart, gap {m['gap_m']:.3f} m <= {m['threshold_m']:.3f} m")
    print(f"\nwrote {rep['output']}\nwrote {js}")


if __name__ == "__main__":
    main()
