"""Snap near-vertical / near-horizontal planes in a surfel cloud into a consistent frame.

Detects big planes by RANSAC (open3d), then for each plane rotates its own inliers about
their centroid so that:
  - planes within `--floor-tol` of horizontal become exactly perpendicular to the up axis
    (hence parallel to each other and orthogonal to every verticalized wall);
  - planes within `--wall-tol` of vertical become exactly parallel to the up axis, and their
    azimuths are snapped to area-weighted wall families clustered modulo 90 deg.
Planes outside both tolerances (sloping roof, beams) are left alone.

Writes a copy; never touches the input.

# ponytail: per-plane correction only -- points not on a detected plane (furniture, clutter
# sitting on the mezzanine floor) keep their original pose, so a corrected floor can end up
# slightly out of contact with the objects on it. Upgrade path if that matters: segment the
# cloud into floor-level slabs and apply each floor's rotation rigidly to everything between
# its floor and ceiling, instead of to the plane inliers only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, cKDTree

AXES = np.eye(3)

PALETTE = [
    {"name": "Crimson Red", "hex": "#E6194B", "rgb": (230, 25, 75), "emoji": "🔴"},
    {"name": "Vivid Green", "hex": "#3CB44B", "rgb": (60, 180, 75), "emoji": "🟢"},
    {"name": "Gold Yellow", "hex": "#FFE119", "rgb": (255, 225, 25), "emoji": "🟡"},
    {"name": "Bright Blue", "hex": "#0082C8", "rgb": (0, 130, 200), "emoji": "🔵"},
    {"name": "Deep Orange", "hex": "#F58231", "rgb": (245, 130, 48), "emoji": "🟠"},
    {"name": "Royal Purple", "hex": "#911EB4", "rgb": (145, 30, 180), "emoji": "🟣"},
    {"name": "Vibrant Cyan", "hex": "#46F0F0", "rgb": (70, 240, 240), "emoji": "🔷"},
    {"name": "Hot Magenta", "hex": "#F032E6", "rgb": (240, 50, 230), "emoji": "🌺"},
    {"name": "Neon Lime", "hex": "#BCF60C", "rgb": (210, 245, 60), "emoji": "🟩"},
    {"name": "Soft Pink", "hex": "#FABED4", "rgb": (250, 190, 212), "emoji": "🌸"},
    {"name": "Teal Green", "hex": "#008080", "rgb": (0, 128, 128), "emoji": "🩵"},
    {"name": "Lavender", "hex": "#DCBEFF", "rgb": (220, 190, 255), "emoji": "🪻"},
    {"name": "Amber Brown", "hex": "#9A6324", "rgb": (154, 99, 36), "emoji": "🟤"},
    {"name": "Warm Beige", "hex": "#FFFAC8", "rgb": (255, 250, 200), "emoji": "🌕"},
    {"name": "Dark Maroon", "hex": "#800000", "rgb": (128, 0, 0), "emoji": "🍷"},
    {"name": "Fresh Mint", "hex": "#AAFFC3", "rgb": (170, 255, 195), "emoji": "🌱"},
    {"name": "Olive Drab", "hex": "#808000", "rgb": (128, 128, 0), "emoji": "🫒"},
    {"name": "Light Coral", "hex": "#FFD8B1", "rgb": (255, 216, 177), "emoji": "🍑"},
    {"name": "Navy Blue", "hex": "#000075", "rgb": (0, 0, 117), "emoji": "🌌"},
]
UNASSIGNED_COLOR = {"name": "Dark Grey (Clutter)", "hex": "#414141", "rgb": (65, 65, 65), "emoji": "⬛"}


def _create_colored_vert(vert: np.ndarray) -> np.ndarray:
    """Ensure vertex array has uint8 red, green, blue fields for visual plane inspection."""
    names = vert.dtype.names or ()
    if "red" in names and "green" in names and "blue" in names:
        return vert.copy()
    descr = list(vert.dtype.descr)
    for c in ("red", "green", "blue"):
        if c not in names:
            descr.append((c, "u1"))
    out = np.zeros(len(vert), dtype=descr)
    for name in names:
        out[name] = vert[name]
    return out


def build_legend_markdown(report: dict) -> str:
    """Build a comprehensive GitHub-flavored Markdown legend for visual analysis."""
    lines = [
        "# Plane Regularization Legend & Analysis",
        "",
        f"- **Source PLY**: `{report.get('source', '')}`",
        f"- **Output PLY**: `{report.get('output', '')}`",
        f"- **Colored Diagnostic PLY**: `{report.get('colored_ply', '')}`",
        f"- **Total Points**: {report.get('points', 0):,} raw -> {report.get('points_out', report.get('points', 0)):,} regularized",
        f"- **Gravity Up-Axis**: {report.get('up_axis')}",
        f"- **Manhattan Global Yaw**: {report.get('manhattan_frame_deg')}°",
        "",
        "### Planar Surfaces Catalog",
        "",
        "| Swatch | Group | Kind | Color Name | Hex | RGB | Points (In -> Out) | OBB Dims (W × H) | OBB Area | Occupied Area | Centroid (X, Y, Z) | Target / Normal | Status |",
        "| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
    ]

    fill_map = {f["group"]: f for f in report.get("fill", [])}
    applied_count = 0
    total_filled_area = 0.0

    for g in report.get("groups", []):
        gid = g["group"]
        kind_raw = g.get("kind", "unknown")
        if kind_raw == "horizontal":
            c_y = g.get("centroid", [0, 0, 0])[1]
            kind_name = f"Horizontal ({'Floor' if c_y < 0 else 'Ceiling'})"
        elif kind_raw == "sloped":
            kind_name = "Sloped Roof"
        elif kind_raw == "vertical":
            kind_name = "Vertical Wall"
        else:
            kind_name = kind_raw.capitalize()

        applied = g.get("applied", False)
        color = PALETTE[gid % len(PALETTE)]
        swatch = color["emoji"]
        cname = color["name"]
        chex = f"`{color['hex']}`"
        crgb = f"`{color['rgb']}`"

        fstat = fill_map.get(gid)
        if fstat:
            pts_str = f"{g['inliers']:,} -> {fstat['added']:,}"
            occ_area_str = f"{fstat['filled_m2']:.2f} m²"
            total_filled_area += fstat["filled_m2"]
        else:
            pts_str = f"{g['inliers']:,}"
            occ_area_str = f"{g.get('area_m2', 0.0):.2f} m²"
            if applied:
                total_filled_area += g.get("area_m2", 0.0)

        obb_dims = g.get("bbox_dims_m", [0, 0])
        dims_str = f"{obb_dims[0]:.2f}m × {obb_dims[1]:.2f}m"
        obb_area_str = f"{g.get('bbox_area_m2', 0.0):.2f} m²"
        c = g.get("centroid", [0, 0, 0])
        centroid_str = f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]"

        target = g.get("target", "none")
        normal = g.get("normal", "")
        rot_deg = g.get("rotation_deg", 0.0)
        if target == "sloped":
            target_str = f"Sloped (rot {rot_deg:.1f}°) {normal}"
        elif target == "manhattan":
            target_str = f"Manhattan {normal}"
        elif target == "horizontal":
            target_str = f"Horizontal {normal}"
        else:
            target_str = str(target)

        status = "**Applied & Filled**" if applied else f"Rejected ({g.get('reason', 'n/a')})"
        if applied:
            applied_count += 1

        lines.append(
            f"| {swatch} | **{gid}** | {kind_name} | {cname} | {chex} | {crgb} | {pts_str} | "
            f"{dims_str} | {obb_area_str} | {occ_area_str} | {centroid_str} | {target_str} | {status} |"
        )

    # Clutter / unassigned points row
    n_out = report.get("points_out", report.get("points", 0))
    n_filled_pts = sum(f["added"] for f in report.get("fill", []))
    n_clutter = n_out - n_filled_pts if n_filled_pts > 0 else n_out - sum(g["inliers"] for g in report.get("groups", []) if g.get("applied"))
    clutter_pct = (n_clutter / max(1, n_out)) * 100

    u_swatch = UNASSIGNED_COLOR["emoji"]
    u_name = UNASSIGNED_COLOR["name"]
    u_hex = f"`{UNASSIGNED_COLOR['hex']}`"
    u_rgb = f"`{UNASSIGNED_COLOR['rgb']}`"
    lines.append(
        f"| {u_swatch} | **Clutter** | Unassigned / Detail | {u_name} | {u_hex} | {u_rgb} | "
        f"{n_clutter:,} ({clutter_pct:.1f}%) | - | - | - | - | Non-planar / Furniture | Preserved |"
    )

    lines.extend([
        "",
        "### Summary",
        f"- **Applied Planar Groups**: {applied_count} surfaces regularized & resampled",
        f"- **Total Regularized Surface Area**: {total_filled_area:.2f} m²",
        f"- **Preserved Clutter / Fine Details**: {n_clutter:,} surfels ({clutter_pct:.1f}% of cloud)",
        "",
        "> [!TIP]",
        "> Open `points3D_depth_planes_colored.ply` in CloudCompare or MeshLab to visually cross-reference each plane with its color swatch above.",
        ""
    ])
    return "\n".join(lines)


def _rotate(vecs: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(N,3) @ R.T via explicit columns -- see backend/CLAUDE.md, matmul is unsafe for big N."""
    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
    out = np.empty_like(vecs)
    for i in range(3):
        out[:, i] = R[i, 0] * x + R[i, 1] * y + R[i, 2] * z
    return out


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a to unit vector b (Rodrigues)."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def detect_up(normals: np.ndarray, tol_deg: float = 10.0, samples: int = 40000) -> np.ndarray:
    """Gravity by hemisphere vote: the direction most surfel normals are either parallel or
    perpendicular to (floors/ceilings + walls). Not restricted to a coordinate axis -- voting
    over the 3 axes alone picks the wrong one when the cloud is even slightly off-frame."""
    rng = np.random.default_rng(0)
    s = normals[rng.choice(len(normals), min(samples, len(normals)), replace=False)]
    k = np.arange(8000) + 0.5
    phi, theta = np.arccos(1 - k / 8000), np.pi * (1 + 5**0.5) * k
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    par, perp = np.cos(np.radians(tol_deg)), np.sin(np.radians(tol_deg))
    scores = np.empty(len(dirs))
    for i in range(0, len(dirs), 1000):  # chunked: the full (40k, 8k) dot product is 2.5 GB
        d = np.abs(s @ dirs[i:i + 1000].T)
        scores[i:i + 1000] = (d > par).sum(axis=0) + (d < perp).sum(axis=0)
    up = dirs[int(np.argmax(scores))]
    # refine on the floor/ceiling normals only (symmetric moment: they point both ways)
    flat = normals[np.abs(normals @ up) > par]
    if len(flat) > 100:
        up = np.linalg.eigh(flat.T @ flat)[1][:, -1]
    up = up / np.linalg.norm(up)
    # the eigenvector's sign is arbitrary; pin it so heights don't flip between runs
    return up * np.sign(up[int(np.argmax(np.abs(up)))])


def plane_from_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane: returns (centroid, unit normal) from the smallest PCA direction."""
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    return centroid, vt[2] / np.linalg.norm(vt[2])


def plane_basis(pts, centroid):
    """The plane's own two in-plane axes (largest PCA directions)."""
    return np.linalg.svd(pts - centroid, full_matrices=False)[2][:2]


def polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for area of 2D polygon vertices in counter-clockwise or clockwise order."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def intersect_lines(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    """Intersect 2D line p1 + t*d1 and p2 + s*d2."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-9:
        return None
    dp = p2 - p1
    t = (dp[0] * d2[1] - dp[1] * d2[0]) / det
    return p1 + t * d1


def compute_quad_metrics(poly: np.ndarray, local_pts: np.ndarray, cell_size: float = 0.10) -> dict:
    """Compute geometric properties of an N-vertex enclosing polygon."""
    edges = np.roll(poly, -1, axis=0) - poly
    edge_lens = [float(np.linalg.norm(e)) for e in edges]

    # N Internal corner angles (degrees)
    angles_deg = []
    n_v = len(poly)
    for i in range(n_v):
        v_prev = poly[(i - 1) % n_v] - poly[i]
        v_next = poly[(i + 1) % n_v] - poly[i]
        l1, l2 = np.linalg.norm(v_prev), np.linalg.norm(v_next)
        if l1 > 1e-6 and l2 > 1e-6:
            cos_a = np.clip(np.dot(v_prev, v_next) / (l1 * l2), -1.0, 1.0)
            ang = float(np.degrees(np.arccos(cos_a)))
        else:
            ang = 90.0
        angles_deg.append(ang)

    poly_area = polygon_area(poly)
    grid = np.floor(local_pts / cell_size).astype(np.int64)
    occ_area = float(len(np.unique(grid, axis=0)) * (cell_size ** 2))
    occ_ratio = float(np.clip(occ_area / max(poly_area, 1e-4), 0.0, 1.0))

    return {
        "vertices_2d": poly,
        "quad_area_m2": round(poly_area, 3),
        "poly_area_m2": round(poly_area, 3),
        "occupied_area_m2": round(occ_area, 3),
        "occupied_ratio": round(occ_ratio, 3),
        "edge_lengths_m": [round(el, 3) for el in edge_lens],
        "internal_angles_deg": [round(a, 1) for a in angles_deg],
    }


def fit_minimum_enclosing_polygon(local_pts: np.ndarray, max_vertices: int = 5) -> tuple[np.ndarray, dict]:
    """Fit a free convex enclosure up to `max_vertices` (e.g. 5-sided polygon) enclosing local_pts that minimizes enclosing area

    (maximizing the internal occupied area ratio). Vertices do not require 90-degree angles.
    Returns (vertices_Nx2, metrics_dict).
    """
    if len(local_pts) < 3:
        pt_min = local_pts.min(axis=0) if len(local_pts) > 0 else np.zeros(2)
        max_xy = local_pts.max(axis=0) if len(local_pts) > 0 else np.ones(2)
        quad = np.array([
            [pt_min[0], pt_min[1]],
            [max_xy[0], pt_min[1]],
            [max_xy[0], max_xy[1]],
            [pt_min[0], max_xy[1]]
        ])
        return quad, compute_quad_metrics(quad, local_pts)

    try:
        hull = ConvexHull(local_pts)
        hull_pts = local_pts[hull.vertices]
        n_h = len(hull_pts)
    except Exception:
        hull_pts = local_pts
        n_h = len(hull_pts)

    if n_h <= max_vertices:
        poly = hull_pts[:max_vertices]
        return poly, compute_quad_metrics(poly, local_pts)

    # Search combinations of tangent lines supporting the convex hull
    edges = []
    for i in range(n_h):
        e = hull_pts[(i + 1) % n_h] - hull_pts[i]
        norm = np.linalg.norm(e)
        if norm > 1e-6:
            edges.append((hull_pts[i], e / norm, i))

    # Subsample edges if hull has many vertices for performance
    if len(edges) > 24:
        lengths = [np.linalg.norm(hull_pts[(i+1)%n_h] - hull_pts[i]) for i in range(n_h)]
        sorted_idx = np.argsort(lengths)[::-1]
        keep_idx = set(sorted_idx[:16])
        for step in np.linspace(0, n_h-1, 12, dtype=int):
            keep_idx.add(step)
        sub_edges = [edges[i] for i in sorted(keep_idx)]
    else:
        sub_edges = edges

    m = len(sub_edges)
    best_area = float('inf')
    best_poly = None
    c = np.mean(hull_pts, axis=0)

    # Prepared oriented lines
    oriented_lines = []
    for (p, d, idx) in sub_edges:
        n = np.array([-d[1], d[0]])
        if np.dot(n, p - c) < 0:
            n, d = -n, np.array([n[1], -n[0]])
        oriented_lines.append((p, d, n))

    def _eval_lines(chosen_lines):
        k = len(chosen_lines)
        verts = []
        for idx in range(k):
            p_a, d_a, _ = chosen_lines[idx]
            p_b, d_b, _ = chosen_lines[(idx + 1) % k]
            if abs(d_a[0] * d_b[1] - d_a[1] * d_b[0]) < 0.12:
                return None
            v = intersect_lines(p_a, d_a, p_b, d_b)
            if v is None:
                return None
            verts.append(v)
        poly = np.array(verts)
        edges_p = np.roll(poly, -1, axis=0) - poly
        crosses = edges_p[:, 0] * np.roll(edges_p[:, 1], -1) - edges_p[:, 1] * np.roll(edges_p[:, 0], -1)
        if not ((crosses > 0).all() or (crosses < 0).all()):
            return None
        p_sign = 1 if crosses[0] > 0 else -1
        for edge_start, edge_vec in zip(poly, edges_p):
            to_hull = hull_pts - edge_start
            cr = edge_vec[0] * to_hull[:, 1] - edge_vec[1] * to_hull[:, 0]
            if (p_sign > 0 and (cr < -1e-4).any()) or (p_sign < 0 and (cr > 1e-4).any()):
                return None
        return poly

    # Search combinations for k=4
    import itertools
    for combo4 in itertools.combinations(oriented_lines, 4):
        poly4 = _eval_lines(combo4)
        if poly4 is not None:
            area4 = polygon_area(poly4)
            if area4 < best_area:
                best_area = area4
                best_poly = poly4

    # Search combinations for k=5 (up to 5 points)
    if max_vertices >= 5:
        # If m is moderate, iterate 5-combinations
        step_edges = oriented_lines if m <= 18 else [oriented_lines[i] for i in np.linspace(0, m - 1, 16, dtype=int)]
        for combo5 in itertools.combinations(step_edges, 5):
            poly5 = _eval_lines(combo5)
            if poly5 is not None:
                area5 = polygon_area(poly5)
                if area5 < best_area:
                    best_area = area5
                    best_poly = poly5

    if best_poly is None:
        w_obb, h_obb, area_obb, center_obb, uv_obb = fit_2d_oriented_bbox(local_pts)
        u, v = uv_obb[0], uv_obb[1]
        half_w, half_h = w_obb * 0.5, h_obb * 0.5
        best_poly = np.array([
            center_obb - half_w * u - half_h * v,
            center_obb + half_w * u - half_h * v,
            center_obb + half_w * u + half_h * v,
            center_obb - half_w * u + half_h * v
        ])

    return best_poly, compute_quad_metrics(best_poly, local_pts)


def fit_minimum_enclosing_quadrilateral(local_pts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Compatibility alias for fit_minimum_enclosing_polygon with max_vertices=5."""
    return fit_minimum_enclosing_polygon(local_pts, max_vertices=5)


def fit_2d_oriented_bbox(local_pts: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Minimum-area 2D oriented bounding box via convex hull and rotating calipers.

    Returns (width, height, area, center_2d, uv_axes) where width <= height.
    """
    if len(local_pts) < 3:
        pt_min = local_pts.min(axis=0)
        pt_max = local_pts.max(axis=0)
        w, h = float(pt_max[0] - pt_min[0]), float(pt_max[1] - pt_min[1])
        w_min, h_max = (min(w, h), max(w, h))
        return w_min, h_max, w * h, (pt_min + pt_max) * 0.5, np.eye(2)

    try:
        hull = ConvexHull(local_pts)
        hull_pts = local_pts[hull.vertices]
        min_area = float("inf")
        best_wh = (0.0, 0.0)
        best_center = np.zeros(2)
        best_uv = np.eye(2)

        n_h = len(hull_pts)
        for i in range(n_h):
            edge = hull_pts[(i + 1) % n_h] - hull_pts[i]
            edge_len = float(np.linalg.norm(edge))
            if edge_len < 1e-8:
                continue
            u = edge / edge_len
            v = np.array([-u[1], u[0]])
            pu = hull_pts @ u
            pv = hull_pts @ v
            u_min, u_max = float(pu.min()), float(pu.max())
            v_min, v_max = float(pv.min()), float(pv.max())
            w, h = u_max - u_min, v_max - v_min
            area = w * h
            if area < min_area:
                min_area = area
                best_wh = (min(w, h), max(w, h))
                best_center = (u_min + u_max) * 0.5 * u + (v_min + v_max) * 0.5 * v
                best_uv = np.stack([u, v], axis=0)

        if min_area < float("inf"):
            return best_wh[0], best_wh[1], min_area, best_center, best_uv
    except Exception:
        pass

    pt_min = local_pts.min(axis=0)
    pt_max = local_pts.max(axis=0)
    w, h = float(pt_max[0] - pt_min[0]), float(pt_max[1] - pt_min[1])
    return min(w, h), max(w, h), w * h, (pt_min + pt_max) * 0.5, np.eye(2)


def occupied_area(local, cell=0.1):
    """Surface area from occupied cells of the plane's own 2D grid, not bounding-box area.
    An L-shaped wall or a floor with a stairwell hole would otherwise read far too big."""
    return len(np.unique(np.floor(local / cell).astype(np.int64), axis=0)) * cell * cell


def extract_planes(xyz, normals, dist_thr, min_inliers, min_extent, normal_tol_deg, max_planes,
                   cluster_eps):
    """Iterative RANSAC. Inliers are additionally required to carry an agreeing surfel normal."""
    o3d.utility.random.seed(0)  # else the plane list shuffles between identical runs
    cos_tol = np.cos(np.radians(normal_tol_deg))
    remaining = np.arange(len(xyz))
    planes = []
    while len(remaining) > min_inliers and len(planes) < max_planes:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[remaining]))
        model, inl = pcd.segment_plane(dist_thr, ransac_n=3, num_iterations=2000)
        inl = np.asarray(inl, dtype=np.int64)
        if len(inl) < min_inliers:
            break
        idx_all = remaining[inl]
        remaining = np.setdiff1d(remaining, idx_all, assume_unique=True)

        n = np.asarray(model[:3], dtype=np.float64)
        n /= np.linalg.norm(n)
        agree = np.abs((normals[idx_all] * n).sum(axis=1)) > cos_tol
        idx = idx_all[agree]
        if len(idx) < min_inliers:
            continue

        # keep the largest connected component -- a diagonal RANSAC cut through room clutter
        # spans a big bbox but shatters into crumbs, a real wall stays one blob
        comp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[idx]))
        lab = np.asarray(comp.cluster_dbscan(eps=cluster_eps, min_points=10))
        if (lab >= 0).any():
            idx = idx[lab == np.argmax(np.bincount(lab[lab >= 0]))]
        if len(idx) < min_inliers:
            continue

        centroid, n = plane_from_points(xyz[idx])
        # in-plane extent, to reject long thin slivers that RANSAC loves
        basis = plane_basis(xyz[idx], centroid)
        local = (xyz[idx] - centroid) @ basis.T
        extent = local.max(axis=0) - local.min(axis=0)
        if extent.min() < min_extent:
            continue
        w_obb, h_obb, area_obb, center_obb, uv_obb = fit_2d_oriented_bbox(local)
        planes.append(dict(idx=idx, centroid=centroid, normal=n, extent=extent,
                           area=occupied_area(local), bbox_dims=(w_obb, h_obb),
                           bbox_area=area_obb))
    return planes


def classify(planes, up, wall_tol_deg, floor_tol_deg):
    """Tag each plane horizontal / vertical / sloped, with its tilt off that ideal in degrees."""
    for p in planes:
        d = abs(float(np.dot(p["normal"], up)))
        from_horizontal = np.degrees(np.arccos(min(d, 1.0)))  # normal vs up
        from_vertical = np.degrees(np.arcsin(min(d, 1.0)))  # normal vs up-plane
        p["tilt_from_horizontal"] = from_horizontal
        p["tilt_from_vertical"] = from_vertical
        if from_horizontal <= floor_tol_deg:
            p["kind"] = "horizontal"
        elif from_vertical <= wall_tol_deg:
            p["kind"] = "vertical"
        else:
            p["kind"] = "sloped"
    return planes


def merge_groups(planes, merge_angle, merge_k, max_gap):
    """Union-find the plane fragments that are one physical surface.

    Same kind, near-parallel, and separated by less than a gap budget that scales with the
    *smaller* plane's size -- two 20x20 cm patches 30 cm apart are not the same surface, two
    3x3 m walls 30 cm apart are. Capped by `max_gap` so two parallel walls on opposite sides
    of a room can never fuse no matter how large they are."""
    parent = list(range(len(planes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            a, b = planes[i], planes[j]
            kind_a = "sloped" if a["kind"] in ("free", "sloped") else a["kind"]
            kind_b = "sloped" if b["kind"] in ("free", "sloped") else b["kind"]
            if kind_a != kind_b:
                continue
            dot = float(np.dot(a["normal"], b["normal"]))
            angle = np.degrees(np.arccos(np.clip(abs(dot), 0, 1)))
            if angle > merge_angle:
                continue
            mean_n = a["normal"] * np.sign(dot or 1.0) + b["normal"]
            mean_n /= np.linalg.norm(mean_n)
            gap = abs(float(np.dot(a["centroid"] - b["centroid"], mean_n)))
            thr = min(merge_k * np.sqrt(min(a["area"], b["area"])), max_gap)
            if gap <= thr:
                parent[find(i)] = find(j)
                pairs.append(dict(planes=[i, j], angle_deg=round(float(angle), 2),
                                  gap_m=round(gap, 4), threshold_m=round(float(thr), 3)))
    groups = {}
    for i in range(len(planes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values()), pairs


def azimuth_of(n, up, e1, e2):
    """Compass direction of a vertical plane, with any tilt off vertical removed."""
    h = n - np.dot(n, up) * up
    return float(np.degrees(np.arctan2(np.dot(h, e2), np.dot(h, e1))))


def wrap90(deg):
    return (deg + 45.0) % 90.0 - 45.0


def fit_groups(xyz, planes, groups, up, e1, e2, min_bbox_area=4.0):
    """One area-weighted plane per group: the merged normal, centroid, total area, and OBB."""
    out = []
    for members in groups:
        w = np.array([planes[m]["area"] for m in members])
        ref = planes[members[0]]["normal"]
        ns = np.array([planes[m]["normal"] * np.sign(np.dot(planes[m]["normal"], ref) or 1.0)
                       for m in members])
        n = (ns * w[:, None]).sum(axis=0)
        n /= np.linalg.norm(n)
        c = (np.array([planes[m]["centroid"] for m in members]) * w[:, None]).sum(axis=0) / w.sum()

        all_idx = np.concatenate([planes[m]["idx"] for m in members])
        pts = xyz[all_idx]
        basis = plane_basis(pts, c)
        local = (pts - c) @ basis.T
        w_obb, h_obb, area_obb, center_obb, uv_obb = fit_2d_oriented_bbox(local)

        keep = area_obb >= min_bbox_area
        out.append(dict(members=members, normal=n, centroid=c, area=float(w.sum()),
                        bbox_dims=(round(float(w_obb), 2), round(float(h_obb), 2)),
                        bbox_area=round(float(area_obb), 2),
                        keep=keep,
                        kind=planes[members[0]]["kind"]))
    return out


def filter_furniture_planes(fitted: list[dict], xyz: np.ndarray, max_furniture_gap: float = 0.70) -> None:
    """Identify and exclude vertical planes that are furniture (e.g. wardrobes, cabinets).

    If two parallel vertical planes face outward along the same direction, the one sitting
    closer to the room center with an offset <= max_furniture_gap (0.70m) from the outer
    perimeter wall is tagged as furniture. Real corridors and alcoves (gap > 0.70m) remain walls.
    """
    c_room = xyz.mean(axis=0)
    verticals = [g for g in fitted if g["kind"] == "vertical" and g.get("keep", True)]
    for i in range(len(verticals)):
        for j in range(len(verticals)):
            if i == j:
                continue
            g_inner, g_outer = verticals[i], verticals[j]
            n_in, n_out = g_inner["normal"], g_outer["normal"]
            dot = float(np.dot(n_in, n_out))
            if abs(dot) < 0.92:
                continue
            n_ref = n_out if dot > 0 else -n_out
            c_in, c_out = g_inner["centroid"], g_outer["centroid"]
            d_in = float(np.dot(c_in - c_room, n_ref))
            d_out = float(np.dot(c_out - c_room, n_ref))
            if d_out > d_in:
                gap = d_out - d_in
                if 0.10 <= gap <= max_furniture_gap:
                    g_inner["kind"] = "furniture"
                    g_inner["keep"] = False
                    g_inner["reason"] = f"furniture: {gap:.2f}m <= {max_furniture_gap:.2f}m in front of wall"


def snap_planes(xyz, normals, planes, up, e1, e2, args):
    """Merge fragments, then square the merged surfaces up: exactly horizontal or exactly
    vertical first, then one shared Manhattan frame so every wall pair sits at an exact
    multiple of 90 deg. The frame is solved globally (area-weighted circular mean in 4-theta
    space) rather than pair by pair -- snapping pairs one at a time fights itself, since
    forcing A onto B then B onto C pulls A back off."""
    groups, pairs = merge_groups(planes, args.merge_angle, args.merge_k, args.max_gap)
    min_bbox_area = getattr(args, "min_bbox_area", 4.0)
    max_furniture_gap = getattr(args, "max_furniture_gap", 0.70)
    fitted = fit_groups(xyz, planes, groups, up, e1, e2, min_bbox_area=min_bbox_area)
    filter_furniture_planes(fitted, xyz, max_furniture_gap=max_furniture_gap)

    # global Manhattan frame from the vertical groups, weighted by area
    verticals = [g for g in fitted if g["kind"] == "vertical" and g.get("keep", True)]
    phi = None
    if verticals:
        az = np.array([azimuth_of(g["normal"], up, e1, e2) for g in verticals])
        w = np.array([g["area"] for g in verticals])
        phi = float(np.degrees(np.angle((w * np.exp(4j * np.radians(az))).sum())) / 4.0)

    report = []
    for gid, g in enumerate(fitted):
        n = g["normal"]
        entry = dict(group=gid, kind=g["kind"], fragments=len(g["members"]),
                     area_m2=round(g["area"], 2),
                     bbox_area_m2=g["bbox_area"],
                     bbox_dims_m=list(g["bbox_dims"]),
                     inliers=int(sum(len(planes[m]["idx"]) for m in g["members"])),
                     centroid=[round(float(v), 3) for v in g["centroid"]])
        if not g.get("keep", True):
            g["applied"] = False
            entry["applied"] = False
            entry["reason"] = g.get("reason", f"bbox area {g['bbox_area']} m2 < {min_bbox_area} m2")
            report.append(entry)
            continue

        if g["kind"] == "horizontal":
            n_t = up * np.sign(np.dot(n, up) or 1.0)
            entry["target"] = "horizontal"
        elif g["kind"] == "vertical":
            az = azimuth_of(n, up, e1, e2)
            off = wrap90(az - phi)
            snapped = abs(off) <= args.manhattan_tol
            az_t = az - off if snapped else az
            n_t = np.cos(np.radians(az_t)) * e1 + np.sin(np.radians(az_t)) * e2
            n_t *= np.sign(np.dot(n_t, n) or 1.0)
            entry.update(target="manhattan" if snapped else "vertical only",
                         azimuth_deg=round(az, 2), azimuth_target_deg=round(az_t, 2),
                         off_frame_deg=round(off, 2))
        else:
            n_t = n
            entry["target"] = "sloped"

        entry["rotation_deg"] = round(float(np.degrees(
            np.arccos(np.clip(abs(np.dot(n, n_t)), 0, 1)))), 3)

        # every fragment ends up on ONE plane: rotate each about its own centroid, then slide
        # it along the target normal onto the group's shared offset
        w = np.array([planes[m]["area"] for m in g["members"]])
        d_target = float((w * np.array([np.dot(planes[m]["centroid"], n_t)
                                        for m in g["members"]])).sum() / w.sum())
        shifts = []
        for m in g["members"]:
            pm = planes[m]
            idx = pm["idx"]
            n_p = pm["normal"] * np.sign(np.dot(pm["normal"], n_t) or 1.0)
            R = rotation_between(n_p, n_t)
            xyz[idx] = _rotate(xyz[idx] - pm["centroid"], R) + pm["centroid"]
            normals[idx] = _rotate(normals[idx], R)
            shift = d_target - float(np.mean(xyz[idx] @ n_t))
            xyz[idx] += shift * n_t
            shifts.append(round(shift, 4))
        g["applied"] = True
        entry.update(applied=True, fragment_shifts_m=shifts,
                     normal=[round(float(v), 4) for v in n_t])
        report.append(entry)

    members = [np.concatenate([planes[m]["idx"] for m in g["members"]]) if g.get("applied")
               else np.empty(0, np.int64) for g in fitted]
    return dict(manhattan_frame_deg=None if phi is None else round(phi, 3),
                merged_pairs=pairs, groups=report), members


def is_point_inside_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized test if 2D points (N, 2) lie inside an N-vertex convex polygon (K, 2)."""
    edges = np.roll(poly, -1, axis=0) - poly
    crosses_hull = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    q_sign = 1.0 if crosses_hull[0] > 0 else -1.0

    inside = np.ones(len(pts), dtype=bool)
    for v_start, e_vec in zip(poly, edges):
        to_pts = pts - v_start
        cr = e_vec[0] * to_pts[:, 1] - e_vec[1] * to_pts[:, 0]
        if q_sign > 0:
            inside &= (cr >= -1e-5)
        else:
            inside &= (cr <= 1e-5)
    return inside


def is_point_inside_quad(pts: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Compatibility alias for is_point_inside_polygon."""
    return is_point_inside_polygon(pts, quad)


def fill_planes(vert, xyz, members, spacing, cell, close, max_hole, min_infill_distance: float = 0.03):
    """Infill empty areas within each plane's bounding polygon with an equidistant grid.

    Preserves ALL original points with their original photogrammetric colors, while generating
    new grid points (at `spacing`, e.g. 3cm) on the fitted plane strictly within the enclosing
    bounding polygon where no original points currently exist (empty areas / holes).
    Guarantees no points are added within `min_infill_distance` (default 3cm radius) of any
    existing inlier point.
    The newly added infill points are assigned the plane's exact uniform average color.
    """
    fresh, stats = [], []
    for gid, idx in enumerate(members):
        if len(idx) == 0:
            continue
        c, n = plane_from_points(xyz[idx])  # snapped planar surface
        u = np.cross(n, AXES[(int(np.argmax(np.abs(n))) + 1) % 3])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        d = xyz[idx] - c
        local = np.stack([d @ u, d @ v], axis=1)

        # 1. Fit up to 5-vertex minimum enclosing polygon on snapped inliers
        poly_2d, metrics = fit_minimum_enclosing_polygon(local, max_vertices=5)

        # 2. Build 2D grid covering the entire bounding polygon
        min_u, max_u = poly_2d[:, 0].min(), poly_2d[:, 0].max()
        min_v, max_v = poly_2d[:, 1].min(), poly_2d[:, 1].max()
        grid_u = np.arange(min_u, max_u + spacing, spacing)
        grid_v = np.arange(min_v, max_v + spacing, spacing)
        L = np.stack(np.meshgrid(grid_u, grid_v, indexing="ij"), axis=-1).reshape(-1, 2)

        # Filter points strictly inside the polygon
        in_poly = is_point_inside_polygon(L, poly_2d)
        L = L[in_poly]
        if len(L) == 0:
            continue

        # 3. Find empty areas: query nearest distance to existing inlier points in local 2D space
        # A candidate grid location is considered empty only if it is farther than max(0.85 * spacing, min_infill_distance)
        # from any existing inlier point (default 3cm radius) to avoid point clustering near original photogrammetric points.
        tree = cKDTree(local)
        dists, nn_idx = tree.query(L, k=1)
        empty_mask = dists >= max(0.85 * spacing, min_infill_distance)
        L_empty = L[empty_mask]
        if len(L_empty) == 0:
            continue

        # 4. Create new vertices for the empty areas
        nn_sample = nn_idx[empty_mask]
        new = vert[idx[nn_sample]].copy()

        # Compute the exact average RGB color of the original plane inliers
        if "red" in new.dtype.names and "green" in new.dtype.names and "blue" in new.dtype.names:
            avg_r = np.uint8(np.clip(np.round(np.mean(vert["red"][idx].astype(np.float64))), 0, 255))
            avg_g = np.uint8(np.clip(np.round(np.mean(vert["green"][idx].astype(np.float64))), 0, 255))
            avg_b = np.uint8(np.clip(np.round(np.mean(vert["blue"][idx].astype(np.float64))), 0, 255))
            new["red"] = avg_r
            new["green"] = avg_g
            new["blue"] = avg_b

        # 3D coordinates on the target plane
        pts_3d = c + L_empty[:, :1] * u + L_empty[:, 1:] * v
        for key, col in zip("xyz", pts_3d.T):
            new[key] = col
        for key, val in zip(("nx", "ny", "nz"), n):
            new[key] = val
        if "scale_u" in new.dtype.names:
            new["scale_u"] = new["scale_v"] = spacing * 0.5

        fresh.append(new)
        stats.append(dict(group=gid, original_pts=int(len(idx)), added=int(len(L_empty)),
                          polygon_area_m2=round(float(metrics["poly_area_m2"]), 2),
                          filled_m2=round(float(len(L_empty)) * spacing * spacing, 2)))

    if not fresh:
        return vert, stats

    # Append all new infill vertices while preserving ALL original vertices untouched
    all_vert = np.concatenate([vert] + fresh)
    return all_vert, stats


# --- storey alignment -------------------------------------------------------------------
# The plane snapping below cannot fix a whole floor that is yawed as a block: it works plane by
# plane, so a storey sitting 5 deg off its neighbour stays 5 deg off. This pass finds that block
# boundary and puts the upper storey back on the lower one.


def wall_yaw(normals, e1, e2):
    """Dominant wall azimuth mod 90 deg, plus the resultant length (0 = no wall structure)."""
    r = np.exp(4j * np.arctan2(normals @ e2, normals @ e1)).mean()
    return float(np.degrees(np.angle(r)) / 4.0), float(abs(r))


def find_storey_split(h, wall_normals_h, wall_normals, e1, e2, min_frac=0.12):
    """Split height whose two halves are each most self-consistent in wall azimuth.

    Scores sum of |resultant| weighted by half size -- a real storey boundary makes both
    halves sharper than any cut inside a single storey does."""
    lo, hi = np.percentile(h, [5, 95])
    best = (None, -1.0)
    for cut in np.arange(lo, hi, 0.05):
        a, b = wall_normals_h < cut, wall_normals_h >= cut
        if min(a.sum(), b.sum()) < min_frac * len(wall_normals_h):
            continue
        score = a.sum() * wall_yaw(wall_normals[a], e1, e2)[1] + b.sum() * wall_yaw(wall_normals[b], e1, e2)[1]
        if score > best[1]:
            best = (float(cut), score)
    return best[0]


def icp_2d(src, dst, theta0, pivot, iters=40, trim=0.7):
    """Trimmed 2D rigid ICP (yaw + slide) on the horizontal projection -- i.e. align the two
    storeys' floor-plan outlines. Trimmed because the storeys' footprints only partly overlap."""
    tree = cKDTree(dst)
    theta, t = theta0, np.zeros(2)
    for _ in range(iters):
        c, si = np.cos(theta), np.sin(theta)
        cur = np.stack([
            c * (src[:, 0] - pivot[0]) - si * (src[:, 1] - pivot[1]) + pivot[0] + t[0],
            si * (src[:, 0] - pivot[0]) + c * (src[:, 1] - pivot[1]) + pivot[1] + t[1],
        ], axis=1)
        d, j = tree.query(cur, workers=-1)
        keep = d <= np.quantile(d, trim)
        a, b = cur[keep], dst[j[keep]]
        ca, cb = a.mean(0), b.mean(0)
        a, b = a - ca, b - cb
        dtheta = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
        if abs(dtheta) < 1e-6 and np.linalg.norm(cb - ca) < 1e-5:
            theta += dtheta
            t += cb - ca
            break
        theta += dtheta
        c2, s2 = np.cos(dtheta), np.sin(dtheta)
        off = cur.mean(0) - pivot
        t += cb - ca + np.array([c2 * off[0] - s2 * off[1], s2 * off[0] + c2 * off[1]]) - off
    return theta, t


def align_storeys(xyz, normals, up, e1, e2, args):
    """Rigidly yaw+slide the upper storey onto the lower one, blended over `--blend` metres
    of height so the shared walls stay continuous instead of stepping at the seam."""
    h = xyz @ up
    wall = np.abs(normals @ up) < np.sin(np.radians(args.wall_tol))
    if wall.sum() < 2000:
        return None
    wh = h[wall]
    cut = args.split_height
    if cut is None:
        cut = find_storey_split(h, wh, normals[wall], e1, e2)
    if cut is None:
        return None

    below, above = wall & (h < cut), wall & (h >= cut)
    yaw_lo, _ = wall_yaw(normals[below], e1, e2)
    yaw_hi, _ = wall_yaw(normals[above], e1, e2)
    dyaw = wrap90(yaw_hi - yaw_lo)  # walls repeat every 90 deg
    if abs(dyaw) < args.min_storey_yaw and not args.force_storey:
        return dict(split_height=round(float(cut), 3), yaw_deg=round(dyaw, 3), applied=False,
                    reason=f"yaw gap {dyaw:.2f} deg below --min-storey-yaw")

    # 2D floor-plan ICP refines the yaw and recovers the slide the normals cannot see
    uv = np.stack([xyz @ e1, xyz @ e2], axis=1)
    rng = np.random.default_rng(0)
    sub = lambda m: uv[m][rng.choice(m.sum(), min(m.sum(), 20000), replace=False)]
    pivot = uv[above].mean(0)
    theta, t = icp_2d(sub(above), sub(below), np.radians(-dyaw), pivot, trim=args.icp_trim)

    # blend over the transition band: the data itself rotates over ~half a metre, and a hard
    # cut would leave a step in every wall that spans both storeys
    b = args.blend
    w = np.clip((h - (cut - b)) / (2 * b), 0.0, 1.0) if b > 0 else (h >= cut).astype(float)
    ang = w * theta
    ca, sa = np.cos(ang), np.sin(ang)
    origin = pivot[0] * e1 + pivot[1] * e2
    rel = xyz - origin
    cross = np.stack([up[1] * rel[:, 2] - up[2] * rel[:, 1],
                      up[2] * rel[:, 0] - up[0] * rel[:, 2],
                      up[0] * rel[:, 1] - up[1] * rel[:, 0]], axis=1)
    dot = rel @ up
    rot = rel * ca[:, None] + cross * sa[:, None] + up[None, :] * (dot * (1 - ca))[:, None]
    xyz[:] = rot + origin + (w * t[0])[:, None] * e1 + (w * t[1])[:, None] * e2

    ncross = np.stack([up[1] * normals[:, 2] - up[2] * normals[:, 1],
                       up[2] * normals[:, 0] - up[0] * normals[:, 2],
                       up[0] * normals[:, 1] - up[1] * normals[:, 0]], axis=1)
    ndot = normals @ up
    normals[:] = normals * ca[:, None] + ncross * sa[:, None] + up[None, :] * (ndot * (1 - ca))[:, None]

    return dict(split_height=round(float(cut), 3), yaw_from_normals_deg=round(dyaw, 3),
                yaw_applied_deg=round(float(np.degrees(theta)), 3),
                slide_m=[round(float(t[0]), 4), round(float(t[1]), 4)],
                blend_m=b, points_above=int((h >= cut).sum()), applied=True)


def regularize(src: Path, dst: Path, args) -> dict:
    ply = PlyData.read(str(src))
    vert = ply["vertex"].data.copy()
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float64)
    normals = np.stack([vert["nx"], vert["ny"], vert["nz"]], axis=1).astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    up = AXES[args.up] if args.up is not None else detect_up(normals)
    e1 = np.cross(up, AXES[(int(np.argmax(np.abs(up))) + 1) % 3])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    eps = args.cluster_eps
    if eps is None:  # 6x the cloud's own point spacing -- a fixed radius shreds sparse clouds
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[:: max(1, len(xyz) // 50000)]))
        eps = 6.0 * float(np.median(np.asarray(pcd.compute_nearest_neighbor_distance())))
    args.cluster_eps = eps

    storeys = None if args.no_storeys else align_storeys(xyz, normals, up, e1, e2, args)

    min_inliers = max(args.min_inliers, int(args.min_inlier_frac * len(xyz)))
    planes = extract_planes(
        xyz, normals, args.dist_thr, min_inliers, args.min_extent, args.normal_tol,
        args.max_planes, args.cluster_eps
    )
    classify(planes, up, args.wall_tol, args.floor_tol)
    snapped, members = snap_planes(xyz, normals, planes, up, e1, e2, args)

    report = dict(
        source=str(src), output=str(dst), points=int(len(xyz)),
        up_axis=[round(float(v), 5) for v in up], cluster_eps_m=round(eps, 4),
        storeys=storeys, **snapped,
    )
    for k, col in zip("xyz", xyz.T):
        vert[k] = col.astype(vert[k].dtype)
    for k, col in zip(("nx", "ny", "nz"), normals.T):
        vert[k] = col.astype(vert[k].dtype)
    if getattr(args, "fill", None):
        min_infill_dist = getattr(args, "fill_min_dist", 0.03)
        vert, report["fill"] = fill_planes(vert, xyz, members, args.fill, args.fill_cell,
                                           args.fill_close, args.fill_max_hole,
                                           min_infill_distance=min_infill_dist)
        report["points_out"] = int(len(vert))

        # Build diagnostic colored cloud from regularized and filled vertices
        colored_vert = _create_colored_vert(vert)
        colored_vert["red"][:] = UNASSIGNED_COLOR["rgb"][0]
        colored_vert["green"][:] = UNASSIGNED_COLOR["rgb"][1]
        colored_vert["blue"][:] = UNASSIGNED_COLOR["rgb"][2]

        # Color original member inliers
        for gid, idx in enumerate(members):
            if len(idx) == 0:
                continue
            color = PALETTE[gid % len(PALETTE)]["rgb"]
            colored_vert["red"][idx] = color[0]
            colored_vert["green"][idx] = color[1]
            colored_vert["blue"][idx] = color[2]

        # Color appended infill points
        n_orig = len(xyz)
        offset = n_orig
        for s in report["fill"]:
            gid = s["group"]
            color = PALETTE[gid % len(PALETTE)]["rgb"]
            colored_vert["red"][offset:offset + s["added"]] = color[0]
            colored_vert["green"][offset:offset + s["added"]] = color[1]
            colored_vert["blue"][offset:offset + s["added"]] = color[2]
            offset += s["added"]
    else:
        report["points_out"] = int(len(vert))
        colored_vert = _create_colored_vert(vert)
        colored_vert["red"][:] = UNASSIGNED_COLOR["rgb"][0]
        colored_vert["green"][:] = UNASSIGNED_COLOR["rgb"][1]
        colored_vert["blue"][:] = UNASSIGNED_COLOR["rgb"][2]
        for gid, idx in enumerate(members):
            if len(idx) == 0:
                continue
            color = PALETTE[gid % len(PALETTE)]["rgb"]
            colored_vert["red"][idx] = color[0]
            colored_vert["green"][idx] = color[1]
            colored_vert["blue"][idx] = color[2]

    # Save regularized point cloud (realistic RGB intact)
    PlyData([PlyElement.describe(vert, "vertex")], text=False).write(str(dst))

    # Save colored diagnostic point cloud (each plane surface has a distinct palette color)
    colored_ply_path = dst.with_name(dst.stem + "_planes_colored.ply")
    PlyData([PlyElement.describe(colored_vert, "vertex")], text=False).write(str(colored_ply_path))
    report["colored_ply"] = str(colored_ply_path)

    # Save companion markdown legend
    legend_md = build_legend_markdown(report)
    legend_path = dst.with_name("planes_legend.md")
    legend_path.write_text(legend_md)
    if dst.stem != "planes":
        dst.with_name(dst.stem + "_legend.md").write_text(legend_md)
    report["legend_file"] = str(legend_path)
    report["legend_md"] = legend_md
    return report


def selftest():
    """Synthetic room: a level floor, a 6 deg tilted mezzanine floor, two walls 4 deg off."""
    rng = np.random.default_rng(0)

    def slab(n, R, offset):
        p = np.zeros((n, 3))
        p[:, 0] = rng.uniform(-2, 2, n)
        p[:, 2] = rng.uniform(-2, 2, n)
        nrm = np.tile([0.0, 1.0, 0.0], (n, 1))
        return p @ R.T + offset, nrm @ R.T

    def wall(n, R, offset):
        p = np.zeros((n, 3))
        p[:, 0] = rng.uniform(-2, 2, n)
        p[:, 1] = rng.uniform(0, 2.4, n)
        nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
        return p @ R.T + offset, nrm @ R.T

    def rot(axis, deg):
        return o3d.geometry.get_rotation_matrix_from_axis_angle(AXES[axis] * np.radians(deg))

    parts = [
        slab(8000, np.eye(3), [0, 0, 0]),
        slab(8000, rot(0, 6.0), [0, 2.6, 0]),
        wall(8000, np.eye(3), [0, 0, -2]),
        wall(8000, rot(1, 4.0), [2, 0, 0]),
    ]
    xyz = np.vstack([p for p, _ in parts])
    nrm = np.vstack([n for _, n in parts])
    xyz += rng.normal(0, 0.004, xyz.shape)

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    arr = np.zeros(len(xyz), dtype=dtype)
    for i, k in enumerate("xyz"):
        arr[k] = xyz[:, i]
    for i, k in enumerate(("nx", "ny", "nz")):
        arr[k] = nrm[:, i]
    tmp = Path("/tmp/regularize_selftest.ply")
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(tmp))

    args = argparse.Namespace(
        no_storeys=False, split_height=None, min_storey_yaw=0.5, force_storey=False,
        blend=0.25, icp_trim=0.7, merge_angle=10.0, merge_k=0.5, max_gap=0.35,
        manhattan_tol=15.0,
        up=1, dist_thr=0.02, cluster_eps=None, min_inliers=1000, min_inlier_frac=0.01,
        min_extent=0.5, normal_tol=25.0, max_planes=12, wall_tol=35.0, floor_tol=20.0,
    )
    rep = regularize(tmp, Path("/tmp/regularize_selftest_out.ply"), args)
    kinds = [g["kind"] for g in rep["groups"]]
    assert kinds.count("horizontal") == 2, kinds
    assert kinds.count("vertical") == 2, kinds

    out = PlyData.read("/tmp/regularize_selftest_out.ply")["vertex"].data
    pts = np.stack([out["x"], out["y"], out["z"]], axis=1).astype(np.float64)
    up = AXES[1]
    normals_out = [plane_from_points(pts[lo:lo + 8000])[1] for lo in (0, 8000, 16000, 24000)]
    for i, n in enumerate(normals_out[:2]):  # both floors exactly horizontal
        off = np.degrees(np.arccos(min(abs(float(np.dot(n, up))), 1.0)))
        assert off < 0.2, f"floor {i} still {off:.2f} deg off horizontal"
    for i, n in enumerate(normals_out[2:]):  # both walls exactly vertical...
        off = np.degrees(np.arcsin(min(abs(float(np.dot(n, up))), 1.0)))
        assert off < 0.2, f"wall {i} still {off:.2f} deg off vertical"
    # ...and parallel to each other (family mean, not necessarily the world axis)
    between = np.degrees(np.arccos(min(abs(float(np.dot(*normals_out[2:]))), 1.0)))
    assert between < 0.2, f"walls {between:.2f} deg apart, expected parallel"
    print("selftest ok:", [(g["kind"], g["rotation_deg"]) for g in rep["groups"]])


def selftest_storeys():
    """Two-storey room whose upper block is yawed 5 deg and slid 8 cm -- the real failure:
    per-plane snapping cannot see it, the storey pass must put the block back."""
    rng = np.random.default_rng(1)
    up = AXES[1]  # same right-handed basis regularize() builds; X,Z here would be left-handed
    e1 = np.cross(up, AXES[2])
    e2 = np.cross(up, e1)

    def room(n, y0, y1, yaw_deg, shift):
        """4 walls of a 4x4 m room plus its floor, yawed about up and slid."""
        pts, nrm = [], []
        for k, (nx, nz, off) in enumerate([(1, 0, 2), (-1, 0, -2), (0, 1, 2), (0, -1, -2)]):
            m = n // 5
            a = rng.uniform(-2, 2, m)
            p = np.zeros((m, 3))
            p[:, 0] = off if nx else a
            p[:, 2] = off if nz else a
            p[:, 1] = rng.uniform(y0, y1, m)
            pts.append(p)
            nrm.append(np.tile([nx, 0.0, nz], (m, 1)))
        m = n // 5
        f = np.zeros((m, 3))
        f[:, 0], f[:, 2], f[:, 1] = rng.uniform(-2, 2, m), rng.uniform(-2, 2, m), y0
        pts.append(f)
        nrm.append(np.tile([0.0, 1.0, 0.0], (m, 1)))
        P, N = np.vstack(pts), np.vstack(nrm)
        c, si = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        R = np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]])
        return P @ R.T + shift, N @ R.T

    lo = room(40000, -1.5, 1.0, 0.0, [0, 0, 0])
    hi = room(30000, 1.0, 2.6, 5.0, [0.08, 0, -0.05])
    xyz = np.vstack([lo[0], hi[0]]) + rng.normal(0, 0.004, (70000, 3))
    nrm = np.vstack([lo[1], hi[1]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    args = argparse.Namespace(wall_tol=35.0, split_height=None, min_storey_yaw=0.5,
                              force_storey=False, blend=0.25, icp_trim=0.7)
    rep = align_storeys(xyz, nrm, up, e1, e2, args)
    assert rep and rep["applied"], rep
    assert abs(rep["split_height"] - 1.0) < 0.2, rep
    assert abs(abs(rep["yaw_applied_deg"]) - 5.0) < 0.5, rep

    h = xyz @ up
    wall = np.abs(nrm @ up) < 0.5
    y_lo = wall_yaw(nrm[wall & (h < 0.5)], e1, e2)[0]
    y_hi = wall_yaw(nrm[wall & (h > 1.4)], e1, e2)[0]
    gap = (y_hi - y_lo + 45.0) % 90.0 - 45.0
    assert abs(gap) < 0.5, f"storeys still {gap:.2f} deg apart"
    # and the slide is gone: upper walls sit on the lower walls' outline in plan
    plan_lo = np.abs(np.abs(xyz[wall & (h < 0.5), 0]) - 2.0)
    plan_hi = np.abs(np.abs(xyz[wall & (h > 1.4), 0]) - 2.0)
    assert np.median(plan_hi) - np.median(plan_lo) < 0.02, (np.median(plan_hi), np.median(plan_lo))
    print(f"storey selftest ok: split {rep['split_height']:.2f} m, "
          f"yaw {rep['yaw_applied_deg']:.2f} deg, slide {rep['slide_m']}, residual {gap:.2f} deg")


def selftest_merge():
    """One wall split into two fragments 4 cm apart and 3 deg off, plus a second wall 86 deg
    from it. The fragments must fuse into one group and the two walls must end up at exactly
    90 deg -- this is the pair the per-plane snapping could never fix."""
    rng = np.random.default_rng(3)
    up = AXES[1]
    e1 = np.cross(up, AXES[2])
    e2 = np.cross(up, e1)

    def patch(n, normal, origin, span_a, span_b):
        """Rectangle through `origin` spanned by two axes orthogonal to `normal`."""
        normal = normal / np.linalg.norm(normal)
        aux = AXES[(int(np.argmax(np.abs(normal))) + 1) % 3]  # never parallel to `normal`
        a = np.cross(normal, aux)
        a /= np.linalg.norm(a)
        b = np.cross(normal, a)
        pts = (origin + np.outer(rng.uniform(*span_a, n), a) + np.outer(rng.uniform(*span_b, n), b))
        return pts, np.tile(normal, (n, 1))

    def yaw(v, deg):
        c, si = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, 0, si], [0, 1, 0], [-si, 0, c]]) @ v

    parts = [
        patch(9000, [1, 0, 0], [0, 0, -1.0], (-2, 2), (-1.2, 1.2)),          # wall A, lower half
        patch(9000, yaw([1, 0, 0], 3.0), [0.04, 0, 1.0], (-2, 2), (-1.2, 1.2)),  # A, upper, off
        patch(9000, yaw([0, 0, 1], -4.0), [0, 0, 2.0], (-2, 2), (-1.2, 1.2)),    # wall B, 86 deg
        patch(9000, [0, 1, 0], [0, -1.5, 0], (-2, 2), (-2, 2)),              # floor
    ]
    xyz = np.vstack([q for q, _ in parts]) + rng.normal(0, 0.003, (36000, 3))
    nrm = np.vstack([n for _, n in parts])

    args = argparse.Namespace(merge_angle=10.0, merge_k=0.5, max_gap=0.35, manhattan_tol=15.0)
    planes = extract_planes(xyz, nrm, 0.02, 1500, 0.5, 20.0, 12, 0.12)
    classify(planes, up, 35.0, 20.0)
    rep, _ = snap_planes(xyz, nrm, planes, up, e1, e2, args)

    fused = [g for g in rep["groups"] if g["fragments"] > 1]
    assert len(fused) == 1 and fused[0]["kind"] == "vertical", rep["groups"]
    walls = [g for g in rep["groups"] if g["kind"] == "vertical"]
    assert len(walls) == 2, walls
    ang = np.degrees(np.arccos(np.clip(abs(np.dot(walls[0]["normal"], walls[1]["normal"])), 0, 1)))
    assert abs(ang - 90.0) < 0.05, f"walls at {ang:.3f} deg, expected 90"
    # the fused fragments really are one plane now, not two 4 cm apart
    print(f"merge selftest ok: fused {fused[0]['fragments']} fragments "
          f"(shifts {fused[0]['fragment_shifts_m']}), walls now {ang:.3f} deg apart")


def selftest_fill():
    """A 2x2 m floor with a 0.3 m sampling gap and a 1 m2 stairwell: close the gap, keep the
    stairwell, land every point exactly on the plane, and do not grow the outline."""
    rng = np.random.default_rng(0)
    p = rng.uniform([0, 0], [2, 2], size=(40000, 2))
    keep = ~(((p[:, 0] > 0.4) & (p[:, 0] < 0.7) & (p[:, 1] > 0.4) & (p[:, 1] < 0.7))
             | ((p[:, 0] > 1.0) & (p[:, 1] > 1.0)))  # small gap, then a 1 m2 opening
    p = p[keep]
    xyz = np.stack([p[:, 0], rng.normal(0, 0.005, len(p)), p[:, 1]], axis=1)  # 5 mm of noise
    vert = np.zeros(len(p), dtype=[(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")]
                    + [(k, "u1") for k in ("red", "green", "blue")])
    out, stats = fill_planes(vert, xyz, [np.arange(len(p))], 0.05, 0.08, 1, 0.25)

    q = np.stack([out["x"], out["z"]], axis=1)
    assert len(stats) == 1 and stats[0]["removed"] == len(p)
    xyz_out = np.stack([out["x"], out["y"], out["z"]], axis=1).astype(np.float64)
    c, n = plane_from_points(xyz_out)  # the fit tilts ~1e-4 rad off Y; flatness is what matters
    assert np.abs((xyz_out - c) @ n).max() < 1e-6, "filled points must lie on one plane"
    assert np.allclose(np.abs(out["ny"]), 1.0), "filled normals must be the plane normal"
    gap = ((q[:, 0] > 0.45) & (q[:, 0] < 0.65) & (q[:, 1] > 0.45) & (q[:, 1] < 0.65)).sum()
    hole = ((q[:, 0] > 1.2) & (q[:, 1] > 1.2)).sum()
    assert gap > 50, f"sampling gap not filled ({gap} pts)"
    assert hole == 0, f"stairwell was filled in ({hole} pts)"
    assert q.min() > -0.12 and q.max() < 2.12, "outline grew"
    d = np.sort(cKDTree(q).query(q, k=2)[0][:, 1])
    assert abs(np.median(d) - 0.02) < 1e-6, f"lattice not equidistant (median {np.median(d):.4f})"
    print(f"fill selftest ok: {stats[0]['removed']} -> {stats[0]['added']} pts, "
          f"{stats[0]['filled_m2']} m2, spacing {np.median(d):.3f} m")


def selftest_sloped():
    """Sloped roof (30 deg pitch) with two coplanar fragments and a hole, plus small clutter."""
    rng = np.random.default_rng(4)

    # 30 deg rotation about X
    c30, s30 = np.cos(np.radians(30)), np.sin(np.radians(30))
    R_pitch = np.array([[1, 0, 0], [0, c30, -s30], [0, s30, c30]])
    nrm_roof = R_pitch @ np.array([0.0, 1.0, 0.0])

    def roof_patch(n, x_span, z_span, offset):
        p = np.zeros((n, 3))
        p[:, 0] = rng.uniform(*x_span, n)
        p[:, 2] = rng.uniform(*z_span, n)
        return p @ R_pitch.T + offset, np.tile(nrm_roof, (n, 1))

    # Two fragments of roof (each 2.5m x 1.5m = 3.75 m2; combined > 7 m2)
    frag1 = roof_patch(6000, (-2.5, 0.0), (-1.5, 1.5), [0, 2.0, 0])
    frag2 = roof_patch(6000, (0.1, 2.5), (-1.5, 1.5), [0, 2.0, 0])
    # Small clutter patch (1.0m x 1.0m = 1.0 m2, should be discarded by 4.0 m2 threshold)
    clutter = roof_patch(1500, (-0.5, 0.5), (-0.5, 0.5), [0, 0.0, 0])

    xyz = np.vstack([frag1[0], frag2[0], clutter[0]]) + rng.normal(0, 0.003, (13500, 3))
    nrm = np.vstack([frag1[1], frag2[1], clutter[1]])

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    arr = np.zeros(len(xyz), dtype=dtype)
    for i, k in enumerate("xyz"):
        arr[k] = xyz[:, i]
    for i, k in enumerate(("nx", "ny", "nz")):
        arr[k] = nrm[:, i]

    tmp = Path("/tmp/regularize_sloped_test.ply")
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(tmp))

    args = default_args(up=1, min_bbox_area=4.0, fill=0.04, fill_max_hole=1.5, no_storeys=True)
    rep = regularize(tmp, Path("/tmp/regularize_sloped_out.ply"), args)

    # Check that roof group was kept and filled, clutter was rejected
    applied_groups = [g for g in rep["groups"] if g.get("applied")]
    assert len(applied_groups) == 1, rep["groups"]
    assert applied_groups[0]["kind"] == "sloped"
    assert applied_groups[0]["bbox_area_m2"] >= 4.0
    print(f"sloped roof selftest ok: group {applied_groups[0]['group']} {applied_groups[0]['bbox_area_m2']} m2 applied")


def report_fill(rep):
    return rep.get("fill") or []


def planes_of(rep):
    return [m for g in rep["groups"] for m in range(g["fragments"])]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", type=Path)
    ap.add_argument("-o", "--output", type=Path)
    ap.add_argument("--up", type=int, choices=[0, 1, 2], default=1,
                    help="up axis index (default: 1, +Y axis)")
    ap.add_argument("--dist-thr", type=float, default=0.02, help="RANSAC inlier distance, m")
    ap.add_argument("--cluster-eps", type=float, help="connectivity radius, m (default: auto)")
    ap.add_argument("--min-inliers", type=int, default=800)
    ap.add_argument("--min-inlier-frac", type=float, default=0.0)
    ap.add_argument("--min-extent", type=float, default=0.5, help="smallest in-plane side, m")
    ap.add_argument("--min-bbox-area", type=float, default=3.8,
                    help="discard candidate planes with 2D OBB area below this, m2")
    ap.add_argument("--max-furniture-gap", type=float, default=0.70,
                    help="parallel vertical planes within this offset from a perimeter wall are furniture, m")
    ap.add_argument("--normal-tol", type=float, default=20.0, help="surfel-normal agreement, deg")
    ap.add_argument("--max-planes", type=int, default=40)
    ap.add_argument("--wall-tol", type=float, default=35.0, help="snap to vertical below this tilt")
    ap.add_argument("--floor-tol", type=float, default=20.0, help="snap to horizontal below this")
    ap.add_argument("--merge-angle", type=float, default=10.0, help="fragments parallel within, deg")
    ap.add_argument("--merge-k", type=float, default=0.5, help="gap budget per sqrt(area), m")
    ap.add_argument("--max-gap", type=float, default=0.35, help="absolute cap on the merge gap, m")
    ap.add_argument("--manhattan-tol", type=float, default=15.0,
                    help="snap a wall to the 90 deg frame when it is within this, deg")
    ap.add_argument("--no-storeys", action="store_true", help="skip the storey alignment pass")
    ap.add_argument("--split-height", type=float, help="storey boundary along up, m (default: auto)")
    ap.add_argument("--min-storey-yaw", type=float, default=0.5, help="below this, leave storeys alone")
    ap.add_argument("--force-storey", action="store_true")
    ap.add_argument("--blend", type=float, default=0.25, help="half-width of the transition band, m")
    ap.add_argument("--icp-trim", type=float, default=0.7, help="correspondence keep fraction")
    ap.add_argument("--fill", type=float, help="re-sample snapped surfaces at this spacing, m")
    ap.add_argument("--fill-min-dist", type=float, default=0.03,
                    help="minimum distance from existing inlier points to allow infill, m (default: 0.03 = 3cm)")
    ap.add_argument("--fill-cell", type=float, default=0.08, help="footprint cell size, m")
    ap.add_argument("--fill-close", type=int, default=1, help="footprint closing radius, cells")
    ap.add_argument("--fill-max-hole", type=float, default=1.5,
                    help="interior gaps up to this area get filled; bigger ones stay open, m2")
    ap.add_argument("--selftest", action="store_true")
    return ap


def default_args(**overrides):
    """The CLI defaults as a namespace, so the pipeline can call regularize() directly."""
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def diagnose_planes_quads(
    depth_ply_path: Path,
    output_dir: Path,
    artifact_dir: Path | None = None,
    min_area: float = 2.0,
    max_vertices: int = 5,
) -> dict:
    """Diagnostic up-to-5-vertex non-orthogonal minimum enclosing polygon plane analysis.

    Extracts dominant planes, fits a free polygon up to 5 vertices (maximizing occupied area ratio),
    renders 2D/3D visual inspection artifacts and generates planes_quad_inspection.md.
    Never alters or mutates the input surfel cloud.
    """
    import os
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ply = PlyData.read(str(depth_ply_path))
    vert = ply["vertex"].data.copy()
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float64)
    normals = np.stack([vert["nx"], vert["ny"], vert["nz"]], axis=1).astype(np.float64)
    n_len = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    normals /= n_len

    up = detect_up(normals)
    eps = None
    try:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[:: max(1, len(xyz) // 40000)]))
        eps = 6.0 * float(np.median(np.asarray(pcd.compute_nearest_neighbor_distance())))
    except Exception:
        eps = 0.08

    planes = extract_planes(xyz, normals, dist_thr=0.025, min_inliers=1000, min_extent=0.35,
                            normal_tol_deg=22.0, max_planes=30, cluster_eps=eps)
    classify(planes, up, wall_tol_deg=35.0, floor_tol_deg=20.0)

    kept = [p for p in planes if p["area"] >= min_area]
    kept.sort(key=lambda p: -p["area"])

    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    catalog = []
    wireframe_pts = []
    wireframe_colors = []
    colored_vert = _create_colored_vert(vert)

    # Base background dimming for diagnostic PLY
    for c in ("red", "green", "blue"):
        colored_vert[c] = (colored_vert[c] * 0.20 + 40).astype(np.uint8)

    md_lines = [
        "# Plane Non-Orthogonal Bounding Enclosures (Polygon Inspection)",
        "",
        f"This artifact presents the **up to {max_vertices}-vertex minimum enclosing polygon** analysis for each detected plane in the scene.",
        f"- **Enclosure Geometry**: Free convex polygon $[V_0, \\dots, V_{{k-1}}]$ ($k \\le {max_vertices}$) with arbitrary non-90° angles.",
        "- **Objective**: Minimizes enclosing area to **maximize internal occupied area ratio** $\\frac{\\text{Occupied Area}}{\\text{Polygon Area}}$.",
        "- **Input Cloud**: [`points3D_depth.ply`](file://" + str(depth_ply_path) + ") *(geometry untouched/unaltered)*.",
        "",
        "---",
        "",
        "## Summary Catalog",
        "",
        "| ID | Swatch | Kind | Color | Inliers | Occupied Area | Enclosure Area | Vertices | Occupied Ratio | Side Lengths | Corner Angles | Normal |",
        "| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for i, p in enumerate(kept):
        palette_entry = PALETTE[i % len(PALETTE)]
        color_rgb = palette_entry["rgb"]
        color_name = palette_entry["name"]
        swatch = palette_entry["emoji"]

        # Color inliers in colored_vert
        for c, val in zip(("red", "green", "blue"), color_rgb):
            colored_vert[c][p["idx"]] = val

        centroid = p["centroid"]
        basis = plane_basis(xyz[p["idx"]], centroid)  # (2, 3)
        local = (xyz[p["idx"]] - centroid) @ basis.T  # (N, 2)

        poly_2d, metrics = fit_minimum_enclosing_polygon(local, max_vertices=max_vertices)
        k_verts = len(poly_2d)
        poly_3d = centroid + poly_2d[:, 0:1] * basis[0:1] + poly_2d[:, 1:2] * basis[1:2]  # (K, 3)

        # Generate 3D wireframe boundary points along edges
        n_interp = 80
        for edge_idx in range(k_verts):
            p_start = poly_3d[edge_idx]
            p_end = poly_3d[(edge_idx + 1) % k_verts]
            t_vals = np.linspace(0, 1, n_interp, endpoint=False)[:, None]
            pts_edge = p_start[None, :] * (1.0 - t_vals) + p_end[None, :] * t_vals
            wireframe_pts.append(pts_edge)
            # Bright white wireframe
            wireframe_colors.append(np.full((n_interp, 3), 255, dtype=np.uint8))

        # Render 2D Matplotlib diagnostic plot
        fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
        ax.set_facecolor("#1A1A1A")
        fig.patch.set_facecolor("#121212")

        # Scatter local inliers (subsample if huge)
        sample_step = max(1, len(local) // 5000)
        ax.scatter(local[::sample_step, 0], local[::sample_step, 1], s=3,
                   c=[np.array(color_rgb) / 255.0], alpha=0.45, label="Surfel Inliers")

        # Convex hull
        try:
            hull = ConvexHull(local)
            hull_closed = np.vstack([local[hull.vertices], local[hull.vertices[0]]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], "--", color="#888888",
                    linewidth=1.2, label="Convex Hull", alpha=0.7)
        except Exception:
            pass

        # Polygon
        poly_closed = np.vstack([poly_2d, poly_2d[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], "-", color=palette_entry["hex"],
                linewidth=2.8, label=f"{k_verts}-Vertex Enclosure")
        ax.fill(poly_2d[:, 0], poly_2d[:, 1], color=palette_entry["hex"], alpha=0.15)

        # Annotate vertices & corner angles
        for v_idx, (vx, vy) in enumerate(poly_2d):
            ax.plot(vx, vy, "o", color="#FFFFFF", markeredgecolor=palette_entry["hex"],
                    markersize=8, markeredgewidth=2)
            ang = metrics["internal_angles_deg"][v_idx]
            ax.annotate(f"V{v_idx}\n({ang:.1f}°)", (vx, vy), textcoords="offset points",
                        xytext=(8, 8), color="#FFFFFF", fontsize=8.5, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#2A2A2A", ec=palette_entry["hex"], alpha=0.9))

        # Side length annotations
        for e_idx in range(k_verts):
            mid_x = 0.5 * (poly_2d[e_idx, 0] + poly_2d[(e_idx + 1) % k_verts, 0])
            mid_y = 0.5 * (poly_2d[e_idx, 1] + poly_2d[(e_idx + 1) % k_verts, 1])
            edge_l = metrics["edge_lengths_m"][e_idx]
            ax.text(mid_x, mid_y, f"{edge_l:.2f}m", color="#FFD700", fontsize=7.5,
                    ha="center", va="center", bbox=dict(boxstyle="square,pad=0.15", fc="#1A1A1A", ec="#555555", alpha=0.85))

        ax.set_title(f"Plane #{i}: {p['kind'].upper()} ({color_name})\n"
                     f"Occupied: {metrics['occupied_area_m2']:.2f} m² | Enclosure: {metrics['poly_area_m2']:.2f} m² ({k_verts}-gon) | "
                     f"Occupied Ratio: {metrics['occupied_ratio']*100:.1f}%",
                     color="#FFFFFF", fontsize=10.5, pad=10)
        ax.set_xlabel("Local In-Plane Axis $U$ (m)", color="#CCCCCC", fontsize=8.5)
        ax.set_ylabel("Local In-Plane Axis $V$ (m)", color="#CCCCCC", fontsize=8.5)
        ax.tick_params(colors="#AAAAAA", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(True, linestyle=":", color="#333333", alpha=0.6)
        ax.legend(loc="upper right", facecolor="#222222", edgecolor="#444444", labelcolor="#FFFFFF", fontsize=8)
        ax.set_aspect("equal", "datalim")

        fig_filename = f"plane_{i}_quad.png"
        fig_path = preview_dir / fig_filename
        plt.tight_layout()
        plt.savefig(fig_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        if artifact_dir:
            shutil.copy2(fig_path, artifact_dir / fig_filename)

        side_lens_str = ", ".join(f"{l:.2f}m" for l in metrics["edge_lengths_m"])
        angles_str = ", ".join(f"{a:.1f}°" for a in metrics["internal_angles_deg"])
        n_vec = [round(float(v), 3) for v in p["normal"]]

        md_lines.append(
            f"| **#{i}** | {swatch} | **{p['kind'].capitalize()}** | {color_name} | {len(p['idx']):,} | "
            f"`{metrics['occupied_area_m2']:.2f} m²` | `{metrics['poly_area_m2']:.2f} m²` | `{k_verts}` | "
            f"**`{metrics['occupied_ratio']*100:.1f}%`** | `{side_lens_str}` | `{angles_str}` | `{n_vec}` |"
        )

        catalog.append({
            "plane_id": i,
            "kind": p["kind"],
            "color": color_name,
            "inliers": int(len(p["idx"])),
            "centroid_3d": [round(float(c), 4) for c in centroid],
            "normal_3d": n_vec,
            "vertices_3d": [[round(float(coord), 4) for coord in v3] for v3 in poly_3d],
            "vertices_2d": [[round(float(coord), 4) for coord in v2] for v2 in poly_2d],
            "metrics": metrics,
            "plot_image": fig_filename,
        })

    # Add per-plane visual sections to markdown report
    md_lines += [
        "",
        "---",
        "",
        "## Detailed Plane Enclosure Visualizations",
        "",
    ]

    for entry in catalog:
        pid = entry["plane_id"]
        cname = entry["color"]
        kind = entry["kind"].capitalize()
        img_name = entry["plot_image"]
        m = entry["metrics"]
        v3 = entry["vertices_3d"]
        k_verts = len(v3)

        img_rel = f"preview/{img_name}" if not artifact_dir else img_name
        angles_formula = ", ".join([f"\\alpha_{j} = {m['internal_angles_deg'][j]}^\\circ" for j in range(k_verts)])
        edges_formula = ", ".join([f"e_{{{j}{(j+1)%k_verts}}} = {m['edge_lengths_m'][j]:.2f}\\text{{ m}}" for j in range(k_verts)])

        v_rows = []
        for j, coord in enumerate(v3):
            v_rows.append(f"| $V_{j}$ | `{coord[0]:.3f}` | `{coord[1]:.3f}` | `{coord[2]:.3f}` |")
        v_rows_str = "\n".join(v_rows)

        md_lines += [
            f"### Plane #{pid}: {kind} ({cname})",
            "",
            f"![Plane #{pid} {k_verts}-Vertex Enclosure]({img_rel})",
            "",
            f"- **Occupied Surface Area**: `{m['occupied_area_m2']:.2f} m²`",
            f"- **{k_verts}-Vertex Enclosure Area**: `{m['poly_area_m2']:.2f} m²`",
            f"- **Internal Occupied Area Ratio**: **`{m['occupied_ratio']*100:.1f}%`**",
            f"- **Corner Angles**: ${angles_formula}$",
            f"- **Edge Lengths**: ${edges_formula}$",
            "",
            f"**3D World Vertex Coordinates ($V_0 \\dots V_{{{k_verts-1}}}$):**",
            "",
            "| Vertex | X (m) | Y (m) | Z (m) |",
            "| :---: | :---: | :---: | :---: |",
            f"{v_rows_str}",
            "",
            "---",
            "",
        ]

    # Write markdown inspection artifact
    report_text = "\n".join(md_lines)
    report_file = output_dir / "planes_quad_inspection.md"
    report_file.write_text(report_text + "\n")
    if artifact_dir:
        (artifact_dir / "planes_quad_inspection.md").write_text(report_text + "\n")

    # Export 3D PLY with colored planes + 3D line wireframes
    if wireframe_pts:
        all_wf_pts = np.concatenate(wireframe_pts, axis=0).astype(np.float32)
        all_wf_colors = np.concatenate(wireframe_colors, axis=0).astype(np.uint8)
        
        # Build combined PLY vertex array
        n_orig = len(colored_vert)
        n_wf = len(all_wf_pts)
        total_v = n_orig + n_wf

        comb_descr = list(colored_vert.dtype.descr)
        comb_arr = np.zeros(total_v, dtype=comb_descr)
        for name in colored_vert.dtype.names:
            comb_arr[name][:n_orig] = colored_vert[name]

        comb_arr["x"][n_orig:] = all_wf_pts[:, 0]
        comb_arr["y"][n_orig:] = all_wf_pts[:, 1]
        comb_arr["z"][n_orig:] = all_wf_pts[:, 2]
        comb_arr["red"][n_orig:] = all_wf_colors[:, 0]
        comb_arr["green"][n_orig:] = all_wf_colors[:, 1]
        comb_arr["blue"][n_orig:] = all_wf_colors[:, 2]

        quad_ply_path = output_dir / "points3D_depth_planes_quads.ply"
        PlyData([PlyElement.describe(comb_arr, "vertex")], text=False).write(str(quad_ply_path))
    else:
        quad_ply_path = output_dir / "points3D_depth_planes_quads.ply"
        PlyData([PlyElement.describe(colored_vert, "vertex")], text=False).write(str(quad_ply_path))

    return {
        "report_path": str(report_file),
        "quad_ply_path": str(quad_ply_path),
        "planes": catalog,
    }


def main():
    args = build_parser().parse_args()

    if args.selftest:
        selftest()
        selftest_storeys()
        selftest_merge()
        selftest_fill()
        selftest_sloped()
        return
    src = args.source
    dst = args.output or src.with_name(src.stem + "_planes_aligned.ply")
    rep = regularize(src, dst, args)
    if rep["storeys"]:
        print("storeys:", json.dumps(rep["storeys"]))
    Path(str(dst.with_suffix("")) + "_report.json").write_text(json.dumps(rep, indent=2))
    print(f"up axis {rep['up_axis']}  points {rep['points']}  "
          f"planes {len(planes_of(rep))} -> groups {len(rep['groups'])}  "
          f"manhattan frame {rep['manhattan_frame_deg']} deg")
    for m in rep["merged_pairs"]:
        print(f"  merged #{m['planes'][0]} + #{m['planes'][1]}: {m['angle_deg']} deg apart, "
              f"gap {m['gap_m']} m <= {m['threshold_m']} m")
    for g in rep["groups"]:
        if not g.get("applied"):
            continue
        print(f"  group {g['group']:2d} {g['kind']:<10} {g['fragments']}frag "
              f"{g['area_m2']:6.2f} m2 (OBB {g['bbox_area_m2']:5.2f} m2) {g['inliers']:7d} pts  "
              f"{g.get('target', ''):<14} rotated {g['rotation_deg']:5.2f} deg  "
              f"shifts {g['fragment_shifts_m']}")
    for f in report_fill(rep):
        print(f"  filled group {f['group']:2d}: {f['removed']:6d} pts -> {f['added']:6d} "
              f"over {f['filled_m2']:6.2f} m2")
    print(f"wrote regularized cloud: {dst}")
    if rep.get("colored_ply"):
        print(f"wrote diagnostic colored cloud: {rep['colored_ply']}")
    if rep.get("legend_file"):
        print(f"wrote planes legend: {rep['legend_file']}")


if __name__ == "__main__":
    main()
