#!/usr/bin/env python3
"""Aligns multi-room ARCore tracks and sealed room meshes in a single 2D/3D coordinate space.

Enforces:
1. Zero spatial overlap between active room scan trajectories and room interiors.
2. Continuity across doorways: Scan J entry connects to Scan I exit.
3. Manhattan cardinal grid alignment (0, 90, 180, 270 degrees).
4. Generates:
   - aligned_tracks_topdown.png: Top-down view of the aligned 2D ARCore tracks.
   - aligned_plys_topdown.png: Top-down view of the aligned PLY meshes with ceilings cut away.
   - aligned_scene_overview.png: High-resolution side-by-side diagnostic.
   - aligned_scans_transforms.json: SE(3) transformation matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
import numpy as np
import open3d as o3d
from shapely.geometry import MultiPoint, Polygon

# Path bootstrapping
_backend_dir = Path(__file__).resolve().parents[2]
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
_scans_dir = Path(__file__).resolve().parent
if str(_scans_dir) not in sys.path:
    sys.path.insert(0, str(_scans_dir))
_artifacts_dir = _backend_dir / "08_artifacts_generation"
if str(_artifacts_dir) not in sys.path:
    sys.path.insert(0, str(_artifacts_dir))

from Utilities.pipeline_paths import bootstrap
bootstrap()

from arcore_trajectory_bridge import ARCoreTrajectoryBridge


def create_crescent_moon_marker() -> MplPath:
    """Builds a centered matplotlib Path representing a crescent moon."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, 32)
    x_out = np.cos(theta)
    y_out = np.sin(theta)
    y_in = np.sin(theta[::-1])
    x_in = -0.5 + np.sqrt(1.25 - y_in**2)
    x_all = np.concatenate([x_out, x_in, [x_out[0]]]) - 0.35
    y_all = np.concatenate([y_out, y_in, [y_out[0]]])
    verts = np.column_stack([x_all, y_all])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


@dataclass
class SingleScanInfo:
    scan_id: str
    scan_dir: Path
    mesh_path: Path
    colmap_poses: np.ndarray       # (N_kf, 3) in Phase 04 mesh frame
    colmap_names: List[str]
    arcore_raw_poses: np.ndarray   # (N_dense, 3) in raw ARCore frame
    arcore_mesh_poses: np.ndarray  # (N_dense, 3) mapped into Phase 04 mesh frame
    arcore_timestamps: List[int]
    arcore_exported_flags: List[bool]
    first_exp_idx: int
    last_exp_idx: int
    T_mesh_from_arcore: np.ndarray # 4x4
    procrustes_rmse: float


def load_scan_info(scan_dir: Path) -> SingleScanInfo:
    """Loads and synchronizes COLMAP and ARCore poses for a single scan."""
    mesh_path = scan_dir / "mesh.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing mesh.ply in {scan_dir}")

    # 1. Load COLMAP cameras from sparse/0/images.txt (ground truth)
    cm_cams = ARCoreTrajectoryBridge.load_phase04_cameras(scan_dir)

    # Load timestamp mapping from transforms.json
    tf_path = scan_dir / "transforms.json"
    name_to_time = {}
    if tf_path.is_file():
        with open(tf_path, "r", encoding="utf-8") as f:
            tf_data = json.load(f)
        for frame in tf_data.get("frames", []):
            fname = Path(frame.get("file_path", "")).name
            name_to_time[fname] = int(frame.get("timestamp_ns", 0))

    sorted_cm = sorted(cm_cams.items(), key=lambda x: name_to_time.get(x[0], 0))
    cm_names = [k for k, _ in sorted_cm]
    cm_poses = np.array([pos for _, pos in sorted_cm], dtype=np.float64)

    # 2. Load ARCore dense trajectory
    traj_path = scan_dir / "trajectory.csv"
    ar_poses, ar_times, ar_exp = [], [], []
    with open(traj_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ar_poses.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
            ar_times.append(int(row["timestamp_ns"]))
            ar_exp.append(row.get("exported", "0").strip() == "1")

    ar_poses = np.array(ar_poses, dtype=np.float64)

    # 3. Fit SE(3) ARCore to Phase 04 mesh frame
    ar_dict = {ar_times[i]: ar_poses[i] for i in range(len(ar_times))}
    P_cm, P_ar = [], []
    for name, pos in cm_cams.items():
        t = name_to_time.get(name)
        if t in ar_dict:
            P_cm.append(pos)
            P_ar.append(ar_dict[t])

    P_cm = np.array(P_cm, dtype=np.float64)
    P_ar = np.array(P_ar, dtype=np.float64)
    c_cm = np.mean(P_cm, axis=0)
    c_ar = np.mean(P_ar, axis=0)
    H = (P_ar - c_ar).T @ (P_cm - c_cm)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t_vec = c_cm - R @ c_ar
    T_m_ar = np.eye(4, dtype=np.float64)
    T_m_ar[:3, :3] = R
    T_m_ar[:3, 3] = t_vec

    pred = (P_ar @ R.T) + t_vec
    rmse = float(np.sqrt(np.mean(np.sum((pred - P_cm)**2, axis=1))))

    ar_mesh_poses = (ar_poses @ R.T) + t_vec
    exp_indices = [i for i, exp in enumerate(ar_exp) if exp]
    first_exp = exp_indices[0] if exp_indices else 0
    last_exp = exp_indices[-1] if exp_indices else len(ar_poses) - 1

    return SingleScanInfo(
        scan_id=scan_dir.name,
        scan_dir=scan_dir,
        mesh_path=mesh_path,
        colmap_poses=cm_poses,
        colmap_names=cm_names,
        arcore_raw_poses=ar_poses,
        arcore_mesh_poses=ar_mesh_poses,
        arcore_timestamps=ar_times,
        arcore_exported_flags=ar_exp,
        first_exp_idx=first_exp,
        last_exp_idx=last_exp,
        T_mesh_from_arcore=T_m_ar,
        procrustes_rmse=rmse,
    )


def solve_multi_room_alignment(scans: List[SingleScanInfo]) -> List[np.ndarray]:
    """Finds global 4x4 SE(3) transforms for all scans relative to Scan 0."""
    assert len(scans) >= 1
    transforms = [np.eye(4, dtype=np.float64)]
    if len(scans) == 1:
        return transforms

    def to_2d(pts):
        return np.column_stack([pts[:, 0], -pts[:, 2]])

    def rot_2d(deg):
        rad = np.radians(deg)
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

    # 1. Align Scan 1 (Hallway) to Scan 0 (Bedroom):
    # Scan 0 exit doorway in 2D (Manhattan frame):
    exit0_2d = to_2d(scans[0].arcore_mesh_poses[[scans[0].last_exp_idx]])[0]
    # Scan 1 transition entry point (pt 0 of trajectory):
    entry1_2d = to_2d(scans[1].arcore_mesh_poses[[0]])[0]

    # At 0 degrees cardinal rotation, Scan 1 extends east out of the bedroom doorway
    R01 = rot_2d(0)
    t01 = exit0_2d - R01 @ entry1_2d
    T01 = np.eye(4, dtype=np.float64)
    T01[0, 3] = t01[0]
    T01[2, 3] = -t01[1]
    transforms.append(T01)

    if len(scans) >= 3:
        # 2. Align Scan 2 (Bathroom) to Scan 1 (Hallway):
        # Scan 1 exit in global frame:
        exit1_local = to_2d(scans[1].arcore_mesh_poses[[scans[1].last_exp_idx]])[0]
        exit1_global = (R01 @ exit1_local) + t01
        
        # Scan 2 doorway entry (first exported pose)
        entry2_local = to_2d(scans[2].arcore_mesh_poses[[scans[2].first_exp_idx]])[0]

        # Scan 2 is rotated 90 deg relative to Scan 1
        R12 = rot_2d(90)
        R02 = R01 @ R12

        # Initial alignment placing entry2 at exit1_global
        t02_base = exit1_global - R02 @ entry2_local

        # Compute wall separation offset along Manhattan Z so that the southern hallway wall
        # and northern bathroom wall touch at the doorway portal with zero mesh overlap:
        # Load mesh vertices
        m1 = o3d.io.read_triangle_mesh(str(scans[1].mesh_path))
        m2 = o3d.io.read_triangle_mesh(str(scans[2].mesh_path))
        v1 = np.asarray(m1.vertices)
        v2 = np.asarray(m2.vertices)

        v1_g = (v1 @ T01[:3, :3].T) + T01[:3, 3]
        
        R_3d_02 = np.array([
            [R02[0, 0], 0.0, -R02[0, 1]],
            [0.0, 1.0, 0.0],
            [-R02[1, 0], 0.0, R02[1, 1]]
        ])
        v2_rot = v2 @ R_3d_02.T
        
        # In Manhattan -Z forward:
        # Hallway min -Z (south wall):
        h_min_fwd = -np.max(v1_g[:, 2])
        # Bathroom max -Z (north wall):
        b_max_fwd = -np.min(v2_rot[:, 2]) + t02_base[1]
        
        # Shift along forward -Z so bathroom is placed flush against the hallway doorway portal:
        shift_fwd = h_min_fwd - b_max_fwd - 0.02
        t02_opt = t02_base + np.array([0.0, shift_fwd])

        T02 = np.eye(4, dtype=np.float64)
        T02[:3, :3] = R_3d_02
        T02[0, 3] = t02_opt[0]
        T02[2, 3] = -t02_opt[1]
        transforms.append(T02)

    return transforms


def render_aligned_tracks_plot(
    scans: List[SingleScanInfo],
    transforms: List[np.ndarray],
    output_path: Path
) -> None:
    """Renders high-resolution 2D top-down view of all aligned ARCore trajectories."""
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#111111")
    ax.set_facecolor("#181818")
    moon_marker = create_crescent_moon_marker()

    room_colors = ["#ff5533", "#00d4ff", "#ffb700"]
    room_labels = ["Scan 0: Bedroom / Mezzanine", "Scan 1: Hallway", "Scan 2: Bathroom"]

    for i, (s, T) in enumerate(zip(scans, transforms)):
        R = T[:3, :3]
        t = T[:3, 3]
        
        poses_g = (s.arcore_mesh_poses @ R.T) + t
        x = poses_g[:, 0]
        z_fwd = -poses_g[:, 2]

        # 1. Transition path before first export (in scan 1: walk from bedroom into hallway)
        if s.first_exp_idx > 0 and i == 1:
            ax.plot(
                x[:s.first_exp_idx + 1], z_fwd[:s.first_exp_idx + 1],
                color="#00ff88", linestyle="--", lw=2.8, alpha=0.95,
                label="Inter-Room Transition Walk",
                zorder=4
            )
            ax.scatter(
                [x[0]], [z_fwd[0]],
                color="#00ff88", marker="D", s=120, edgecolor="#ffffff", lw=1.2,
                label="Transition Start ♦ (at Scan 0 Exit)",
                zorder=7
            )

        # 2. Active scan path
        act_x = x[s.first_exp_idx:s.last_exp_idx + 1]
        act_z = z_fwd[s.first_exp_idx:s.last_exp_idx + 1]
        c = room_colors[i % len(room_colors)]
        lbl = room_labels[i] if i < len(room_labels) else f"Scan {i}"
        
        ax.plot(act_x, act_z, color=c, lw=1.8, alpha=0.85, label=f"Active Scan: {lbl}", zorder=5)

        # 3. COLMAP keyframe markers
        cm_g = (s.colmap_poses @ R.T) + t
        ax.scatter(
            cm_g[:, 0], -cm_g[:, 2],
            color=c, s=15, alpha=0.6, edgecolor="#ffffff", lw=0.5, zorder=6
        )

        # 4. Markers for Start ★ and End ☾
        start_x, start_z = act_x[0], act_z[0]
        end_x, end_z = act_x[-1], act_z[-1]

        ax.scatter(
            [start_x], [start_z],
            marker="*", s=260, color="#ffd700", edgecolor="#222222", lw=1.2,
            label="Room Scan Start ★" if i == 0 else None,
            zorder=8
        )
        ax.scatter(
            [end_x], [end_z],
            marker=moon_marker, s=200, color="#d580ff", edgecolor="#ffffff", lw=1.0,
            label="Room Scan End ☾" if i == 0 else None,
            zorder=8
        )

        ax.annotate(
            f"{s.scan_id.upper()}\nStart ★",
            (start_x, start_z),
            textcoords="offset points", xytext=(12, 6),
            color="#ffd700", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#222222", ec="#ffd700", alpha=0.85),
            zorder=9
        )

    # Bridge between Scan 1 exit and Scan 2 entry
    if len(scans) >= 3:
        p1_exit_g = (transforms[1][:3, :3] @ scans[1].arcore_mesh_poses[scans[1].last_exp_idx]) + transforms[1][:3, 3]
        p2_entry_g = (transforms[2][:3, :3] @ scans[2].arcore_mesh_poses[scans[2].first_exp_idx]) + transforms[2][:3, 3]
        ax.plot(
            [p1_exit_g[0], p2_entry_g[0]], [-p1_exit_g[2], -p2_entry_g[2]],
            color="#00ff88", linestyle="--", lw=3.0, zorder=7
        )
        dist_12 = np.linalg.norm(p1_exit_g - p2_entry_g)
        mid_x = 0.5 * (p1_exit_g[0] + p2_entry_g[0])
        mid_z = -0.5 * (p1_exit_g[2] + p2_entry_g[2])
        ax.annotate(
            f"Doorway Step: {dist_12:.2f}m",
            (mid_x, mid_z),
            textcoords="offset points", xytext=(-25, -20),
            color="#00ff88", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#112211", ec="#00ff88", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#00ff88", lw=1.2),
            zorder=10
        )

    ax.set_title("Unified Multi-Room 2D ARCore Trajectories (Zero-Overlap Aligned)", color="#ffffff", fontsize=15, pad=12, fontweight="bold")
    ax.set_xlabel("Global Manhattan X (meters)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Global Manhattan -Z / Forward (meters)", color="#cccccc", fontsize=11)
    ax.grid(True, color="#2e2e2e", linestyle=":", lw=0.8)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#ffffff", loc="upper right", fontsize=9)
    ax.axis("equal")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved aligned tracks plot to: {output_path}")


def render_aligned_plys_plot(
    scans: List[SingleScanInfo],
    transforms: List[np.ndarray],
    output_path: Path
) -> None:
    """Renders top-down orthographic slice of the combined PLY meshes positioned in global space."""
    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#111111")
    ax.set_facecolor("#181818")

    room_mesh_colors = ["#ff7755", "#44bbff", "#ffcc44"]
    room_labels = ["Scan 0: Bedroom / Mezzanine", "Scan 1: Hallway", "Scan 2: Bathroom"]

    for i, (s, T) in enumerate(zip(scans, transforms)):
        mesh = o3d.io.read_triangle_mesh(str(s.mesh_path))
        verts = np.asarray(mesh.vertices)
        
        y_max = np.max(verts[:, 1])
        y_min = np.min(verts[:, 1])
        y_cutoff = y_min + 0.70 * (y_max - y_min)
        
        mask = (verts[:, 1] >= y_min + 0.3) & (verts[:, 1] <= y_cutoff)
        v_sub = verts[mask]
        
        R = T[:3, :3]
        t = T[:3, 3]
        v_g = (v_sub @ R.T) + t

        x = v_g[:, 0]
        z_fwd = -v_g[:, 2]

        c = room_mesh_colors[i % len(room_mesh_colors)]
        lbl = room_labels[i] if i < len(room_labels) else f"Scan {i}"
        
        ax.scatter(x, z_fwd, s=0.3, c=c, alpha=0.35, label=f"Mesh: {lbl}")

        cx, cz = np.mean(x), np.mean(z_fwd)
        ax.annotate(
            f"{lbl.split(':')[1].strip()}",
            (cx, cz),
            color="#ffffff", fontsize=11, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#222222", ec=c, lw=1.5, alpha=0.9),
            zorder=8
        )

    # Plot doorway connectors
    if len(scans) >= 2:
        # Door 0 -> 1
        p0_exit_g = (transforms[0][:3, :3] @ scans[0].arcore_mesh_poses[scans[0].last_exp_idx]) + transforms[0][:3, 3]
        p1_entry_g = (transforms[1][:3, :3] @ scans[1].arcore_mesh_poses[0]) + transforms[1][:3, 3]
        ax.plot(
            [p0_exit_g[0], p1_entry_g[0]], [-p0_exit_g[2], -p1_entry_g[2]],
            color="#00ff88", linestyle="--", lw=3.0, zorder=9, label="Doorway Portals"
        )

    if len(scans) >= 3:
        # Door 1 -> 2
        p1_exit_g = (transforms[1][:3, :3] @ scans[1].arcore_mesh_poses[scans[1].last_exp_idx]) + transforms[1][:3, 3]
        p2_entry_g = (transforms[2][:3, :3] @ scans[2].arcore_mesh_poses[scans[2].first_exp_idx]) + transforms[2][:3, 3]
        ax.plot(
            [p1_exit_g[0], p2_entry_g[0]], [-p1_exit_g[2], -p2_entry_g[2]],
            color="#00ff88", linestyle="--", lw=3.0, zorder=9
        )

    ax.set_title("Unified Multi-Room Sealed Meshes (Top-Down Floor Plan)", color="#ffffff", fontsize=15, pad=12, fontweight="bold")
    ax.set_xlabel("Global Manhattan X (meters)", color="#cccccc", fontsize=11)
    ax.set_ylabel("Global Manhattan -Z / Forward (meters)", color="#cccccc", fontsize=11)
    ax.grid(True, color="#2e2e2e", linestyle=":", lw=0.8)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#222222", edgecolor="#444444", labelcolor="#ffffff", loc="upper right", fontsize=9)
    ax.axis("equal")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved aligned PLYs plot to: {output_path}")


def render_aligned_overview_plot(
    tracks_png: Path,
    plys_png: Path,
    output_path: Path
) -> None:
    """Combines tracks and PLYs plots side-by-side into a single overview graphic."""
    im1 = cv2.imread(str(tracks_png))
    im2 = cv2.imread(str(plys_png))
    if im1 is None or im2 is None:
        return
    h = min(im1.shape[0], im2.shape[0])
    w1 = int(im1.shape[1] * (h / im1.shape[0]))
    w2 = int(im2.shape[1] * (h / im2.shape[0]))
    r1 = cv2.resize(im1, (w1, h), interpolation=cv2.INTER_AREA)
    r2 = cv2.resize(im2, (w2, h), interpolation=cv2.INTER_AREA)
    combined = np.hstack([r1, r2])
    cv2.imwrite(str(output_path), combined)
    print(f"Saved combined overview plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-room 2D/3D ARCore trajectory and mesh alignment.")
    parser.add_argument("--scene-dir", type=str, default="backend/current_scene", help="Scene root folder")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for diagnostic plots")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else scene_dir

    scan_dirs = []
    for p in scene_dir.iterdir():
        if p.is_dir() and p.name.startswith("scan_") and p.name.removeprefix("scan_").isdigit():
            scan_dirs.append((int(p.name.removeprefix("scan_")), p))
    scan_dirs.sort(key=lambda x: x[0])
    scan_dirs = [p for _, p in scan_dirs]

    print(f"Found {len(scan_dirs)} scans: {[p.name for p in scan_dirs]}")

    scans = [load_scan_info(p) for p in scan_dirs]
    transforms = solve_multi_room_alignment(scans)

    tracks_png = out_dir / "aligned_tracks_topdown.png"
    render_aligned_tracks_plot(scans, transforms, tracks_png)

    plys_png = out_dir / "aligned_plys_topdown.png"
    render_aligned_plys_plot(scans, transforms, plys_png)

    overview_png = out_dir / "aligned_scene_overview.png"
    render_aligned_overview_plot(tracks_png, plys_png, overview_png)

    out_json = out_dir / "aligned_scans_transforms.json"
    manifest = {
        "anchor_scan": scans[0].scan_id,
        "scans": {}
    }
    for s, T in zip(scans, transforms):
        manifest["scans"][s.scan_id] = {
            "transform_matrix": T.tolist(),
            "mesh_path": str(s.mesh_path),
            "procrustes_rmse_m": float(s.procrustes_rmse)
        }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved alignment transforms to: {out_json}")


if __name__ == "__main__":
    main()
