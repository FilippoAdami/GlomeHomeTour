#!/usr/bin/env python3
"""Visualize top-down views of sealed room meshes with overlaid ARCore and COLMAP paths.

For each room scan in a multi-room capture:
1. Renders an orthographic top-down view of the Phase 04 sealed mesh (with ceiling
   cut away so the floor plan, walls, doorways, and interior structures are visible).
2. Uses the ground-truth Phase 04 Manhattan camera poses from sparse/0/images.txt
   (avoiding any un-applied in-place yaw rotations in transforms.json).
3. Overlays the transition path from the previous scan in GREEN (poses from trajectory start
   up to the first exported keyframe, where room scanning began).
4. Overlays the active room scan path in RED (poses during exported keyframing).
5. Overlays the COLMAP keyframe path in BLUE with camera markers.
6. Marks the scan starting point with a gold STAR (★) and ending point with a purple crescent MOON (☾).
7. Overlays in GREEN the transition to the next scan if consecutive scans share ARCore tracking.
8. Saves one high-resolution image per scan.
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
import numpy as np
import open3d as o3d

# Ensure backend root is on sys.path
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
from extract_cross_section_views import render_cut_view, slice_open3d_mesh_plane


def create_crescent_moon_marker() -> MplPath:
    """Builds a centered matplotlib Path representing a crescent moon."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, 32)
    # Outer circular arc of radius 1.0 on the right
    x_out = np.cos(theta)
    y_out = np.sin(theta)
    # Inner circular cutout biting from the left
    y_in = np.sin(theta[::-1])
    x_in = -0.5 + np.sqrt(1.25 - y_in**2)

    # Center horizontally at the centroid (offset ~0.35)
    x_all = np.concatenate([x_out, x_in, [x_out[0]]]) - 0.35
    y_all = np.concatenate([y_out, y_in, [y_out[0]]])

    verts = np.column_stack([x_all, y_all])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(verts, codes)


@dataclass
class ScanPathsData:
    """Data structures for a single room scan."""
    scan_id: str
    scan_dir: Path
    mesh_path: Path
    colmap_poses: np.ndarray       # (N_kf, 3) keyframe camera centers in Phase 04 mesh frame
    colmap_timestamps: List[int]
    colmap_names: List[str]
    arcore_raw_poses: np.ndarray   # (N_dense, 3) raw ARCore translation in ARCore frame
    arcore_timestamps: List[int]
    arcore_exported_flags: List[bool]
    arcore_mesh_poses: np.ndarray  # (N_dense, 3) ARCore translation mapped to mesh frame
    T_mesh_from_arcore: np.ndarray # 4x4 SE(3) transformation
    procrustes_rmse: float         # Fit RMSE between matching keyframes (meters)
    first_exported_idx: int        # Index where exported == 1 starts
    last_exported_idx: int         # Index where exported == 1 ends
    initial_point_mesh: np.ndarray # [X, Y, Z] at start of recording (point 0)
    start_point_mesh: np.ndarray   # [X, Y, Z] at start of active room scan (first exported pose)
    end_point_mesh: np.ndarray     # [X, Y, Z] at end of active room scan (last exported pose)
    loop_closure_dist: float       # ||start - end|| of the active room scan (meters)
    transition_duration_s: float   # Duration from pt 0 to first exported keyframe (seconds)
    scan_duration_s: float         # Duration of active room scanning (seconds)
    total_duration_s: float        # Total duration from pt 0 to last pose (seconds)
    next_scan_transition: Optional[Dict[str, Any]] = None


class ScanPathVisualizer:
    """Extracts topdown mesh views and plots trajectory diagnostics for all room scans."""

    def __init__(self, scene_dir: Path, output_dir: Optional[Path] = None):
        self.scene_dir = Path(scene_dir).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else self.scene_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.moon_marker = create_crescent_moon_marker()

    def discover_scans(self) -> List[Path]:
        """Discovers all scan directories (e.g. scan_0, scan_1, ...) sorted numerically."""
        scans = []
        for p in self.scene_dir.iterdir():
            if p.is_dir() and p.name.startswith("scan_"):
                suffix = p.name.removeprefix("scan_")
                if suffix.isdigit():
                    scans.append((int(suffix), p))
        scans.sort(key=lambda x: x[0])
        return [p for _, p in scans]

    @staticmethod
    def load_colmap_keyframes(scan_dir: Path) -> Tuple[np.ndarray, List[int], List[str]]:
        """Loads COLMAP keyframe positions in Phase 04 mesh frame directly from sparse/0/images.txt (ground truth)."""
        images_txt = scan_dir / "sparse" / "0" / "images.txt"
        tf_path = scan_dir / "transforms.json"

        # Map frame file names to timestamps
        name_to_time = {}
        if tf_path.is_file():
            with open(tf_path, "r", encoding="utf-8") as f:
                tf_data = json.load(f)
            for frame in tf_data.get("frames", []):
                fname = Path(frame.get("file_path", "")).name
                name_to_time[fname] = int(frame.get("timestamp_ns", 0))

        if images_txt.is_file():
            p04_cams = ARCoreTrajectoryBridge.load_phase04_cameras(scan_dir)
            # Sort chronologically by timestamp
            sorted_items = sorted(p04_cams.items(), key=lambda x: name_to_time.get(x[0], 0))
            names = [k for k, _ in sorted_items]
            positions = [pos for _, pos in sorted_items]
            timestamps = [name_to_time.get(k, 0) for k in names]
            return np.array(positions, dtype=np.float64), timestamps, names

        # Fallback to transforms.json if images.txt is missing
        if tf_path.is_file():
            positions = []
            timestamps = []
            names = []
            with open(tf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for frame in data.get("frames", []):
                mat = np.array(frame["transform_matrix"], dtype=np.float64)
                positions.append(mat[:3, 3])
                timestamps.append(int(frame.get("timestamp_ns", 0)))
                names.append(Path(frame.get("file_path", "")).name)
            return np.array(positions, dtype=np.float64), timestamps, names

        raise FileNotFoundError(f"Missing images.txt or transforms.json in {scan_dir}")

    @staticmethod
    def load_dense_arcore_trajectory(scan_dir: Path) -> Tuple[np.ndarray, List[int], List[bool]]:
        """Loads high-frequency raw ARCore trajectory poses and exported flags from trajectory.csv."""
        traj_path = scan_dir / "trajectory.csv"
        if not traj_path.is_file():
            raise FileNotFoundError(f"Missing trajectory.csv in {scan_dir}")

        positions = []
        timestamps = []
        exported_flags = []
        with open(traj_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tx = float(row["tx"])
                ty = float(row["ty"])
                tz = float(row["tz"])
                t_ns = int(row["timestamp_ns"])
                exp = (row.get("exported", "0").strip() == "1")
                positions.append([tx, ty, tz])
                timestamps.append(t_ns)
                exported_flags.append(exp)

        return np.array(positions, dtype=np.float64), timestamps, exported_flags

    @classmethod
    def fit_arcore_to_mesh_rigid_transform(
        cls,
        arcore_poses: np.ndarray,
        arcore_timestamps: List[int],
        colmap_poses: np.ndarray,
        colmap_timestamps: List[int],
    ) -> Tuple[np.ndarray, float]:
        """Calculates SE(3) rigid transformation mapping raw ARCore coordinates into the Phase 04 mesh frame."""
        ar_dict = {t: arcore_poses[i] for i, t in enumerate(arcore_timestamps)}
        
        P_ar = []
        P_cm = []
        for i, t in enumerate(colmap_timestamps):
            if t in ar_dict:
                P_ar.append(ar_dict[t])
                P_cm.append(colmap_poses[i])

        if len(P_ar) < 3:
            n_match = min(len(arcore_poses), len(colmap_poses))
            indices_ar = np.linspace(0, len(arcore_poses) - 1, n_match, dtype=int)
            indices_cm = np.linspace(0, len(colmap_poses) - 1, n_match, dtype=int)
            P_ar = arcore_poses[indices_ar]
            P_cm = colmap_poses[indices_cm]
        else:
            P_ar = np.array(P_ar, dtype=np.float64)
            P_cm = np.array(P_cm, dtype=np.float64)

        # Procrustes SE(3) alignment: P_cm = R @ P_ar + t
        c_ar = np.mean(P_ar, axis=0)
        c_cm = np.mean(P_cm, axis=0)

        H = (P_ar - c_ar).T @ (P_cm - c_cm)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t_vec = c_cm - R @ c_ar

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t_vec

        pred = (P_ar @ R.T) + t_vec
        rmse = float(np.sqrt(np.mean(np.sum((pred - P_cm)**2, axis=1))))
        return T, rmse

    def load_all_scans(self) -> List[ScanPathsData]:
        """Loads and prepares trajectory and mesh data for all scans in scene."""
        scan_dirs = self.discover_scans()
        if not scan_dirs:
            raise FileNotFoundError(f"No scan_* directories found in {self.scene_dir}")

        scans_data = []
        for s_dir in scan_dirs:
            mesh_path = s_dir / "mesh.ply"
            if not mesh_path.is_file():
                print(f"Warning: mesh.ply not found in {s_dir}, skipping.")
                continue

            cm_poses, cm_times, cm_names = self.load_colmap_keyframes(s_dir)
            ar_poses, ar_times, ar_exp_flags = self.load_dense_arcore_trajectory(s_dir)

            T_m_ar, rmse = self.fit_arcore_to_mesh_rigid_transform(
                ar_poses, ar_times, cm_poses, cm_times
            )

            # Transform all dense ARCore poses into the mesh frame
            R = T_m_ar[:3, :3]
            t = T_m_ar[:3, 3]
            ar_mesh_poses = (ar_poses @ R.T) + t

            # Identify transition vs active scanning indices
            exp_indices = [i for i, exp in enumerate(ar_exp_flags) if exp]
            if exp_indices:
                first_exp_idx = exp_indices[0]
                last_exp_idx = exp_indices[-1]
            else:
                first_exp_idx = 0
                last_exp_idx = len(ar_poses) - 1

            initial_pt = ar_mesh_poses[0]
            start_scan_pt = ar_mesh_poses[first_exp_idx]
            end_scan_pt = ar_mesh_poses[last_exp_idx]

            loop_closure = float(np.linalg.norm(start_scan_pt - end_scan_pt))
            t_start = ar_times[0]
            t_first_exp = ar_times[first_exp_idx]
            t_last_exp = ar_times[last_exp_idx]
            t_end = ar_times[-1]

            trans_dur = (t_first_exp - t_start) / 1e9
            scan_dur = (t_last_exp - t_first_exp) / 1e9
            total_dur = (t_end - t_start) / 1e9

            scans_data.append(ScanPathsData(
                scan_id=s_dir.name,
                scan_dir=s_dir,
                mesh_path=mesh_path,
                colmap_poses=cm_poses,
                colmap_timestamps=cm_times,
                colmap_names=cm_names,
                arcore_raw_poses=ar_poses,
                arcore_timestamps=ar_times,
                arcore_exported_flags=ar_exp_flags,
                arcore_mesh_poses=ar_mesh_poses,
                T_mesh_from_arcore=T_m_ar,
                procrustes_rmse=rmse,
                first_exported_idx=first_exp_idx,
                last_exported_idx=last_exp_idx,
                initial_point_mesh=initial_pt,
                start_point_mesh=start_scan_pt,
                end_point_mesh=end_scan_pt,
                loop_closure_dist=loop_closure,
                transition_duration_s=trans_dur,
                scan_duration_s=scan_dur,
                total_duration_s=total_dur,
            ))

        # Analyze inter-scan transitions to next scan
        for i in range(len(scans_data)):
            curr = scans_data[i]
            if i + 1 < len(scans_data):
                nxt = scans_data[i + 1]
                t_end_curr = curr.arcore_timestamps[-1]
                t_start_nxt = nxt.arcore_timestamps[0]
                gap_s = (t_start_nxt - t_end_curr) / 1e9

                # Check if consecutive scans share ARCore frame
                ar_disp = np.linalg.norm(nxt.arcore_raw_poses[0] - curr.arcore_raw_poses[-1])
                is_same_session = (0 <= gap_s < 120.0) and (ar_disp < 3.0)

                if is_same_session:
                    R_curr = curr.T_mesh_from_arcore[:3, :3]
                    t_curr = curr.T_mesh_from_arcore[:3, 3]
                    next_init_in_curr_mesh = (R_curr @ nxt.arcore_raw_poses[0]) + t_curr
                    curr.next_scan_transition = {
                        "target_scan_id": nxt.scan_id,
                        "session_shared": True,
                        "time_gap_s": gap_s,
                        "arcore_displacement_m": float(ar_disp),
                        "next_start_point_mesh": next_init_in_curr_mesh,
                        "has_recorded_trajectory": False,
                        "description": f"Shared ARCore session ({gap_s:.1f}s gap, {ar_disp:.2f}m displacement to {nxt.scan_id} start)."
                    }
                else:
                    curr.next_scan_transition = {
                        "target_scan_id": nxt.scan_id,
                        "session_shared": False,
                        "time_gap_s": gap_s,
                        "arcore_displacement_m": None,
                        "next_start_point_mesh": None,
                        "has_recorded_trajectory": False,
                        "description": f"Not in same session ({gap_s/3600:.1f}h gap). Independent ARCore origins; transition path NOT PRESENT in app outputs."
                    }
            else:
                curr.next_scan_transition = {
                    "target_scan_id": None,
                    "session_shared": False,
                    "time_gap_s": 0.0,
                    "arcore_displacement_m": None,
                    "next_start_point_mesh": None,
                    "has_recorded_trajectory": False,
                    "description": "Terminal scan (no subsequent scan)."
                }

        return scans_data

    @staticmethod
    def render_mesh_topdown(
        mesh_path: Path,
        ceiling_mode: str = "normal_cut",
        resolution: int = 1024,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Renders an orthographic top-down view of the mesh using CPU raycasting.
        
        Returns:
            image: (H, W, 3) uint8 RGB image
            extent: (x_min, x_max, neg_z_min, neg_z_max) in Phase 04 meters.
        """
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_triangle_normals()

        bmin = mesh.get_min_bound()
        bmax = mesh.get_max_bound()
        center = (bmin + bmax) / 2.0

        if ceiling_mode == "normal_cut":
            # Remove downward-pointing ceiling faces in the upper portion
            fn = np.asarray(mesh.triangle_normals)
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            fc = verts[faces].mean(axis=1)

            # Identify ceiling faces: normal pointing down (n_y < -0.3) in top 55% of room height
            cut_y = bmin[1] + 0.45 * (bmax[1] - bmin[1])
            is_ceiling = (fn[:, 1] < -0.3) & (fc[:, 1] > cut_y)
            keep_faces = ~is_ceiling

            used_verts, inv = np.unique(faces[keep_faces], return_inverse=True)
            sub_m = o3d.geometry.TriangleMesh()
            sub_m.vertices = o3d.utility.Vector3dVector(verts[used_verts])
            sub_m.triangles = o3d.utility.Vector3iVector(inv.reshape(-1, 3))
            if mesh.has_vertex_colors():
                sub_m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors)[used_verts])
            sub_m.compute_vertex_normals()
            active_mesh = sub_m
        elif ceiling_mode == "plane_cut":
            top_y = bmax[1] - 0.20 * (bmax[1] - bmin[1])
            active_mesh = slice_open3d_mesh_plane(
                mesh, np.array([0.0, -1.0, 0.0]), np.array([center[0], top_y, center[2]])
            )
        else:
            active_mesh = mesh

        # Camera positioned above the center looking down along -Y
        span = max(bmax[0] - bmin[0], bmax[2] - bmin[2])
        dist = span * 1.5
        eye = np.array([center[0], bmax[1] + dist, center[2]], dtype=np.float64)
        target = np.array([center[0], center[1], center[2]], dtype=np.float64)
        up = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        fov_deg = 60.0
        half_size = np.linalg.norm(target - eye) * np.tan(np.radians(fov_deg) / 2.0)
        img = render_cut_view(active_mesh, eye, target, up, resolution=resolution, orthographic=True, fov_deg=fov_deg)

        x_min = float(center[0] - half_size)
        x_max = float(center[0] + half_size)
        neg_z_min = float(-center[2] - half_size)
        neg_z_max = float(-center[2] + half_size)

        return img, (x_min, x_max, neg_z_min, neg_z_max)

    def visualize_scan(
        self,
        scan_data: ScanPathsData,
        ceiling_mode: str = "normal_cut",
        resolution: int = 1024,
        dpi: int = 150,
    ) -> Path:
        """Renders and saves the top-down visualization image for a single scan."""
        print(f"[{scan_data.scan_id}] Rendering top-down mesh view...")
        img, (x_min, x_max, neg_z_min, neg_z_max) = self.render_mesh_topdown(
            scan_data.mesh_path, ceiling_mode=ceiling_mode, resolution=resolution
        )

        fig, ax = plt.subplots(figsize=(12, 12), facecolor="#121212")
        ax.set_facecolor("#1A1A1A")

        # 1. Background top-down mesh image
        ax.imshow(
            img,
            extent=[x_min, x_max, neg_z_min, neg_z_max],
            origin="upper",
            aspect="equal",
            zorder=1,
        )

        f_idx = scan_data.first_exported_idx
        l_idx = scan_data.last_exported_idx

        # 2. GREEN: Transition path from previous scan / entrance into the room (pt 0 to first_exp)
        if f_idx > 0:
            trans_x = scan_data.arcore_mesh_poses[:f_idx + 1, 0]
            trans_neg_z = -scan_data.arcore_mesh_poses[:f_idx + 1, 2]
            ax.plot(
                trans_x, trans_neg_z,
                color="#00E676",
                linewidth=2.5,
                linestyle="--",
                alpha=0.90,
                zorder=6,
                label=f"Transition path into room ({f_idx} pts, {scan_data.transition_duration_s:.1f}s, green)",
            )
            # Mark the initial position where recording started
            init_x = scan_data.initial_point_mesh[0]
            init_neg_z = -scan_data.initial_point_mesh[2]
            ax.scatter(
                [init_x], [init_neg_z],
                marker="D",
                s=160,
                color="#00E676",
                edgecolor="black",
                linewidth=1.5,
                zorder=10,
                label="Transition Start ♦",
            )
            ax.annotate(
                " Transition Start ♦",
                xy=(init_x, init_neg_z),
                xytext=(8, -14),
                textcoords="offset points",
                color="#00E676",
                fontweight="bold",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="#1B5E20", ec="#00E676", alpha=0.9),
                zorder=11,
            )

        # 3. RED: Active room scanning path (first_exp to last_exp)
        scan_x = scan_data.arcore_mesh_poses[f_idx:l_idx + 1, 0]
        scan_neg_z = -scan_data.arcore_mesh_poses[f_idx:l_idx + 1, 2]
        ax.plot(
            scan_x, scan_neg_z,
            color="#FF1744",
            linewidth=1.8,
            alpha=0.85,
            zorder=5,
            label=f"ARCore Room Scan ({len(scan_x)} poses, red)",
        )

        # 4. BLUE: COLMAP keyframe path
        cm_x = scan_data.colmap_poses[:, 0]
        cm_neg_z = -scan_data.colmap_poses[:, 2]
        ax.plot(
            cm_x, cm_neg_z,
            color="#00E5FF",
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            markerfacecolor="#00B0FF",
            markeredgecolor="white",
            markeredgewidth=0.5,
            alpha=0.92,
            zorder=7,
            label=f"COLMAP Path ({len(cm_x)} keyframes, blue)",
        )

        # 5. Start point marked with a gold STAR (★) at room scan start
        start_x = scan_data.start_point_mesh[0]
        start_neg_z = -scan_data.start_point_mesh[2]
        ax.scatter(
            [start_x], [start_neg_z],
            marker="*",
            s=450,
            color="#FFD700",
            edgecolor="#000000",
            linewidth=1.8,
            zorder=12,
            label="Room Scan Start ★",
        )
        ax.annotate(
            " Start ★",
            xy=(start_x, start_neg_z),
            xytext=(10, 10),
            textcoords="offset points",
            color="#FFD700",
            fontweight="bold",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="#212121", ec="#FFD700", alpha=0.9),
            zorder=13,
        )

        # 6. End point marked with a crescent MOON (☾) at room scan end
        end_x = scan_data.end_point_mesh[0]
        end_neg_z = -scan_data.end_point_mesh[2]
        ax.scatter(
            [end_x], [end_neg_z],
            marker=self.moon_marker,
            s=450,
            color="#E040FB",
            edgecolor="#FFFFFF",
            linewidth=1.5,
            zorder=12,
            label=f"Room Scan End ☾ (gap: {scan_data.loop_closure_dist:.2f}m)",
        )
        ax.annotate(
            " End ☾",
            xy=(end_x, end_neg_z),
            xytext=(10, -18),
            textcoords="offset points",
            color="#E040FB",
            fontweight="bold",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="#212121", ec="#E040FB", alpha=0.9),
            zorder=13,
        )

        # Dotted line between Start and End to show loop closure
        ax.plot(
            [start_x, end_x], [start_neg_z, end_neg_z],
            color="#FFD54F",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            zorder=6,
        )

        # 7. Inter-scan transition connector to NEXT scan (if shared ARCore session)
        trans = scan_data.next_scan_transition
        if trans and trans.get("session_shared") and trans.get("next_start_point_mesh") is not None:
            nxt_pt = trans["next_start_point_mesh"]
            nxt_x = nxt_pt[0]
            nxt_neg_z = -nxt_pt[2]

            ax.plot(
                [end_x, nxt_x], [end_neg_z, nxt_neg_z],
                color="#76FF03",
                linestyle="--",
                linewidth=2.8,
                zorder=8,
                label=f"Exit step to {trans['target_scan_id']} ({trans['arcore_displacement_m']:.2f}m, green)",
            )
            ax.scatter(
                [nxt_x], [nxt_neg_z],
                marker="s",
                s=100,
                color="#76FF03",
                edgecolor="black",
                linewidth=1.2,
                zorder=9,
            )

        # Formatting, labels, and grid
        ax.set_title(
            f"Top-Down Path Diagnostics — {scan_data.scan_id.upper()}\n"
            f"Sealed Mesh ({ceiling_mode}) with Transition (Green), ARCore Scan (Red) & COLMAP (Blue)",
            color="#ECEFF1",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Phase 04 Manhattan X (meters)", color="#B0BEC5", fontsize=11)
        ax.set_ylabel("Phase 04 Manhattan -Z / Forward (meters)", color="#B0BEC5", fontsize=11)

        ax.tick_params(colors="#90A4AE", labelsize=10)
        ax.grid(True, color="#37474F", linestyle="--", linewidth=0.6, alpha=0.6)

        # Legend
        ax.legend(
            loc="upper right",
            facecolor="#212121",
            edgecolor="#546E7A",
            fontsize=9.5,
            labelcolor="#ECEFF1",
            framealpha=0.92,
        )

        # Diagnostic info box at bottom left
        transition_info = trans.get("description", "None") if trans else "None"
        info_text = (
            f"Transition Into Room: {scan_data.transition_duration_s:.1f}s ({f_idx} poses)\n"
            f"Active Room Scan: {scan_data.scan_duration_s:.1f}s ({l_idx - f_idx + 1} poses)\n"
            f"COLMAP Keyframes: {len(scan_data.colmap_poses)}\n"
            f"ARCore/COLMAP Fit RMSE: {scan_data.procrustes_rmse*100:.1f} cm\n"
            f"Loop Closure Gap (Start ★ - End ☾): {scan_data.loop_closure_dist*100:.1f} cm\n"
            f"Next Scan: {transition_info}"
        )
        ax.text(
            0.02, 0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#CFD8DC",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="#212121", ec="#37474F", alpha=0.92),
            zorder=12,
        )

        out_file = self.output_dir / f"{scan_data.scan_id}_topdown_paths.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        print(f"[{scan_data.scan_id}] Saved visualization to {out_file}")
        return out_file

    def run(self, ceiling_mode: str = "normal_cut", resolution: int = 1024, dpi: int = 150) -> List[Path]:
        """Runs visualization generation for all scans and writes summary report."""
        print(f"Loading scans from {self.scene_dir}...")
        scans_data = self.load_all_scans()
        print(f"Found {len(scans_data)} scans with valid meshes.")

        output_images = []
        summary = {
            "scene_dir": str(self.scene_dir),
            "scans_count": len(scans_data),
            "ceiling_mode": ceiling_mode,
            "scans": []
        }

        for s_data in scans_data:
            out_img = self.visualize_scan(
                s_data,
                ceiling_mode=ceiling_mode,
                resolution=resolution,
                dpi=dpi,
            )
            output_images.append(out_img)

            summary["scans"].append({
                "scan_id": s_data.scan_id,
                "output_image": str(out_img),
                "transition_duration_s": s_data.transition_duration_s,
                "scan_duration_s": s_data.scan_duration_s,
                "keyframe_count": len(s_data.colmap_poses),
                "dense_pose_count": len(s_data.arcore_mesh_poses),
                "arcore_colmap_rmse_m": s_data.procrustes_rmse,
                "loop_closure_dist_m": s_data.loop_closure_dist,
                "transition": s_data.next_scan_transition,
            })

        summary_file = self.output_dir / "scan_paths_summary.json"
        summary_file.write_text(
            json.dumps(summary, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)),
            encoding="utf-8",
        )
        print(f"Wrote summary report to {summary_file}")
        return output_images


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-dir", type=Path, default=Path("backend/current_scene"), help="Path to scene directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save output images (default: scene-dir)")
    parser.add_argument("--ceiling-mode", choices=["normal_cut", "plane_cut", "none"], default="normal_cut",
                        help="Ceiling removal method: normal_cut (preserves walls & mezzanine), plane_cut, none")
    parser.add_argument("--resolution", type=int, default=1024, help="Render resolution (px)")
    parser.add_argument("--dpi", type=int, default=150, help="Plot image DPI")
    args = parser.parse_args(argv)

    visualizer = ScanPathVisualizer(scene_dir=args.scene_dir, output_dir=args.output_dir)
    visualizer.run(ceiling_mode=args.ceiling_mode, resolution=args.resolution, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
