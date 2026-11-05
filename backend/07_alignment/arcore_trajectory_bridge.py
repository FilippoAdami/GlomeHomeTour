"""ARCore trajectory bridging and coarse multi-room SE(3) pose initialization.

Calculates the rigid transformation between the mobile ARCore tracking space
and the Phase 04 / Phase 06 leveled Manhattan coordinate system, then computes
the coarse relative transformation linking consecutive room scans.
Automatically discovers the initial wall thickness from the trajectory traversal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BridgeResult:
    """Coarse initial alignment from ARCore trajectory bridging."""
    relative_transform: np.ndarray  # 4x4 transform mapping Scan J to Scan I coordinate frame
    yaw_cardinal_deg: float         # Snapped Manhattan cardinal yaw (0, 90, 180, 270)
    raw_yaw_deg: float              # Raw estimated yaw before Manhattan snapping
    translation: np.ndarray         # [tx, ty, tz] in Phase 04/06 coordinate frame
    exit_point_i: np.ndarray        # Estimated doorway exit point in Scan I frame
    entry_point_j: np.ndarray       # Estimated doorway entry point in Scan J frame
    transit_vector: np.ndarray      # Displacement vector between exit and entry
    automatically_discovered_thickness: float # Discovered initial wall thickness


class ARCoreTrajectoryBridge:
    """Estimates coarse relative SE(3) transform between two room scans using ARCore poses."""

    @staticmethod
    def load_phase04_cameras(scan_dir: Path) -> Dict[str, np.ndarray]:
        """Loads camera centers in Phase 04 / Phase 06 frame from sparse/0/images.txt."""
        images_txt = scan_dir / "sparse" / "0" / "images.txt"
        if not images_txt.is_file():
            raise FileNotFoundError(f"Missing COLMAP images file: {images_txt}")

        cameras = {}
        with open(images_txt, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[-1].endswith(".jpg"):
                    qw, qx, qy, qz = [float(x) for x in parts[1:5]]
                    tx, ty, tz = [float(x) for x in parts[5:8]]
                    name = parts[9]
                    R = np.array([
                        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                    ])
                    t = np.array([tx, ty, tz])
                    cam_center = -R.T @ t
                    cameras[name] = cam_center
        return cameras

    @staticmethod
    def load_arcore_cameras(scan_dir: Path) -> Dict[str, np.ndarray]:
        """Loads camera centers in ARCore world frame from transforms.json."""
        transforms_json = scan_dir / "transforms.json"
        if not transforms_json.is_file():
            raise FileNotFoundError(f"Missing transforms.json: {transforms_json}")

        with open(transforms_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        cameras = {}
        for frame in data.get("frames", []):
            file_path = frame.get("file_path", "")
            name = Path(file_path).name
            matrix = np.array(frame["transform_matrix"], dtype=np.float64)
            cameras[name] = matrix[:3, 3]
        return cameras

    @classmethod
    def estimate_arcore_to_phase04_rigid_transform(cls, scan_dir: Path) -> np.ndarray:
        """Computes the 4x4 rigid transformation mapping ARCore coordinates to Phase 04 coordinates."""
        arcore_cams = cls.load_arcore_cameras(scan_dir)
        phase04_cams = cls.load_phase04_cameras(scan_dir)

        common_keys = sorted(set(arcore_cams.keys()) & set(phase04_cams.keys()))
        if len(common_keys) < 3:
            raise ValueError(f"Too few matching frames ({len(common_keys)}) between ARCore and Phase 04 in {scan_dir}")

        P_arcore = np.array([arcore_cams[k] for k in common_keys], dtype=np.float64)
        P_phase04 = np.array([phase04_cams[k] for k in common_keys], dtype=np.float64)

        c_ar = np.mean(P_arcore, axis=0)
        c_p4 = np.mean(P_phase04, axis=0)

        H = (P_arcore - c_ar).T @ (P_phase04 - c_p4)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        t = c_p4 - R @ c_ar

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    @staticmethod
    def get_trajectory_endpoints(scan_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the start and end camera positions in ARCore frame from trajectory.csv or transforms.json."""
        traj_csv = scan_dir / "trajectory.csv"
        if traj_csv.is_file():
            try:
                import pandas as pd
                df = pd.read_csv(traj_csv)
                if not df.empty:
                    start_pt = df[["tx", "ty", "tz"]].iloc[0].values.astype(np.float64)
                    end_pt = df[["tx", "ty", "tz"]].iloc[-1].values.astype(np.float64)
                    return start_pt, end_pt
            except Exception:
                pass

        cams = ARCoreTrajectoryBridge.load_arcore_cameras(scan_dir)
        keys = sorted(cams.keys())
        return cams[keys[0]], cams[keys[-1]]

    def estimate_coarse_relative_transform(
        self,
        scan_dir_i: Path,
        scan_dir_j: Path,
        wall_normal_hint: Optional[np.ndarray] = None,
    ) -> BridgeResult:
        """Estimates coarse SE(3) transform mapping Scan J coordinates into Scan I coordinates.

        Aligns the start of Scan J with the end of Scan I, automatically discovering
        the initial wall thickness from the trajectory transit displacement.
        """
        T_p4_ar_i = self.estimate_arcore_to_phase04_rigid_transform(scan_dir_i)
        T_p4_ar_j = self.estimate_arcore_to_phase04_rigid_transform(scan_dir_j)

        _, end_ar_i = self.get_trajectory_endpoints(scan_dir_i)
        start_ar_j, _ = self.get_trajectory_endpoints(scan_dir_j)

        exit_p4_i = (T_p4_ar_i[:3, :3] @ end_ar_i) + T_p4_ar_i[:3, 3]
        entry_p4_j = (T_p4_ar_j[:3, :3] @ start_ar_j) + T_p4_ar_j[:3, 3]

        R_i = T_p4_ar_i[:3, :3]
        R_j = T_p4_ar_j[:3, :3]
        R_rel_raw = R_i @ R_j.T

        raw_yaw_rad = np.arctan2(R_rel_raw[0, 2], R_rel_raw[2, 2])
        raw_yaw_deg = float(np.degrees(raw_yaw_rad))

        cardinal_yaw_deg = float(round(raw_yaw_deg / 90.0) * 90.0) % 360.0
        snapped_yaw_rad = np.radians(cardinal_yaw_deg)

        cos_y = np.cos(snapped_yaw_rad)
        sin_y = np.sin(snapped_yaw_rad)
        R_cardinal = np.array([
            [cos_y, 0.0, sin_y],
            [0.0,   1.0, 0.0],
            [-sin_y, 0.0, cos_y]
        ], dtype=np.float64)

        transit_dir = wall_normal_hint if wall_normal_hint is not None else np.array([0.0, 0.0, 1.0])
        norm_td = np.linalg.norm(transit_dir)
        if norm_td > 1e-6:
            transit_dir = transit_dir / norm_td

        # Automatically discover initial wall thickness from trajectory displacement along transit normal
        disp_p4 = exit_p4_i - (R_cardinal @ entry_p4_j)
        proj_thickness = float(abs(disp_p4 @ transit_dir))
        # If trajectory is from separate sessions starting at origin, clamp to architectural range [0.10, 0.50]
        if 0.08 <= proj_thickness <= 0.60:
            discovered_thickness = proj_thickness
        else:
            discovered_thickness = 0.25  # Geometric midpoint initialization

        offset_vector = transit_dir * discovered_thickness
        target_entry_in_i = exit_p4_i + offset_vector
        translation = target_entry_in_i - (R_cardinal @ entry_p4_j)
        translation[1] = 0.0

        rel_transform = np.eye(4, dtype=np.float64)
        rel_transform[:3, :3] = R_cardinal
        rel_transform[:3, 3] = translation

        transit_vec = target_entry_in_i - exit_p4_i

        return BridgeResult(
            relative_transform=rel_transform,
            yaw_cardinal_deg=cardinal_yaw_deg,
            raw_yaw_deg=raw_yaw_deg,
            translation=translation,
            exit_point_i=exit_p4_i,
            entry_point_j=entry_p4_j,
            transit_vector=transit_vec,
            automatically_discovered_thickness=discovered_thickness,
        )
