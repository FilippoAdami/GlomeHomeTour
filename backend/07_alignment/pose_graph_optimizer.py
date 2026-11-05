"""Multi-room pose graph optimizer and unified multi-asset scene delivery.

Builds a global pose graph from pairwise sequential room alignments and
loop-closure edges, solves for global house-wide coordinate frames,
and produces the three canonical aligned multi-room deliverables:
1. aligned_sealed_meshes.ply (all Phase 04 sealed meshes aligned)
2. aligned_regularized_meshes.ply (all Phase 06 regularized meshes aligned)
3. aligned_sealed_gaussians.ply (all Phase 04 sealed 3D Gaussian Splats aligned)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import plyfile
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import trimesh


@dataclass
class AlignmentEdge:
    """Directed edge in the multi-room alignment graph."""
    from_scan: str
    to_scan: str
    relative_transform: np.ndarray  # Transforms 'to_scan' into 'from_scan' frame
    weight: float = 1.0
    wall_thickness: float = 0.40
    door_residual_m: float = 0.0
    is_loop_closure: bool = False


class PoseGraphOptimizer:
    """Assembles pairwise scan registrations into unified multi-asset property deliverables."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.edges: List[AlignmentEdge] = []
        self.global_transforms: Dict[str, np.ndarray] = {}

    def add_edge(self, edge: AlignmentEdge) -> None:
        """Adds a pairwise alignment edge between two scans."""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.from_scan,
            edge.to_scan,
            transform=edge.relative_transform,
            weight=edge.weight,
            wall_thickness=edge.wall_thickness,
        )

    def solve_global_poses(self, anchor_scan: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Computes global 4x4 coordinate transformations for all scans."""
        nodes = sorted(list(self.graph.nodes()))
        if not nodes:
            return {}

        if anchor_scan is None or anchor_scan not in nodes:
            anchor_scan = nodes[0]

        global_poses = {anchor_scan: np.eye(4, dtype=np.float64)}
        visited = {anchor_scan}
        queue = [anchor_scan]

        while queue:
            curr = queue.pop(0)
            curr_global = global_poses[curr]

            for _, nbr, data in self.graph.out_edges(curr, data=True):
                if nbr not in visited:
                    T_curr_nbr = data["transform"]
                    global_poses[nbr] = curr_global @ T_curr_nbr
                    visited.add(nbr)
                    queue.append(nbr)

            for nbr, _, data in self.graph.in_edges(curr, data=True):
                if nbr not in visited:
                    T_nbr_curr = data["transform"]
                    global_poses[nbr] = curr_global @ np.linalg.inv(T_nbr_curr)
                    visited.add(nbr)
                    queue.append(nbr)

        if len(self.edges) > len(nodes) - 1:
            global_poses = self._optimize_pose_graph(nodes, anchor_scan, global_poses)

        self.global_transforms = global_poses
        return global_poses

    def _optimize_pose_graph(
        self,
        nodes: List[str],
        anchor: str,
        init_poses: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Levenberg-Marquardt / Nelder-Mead optimization over planar pose parameters [tx, tz, yaw]."""
        free_nodes = [n for n in nodes if n != anchor]
        if not free_nodes:
            return init_poses

        def pack_state(poses: Dict[str, np.ndarray]) -> np.ndarray:
            vec = []
            for n in free_nodes:
                T = poses[n]
                yaw_rad = np.arctan2(T[0, 2], T[2, 2])
                vec.extend([T[0, 3], T[2, 3], float(np.degrees(yaw_rad))])
            return np.array(vec, dtype=np.float64)

        def unpack_state(vec: np.ndarray) -> Dict[str, np.ndarray]:
            poses = {anchor: np.eye(4, dtype=np.float64)}
            idx = 0
            for n in free_nodes:
                tx, tz, yaw_deg = vec[idx : idx + 3]
                idx += 3
                rad = np.radians(yaw_deg)
                cos_y = np.cos(rad)
                sin_y = np.sin(rad)
                T = np.array([
                    [cos_y, 0.0, sin_y, tx],
                    [0.0,   1.0, 0.0,   0.0],
                    [-sin_y, 0.0, cos_y, tz],
                    [0.0,   0.0, 0.0,   1.0]
                ], dtype=np.float64)
                poses[n] = T
            return poses

        def error_func(vec: np.ndarray) -> float:
            current_poses = unpack_state(vec)
            total_err = 0.0
            for edge in self.edges:
                T_i = current_poses[edge.from_scan]
                T_j = current_poses[edge.to_scan]
                T_meas = np.linalg.inv(T_i) @ T_j
                T_target = edge.relative_transform

                dt = T_meas[:3, 3] - T_target[:3, 3]
                t_err = np.sum(dt ** 2)

                R_diff = T_meas[:3, :3] @ T_target[:3, :3].T
                rot_err = (np.trace(R_diff) - 3.0) ** 2

                total_err += edge.weight * (t_err + 0.5 * rot_err)
            return float(total_err)

        x0 = pack_state(init_poses)
        res = minimize(error_func, x0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-3})
        return unpack_state(res.x)

    def assemble_aligned_sealed_meshes(
        self,
        scan_directories: Dict[str, Path],
        output_path: Path,
    ) -> Optional[Path]:
        """Deliverable 1: Merges all Phase 04 sealed meshes (mesh.ply) aligned into one single PLY."""
        meshes = []
        for scan_name, scan_dir in sorted(scan_directories.items()):
            mesh_file = scan_dir / "mesh.ply"
            if mesh_file.is_file():
                try:
                    m = trimesh.load(mesh_file, process=False)
                    if isinstance(m, trimesh.Trimesh):
                        T = self.global_transforms.get(scan_name, np.eye(4))
                        m_trans = m.copy()
                        m_trans.apply_transform(T)
                        meshes.append(m_trans)
                except Exception:
                    pass

        if meshes:
            merged = trimesh.util.concatenate(meshes)
            merged.export(output_path)
            return output_path
        return None

    def assemble_aligned_regularized_meshes(
        self,
        scan_directories: Dict[str, Path],
        output_path: Path,
    ) -> Optional[Path]:
        """Deliverable 2: Merges all Phase 06 regularized meshes (complete.ply) aligned into one single PLY."""
        meshes = []
        for scan_name, scan_dir in sorted(scan_directories.items()):
            mesh_file = scan_dir / "mesh_elaboration" / "complete.ply"
            if not mesh_file.is_file():
                mesh_file = scan_dir / "mesh.ply"
            if mesh_file.is_file():
                try:
                    m = trimesh.load(mesh_file, process=False)
                    if isinstance(m, trimesh.Trimesh):
                        T = self.global_transforms.get(scan_name, np.eye(4))
                        m_trans = m.copy()
                        m_trans.apply_transform(T)
                        meshes.append(m_trans)
                except Exception:
                    pass

        if meshes:
            merged = trimesh.util.concatenate(meshes)
            merged.export(output_path)
            return output_path
        return None

    def assemble_aligned_sealed_gaussians(
        self,
        scan_directories: Dict[str, Path],
        output_path: Path,
    ) -> Optional[Path]:
        """Deliverable 3: Merges all Phase 04 sealed Gaussian Splat models (gaussian_splat_sealed.ply) aligned into one single PLY."""
        all_vertex_arrays = []

        for scan_name, scan_dir in sorted(scan_directories.items()):
            gs_file = scan_dir / "gaussian_splat" / "gaussian_splat_sealed.ply"
            if not gs_file.is_file():
                gs_file = scan_dir / "gaussian_splat" / "gaussian_splat.ply"
            if not gs_file.is_file():
                continue

            try:
                ply = plyfile.PlyData.read(gs_file)
                v = ply["vertex"].data.copy()
                T = self.global_transforms.get(scan_name, np.eye(4))
                R = T[:3, :3]
                t = T[:3, 3]

                # 1. Transform positions: x' = R @ x + t
                xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
                xyz_new = (R @ xyz.T).T + t
                v["x"] = xyz_new[:, 0].astype(np.float32)
                v["y"] = xyz_new[:, 1].astype(np.float32)
                v["z"] = xyz_new[:, 2].astype(np.float32)

                # 2. Transform normals if present
                if "nx" in v.dtype.names and "ny" in v.dtype.names and "nz" in v.dtype.names:
                    norm = np.stack([v["nx"], v["ny"], v["nz"]], axis=-1)
                    norm_new = (R @ norm.T).T
                    v["nx"] = norm_new[:, 0].astype(np.float32)
                    v["ny"] = norm_new[:, 1].astype(np.float32)
                    v["nz"] = norm_new[:, 2].astype(np.float32)

                # 3. Transform rotation quaternions: q' = q_R ⊗ q_G
                # In 3DGS, rot_0 is w, rot_1 is x, rot_2 is y, rot_3 is z
                r_scipy = Rotation.from_matrix(R)
                qx, qy, qz, qw = r_scipy.as_quat()

                w_G = v["rot_0"]
                x_G = v["rot_1"]
                y_G = v["rot_2"]
                z_G = v["rot_3"]

                w_new = qw * w_G - qx * x_G - qy * y_G - qz * z_G
                x_new = qw * x_G + qx * w_G + qy * z_G - qz * y_G
                y_new = qw * y_G - qx * z_G + qy * w_G + qz * x_G
                z_new = qw * z_G + qx * y_G - qy * x_G + qz * w_G

                norms = np.sqrt(w_new**2 + x_new**2 + y_new**2 + z_new**2)
                norms[norms == 0] = 1.0

                v["rot_0"] = (w_new / norms).astype(np.float32)
                v["rot_1"] = (x_new / norms).astype(np.float32)
                v["rot_2"] = (y_new / norms).astype(np.float32)
                v["rot_3"] = (z_new / norms).astype(np.float32)

                all_vertex_arrays.append(v)
            except Exception:
                pass

        if all_vertex_arrays:
            merged_data = np.concatenate(all_vertex_arrays)
            merged_el = plyfile.PlyElement.describe(merged_data, "vertex")
            merged_ply = plyfile.PlyData([merged_el], text=False, byte_order="<")
            merged_ply.write(output_path)
            return output_path
        return None

    def assemble_all_deliverables(
        self,
        scan_directories: Dict[str, Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Generates all 3 canonical aligned multi-room deliverables and alignment manifest."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.global_transforms:
            self.solve_global_poses()

        # Deliverable 1: Aligned sealed meshes (Phase 04)
        p1 = output_dir / "aligned_sealed_meshes.ply"
        path_sealed = self.assemble_aligned_sealed_meshes(scan_directories, p1)

        # Deliverable 2: Aligned regularized meshes (Phase 06)
        p2 = output_dir / "aligned_regularized_meshes.ply"
        path_reg = self.assemble_aligned_regularized_meshes(scan_directories, p2)

        # Deliverable 3: Aligned sealed Gaussian Splats (Phase 04)
        p3 = output_dir / "aligned_sealed_gaussians.ply"
        path_gs = self.assemble_aligned_sealed_gaussians(scan_directories, p3)

        # Also preserve property_complete.ply compatibility alias pointing to regularized mesh
        compat_path = output_dir / "property_complete.ply"
        if path_reg and path_reg.is_file():
            import shutil
            shutil.copy2(path_reg, compat_path)

        scan_reports = {
            sname: {
                "global_transform": self.global_transforms.get(sname, np.eye(4)).tolist(),
                "scan_dir": str(sdir),
            }
            for sname, sdir in sorted(scan_directories.items())
        }

        manifest = {
            "schema_version": "1.0.0",
            "anchor_scan": sorted(list(self.global_transforms.keys()))[0],
            "scans": scan_reports,
            "edges": [
                {
                    "from_scan": e.from_scan,
                    "to_scan": e.to_scan,
                    "relative_transform": e.relative_transform.tolist(),
                    "automatically_discovered_wall_thickness_m": e.wall_thickness,
                    "door_residual_m": e.door_residual_m,
                    "is_loop_closure": e.is_loop_closure,
                }
                for e in self.edges
            ],
            "deliverables": {
                "aligned_sealed_meshes_ply": str(path_sealed) if path_sealed else None,
                "aligned_regularized_meshes_ply": str(path_reg) if path_reg else None,
                "aligned_sealed_gaussians_ply": str(path_gs) if path_gs else None,
            },
        }

        manifest_path = output_dir / "multi_room_alignment.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
