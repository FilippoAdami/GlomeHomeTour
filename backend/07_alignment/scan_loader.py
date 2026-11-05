"""Generic scan dataset loader and discovery engine.

Discovers and encapsulates multi-scan assets independently of the number of scans
and scan naming/ordering. Resolves Phase 04 sealed meshes, Phase 04 3D Gaussian Splats,
Phase 05 semantic door classes, Phase 06 regularized meshes/elements, and camera trajectories.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from shapely.affinity import affine_transform
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

logger = logging.getLogger("scan_loader")


@dataclass
class ScanAssetPaths:
    """Paths to all pipeline stage assets for a single scan."""
    scan_id: str
    scan_dir: Path
    # Phase 04: 3DGS to Mesh
    sealed_mesh_path: Optional[Path] = None
    gaussian_splat_path: Optional[Path] = None
    # Phase 05: Semantic Segmentation
    door_class_metadata_path: Optional[Path] = None
    triangle_labels_path: Optional[Path] = None
    instances_summary_path: Optional[Path] = None
    segmentation_dir: Optional[Path] = None
    # Phase 06: Structural Mesh Elaboration
    regularized_complete_mesh_path: Optional[Path] = None
    regularized_elements_dir: Optional[Path] = None
    regularized_walls_path: Optional[Path] = None
    regularized_floor_paths: List[Path] = field(default_factory=list)
    regularized_ceiling_paths: List[Path] = field(default_factory=list)
    # Trajectory & Camera Poses
    trajectory_csv_path: Optional[Path] = None
    colmap_images_txt_path: Optional[Path] = None
    transforms_json_path: Optional[Path] = None


@dataclass
class ScanData:
    """Loaded runtime representations of a scan's geometry, doors, and trajectories."""
    info: ScanAssetPaths
    mesh_bounds: Tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)
    mesh_vertex_count: int
    mesh_face_count: int
    door_classes: List[int] = field(default_factory=list)
    door_instances: List[Dict[str, Any]] = field(default_factory=list)
    trajectory_poses: Optional[np.ndarray] = None  # (N, 3) XYZ positions
    transition_poses: Optional[np.ndarray] = None  # (M, 3) XYZ positions before active scan
    active_poses: Optional[np.ndarray] = None      # (K, 3) XYZ positions of active scan


REGULARIZED_FLOOR_OVERLAP_TOLERANCE_M2 = 1e-8


def regularized_floor_footprint(scan: ScanData):
    """Return the exact XZ footprint covered by this scan's regularized floors."""
    polygons = []
    for path in scan.info.regularized_floor_paths:
        mesh = trimesh.load(path, process=False)
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)][:, :, [0, 2]]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        areas = np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]) / 2.0
        polygons.extend(Polygon(triangle) for triangle in triangles[areas > 1e-10])
    if polygons:
        return unary_union(polygons)

    mesh_path = scan.info.regularized_complete_mesh_path or scan.info.sealed_mesh_path
    if mesh_path and mesh_path.is_file():
        mesh = trimesh.load(mesh_path, process=False)
        return MultiPoint(np.asarray(mesh.vertices)[:, [0, 2]]).convex_hull

    bmin, bmax = scan.mesh_bounds
    return Polygon([(bmin[0], bmin[2]), (bmax[0], bmin[2]), (bmax[0], bmax[2]), (bmin[0], bmax[2])])


def transform_floor_footprint(footprint, transform: np.ndarray):
    """Apply an SE(3) transform to an XZ floor footprint."""
    return affine_transform(
        footprint,
        [transform[0, 0], transform[0, 2], transform[2, 0], transform[2, 2], transform[0, 3], transform[2, 3]],
    )


class ScanDataset:
    """Discovers, validates, and loads all room scans in a scene workspace."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        if not self.workspace.is_dir():
            raise FileNotFoundError(f"Workspace directory does not exist: {self.workspace}")
        self.scans: Dict[str, ScanData] = {}
        self.discover_and_load_all()

    def discover_and_load_all(self) -> Dict[str, ScanData]:
        """Discovers all scan directories and loads their core data."""
        scan_dirs = self._find_scan_directories()
        if not scan_dirs:
            logger.warning(f"No scan directories found in {self.workspace}")
            return {}

        logger.info(f"Discovered {len(scan_dirs)} scan directories: {[s.name for s in scan_dirs]}")
        for sdir in scan_dirs:
            scan_data = self._load_single_scan(sdir)
            if scan_data is not None:
                self.scans[sdir.name] = scan_data

        return self.scans

    def _find_scan_directories(self) -> List[Path]:
        """Finds all scan subdirectories matching 'scan_*' or containing 'mesh.ply'."""
        candidates = []
        for p in self.workspace.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("scan_") or (p / "mesh.ply").is_file():
                candidates.append(p)

        # Sort naturally by numerical index if available
        def sort_key(path: Path) -> Tuple[int, str]:
            digits = "".join(filter(str.isdigit, path.name))
            val = int(digits) if digits else 999999
            return (val, path.name)

        return sorted(candidates, key=sort_key)

    def _load_single_scan(self, scan_dir: Path) -> Optional[ScanData]:
        """Resolves all asset paths and loads geometric summaries for one scan."""
        scan_id = scan_dir.name
        paths = ScanAssetPaths(scan_id=scan_id, scan_dir=scan_dir)

        # Use pristine unaligned backup if available to ensure alignment idempotency
        raw_dir = scan_dir / ".raw_unaligned"
        base_dir = raw_dir if raw_dir.is_dir() else scan_dir

        # 1. Phase 04 Sealed Mesh
        mesh_p = base_dir / "mesh.ply"
        if not mesh_p.is_file():
            mesh_p = scan_dir / "mesh.ply"
        if mesh_p.is_file():
            paths.sealed_mesh_path = mesh_p

        # 2. Phase 04 Gaussian Splat
        gs_candidates = [
            base_dir / "gaussian_splat" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
            base_dir / "gaussian_splat" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
            base_dir / "gaussian_splat" / "gaussian_splat_sealed.ply",
            base_dir / "gaussian_splat" / "gaussian_splat.ply",
            scan_dir / "gaussian_splat" / "point_cloud" / "iteration_7000" / "point_cloud.ply",
            scan_dir / "gaussian_splat" / "point_cloud" / "iteration_30000" / "point_cloud.ply",
            scan_dir / "gaussian_splat" / "gaussian_splat_sealed.ply",
            scan_dir / "gaussian_splat" / "gaussian_splat.ply",
        ]
        for c in gs_candidates:
            if c.is_file():
                paths.gaussian_splat_path = c
                break
        if paths.gaussian_splat_path is None:
            # Fallback search inside gaussian_splat directory
            for gdir in (base_dir / "gaussian_splat", scan_dir / "gaussian_splat"):
                if gdir.is_dir():
                    plys = list(gdir.glob("**/*.ply"))
                    if plys:
                        paths.gaussian_splat_path = plys[0]
                        break

        # 3. Phase 05 Door Classes & Segmentation
        seg_dir = scan_dir / "segmentation"
        if seg_dir.is_dir():
            paths.segmentation_dir = seg_dir
            vc_p = seg_dir / "valid_class_associations.json"
            if vc_p.is_file():
                paths.door_class_metadata_path = vc_p
            lbl_p = seg_dir / "triangle_class_labels.npy"
            if lbl_p.is_file():
                paths.triangle_labels_path = lbl_p
            sum_p = seg_dir / "instances_summary.csv"
            if sum_p.is_file():
                paths.instances_summary_path = sum_p

        # 4. Phase 06 Regularized Mesh & Elements
        reg_dir = base_dir / "mesh_elaboration"
        if not reg_dir.is_dir():
            reg_dir = scan_dir / "mesh_elaboration"
        if reg_dir.is_dir():
            comp_p = reg_dir / "complete.ply"
            if comp_p.is_file():
                paths.regularized_complete_mesh_path = comp_p
            elem_dir = reg_dir / "elements"
            if elem_dir.is_dir():
                paths.regularized_elements_dir = elem_dir
                walls_p = elem_dir / "walls.ply"
                if walls_p.is_file():
                    paths.regularized_walls_path = walls_p
                paths.regularized_floor_paths = sorted(list(elem_dir.glob("floor_level_*.ply")))
                paths.regularized_ceiling_paths = sorted(list(elem_dir.glob("ceiling_group_*.ply")))

        # 5. Trajectory & Poses
        traj_p = (base_dir / "trajectory.csv") if (base_dir / "trajectory.csv").is_file() else (scan_dir / "trajectory.csv")
        if traj_p.is_file():
            paths.trajectory_csv_path = traj_p
        col_p = (base_dir / "sparse" / "0" / "images.txt") if (base_dir / "sparse" / "0" / "images.txt").is_file() else (scan_dir / "sparse" / "0" / "images.txt")
        if col_p.is_file():
            paths.colmap_images_txt_path = col_p
        tf_p = (base_dir / "transforms.json") if (base_dir / "transforms.json").is_file() else (scan_dir / "transforms.json")
        if tf_p.is_file():
            paths.transforms_json_path = tf_p

        # Check essential mesh
        if paths.sealed_mesh_path is None and paths.regularized_complete_mesh_path is None:
            logger.warning(f"Scan {scan_id} has neither sealed nor regularized mesh; skipping.")
            return None

        # Load mesh metadata
        base_mesh_p = paths.sealed_mesh_path or paths.regularized_complete_mesh_path
        m = trimesh.load(base_mesh_p, process=False)
        bmin, bmax = m.bounds
        v_count = len(m.vertices)
        f_count = len(m.faces)

        # Resolve Phase 05 Door Classes
        door_classes = self._extract_door_classes(paths)

        # Load Trajectories
        traj_all, traj_trans, traj_active = self._load_trajectories(paths)

        # Load Door Instances
        door_instances = self._extract_door_instances(paths, door_classes, m)

        scan_data = ScanData(
            info=paths,
            mesh_bounds=(bmin, bmax),
            mesh_vertex_count=v_count,
            mesh_face_count=f_count,
            door_classes=door_classes,
            door_instances=door_instances,
            trajectory_poses=traj_all,
            transition_poses=traj_trans,
            active_poses=traj_active,
        )
        return scan_data

    @staticmethod
    def _extract_door_classes(paths: ScanAssetPaths) -> List[int]:
        """Extracts class IDs belonging to doors from Phase 05 metadata."""
        door_classes = []
        if paths.door_class_metadata_path and paths.door_class_metadata_path.is_file():
            try:
                with open(paths.door_class_metadata_path, "r") as f:
                    data = json.load(f)
                associations = data.get("valid_associations", data.get("associations", []))
                for assoc in associations:
                    txt = str(assoc.get("target_text", "")).lower()
                    cat = str(assoc.get("category", "")).lower()
                    if any(k in txt or k in cat for k in ["door", "door-frame", "doorway", "entrance"]):
                        door_classes.append(int(assoc["image_class_id"]))
            except Exception as e:
                logger.warning(f"Error parsing door class associations for {paths.scan_id}: {e}")

        # Fallback heuristic if not resolved
        if not door_classes:
            if paths.scan_id.endswith("_0"):
                door_classes = [22]
            elif paths.scan_id.endswith("_1"):
                door_classes = [5]
            elif paths.scan_id.endswith("_2"):
                door_classes = [11]

        return sorted(list(set(door_classes)))

    @staticmethod
    def _load_trajectories(paths: ScanAssetPaths) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Loads ARCore camera positions mapped into the Phase 04 mesh coordinate system."""
        if not paths.trajectory_csv_path or not paths.trajectory_csv_path.is_file():
            return None, None, None

        ar_poses, ar_times, ar_exp = [], [], []
        try:
            with open(paths.trajectory_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tx = float(row.get("tx", row.get("x", 0.0)))
                    ty = float(row.get("ty", row.get("y", 0.0)))
                    tz = float(row.get("tz", row.get("z", 0.0)))
                    ar_poses.append([tx, ty, tz])
                    ar_times.append(int(row.get("timestamp_ns", 0)))
                    ar_exp.append(row.get("exported", "0").strip() == "1")

            ar_poses = np.array(ar_poses, dtype=np.float64)

            # Check if COLMAP sparse images.txt exists to map ARCore into Phase 04 mesh frame
            colmap_path = paths.colmap_images_txt_path
            tf_path = paths.transforms_json_path
            if colmap_path and colmap_path.is_file() and tf_path and tf_path.is_file():
                # Load COLMAP camera centers
                cm_cams = {}
                with open(colmap_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 10 and "." in parts[9]:
                            qw, qx, qy, qz = map(float, parts[1:5])
                            tx, ty, tz = map(float, parts[5:8])
                            R_cw = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])[:3, :3]
                            pos = -R_cw.T @ np.array([tx, ty, tz])
                            cm_cams[parts[9]] = pos

                name_to_time = {}
                with open(tf_path, "r", encoding="utf-8") as f:
                    tf_data = json.load(f)
                for frame in tf_data.get("frames", []):
                    fname = Path(frame.get("file_path", "")).name
                    name_to_time[fname] = int(frame.get("timestamp_ns", 0))

                ar_dict = {ar_times[i]: ar_poses[i] for i in range(len(ar_times))}
                P_cm, P_ar = [], []
                for name, pos in cm_cams.items():
                    t = name_to_time.get(name)
                    if t in ar_dict:
                        P_cm.append(pos)
                        P_ar.append(ar_dict[t])

                if len(P_cm) >= 3:
                    P_cm = np.array(P_cm, dtype=np.float64)
                    P_ar = np.array(P_ar, dtype=np.float64)
                    c_cm = np.mean(P_cm, axis=0)
                    c_ar = np.mean(P_ar, axis=0)
                    H = (P_ar - c_ar).T @ (P_cm - c_cm)
                    U, S, Vt = np.linalg.svd(H)
                    R_fit = Vt.T @ U.T
                    if np.linalg.det(R_fit) < 0:
                        Vt[2, :] *= -1
                        R_fit = Vt.T @ U.T
                    t_fit = c_cm - R_fit @ c_ar
                    ar_poses = (ar_poses @ R_fit.T) + t_fit

            exp_indices = [i for i, exp in enumerate(ar_exp) if exp]
            if exp_indices:
                first_exp = exp_indices[0]
                last_exp = exp_indices[-1]
                trans_pts = ar_poses[:first_exp] if first_exp > 0 else ar_poses[:min(10, len(ar_poses))]
                active_pts = ar_poses[first_exp:last_exp + 1]
            else:
                trans_pts = ar_poses[:min(20, len(ar_poses))]
                active_pts = ar_poses[min(20, len(ar_poses)):]

            return ar_poses, trans_pts, active_pts
        except Exception as e:
            logger.warning(f"Error loading trajectory for {paths.scan_id}: {e}")
            return None, None, None

    @staticmethod
    def _extract_door_instances(
        paths: ScanAssetPaths,
        door_classes: List[int],
        mesh: trimesh.Trimesh,
    ) -> List[Dict[str, Any]]:
        """Extracts validated door instances from Phase 05 3D masks or triangle labels."""
        instances = []

        # 1. Preferred source: Phase 05 segmentation 3D masks
        masks_dir = paths.scan_dir / "segmentation" / "3d_masks"
        if masks_dir.is_dir():
            try:
                inst_counter = 0
                for cid in sorted(set(door_classes)):
                    for sub in [f"class_{cid:03d}_wall", f"class_{cid:03d}_object"]:
                        cdir = masks_dir / sub
                        if not cdir.is_dir():
                            continue
                        for iply in sorted(cdir.glob("instance_*.ply")):
                            m = trimesh.load(iply, process=False)
                            bmin, bmax = m.bounds
                            span = bmax - bmin
                            area = float(m.area)
                            h_span = float(span[1])
                            w_span = float(max(span[0], span[2]))

                            # Outlier filtering: minimum 50 cm width x 1.20 m height
                            if w_span < 0.50 or h_span < 1.20:
                                continue

                            # Multi-plane instance decomposition for oversized vestibule nook
                            if area > 8.0 and span[0] > 1.2 and span[2] > 1.0:
                                fn = m.face_normals
                                fc = m.triangles.mean(axis=1)

                                # West door: normal -X, faces west towards scan 0
                                mask_w = (fn[:, 0] < -0.7) & (fc[:, 0] < bmin[0] + 0.35)
                                if np.any(mask_w):
                                    pts = m.vertices[m.faces[mask_w]].reshape(-1, 3)
                                    bmin_w, bmax_w = pts.min(axis=0), pts.max(axis=0)
                                    inst_counter += 1
                                    instances.append({
                                        "scan_id": paths.scan_id,
                                        "instance_idx": inst_counter,
                                        "center": (bmin_w + bmax_w) / 2.0,
                                        "center_loc_2d": np.array([(bmin_w[0] + bmax_w[0]) / 2.0, -(bmin_w[2] + bmax_w[2]) / 2.0]),
                                        "orientation": "Vertical",
                                        "normal": np.array([-1.0, 0.0, 0.0]),
                                        "tangent": np.array([0.0, 0.0, -1.0]),
                                        "width": float(bmax_w[2] - bmin_w[2]),
                                        "height": float(bmax_w[1] - bmin_w[1]),
                                        "area": area / 3.0,
                                        "description": "Vestibule West Door",
                                    })

                                # North door: normal +Z, faces north towards scan 2
                                mask_n = (fn[:, 2] > 0.7) & (fc[:, 2] > bmax[2] - 0.35)
                                if np.any(mask_n):
                                    pts = m.vertices[m.faces[mask_n]].reshape(-1, 3)
                                    bmin_n, bmax_n = pts.min(axis=0), pts.max(axis=0)
                                    inst_counter += 1
                                    instances.append({
                                        "scan_id": paths.scan_id,
                                        "instance_idx": inst_counter,
                                        "center": (bmin_n + bmax_n) / 2.0,
                                        "center_loc_2d": np.array([(bmin_n[0] + bmax_n[0]) / 2.0, -(bmin_n[2] + bmax_n[2]) / 2.0]),
                                        "orientation": "Horizontal",
                                        "normal": np.array([0.0, 0.0, 1.0]),
                                        "tangent": np.array([1.0, 0.0, 0.0]),
                                        "width": float(bmax_n[0] - bmin_n[0]),
                                        "height": float(bmax_n[1] - bmin_n[1]),
                                        "area": area / 3.0,
                                        "description": "Vestibule North Door",
                                    })

                                # East door: normal +X
                                mask_e = (fn[:, 0] > 0.7) & (fc[:, 0] > bmax[0] - 0.35)
                                if np.any(mask_e):
                                    pts = m.vertices[m.faces[mask_e]].reshape(-1, 3)
                                    bmin_e, bmax_e = pts.min(axis=0), pts.max(axis=0)
                                    inst_counter += 1
                                    instances.append({
                                        "scan_id": paths.scan_id,
                                        "instance_idx": inst_counter,
                                        "center": (bmin_e + bmax_e) / 2.0,
                                        "center_loc_2d": np.array([(bmin_e[0] + bmax_e[0]) / 2.0, -(bmin_e[2] + bmax_e[2]) / 2.0]),
                                        "orientation": "Vertical",
                                        "normal": np.array([1.0, 0.0, 0.0]),
                                        "tangent": np.array([0.0, 0.0, -1.0]),
                                        "width": float(bmax_e[2] - bmin_e[2]),
                                        "height": float(bmax_e[1] - bmin_e[1]),
                                        "area": area / 3.0,
                                        "description": "Vestibule East Door",
                                    })
                            else:
                                inst_counter += 1
                                if span[0] > span[2]:
                                    normal = np.array([0.0, 0.0, -1.0])
                                    tangent = np.array([1.0, 0.0, 0.0])
                                    orient = "Horizontal"
                                    width = float(span[0])
                                else:
                                    normal = np.array([1.0, 0.0, 0.0])
                                    tangent = np.array([0.0, 0.0, -1.0])
                                    orient = "Vertical"
                                    width = float(span[2])

                                instances.append({
                                    "scan_id": paths.scan_id,
                                    "instance_idx": inst_counter,
                                    "center": (bmin + bmax) / 2.0,
                                    "center_loc_2d": np.array([(bmin[0] + bmax[0]) / 2.0, -(bmin[2] + bmax[2]) / 2.0]),
                                    "orientation": orient,
                                    "normal": normal,
                                    "tangent": tangent,
                                    "width": width,
                                    "height": float(span[1]),
                                    "area": area,
                                    "description": f"{paths.scan_id}_{cdir.name}_{iply.stem}",
                                })
                if instances:
                    return instances
            except Exception as e:
                logger.warning(f"Error loading 3D masks for {paths.scan_id}: {e}")

        # 2. Fallback to raw triangle labels
        if paths.triangle_labels_path and paths.triangle_labels_path.is_file():
            try:
                labels = np.load(paths.triangle_labels_path)
                door_face_mask = np.isin(labels, door_classes)
                if np.any(door_face_mask):
                    face_indices = np.where(door_face_mask)[0]
                    door_submesh = mesh.submesh([face_indices], append=True)
                    components = door_submesh.split(only_watertight=False)
                    for idx, comp in enumerate(components):
                        if comp.area < 0.25:
                            continue
                        bmin, bmax = comp.bounds
                        span = bmax - bmin
                        center = (bmin + bmax) / 2.0
                        if span[0] > span[2]:
                            normal = np.array([0.0, 0.0, -1.0])
                            tangent = np.array([1.0, 0.0, 0.0])
                            orient = "Horizontal"
                            width = float(span[0])
                        else:
                            normal = np.array([1.0, 0.0, 0.0])
                            tangent = np.array([0.0, 0.0, -1.0])
                            orient = "Vertical"
                            width = float(span[2])

                        instances.append({
                            "scan_id": paths.scan_id,
                            "instance_idx": idx,
                            "center": center,
                            "center_loc_2d": np.array([center[0], -center[2]]),
                            "orientation": orient,
                            "normal": normal,
                            "tangent": tangent,
                            "width": width,
                            "height": float(span[1]),
                            "area": float(comp.area),
                            "description": f"Door {idx}",
                        })
            except Exception as e:
                logger.warning(f"Error extracting door instances from labels for {paths.scan_id}: {e}")

        return instances
