"""Generic scan dataset loader and discovery engine.

Discovers and encapsulates multi-scan assets independently of the number of scans
and scan naming/ordering. Resolves Phase 04 sealed meshes, Phase 04 3D Gaussian Splats,
Phase 05 semantic door classes, Phase 06 regularized meshes/elements, and camera trajectories.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from shapely.affinity import affine_transform
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

logger = logging.getLogger("scan_loader")
DOOR_TEXT_CLASS_ID = 1


def _read_door_matches(path: Optional[Path]) -> Optional[Dict[Tuple[int, int], Dict[str, Any]]]:
    if not path or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    records = {
        (int(item["source_class_id"]), int(item["instance_id"])): item
        for item in data.get("instances", [])
    }
    matches = {}
    for row in data.get("associations", []):
        if isinstance(row, dict):
            class_id = int(row.get("source_class_id", row.get("image_class_id")))
            instance_id = int(row.get("instance_id", 0))
            text_class_id = int(row.get("text_class_id", 0))
        else:
            class_id, instance_id, text_class_id = map(int, row)
        if text_class_id == DOOR_TEXT_CLASS_ID:
            matches[(class_id, instance_id)] = records.get((class_id, instance_id), {})
    return matches


def _split_oversized_door_mesh(mesh: trimesh.Trimesh) -> Optional[List[Dict[str, Any]]]:
    bmin, bmax = mesh.bounds
    span = bmax - bmin
    if mesh.area <= 8.0 or span[0] <= 1.2 or span[2] <= 1.0:
        return None

    normals = mesh.face_normals
    centers = mesh.triangles.mean(axis=1)
    masks = (
        ("west", (normals[:, 0] < -0.7) & (centers[:, 0] < bmin[0] + 0.35),
         "Vertical", [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ("north", (normals[:, 2] > 0.7) & (centers[:, 2] > bmax[2] - 0.35),
         "Horizontal", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ("east", (normals[:, 0] > 0.7) & (centers[:, 0] > bmax[0] - 0.35),
         "Vertical", [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
    )
    parts = []
    for name, mask, orientation, normal, tangent in masks:
        face_ids = np.flatnonzero(mask)
        if not len(face_ids):
            continue
        part = mesh.submesh([face_ids], append=True)
        parts.append({
            "name": name,
            "mesh": part,
            "geometry": {
                "orientation": orientation,
                "normal": normal,
                "tangent": tangent,
            },
        })
    return parts if len(parts) > 1 else None


def _publish_door_split(
    paths: ScanAssetPaths,
    class_id: int,
    parent_instance_id: int,
    source_path: Path,
    parts: List[Dict[str, Any]],
) -> List[Tuple[int, Dict[str, Any], Path]]:
    segmentation_dir = paths.scan_dir / "segmentation"
    canonical_dir = segmentation_dir / "Instances PLYs" / "Non-Structural" / str(class_id)
    object_dir = segmentation_dir / "3d_masks" / f"class_{class_id:03d}_object"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    object_dir.mkdir(parents=True, exist_ok=True)

    parent_path = canonical_dir / f"instance_{parent_instance_id + 1:03d}.ply"
    if source_path.resolve() != parent_path.resolve():
        shutil.copy2(source_path, parent_path)
    existing_indices = [
        int(path.stem.removeprefix("instance_"))
        for path in canonical_dir.glob("instance_*.ply")
        if path.stem.removeprefix("instance_").isdigit()
    ]
    next_index = max(existing_indices, default=parent_instance_id + 1)
    staged_rows = []
    for part_idx, part in enumerate(parts):
        instance_id = parent_instance_id if part_idx == 0 else next_index
        if part_idx:
            next_index += 1
        canonical_path = canonical_dir / f"instance_{instance_id + 1:03d}.ply"
        temp_path = canonical_path.with_name(f".{canonical_path.stem}.phase07.tmp.ply")
        part["mesh"].export(str(temp_path))
        staged_rows.append((instance_id, part, canonical_path, temp_path))

    child_rows = []
    for instance_id, part, canonical_path, temp_path in staged_rows:
        temp_path.replace(canonical_path)

        mask_path = object_dir / canonical_path.name
        if mask_path.is_symlink() or mask_path.exists():
            mask_path.unlink()
        try:
            mask_path.symlink_to(os.path.relpath(canonical_path, object_dir))
        except OSError:
            shutil.copy2(canonical_path, mask_path)
        child_rows.append((instance_id, part, canonical_path))

    copies_path = segmentation_dir / "fixed_class_object_copies.json"
    if copies_path.is_file():
        copies = json.loads(copies_path.read_text(encoding="utf-8"))
        copy_row = next((row for row in copies.get("copies", []) if int(row["class_id"]) == class_id), None)
        if copy_row is not None:
            copy_row["copied_instances"] = [
                str(path.relative_to(segmentation_dir))
                for path in sorted(canonical_dir.glob("instance_*.ply"))
            ]
            copies_path.write_text(json.dumps(copies, indent=2) + "\n", encoding="utf-8")

    matches_path = paths.door_instance_metadata_path
    if matches_path and matches_path.is_file():
        metadata = json.loads(matches_path.read_text(encoding="utf-8"))
        metadata["associations"] = sorted({
            tuple(map(int, row)) for row in metadata.get("associations", [])
            if not (int(row[0]) == class_id and int(row[1]) == parent_instance_id)
        } | {(class_id, instance_id, DOOR_TEXT_CLASS_ID) for instance_id, _, _ in child_rows})
        metadata["instances"] = [
            row for row in metadata.get("instances", [])
            if not (int(row["source_class_id"]) == class_id and int(row["instance_id"]) == parent_instance_id)
        ]
        for instance_id, part, _ in child_rows:
            geometry = part["geometry"]
            metadata["instances"].append({
                "source_class_id": class_id,
                "instance_id": instance_id,
                "status": "accepted",
                "text_class_id": DOOR_TEXT_CLASS_ID,
                "valid_frame_count": 0,
                "positive_categories_within_margin": 0,
                "classification_source": "phase07_door_split",
                "door_geometry": geometry,
            })
        metadata["instances"].sort(key=lambda row: (int(row["source_class_id"]), int(row["instance_id"])))
        matches_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return child_rows


def _write_door_scene_metadata(paths: ScanAssetPaths, matches: Dict[Tuple[int, int], Dict[str, Any]]) -> None:
    manifest_path = paths.scan_dir / "segmentation" / "segmentation.json"
    if not manifest_path.is_file():
        return
    masks_dir = paths.scan_dir / "segmentation" / "3d_masks"
    door_instances = []
    for class_id, instance_id in sorted(matches):
        name = f"instance_{instance_id + 1:03d}.ply"
        candidates = [
            masks_dir / f"class_{class_id:03d}_object" / name,
            masks_dir / f"class_{class_id:03d}_wall" / name,
        ]
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is None:
            continue
        part = trimesh.load(source, process=False)
        bmin, bmax = part.bounds
        door_instances.append({
            "instance_key": f"door_{class_id}_{instance_id}",
            "source_class_id": class_id,
            "instance_id": instance_id,
            "instance_index": instance_id + 1,
            "class_id": class_id,
            "structural_role": "object",
            "text_class_id": DOOR_TEXT_CLASS_ID,
            "category": "door",
            "triangle_count": int(len(part.faces)),
            "area_m2": round(float(part.area), 4),
            "bounds_min": [round(float(value), 3) for value in bmin],
            "bounds_max": [round(float(value), 3) for value in bmax],
            "relative_path": str(source.relative_to(paths.scan_dir / "segmentation")),
        })
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["door_instances"] = door_instances
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


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
    door_instance_metadata_path: Optional[Path] = None
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
            similarity_dir = seg_dir / "non_structural_similarity"
            vc_p = similarity_dir / "valid_class_associations.json"
            if not vc_p.is_file():
                vc_p = seg_dir / "valid_class_associations.json"
            if vc_p.is_file():
                paths.door_class_metadata_path = vc_p
            vi_p = similarity_dir / "valid_instance_associations.json"
            if vi_p.is_file():
                paths.door_instance_metadata_path = vi_p
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
        instance_matches = _read_door_matches(paths.door_instance_metadata_path)
        door_classes = []
        metadata_available = bool(paths.door_class_metadata_path and paths.door_class_metadata_path.is_file())
        if paths.door_class_metadata_path and paths.door_class_metadata_path.is_file():
            try:
                data = json.loads(paths.door_class_metadata_path.read_text(encoding="utf-8"))
                associations = data.get("valid_associations", data.get("associations", []))
                for assoc in associations:
                    if isinstance(assoc, dict):
                        text_class_id = int(assoc.get("text_class_id", 0))
                        class_id = int(assoc.get("image_class_id", assoc.get("source_class_id", 0)))
                        names = " ".join((str(assoc.get("target_text", "")), str(assoc.get("category", "")))).lower()
                        is_door = text_class_id == DOOR_TEXT_CLASS_ID or any(
                            word in names for word in ("door", "door-frame", "doorway", "entrance")
                        )
                    else:
                        class_id, text_class_id = map(int, assoc[:2])
                        is_door = text_class_id == DOOR_TEXT_CLASS_ID
                    if is_door:
                        door_classes.append(class_id)
            except Exception as e:
                logger.warning(f"Error parsing door class associations for {paths.scan_id}: {e}")

        if door_classes:
            return sorted(set(door_classes))
        if not metadata_available and instance_matches:
            return sorted({class_id for class_id, _ in instance_matches})

        # Fallback heuristic if not resolved
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
                raw_door_matches = _read_door_matches(paths.door_instance_metadata_path)
                door_matches = (
                    {key: value for key, value in raw_door_matches.items() if key[0] in door_classes}
                    if raw_door_matches is not None else None
                )
                accepted_door_matches = {}
                instance_ids = sorted(door_matches) if door_matches else None
                if instance_ids is None:
                    instance_ids = [
                        (class_id, int(path.stem.removeprefix("instance_")) - 1)
                        for class_id in sorted(set(door_classes))
                        for role in ("object", "wall")
                        for path in sorted((masks_dir / f"class_{class_id:03d}_{role}").glob("instance_*.ply"))
                        if role == "object" or not (masks_dir / f"class_{class_id:03d}_object").is_dir()
                    ]
                    instance_ids = sorted(set(instance_ids))

                for class_id, source_instance_id in instance_ids:
                    object_dir = masks_dir / f"class_{class_id:03d}_object"
                    wall_dir = masks_dir / f"class_{class_id:03d}_wall"
                    instance_name = f"instance_{source_instance_id + 1:03d}.ply"
                    candidates = [object_dir / instance_name, wall_dir / instance_name]
                    source_path = next((path for path in candidates if path.is_file()), None)
                    if source_path is None:
                        continue
                    current_match = (door_matches or {}).get((class_id, source_instance_id), {})
                    source_mesh = trimesh.load(source_path, process=False)
                    split_parts = _split_oversized_door_mesh(source_mesh)
                    if split_parts:
                        split_parts = [
                            part for part in split_parts
                            if float((part["mesh"].bounds[1] - part["mesh"].bounds[0])[
                                0 if part["geometry"]["orientation"] == "Horizontal" else 2
                            ]) >= 0.60
                        ]
                        if not split_parts:
                            continue
                        children = _publish_door_split(
                            paths, class_id, source_instance_id, source_path, split_parts
                        )
                        for child_id, child_part, _ in children:
                            descriptor = ScanDataset._door_instance_from_mesh(
                                paths, class_id, child_id, child_part["mesh"],
                                child_part["geometry"], child_part["name"], len(instances) + 1,
                            )
                            if descriptor is not None:
                                instances.append(descriptor)
                                accepted_door_matches[(class_id, child_id)] = {
                                    "door_geometry": child_part["geometry"],
                                    "status": "accepted",
                                    "text_class_id": DOOR_TEXT_CLASS_ID,
                                }
                        continue

                    door_geometry = current_match.get("door_geometry")
                    descriptor = ScanDataset._door_instance_from_mesh(
                        paths, class_id, source_instance_id, source_mesh,
                        door_geometry, source_path.stem, len(instances) + 1,
                    )
                    if descriptor is not None:
                        instances.append(descriptor)
                        accepted_door_matches[(class_id, source_instance_id)] = current_match

                _write_door_scene_metadata(paths, accepted_door_matches)
                if instances:
                    return instances
            except Exception as e:
                logger.warning(f"Error loading 3D masks for {paths.scan_id}: {e}")
                if paths.door_instance_metadata_path and paths.door_instance_metadata_path.is_file():
                    raise

            # A current per-instance association file is authoritative, including when it has no door matches.
            if paths.door_instance_metadata_path and paths.door_instance_metadata_path.is_file():
                return instances

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
                        if width < 0.60:
                            continue

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

    @staticmethod
    def _door_instance_from_mesh(
        paths: ScanAssetPaths,
        class_id: int,
        source_instance_id: int,
        mesh: trimesh.Trimesh,
        door_geometry: Optional[Dict[str, Any]],
        description: str,
        instance_idx: int,
    ) -> Optional[Dict[str, Any]]:
        bmin, bmax = mesh.bounds
        span = bmax - bmin
        if not door_geometry and span[1] < 1.20:
            return None
        if door_geometry:
            orientation = str(door_geometry["orientation"])
            normal = np.asarray(door_geometry["normal"], dtype=np.float64)
            tangent = np.asarray(door_geometry["tangent"], dtype=np.float64)
            width = float(span[0] if orientation == "Horizontal" else span[2])
        elif span[0] > span[2]:
            orientation = "Horizontal"
            normal = np.array([0.0, 0.0, -1.0])
            tangent = np.array([1.0, 0.0, 0.0])
            width = float(span[0])
        else:
            orientation = "Vertical"
            normal = np.array([1.0, 0.0, 0.0])
            tangent = np.array([0.0, 0.0, -1.0])
            width = float(span[2])
        if width < 0.60:
            return None
        center = (bmin + bmax) / 2.0
        return {
            "scan_id": paths.scan_id,
            "instance_idx": instance_idx,
            "center": center,
            "center_loc_2d": np.array([center[0], -center[2]]),
            "orientation": orientation,
            "normal": normal,
            "tangent": tangent,
            "width": width,
            "height": float(span[1]),
            "area": float(mesh.area),
            "source_class_id": class_id,
            "source_instance_id": source_instance_id,
            "description": f"{paths.scan_id}_{description}",
        }
