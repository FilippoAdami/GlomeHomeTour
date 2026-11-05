"""Single-Scan 3D Asset & Camera Pose Synchronization Engine.

Applies solved global SE(3) coordinate transformations directly to each individual scan's:
1. Triangular Meshes:
   - Sealed watertight mesh (mesh.ply)
   - Phase 06 regularized complete mesh (mesh_elaboration/complete.ply)
   - Phase 06 structural elements (mesh_elaboration/elements/*.ply)
2. 3D Gaussian Splats:
   - gaussian_splat/*.ply (positions, normals, quaternion rotations q' = q_R ⊗ q_G)
3. Camera Poses:
   - ARCore transit trajectory (trajectory.csv: positions and orientations)
   - NeRF / Instant-NGP / 3DGS camera transforms (transforms.json: T_c2w' = T_scan * T_c2w)
   - COLMAP reconstruction (sparse/0/: images, points3D, cameras via pycolmap)

Maintains full idempotency by snapshotting pristine source assets into .raw_unaligned/
on first alignment.
"""
from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import plyfile
from scipy.spatial.transform import Rotation
import trimesh

try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False

logger = logging.getLogger("single_scan_synchronizer")


def _copy_transformed_output(source: Path, destinations: List[Path]) -> None:
    for destination in destinations:
        if source.resolve() == destination.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_mesh_occupancy_scenes(meshes: Dict[str, trimesh.Trimesh]) -> Dict[str, Any]:
    """Build Open3D point-in-mesh scenes for aligned, sealed scan meshes."""
    if not meshes:
        return {}

    import open3d as o3d

    scenes = {}
    for scan_id, mesh in meshes.items():
        if not mesh.is_watertight:
            raise ValueError(f"Cannot test Gaussian containment: sealed mesh for {scan_id} is not watertight")
        legacy_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        )
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))
        scenes[scan_id] = (scene, mesh.bounds)
    return scenes


def filter_gaussians_inside_other_meshes(
    vertices: np.ndarray,
    scan_id: str,
    mesh_scenes: Dict[str, Any],
) -> np.ndarray:
    """Drop Gaussian centers enclosed by another scan's aligned sealed mesh."""
    if not mesh_scenes:
        return vertices

    import open3d as o3d

    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    keep = np.ones(len(vertices), dtype=bool)
    for other_scan, (scene, bounds) in mesh_scenes.items():
        if other_scan == scan_id:
            continue
        if not keep.any():
            break
        in_bounds = np.all((xyz >= bounds[0]) & (xyz <= bounds[1]), axis=1)
        candidate_indices = np.flatnonzero(keep & in_bounds)
        if not len(candidate_indices):
            continue
        occupied = scene.compute_occupancy(
            o3d.core.Tensor(xyz[candidate_indices], dtype=o3d.core.Dtype.Float32), nsamples=3
        ).numpy().astype(bool)
        keep[candidate_indices[occupied]] = False

    removed = int((~keep).sum())
    if removed:
        logger.info("Removed %d Gaussian(s) from %s inside another scan's sealed mesh", removed, scan_id)
    return vertices[keep]


def backup_raw_scan_if_needed(scan_dir: Path) -> Path:
    """Creates a pristine backup of unaligned assets in .raw_unaligned/ if not present."""
    backup_dir = scan_dir / ".raw_unaligned"
    if backup_dir.is_dir():
        return backup_dir

    logger.info(f"Creating pristine unaligned backup for {scan_dir.name} in .raw_unaligned/...")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1. mesh.ply
    if (scan_dir / "mesh.ply").is_file():
        shutil.copy2(scan_dir / "mesh.ply", backup_dir / "mesh.ply")

    # 2. gaussian_splat
    gs_dir = scan_dir / "gaussian_splat"
    if gs_dir.is_dir():
        shutil.copytree(gs_dir, backup_dir / "gaussian_splat", dirs_exist_ok=True)

    # 3. mesh_elaboration
    me_dir = scan_dir / "mesh_elaboration"
    if me_dir.is_dir():
        shutil.copytree(me_dir, backup_dir / "mesh_elaboration", dirs_exist_ok=True)

    # 4. trajectory.csv
    if (scan_dir / "trajectory.csv").is_file():
        shutil.copy2(scan_dir / "trajectory.csv", backup_dir / "trajectory.csv")

    # 5. transforms.json
    if (scan_dir / "transforms.json").is_file():
        shutil.copy2(scan_dir / "transforms.json", backup_dir / "transforms.json")

    # 6. sparse
    sp_dir = scan_dir / "sparse"
    if sp_dir.is_dir():
        shutil.copytree(sp_dir, backup_dir / "sparse", dirs_exist_ok=True)

    return backup_dir


def transform_mesh_ply(
    src_ply: Path,
    dst_ply: Path,
    transform: np.ndarray,
    additional_dst_paths: Optional[List[Path]] = None,
) -> None:
    """Transforms a triangular mesh PLY using a 4x4 SE(3) matrix."""
    if not src_ply.is_file():
        return
    m = trimesh.load(src_ply, process=False)
    m.apply_transform(transform)
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(dst_ply))
    _copy_transformed_output(dst_ply, additional_dst_paths or [])
    logger.debug(f"Transformed mesh {src_ply.name} -> {dst_ply}")


def transform_gaussian_splat_vertices(
    vertices: np.ndarray,
    transform: np.ndarray,
    mesh_scenes: Optional[Dict[str, Any]] = None,
    scan_id: Optional[str] = None,
) -> np.ndarray:
    """Transform and optionally filter one structured Gaussian vertex array."""
    v = vertices.copy()
    R = transform[:3, :3]
    t = transform[:3, 3]

    # 1. Positions
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    xyz_new = (R @ xyz.T).T + t
    v["x"] = xyz_new[:, 0].astype(np.float32)
    v["y"] = xyz_new[:, 1].astype(np.float32)
    v["z"] = xyz_new[:, 2].astype(np.float32)

    if mesh_scenes and scan_id is not None:
        v = filter_gaussians_inside_other_meshes(v, scan_id, mesh_scenes)

    # 2. Normals
    if "nx" in v.dtype.names and "ny" in v.dtype.names and "nz" in v.dtype.names:
        norm = np.stack([v["nx"], v["ny"], v["nz"]], axis=-1)
        norm_new = (R @ norm.T).T
        v["nx"] = norm_new[:, 0].astype(np.float32)
        v["ny"] = norm_new[:, 1].astype(np.float32)
        v["nz"] = norm_new[:, 2].astype(np.float32)

    # 3. Quaternions: q' = q_R ⊗ q_G
    # In 3DGS PLY format: rot_0 is w, rot_1 is x, rot_2 is y, rot_3 is z
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

    return v


def write_gaussian_splat_vertices(vertices: np.ndarray, dst_ply: Path) -> None:
    el = plyfile.PlyElement.describe(vertices, "vertex")
    out_ply = plyfile.PlyData([el], text=False, byte_order="<")
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    out_ply.write(str(dst_ply))


def transform_gaussian_splat_ply(
    src_ply: Path,
    dst_ply: Path,
    transform: np.ndarray,
    mesh_scenes: Optional[Dict[str, Any]] = None,
    scan_id: Optional[str] = None,
    additional_dst_paths: Optional[List[Path]] = None,
) -> None:
    """Transforms a 3D Gaussian Splat PLY with positions, normals, and quaternion composition."""
    if not src_ply.is_file():
        return

    vertices = plyfile.PlyData.read(str(src_ply))["vertex"].data
    v = transform_gaussian_splat_vertices(vertices, transform, mesh_scenes, scan_id)
    write_gaussian_splat_vertices(v, dst_ply)
    _copy_transformed_output(dst_ply, additional_dst_paths or [])
    logger.debug(f"Transformed Gaussian Splat {src_ply.name} ({len(v)} splats) -> {dst_ply}")


def transform_trajectory_csv(
    src_csv: Path,
    dst_csv: Path,
    transform: np.ndarray,
    additional_dst_paths: Optional[List[Path]] = None,
) -> None:
    """Transforms ARCore trajectory poses (positions and quaternions)."""
    if not src_csv.is_file():
        return

    R = transform[:3, :3]
    t = transform[:3, 3]
    r_rot = Rotation.from_matrix(R)

    with open(src_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    col_map = {col.strip(): idx for idx, col in enumerate(header)}
    tx_i = col_map.get("tx")
    ty_i = col_map.get("ty")
    tz_i = col_map.get("tz")
    qx_i = col_map.get("qx")
    qy_i = col_map.get("qy")
    qz_i = col_map.get("qz")
    qw_i = col_map.get("qw")

    if None in (tx_i, ty_i, tz_i, qx_i, qy_i, qz_i, qw_i):
        logger.warning(f"trajectory.csv missing required pose columns in {src_csv}")
        return

    new_rows = []
    for row in rows:
        if not row:
            continue
        row_copy = list(row)
        p = np.array([float(row[tx_i]), float(row[ty_i]), float(row[tz_i])], dtype=np.float64)
        p_new = R @ p + t
        row_copy[tx_i] = f"{p_new[0]:.6f}"
        row_copy[ty_i] = f"{p_new[1]:.6f}"
        row_copy[tz_i] = f"{p_new[2]:.6f}"

        q_orig = [float(row[qx_i]), float(row[qy_i]), float(row[qz_i]), float(row[qw_i])]
        rot_orig = Rotation.from_quat(q_orig)
        rot_new = r_rot * rot_orig
        q_new = rot_new.as_quat()
        row_copy[qx_i] = f"{q_new[0]:.6f}"
        row_copy[qy_i] = f"{q_new[1]:.6f}"
        row_copy[qz_i] = f"{q_new[2]:.6f}"
        row_copy[qw_i] = f"{q_new[3]:.6f}"
        new_rows.append(row_copy)

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_rows)
    _copy_transformed_output(dst_csv, additional_dst_paths or [])
    logger.debug(f"Transformed {len(new_rows)} ARCore trajectory poses -> {dst_csv}")


def transform_transforms_json(
    src_json: Path,
    dst_json: Path,
    transform: np.ndarray,
    additional_dst_paths: Optional[List[Path]] = None,
) -> None:
    """Transforms NeRF / COLMAP camera-to-world frame matrices T_c2w' = T_scan * T_c2w."""
    if not src_json.is_file():
        return

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for frame in data.get("frames", []):
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        mat_new = transform @ mat
        frame["transform_matrix"] = mat_new.tolist()

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _copy_transformed_output(dst_json, additional_dst_paths or [])
    logger.debug(f"Transformed {len(data.get('frames', []))} camera poses in {dst_json}")


def transform_colmap_sparse(
    src_sparse: Path,
    dst_sparse: Path,
    transform: np.ndarray,
    additional_dst_sparse_paths: Optional[List[Path]] = None,
) -> None:
    """Transforms COLMAP sparse model (images, points3D, cameras) using pycolmap."""
    sub_0_src = src_sparse / "0" if (src_sparse / "0").is_dir() else src_sparse
    if not (sub_0_src / "images.bin").is_file() and not (sub_0_src / "images.txt").is_file():
        return

    destination_roots = [dst_sparse, *(additional_dst_sparse_paths or [])]
    sub_0_destinations = [root / "0" for root in destination_roots]
    for destination in sub_0_destinations:
        destination.mkdir(parents=True, exist_ok=True)

    if HAS_PYCOLMAP:
        try:
            recon = pycolmap.Reconstruction(str(sub_0_src))
            sim3 = pycolmap.Sim3d(transform[:3, :])
            recon.transform(sim3)
            for sub_0_dst in sub_0_destinations:
                try:
                    recon.write_text(str(sub_0_dst))
                    recon.write_binary(str(sub_0_dst))
                except Exception as e:
                    logger.warning(f"pycolmap error transforming {src_sparse}: {e}")
            logger.debug(
                f"Transformed COLMAP reconstruction ({len(recon.images)} cameras, "
                f"{len(recon.points3D)} points)"
            )
        except Exception as e:
            logger.warning(f"pycolmap error transforming {src_sparse}: {e}")
    else:
        logger.warning("pycolmap not installed; skipping binary COLMAP model transformation.")

    # Also transform points3D.ply if present
    pts_ply = sub_0_src / "points3D.ply"
    if pts_ply.is_file():
        try:
            m = trimesh.load(pts_ply, process=False)
            m.apply_transform(transform)
            transformed_ply = sub_0_destinations[0] / "points3D.ply"
            m.export(str(transformed_ply))
            _copy_transformed_output(
                transformed_ply,
                [destination / "points3D.ply" for destination in sub_0_destinations[1:]],
            )
        except Exception as e:
            logger.warning(f"Error transforming points3D.ply: {e}")


def apply_transformation_to_single_scan(
    scan_dir: Path,
    transform: np.ndarray,
    export_dir: Optional[Path] = None,
    gaussian_mesh_scenes: Optional[Dict[str, Any]] = None,
    pretransformed_gaussian_splats: Optional[Dict[Path, Path]] = None,
) -> Dict[str, Any]:
    """Applies SE(3) transformation to all single PLYs and camera poses for a scan.

    Updates scan_dir in-place while keeping pristine originals in scan_dir/.raw_unaligned/.
    Optionally also exports a synchronized copy to export_dir.
    """
    scan_dir = Path(scan_dir).resolve()
    raw_dir = backup_raw_scan_if_needed(scan_dir)

    is_identity = np.allclose(transform, np.eye(4), atol=1e-7)
    logger.info(f"Synchronizing single scan assets for {scan_dir.name} (is_identity={is_identity})...")

    targets = [scan_dir]
    if export_dir is not None:
        export_p = Path(export_dir).resolve()
        export_p.mkdir(parents=True, exist_ok=True)
        if export_p != scan_dir:
            targets.append(export_p)

    primary_target = targets[0]
    additional_targets = targets[1:]

    updated_assets = []

    # Transform each source once. The aligned export is copied from that result.
    src_mesh = raw_dir / "mesh.ply"
    if src_mesh.is_file():
        dst_mesh = primary_target / "mesh.ply"
        transform_mesh_ply(
            src_mesh, dst_mesh, transform,
            [target / "mesh.ply" for target in additional_targets],
        )
        updated_assets.extend(str(target / "mesh.ply") for target in targets)

    src_comp = raw_dir / "mesh_elaboration" / "complete.ply"
    if src_comp.is_file():
        dst_comp = primary_target / "mesh_elaboration" / "complete.ply"
        transform_mesh_ply(
            src_comp, dst_comp, transform,
            [target / "mesh_elaboration" / "complete.ply" for target in additional_targets],
        )
        updated_assets.extend(str(target / "mesh_elaboration" / "complete.ply") for target in targets)

    src_elem = raw_dir / "mesh_elaboration" / "elements"
    if src_elem.is_dir():
        primary_elements = primary_target / "mesh_elaboration" / "elements"
        primary_elements.mkdir(parents=True, exist_ok=True)
        for elem_ply in src_elem.glob("*.ply"):
            rel = elem_ply.name
            transform_mesh_ply(
                elem_ply, primary_elements / rel, transform,
                [target / "mesh_elaboration" / "elements" / rel for target in additional_targets],
            )
            updated_assets.extend(
                str(target / "mesh_elaboration" / "elements" / rel) for target in targets
            )

    src_gs_dir = raw_dir / "gaussian_splat"
    if src_gs_dir.is_dir():
        primary_gs_dir = primary_target / "gaussian_splat"
        primary_gs_dir.mkdir(parents=True, exist_ok=True)
        for gs_ply in src_gs_dir.glob("**/*.ply"):
            rel = gs_ply.relative_to(src_gs_dir)
            cached_output = (pretransformed_gaussian_splats or {}).get(rel)
            destinations = [primary_gs_dir / rel,
                            *(target / "gaussian_splat" / rel for target in additional_targets)]
            if cached_output is not None:
                if not cached_output.is_file():
                    raise FileNotFoundError(f"Missing transformed Gaussian splat: {cached_output}")
                _copy_transformed_output(cached_output, destinations)
            else:
                transform_gaussian_splat_ply(
                    gs_ply, destinations[0], transform, gaussian_mesh_scenes, scan_dir.name,
                    destinations[1:],
                )
            updated_assets.extend(str(target / "gaussian_splat" / rel) for target in targets)

    src_traj = raw_dir / "trajectory.csv"
    if src_traj.is_file():
        transform_trajectory_csv(
            src_traj, primary_target / "trajectory.csv", transform,
            [target / "trajectory.csv" for target in additional_targets],
        )
        updated_assets.extend(str(target / "trajectory.csv") for target in targets)

    src_tf = raw_dir / "transforms.json"
    if src_tf.is_file():
        transform_transforms_json(
            src_tf, primary_target / "transforms.json", transform,
            [target / "transforms.json" for target in additional_targets],
        )
        updated_assets.extend(str(target / "transforms.json") for target in targets)

    src_sp = raw_dir / "sparse"
    if src_sp.is_dir():
        transform_colmap_sparse(
            src_sp, primary_target / "sparse", transform,
            [target / "sparse" for target in additional_targets],
        )
        updated_assets.extend(str(target / "sparse") for target in targets)

    return {
        "scan_id": scan_dir.name,
        "is_identity": is_identity,
        "transform_matrix": transform.tolist(),
        "updated_assets_count": len(updated_assets),
        "target_directories": [str(t) for t in targets],
    }
