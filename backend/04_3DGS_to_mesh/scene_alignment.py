"""Apply one rigid world-frame correction to Phase 04 geometry and pose products."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation


def _colmap_up(images_path: Path) -> np.ndarray:
    lines = images_path.read_text(encoding="utf-8").splitlines()
    ups = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10 or index + 1 >= len(lines):
            raise ValueError(f"Malformed COLMAP image record: {line}")
        q = np.asarray(fields[1:5], dtype=np.float64)
        camera_up = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().T @ [0., -1., 0.]
        ups.append(camera_up / np.linalg.norm(camera_up))
        index += 2  # Each image record is followed by a 2D observation line.
    if not ups:
        raise ValueError(f"No registered COLMAP poses in {images_path}")
    mean = np.mean(ups, axis=0)
    agreement = float(np.linalg.norm(mean))
    if agreement < .5:
        raise ValueError(f"COLMAP camera poses do not agree on an up direction ({agreement:.3f})")
    return mean / agreement


def _rotation_to_up(up: np.ndarray) -> np.ndarray:
    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    target = np.array([0., 1., 0.])
    cross = np.cross(up, target)
    cosine = float(np.clip(up @ target, -1., 1.))
    if np.linalg.norm(cross) < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1., -1., -1.])
    skew = np.array([[0., -cross[2], cross[1]],
                     [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]])
    return np.eye(3) + skew + skew @ skew / (1. + cosine)


def _wall_up(normals: np.ndarray, areas: np.ndarray, expected_up: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(normals).all(axis=1) & np.isfinite(areas) & (areas > 1e-10)
    normals, areas = normals[valid].copy(), areas[valid]
    if len(normals) < 2:
        return None
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    first = np.argmax(np.abs(normals) > 1e-8, axis=1)
    normals *= np.where(normals[np.arange(len(normals)), first] < 0., -1., 1.)[:, None]
    _, inverse = np.unique(np.rint(normals / .04).astype(np.int16), axis=0, return_inverse=True)
    weights = np.bincount(inverse, weights=areas)
    sums = np.zeros((len(weights), 3), dtype=np.float64)
    np.add.at(sums, inverse, normals * areas[:, None])
    modes = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
    candidates = np.argsort(weights)[-64:]
    best, support = None, 0.
    for i, first_mode in enumerate(candidates):
        for second_mode in candidates[:i]:
            if abs(float(modes[first_mode] @ modes[second_mode])) > .45:
                continue
            up = np.cross(modes[first_mode], modes[second_mode])
            length = float(np.linalg.norm(up))
            if length < 1e-8:
                continue
            up /= length
            if abs(float(up @ expected_up)) < math.cos(math.radians(25.)):
                continue
            score = float(weights[np.abs(modes @ up) <= math.sin(math.radians(10.))].sum())
            if score > support:
                best, support = up, score
    if best is None:
        return None
    inliers = np.abs(normals @ best) <= math.sin(math.radians(10.))
    for _ in range(2):
        if not inliers.any():
            return None
        covariance = (normals[inliers] * areas[inliers, None]).T @ normals[inliers]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[1] < .12 * eigenvalues.sum() or eigenvalues[1] / max(eigenvalues[2], 1e-12) < .35:
            return None
        best = eigenvectors[:, 0]
        inliers = np.abs(normals @ best) <= math.sin(math.radians(10.))
    if areas[inliers].sum() < .25 * areas.sum():
        return None
    if best @ expected_up < 0.:
        best = -best
    return best / np.linalg.norm(best)


def _wall_yaw(normals: np.ndarray, areas: np.ndarray) -> float:
    horizontal = normals[:, [0, 2]]
    keep = np.linalg.norm(horizontal, axis=1) > .85
    if not keep.any():
        return 0.
    angles = np.mod(np.arctan2(horizontal[keep, 1], horizontal[keep, 0]), np.pi / 2.)
    weights = areas[keep]
    edges = np.linspace(0., np.pi / 2., 181)
    hist, _ = np.histogram(angles, bins=edges, weights=weights)
    peak_i = int(np.argmax(hist))
    peak = (edges[peak_i] + edges[peak_i + 1]) * .5
    delta = np.abs(np.angle(np.exp(4j * (angles - peak)))) <= math.radians(12.)
    return float((.25 * np.angle(np.sum(weights[delta] * np.exp(4j * angles[delta])))) % (np.pi / 2.))


def _floor_reference_y(centers: np.ndarray, normals: np.ndarray, areas: np.ndarray) -> float:
    horizontal = np.abs(normals[:, 1]) >= math.cos(math.radians(20.))
    if not horizontal.any():
        return float(np.min(centers[:, 1]))
    y, weights = centers[horizontal, 1], areas[horizontal]
    order = np.argsort(y)
    groups: list[list[int]] = [[int(order[0])]]
    for index in order[1:]:
        if y[index] - y[groups[-1][0]] > .04:
            groups.append([])
        groups[-1].append(int(index))
    minimum_support = .02 * float(weights.sum())
    for group in groups:
        if weights[group].sum() >= minimum_support:
            return float(np.average(y[group], weights=weights[group]))
    return float(np.min(y))


def estimate_alignment(vertices: np.ndarray, triangles: np.ndarray, images_path: Path) -> dict:
    """Estimate mesh-to-aligned-world transform from wall and floor support."""
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    corners = vertices[triangles]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-10
    if not valid.any():
        raise ValueError("Cannot align a mesh without nondegenerate triangles")
    normals = cross[valid] / double_area[valid, None]
    areas = double_area[valid] * .5
    camera_up = _colmap_up(images_path)
    up = _wall_up(normals, areas, camera_up)
    if up is None:
        up = camera_up
        up_source = "COLMAP camera poses"
    else:
        up_source = "mesh wall-normal families"
    level = _rotation_to_up(up)
    level_normals = normals @ level.T
    yaw = _wall_yaw(level_normals, areas)
    c, s = math.cos(yaw), math.sin(yaw)
    align_yaw = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    rotation = align_yaw @ level
    centers = corners[valid].mean(axis=1) @ rotation.T
    aligned_normals = normals @ rotation.T
    floor_y = _floor_reference_y(centers, aligned_normals, areas)
    translation = np.array([0., -floor_y, 0.])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {"transform_matrix": transform.tolist(), "rotation_matrix": rotation.tolist(),
            "translation_world_units": translation.tolist(), "up_source": up_source,
            "yaw_correction_deg": round(math.degrees(yaw), 6),
            "floor_y_before_world_units": round(floor_y, 8),
            "floor_y_after_world_units": round(float((vertices @ rotation.T + translation)[..., 1].min()), 8)}


def transform_mesh_ply(path: Path, transform: np.ndarray) -> None:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY has no x/y/z vertex properties: {path}")
    rotation, translation = transform[:3, :3], transform[:3, 3]
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    moved = xyz @ rotation.T + translation
    vertex["x"], vertex["y"], vertex["z"] = moved.T
    if {"nx", "ny", "nz"}.issubset(names):
        normals = np.column_stack([vertex[name] for name in ("nx", "ny", "nz")]) @ rotation.T
        vertex["nx"], vertex["ny"], vertex["nz"] = normals.T
    temporary = path.with_name(path.name + ".aligned.tmp")
    ply.write(str(temporary))
    temporary.replace(path)


def _rotate_sh(sh0, shn, rotation: np.ndarray):
    import torch
    if shn.shape[1] == 0:
        return sh0, shn
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    basis_count = shn.shape[1] + 1
    if math.isqrt(basis_count) ** 2 != basis_count:
        raise ValueError(f"Unsupported Gaussian SH coefficient count: {basis_count}")
    sample_count = max(64, 4 * basis_count)
    index = np.arange(sample_count, dtype=np.float64) + .5
    z = 1. - 2. * index / sample_count
    phi = np.pi * (1. + 5. ** .5) * index
    radius = np.sqrt(1. - z * z)
    directions = np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    directions_old = directions @ rotation
    device, dtype = shn.device, shn.dtype
    with torch.no_grad():
        dirs_new = torch.as_tensor(directions, dtype=torch.float64)
        dirs_old = torch.as_tensor(directions_old, dtype=torch.float64)
        basis_new = _eval_sh_bases_fast(basis_count, dirs_new).cpu().numpy()
        basis_old = _eval_sh_bases_fast(basis_count, dirs_old).cpu().numpy()
    coeff_rotation = np.linalg.lstsq(basis_new, basis_old, rcond=None)[0]
    coeff_rotation[0, :] = 0.
    coeff_rotation[:, 0] = 0.
    coeff_rotation[0, 0] = 1.
    matrix = torch.as_tensor(coeff_rotation[1:, 1:], dtype=dtype, device=device)
    with torch.no_grad():
        for start in range(0, len(shn), 65536):
            stop = min(start + 65536, len(shn))
            shn[start:stop] = torch.einsum("ij,njc->nic", matrix, shn[start:stop].clone())
    return sh0, shn


def transform_splats(checkpoint_path: Path, ply_path: Path, transform: np.ndarray) -> None:
    import torch
    from infill_gsplat import export_splats_ply

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    splats = checkpoint.get("splats")
    if not isinstance(splats, dict) or not {"means", "quats", "sh0", "shN"}.issubset(splats):
        raise ValueError(f"Unsupported Gaussian checkpoint structure: {checkpoint_path}")
    rotation = transform[:3, :3]
    translation = torch.as_tensor(transform[:3, 3], dtype=splats["means"].dtype,
                                  device=splats["means"].device)
    r = Rotation.from_matrix(rotation).as_quat()
    delta = torch.as_tensor([r[3], r[0], r[1], r[2]], dtype=splats["quats"].dtype,
                            device=splats["quats"].device)
    old = splats["quats"]
    w1, v1 = delta[:1], delta[1:]
    w2, v2 = old[:, :1], old[:, 1:]
    splats["means"] = splats["means"] @ torch.as_tensor(rotation.T, dtype=splats["means"].dtype) + translation
    splats["quats"] = torch.nn.functional.normalize(torch.cat((w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
            w1 * v2 + w2 * v1 + torch.cross(v1.expand_as(v2), v2, dim=-1)), dim=-1), dim=-1)
    splats["sh0"], splats["shN"] = _rotate_sh(splats["sh0"], splats["shN"], rotation)
    checkpoint["splats"] = splats
    temporary = checkpoint_path.with_name(checkpoint_path.name + ".aligned.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    export_splats_ply(ply_path, splats)


def _transform_colmap(images_path: Path, points_path: Path, output: Path,
                      transform: np.ndarray) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    lines = images_path.read_text(encoding="utf-8").splitlines()
    transformed, index = [], 0
    while index < len(lines):
        line = lines[index]
        if not line or line.startswith("#"):
            transformed.append(line)
            index += 1
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10 or index + 1 >= len(lines):
            raise ValueError(f"Malformed COLMAP image record: {line}")
        q = np.asarray(fields[1:5], dtype=np.float64)
        world_to_camera = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        new_rotation = world_to_camera @ rotation.T
        new_translation = np.asarray(fields[5:8], dtype=np.float64) - new_rotation @ translation
        q_new = Rotation.from_matrix(new_rotation).as_quat()
        fields[1:8] = [f"{value:.12g}" for value in (q_new[3], *q_new[:3], *new_translation)]
        transformed.extend((" ".join(fields), lines[index + 1]))
        index += 2
    (output / "images.txt").write_text("\n".join(transformed) + "\n", encoding="utf-8")
    point_lines = []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            point_lines.append(line)
            continue
        fields = line.split()
        xyz = rotation @ np.asarray(fields[1:4], dtype=np.float64) + translation
        fields[1:4] = [f"{value:.12g}" for value in xyz]
        point_lines.append(" ".join(fields))
    (output / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")


def write_aligned_inputs(scene: Path, output: Path, transform: np.ndarray,
                         alignment: dict) -> None:
    sparse = scene / "sparse" / "0"
    aligned_sparse = output / "sparse" / "0"
    aligned_sparse.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sparse / "cameras.txt", output / "sparse" / "0" / "cameras.txt")
    _transform_colmap(sparse / "images.txt", sparse / "points3D.txt", aligned_sparse, transform)
    depth_pose = scene / "depth" / "poses_da3.npz"
    if depth_pose.is_file():
        with np.load(depth_pose) as source:
            data = {key: source[key] for key in source.files}
        if "w2c" in data:
            data["w2c"] = data["w2c"] @ np.linalg.inv(transform).astype(data["w2c"].dtype)
        (output / "depth").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "depth" / "poses_da3.npz", **data)
    alignment.update({"sparse_dir": str(aligned_sparse),
                      "depth_poses_path": str(output / "depth" / "poses_da3.npz")
                      if depth_pose.is_file() else None})
    (output / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")
