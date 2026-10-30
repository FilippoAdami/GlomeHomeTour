"""Read-only inputs shared by semantic-segmentation entry points."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = BACKEND / "current_scene"


def _is_scene(path: Path) -> bool:
    sparse = path / "sparse" / "0"
    return (path / "images").is_dir() and (sparse / "cameras.txt").is_file() and (sparse / "images.txt").is_file()


def discover_scans(workspace: Path) -> list[Path]:
    """Discover all scan directories under workspace or return workspace itself if it is a single scene."""
    workspace = Path(workspace)
    if not workspace.is_dir():
        if (BACKEND / workspace).is_dir():
            workspace = BACKEND / workspace
    if _is_scene(workspace):
        return [workspace]
    if not workspace.is_dir():
        raise FileNotFoundError(f"Scene workspace does not exist: {workspace}")
    scans = sorted(
        (path for path in workspace.iterdir()
         if path.is_dir() and path.name.startswith("scan_") and path.name[5:].isdigit()),
        key=lambda path: int(path.name[5:]),
    )
    if scans:
        invalid = [s.name for s in scans if not _is_scene(s)]
        if invalid:
            raise FileNotFoundError(f"Scans in {workspace} are missing images or sparse/0 COLMAP text inputs: {invalid}")
        return scans
    raise FileNotFoundError(f"No scene or scan_N directories with images and sparse/0 COLMAP text inputs in {workspace}")


def resolve_scene(workspace: Path) -> Path:
    """Resolve one scan directory without choosing an ambiguous scan silently."""
    workspace = Path(workspace)
    if not workspace.is_dir():
        if (BACKEND / workspace).is_dir():
            workspace = BACKEND / workspace
    if _is_scene(workspace):
        return workspace
    if not workspace.is_dir():
        raise FileNotFoundError(f"Scene workspace does not exist: {workspace}")
    scans = sorted(
        (path for path in workspace.iterdir()
         if path.is_dir() and path.name.startswith("scan_") and path.name[5:].isdigit()),
        key=lambda path: int(path.name[5:]),
    )
    if len(scans) > 1:
        names = ", ".join(scan.name for scan in scans)
        raise ValueError(f"Multiple scans in {workspace}: {names}; choose one scan_N directory")
    if len(scans) == 1:
        if _is_scene(scans[0]):
            return scans[0]
        raise FileNotFoundError(f"Scan is missing images or sparse/0 COLMAP text inputs: {scans[0]}")
    raise FileNotFoundError(f"No scene with images and sparse/0 COLMAP text inputs in {workspace}")


def stage_directory(workspace: Path) -> Path:
    return resolve_scene(workspace) / "segmentation"


def resolve_stage_output(workspace: Path, subdir: str | None = None, create: bool = True) -> Path:
    """Resolve stage output directory under the scene, optionally creating a subdirectory."""
    out = stage_directory(workspace)
    if subdir:
        out = out / subdir
    if create:
        out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_mesh(workspace: Path, mesh_path: Path | None = None) -> Path:
    """Resolve the Stage 04 authoritative mesh.ply, respecting explicit overrides."""
    if mesh_path is not None:
        p = Path(mesh_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit mesh path does not exist: {p}")
    scene = resolve_scene(workspace)
    # Stage 04 authoritative sealed mesh
    candidate = scene / "mesh.ply"
    if candidate.is_file():
        return candidate
    # Legacy experimental fallback
    fallback = scene / "05_semantic_segmentation/inputs/mesh.ply"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No mesh.ply found in scan {scene} or fallback {fallback}")


def resolve_mesh_report(workspace: Path) -> Path | None:
    """Resolve the Stage 04 consolidated mesh.json report."""
    scene = resolve_scene(workspace)
    candidate = scene / "mesh.json"
    return candidate if candidate.is_file() else None


def resolve_sparse_dir(workspace: Path) -> Path:
    """Resolve in-place Phase 04 poses, legacy aligned copies, or Stage 01 poses."""
    scene = resolve_scene(workspace)
    candidate = scene / "sparse" / "0"
    report = resolve_mesh_report(scene)
    if report is not None:
        try:
            alignment = json.loads(report.read_text(encoding="utf-8")).get("mesh_alignment", {})
            if alignment.get("poses_written_in_place") and (candidate / "images.txt").is_file():
                return candidate
        except (OSError, ValueError):
            pass
    aligned = scene / "phase04_alignment" / "sparse" / "0"
    if (scene / "phase04_alignment" / "alignment.json").is_file() \
            and (aligned / "images.txt").is_file():
        return aligned
    if candidate.is_dir() and (candidate / "images.txt").is_file():
        return candidate
    fallback = scene / "05_semantic_segmentation/inputs/sparse/0"
    if fallback.is_dir() and (fallback / "images.txt").is_file():
        return fallback
    raise FileNotFoundError(f"No COLMAP sparse/0 directory in {scene}")


def resolve_images_dir(workspace: Path) -> Path:
    """Resolve the registered images directory."""
    scene = resolve_scene(workspace)
    candidate = scene / "images"
    if candidate.is_dir():
        return candidate
    fallback = scene / "05_semantic_segmentation/inputs/images"
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"No images directory in {scene}")


def resolve_depth_dir(workspace: Path) -> Path | None:
    """Resolve the Stage 02 depth directory containing maps, normals, and poses_da3.npz."""
    scene = resolve_scene(workspace)
    candidate = scene / "depth"
    if candidate.is_dir() and (candidate / "poses_da3.npz").is_file():
        return candidate
    workspace = Path(workspace)
    alt = workspace / "02_depth_estimation" / "depth"
    if alt.is_dir() and (alt / "poses_da3.npz").is_file():
        return alt
    return candidate if candidate.is_dir() else None


def _camera_source_sizes(scene: Path) -> dict[str, tuple[int, int]]:
    sparse = resolve_sparse_dir(scene)
    sizes: dict[int, tuple[int, int]] = {}
    for line in (sparse / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("malformed COLMAP cameras.txt record")
        try:
            sizes[int(fields[0])] = (int(fields[2]), int(fields[3]))
        except ValueError as exc:
            raise ValueError("malformed COLMAP cameras.txt record") from exc

    result: dict[str, tuple[int, int]] = {}
    lines = (sparse / "images.txt").read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        if not lines[index] or lines[index].startswith("#"):
            index += 1
            continue
        if index + 1 >= len(lines):
            raise ValueError("malformed COLMAP images.txt: image header has no points line")
        fields = lines[index].split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError("malformed COLMAP images.txt image header")
        try:
            camera_id = int(fields[8])
        except ValueError as exc:
            raise ValueError("malformed COLMAP images.txt image header") from exc
        if camera_id not in sizes:
            raise ValueError(f"COLMAP image references unknown camera {camera_id}")
        if fields[9] in result:
            raise ValueError(f"duplicate COLMAP image name: {fields[9]}")
        result[fields[9]] = sizes[camera_id]
        index += 2
    return result


def load_scene_cameras(workspace: Path, resolution: int = 1) -> list[dict]:
    """Load stage-03 cameras and validate source image dimensions and NPZ-safe stems."""
    if not isinstance(resolution, int) or resolution < 1:
        raise ValueError("resolution must be a positive integer")
    scene = resolve_scene(workspace)
    stage03 = BACKEND / "03_FastGS_DNSplatter"
    if str(stage03) not in sys.path:
        sys.path.insert(0, str(stage03))
    from gsplat_train import load_cameras  # local: importing this helper stays lightweight

    cameras = load_cameras(scene, resolution=resolution, sparse_dir=resolve_sparse_dir(scene))
    expected_sizes = _camera_source_sizes(scene)
    stems: set[str] = set()
    for camera in cameras:
        image_path = Path(camera["image_path"])
        name = image_path.relative_to(scene / "images").as_posix()
        if image_path.stem in stems:
            raise ValueError("COLMAP image stems must be unique for semantic-segmentation NPZ keys")
        stems.add(image_path.stem)
        if name not in expected_sizes:
            raise ValueError(f"stage 03 returned image absent from COLMAP images.txt: {name}")
        with Image.open(image_path) as image:
            actual = image.size
        source_size = expected_sizes[name]
        if actual != source_size:
            raise ValueError(f"COLMAP camera dimensions for {name} are {source_size}, image is {actual}")
        expected = (source_size[0] // resolution, source_size[1] // resolution)
        if (camera["width"], camera["height"]) != expected:
            raise ValueError(f"stage 03 camera dimensions disagree for {name}")
    return cameras


def registered_images(workspace: Path) -> list[Path]:
    """Return registered RGB files sorted by their COLMAP-relative names."""
    scene = resolve_scene(workspace)
    images = [Path(camera["image_path"]) for camera in load_scene_cameras(scene)]
    return sorted(images, key=lambda path: path.relative_to(scene / "images").as_posix())


def colmap_point_counts(images_txt: Path) -> dict[str, int]:
    """Count triangulated (non--1) POINTS2D observations for every image record."""
    images_txt = Path(images_txt)
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    result: dict[str, int] = {}
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith("#"):
            index += 1
            continue
        if not lines[index].strip():
            raise ValueError("malformed COLMAP images.txt: unexpected empty image header")
        if index + 1 >= len(lines):
            raise ValueError("malformed COLMAP images.txt: image header has no points line")
        header, observations = lines[index], lines[index + 1]
        fields = header.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError("malformed COLMAP images.txt image header")
        try:
            int(fields[0])
            [float(value) for value in fields[1:8]]
            int(fields[8])
        except ValueError as exc:
            raise ValueError("malformed COLMAP images.txt image header") from exc
        name = fields[9]
        if name in result:
            raise ValueError(f"duplicate COLMAP image name: {name}")
        values = observations.split()
        if len(values) % 3:
            raise ValueError("malformed COLMAP images.txt POINTS2D record")
        count = 0
        for x, y, point_id in zip(values[::3], values[1::3], values[2::3]):
            try:
                float(x), float(y)
                if int(point_id) != -1:
                    count += 1
            except ValueError as exc:
                raise ValueError("malformed COLMAP images.txt POINTS2D record") from exc
        result[name] = count
        index += 2
    if not result:
        raise ValueError("COLMAP images.txt has no image records")
    return result


def load_mesh_transform(path: Path | None = None) -> np.ndarray:
    """Load an explicit proper rigid mesh-to-camera transform, or identity."""
    if path is None:
        return np.eye(4, dtype=np.float64)
    transform = np.asarray(np.load(Path(path)), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("mesh transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-8):
        raise ValueError("mesh transform must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("mesh transform rotation must be proper orthonormal")
    return transform


def resolve_mesh_transform(workspace: Path, path: Path | None = None) -> np.ndarray:
    """Resolve mesh-to-camera transform; defaults to identity since Stage 04 mesh is in COLMAP space."""
    if path is not None:
        p = Path(path)
        if p.is_file():
            return load_mesh_transform(p)
    try:
        scene = resolve_scene(workspace)
        candidate = scene / "05_semantic_segmentation/inputs/T_mesh_to_raw.npy"
        if candidate.is_file():
            return load_mesh_transform(candidate)
    except Exception:
        pass
    return np.eye(4, dtype=np.float64)



def validate_labels(labels: np.ndarray, camera: dict, classes=None) -> np.ndarray:
    """Validate a full-resolution nonnegative integer label map for one camera."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape != (camera["height"], camera["width"]):
        raise ValueError("labels must match the camera height and width")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
        raise ValueError("labels must have an integer dtype")
    if np.any(labels < 0):
        raise ValueError("labels must be nonnegative")
    if classes is not None:
        allowed = {int(value) for value in classes}
        unknown = set(np.unique(labels)) - {0} - allowed
        if unknown:
            raise ValueError(f"labels contain classes outside the allowed set: {sorted(unknown)}")
    return labels


def dino_grid(height: int, width: int, max_side: int = 896, patch: int = 14) -> tuple[int, int]:
    """Return the ViT patch grid used by ``run_dinov2_patches.py``."""
    if min(height, width, max_side, patch) < 1:
        raise ValueError("image dimensions, max_side, and patch must be positive")
    scale = max_side / max(width, height)
    return max(1, round(height * scale / patch)), max(1, round(width * scale / patch))
