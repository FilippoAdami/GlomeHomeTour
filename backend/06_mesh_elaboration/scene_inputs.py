"""Read-only input resolution shared by Stage 06 mesh elaboration entry points.

Discovers and validates locked upstream outputs:
- Stage 04: Authoritative sealed mesh (mesh.ply) and report (mesh.json)
- Stage 05: triangle_class_labels.npy, class_metadata.json, segmentation.json, Instances PLYs/, 3d_masks/
- Stage 01/02: Optional camera poses, depth maps, and scene_size.txt
- Stage 03/04: Optional sealed 3D Gaussian Splat model
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple


BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = BACKEND / "current_scene"


def _is_scene(path: Path) -> bool:
    """Check if a directory looks like a scan root (has mesh.ply, sparse/0, or segmentation)."""
    if not path.is_dir():
        return False
    has_mesh = (path / "mesh.ply").is_file()
    has_sparse = (path / "sparse" / "0" / "cameras.txt").is_file()
    has_segmentation = (path / "segmentation").is_dir()
    return has_mesh or has_sparse or has_segmentation


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
        valid_scans = [s for s in scans if _is_scene(s)]
        if valid_scans:
            return valid_scans
        raise FileNotFoundError(f"Scans in {workspace} do not contain expected scene artifacts (mesh.ply, sparse/0, or segmentation/): {[s.name for s in scans]}")
    raise FileNotFoundError(f"No scene or scan_N directories found in {workspace}")


def resolve_scene(workspace: Path) -> Path:
    """Resolve exactly one scan directory without silently guessing when multiple exist."""
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
        raise ValueError(f"Multiple scans in {workspace}: {names}; specify one scan directory")
    if len(scans) == 1:
        if _is_scene(scans[0]):
            return scans[0]
        raise FileNotFoundError(f"Scan directory is missing expected scene artifacts: {scans[0]}")
    raise FileNotFoundError(f"No valid scan or scene directory found in {workspace}")


def resolve_mesh(scene: Path, mesh_path: Path | None = None) -> Path:
    """Resolve the Stage 04 authoritative sealed mesh.ply, respecting explicit overrides."""
    if mesh_path is not None:
        p = Path(mesh_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit mesh path does not exist: {p}")
    candidate = scene / "mesh.ply"
    if candidate.is_file():
        return candidate
    # Fallback to compatibility location
    fallback = scene / "06_mesh_elaboration" / "original_mesh.ply"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No mesh.ply found in scan {scene}")


def resolve_labels(scene: Path, labels_path: Path | None = None) -> Path:
    """Resolve the Stage 05 triangle_class_labels.npy, respecting explicit overrides."""
    if labels_path is not None:
        p = Path(labels_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit labels path does not exist: {p}")
    candidate = scene / "segmentation" / "triangle_class_labels.npy"
    if candidate.is_file():
        return candidate
    fallback = scene / "06_mesh_elaboration" / "generic_graph_regions" / "triangle_class_labels.npy"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No triangle_class_labels.npy found in {scene / 'segmentation'}")


def resolve_structural_metadata(scene: Path, structural_json: Path | None = None) -> Path:
    """Resolve Stage 05 class_metadata.json (or segmentation.json / structural_classes.json)."""
    if structural_json is not None:
        p = Path(structural_json)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit structural metadata JSON path does not exist: {p}")
    seg_dir = scene / "segmentation"
    for filename in ("class_metadata.json", "segmentation.json", "structural_classes.json"):
        candidate = seg_dir / filename
        if candidate.is_file():
            return candidate
    fallback = scene / "06_mesh_elaboration" / "generic_graph_regions" / "structural_classes.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No structural metadata JSON found in {seg_dir}")


def resolve_segmentation_output(scene: Path, segmentation_output: Path | None = None) -> Optional[Path]:
    """Resolve Stage 05 segmentation delivery output directory."""
    if segmentation_output is not None:
        p = Path(segmentation_output)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Explicit segmentation output directory does not exist: {p}")
    candidate = scene / "segmentation"
    if candidate.is_dir() and (candidate / "class_metadata.json").is_file():
        return candidate
    fallback = scene / "05_semantic_segmentation" / "mesh_objects_step3" / "segmentation_output"
    if fallback.is_dir():
        return fallback
    return candidate if candidate.is_dir() else None


def resolve_output_dir(scene: Path, output_dir: Path | None = None, subdir: str = "mesh_elaboration") -> Path:
    """Resolve Stage 06 output directory, defaulting to <scene>/mesh_elaboration."""
    if output_dir is not None:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    out = scene / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_scene_size_file(scene: Path, scene_size_file: Path | None = None) -> Optional[Path]:
    """Resolve scene size file (from Stage 01/02) if available."""
    if scene_size_file is not None:
        p = Path(scene_size_file)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit scene size file does not exist: {p}")
    for cand in (
        scene / "depth" / "scene_size.txt",
        scene / "scene_size.json",
        scene / "metric_scale.json",
    ):
        if cand.is_file():
            return cand
    return None


def resolve_gaussian_splat(scene: Path, splat_path: Path | None = None) -> Optional[Path]:
    """Resolve Stage 03/04 sealed Gaussian Splat model PLY if available."""
    if splat_path is not None:
        p = Path(splat_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicit Gaussian Splat model does not exist: {p}")
    gs_dir = scene / "gaussian_splat"
    if not gs_dir.is_dir():
        return None
    # 1. Sealed sibling from Stage 04
    sealed = gs_dir / "gaussian_splat_sealed.ply"
    if sealed.is_file():
        return sealed
    direct = gs_dir / "gaussian_splat.ply"
    if direct.is_file():
        return direct
    # 2. Stage 03 trained iterations
    pts = sorted(gs_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if pts:
        return pts[-1]
    return None
