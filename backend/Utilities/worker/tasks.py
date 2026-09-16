"""GlomeHomeTour: Queue Worker Tasks for GPU Reconstruction & CAD Mesh Generation.

Runs asynchronous RQ jobs on the ROCm host with automatic VRAM cleanup.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from mesh_generation.pipeline import MeshGenerationPipeline
from mesh_generation.types import MeshPipelineConfig
from mesh_generation.io_adapter import load_splats_ply, load_transforms_json


def generate_cad_mesh_task(
    scene_id: str,
    ply_path: str,
    transforms_path: str,
    output_dir: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Asynchronous worker task: converts 2DGS radiance fields into watertight CAD/BIM package."""
    config = MeshPipelineConfig(device=device)
    pipeline = MeshGenerationPipeline(config=config)

    surfels = load_splats_ply(ply_path, device=device)
    transforms = load_transforms_json(transforms_path, device=device)

    manifest, paths = pipeline.run(
        surfels=surfels,
        transforms=transforms,
        output_dir=output_dir,
        scene_id=scene_id,
    )

    return {
        "status": "completed",
        "scene_id": scene_id,
        "total_triangles": manifest.total_triangles,
        "total_vertices": manifest.total_vertices,
        "manifest_path": str(paths["manifest"]),
        "glb_path": str(paths["glb"]),
        "dxf_path": str(paths["dxf"]),
        "ifc_path": str(paths["ifc"]),
    }
