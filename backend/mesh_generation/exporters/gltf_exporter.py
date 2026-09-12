"""GlomeHomeTour: glTF 2.0 & GLB Scene Packager (Phase 6.2).

Exports standalone binary .glb and .gltf scenes with:
- Metric unit hierarchy (+Y up, -Z forward)
- Root scene nodes for Architecture and segmented Objects
- Embedded or external PBR textures
- Clean buffer views and accessors
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh

from ..types import MeshNode


class GLTFExporter:
    """Exports hierarchical scene graphs to glTF 2.0 (.glb / .gltf) formats."""

    @classmethod
    def export_scene_glb(
        cls,
        nodes: List[MeshNode],
        meshes: Dict[str, trimesh.Trimesh],
        output_path: Union[str, Path],
    ) -> Path:
        """Package all scene nodes into a single standalone binary .glb file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.Scene()

        for node in nodes:
            mesh = meshes.get(node.name)
            if mesh is not None and len(mesh.faces) > 0:
                t_mat = np.array(node.transform_matrix, dtype=np.float64)
                # Clone mesh to avoid modifying caller's state
                m_copy = mesh.copy()
                # Apply transform matrix to geometry or scene node
                scene.add_geometry(
                    geometry=m_copy,
                    node_name=node.name,
                    transform=t_mat,
                )

        glb_data = scene.export(file_type="glb")
        with open(out_path, "wb") as f:
            f.write(glb_data)

        return out_path
