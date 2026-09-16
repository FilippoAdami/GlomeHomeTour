"""GlomeHomeTour: End-to-End Mesh Generation Pipeline (Phase 7.2).

Orchestrates all phases:
- Phase 1: Ingests 2DGS surfels (`splats.ply`) and metric poses (`transforms.json`).
- Phase 2: Hybrid Geometric-Semantic 3D instance segmentation.
- Phase 3: Architectural infill, Manhattan RANSAC planar regularization, and watertight room shell extraction.
- Phase 4: Prototype de-duplication, canonical multi-view rendering, and Pixal3D generative object completion.
- Phase 5: UV chart unwrapping and 2048x2048 PBR texture baking.
- Phase 6: Hierarchical scene assembly and multi-format export (.glb, .dxf, .ifc, mesh_manifest.json).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import trimesh

from mesh_types import (
    BoundingBox3D,
    MeshManifest,
    MeshNode,
    MeshPipelineConfig,
    SurfelCloudTorch,
)
from io_adapter import (
    TransformsDataset,
    load_splats_ply,
    load_transforms_json,
)
from segmentation import (
    GeometricArchitectureFilter,
    SemanticInstanceSegmenter,
)
from planar_snapping import ManhattanPlanarRANSAC
from architectural_infill import (
    ArchitecturalInfillEngine,
    InfillResult,
)
from prototype_cluster import (
    PrototypeClusterEngine,
    PrototypeGroup,
)
from object_reconstruction import (
    CanonicalViewRenderer,
    OBBAligner,
    Pixal3DReconstructionEngine,
)
from texture_baking import PBRTextureBaker
from scene_assembler import SceneAssembler


class MeshGenerationPipeline:
    """Complete automated pipeline from 2DGS radiance fields to CAD/BIM packages."""

    def __init__(self, config: Optional[MeshPipelineConfig] = None):
        self.config = config or MeshPipelineConfig()
        self.segmenter = SemanticInstanceSegmenter(config=self.config)
        self.infill_engine = ArchitecturalInfillEngine()
        self.cluster_engine = PrototypeClusterEngine(
            color_sim_threshold=self.config.prototype_dedup_threshold
        )
        self.renderer = CanonicalViewRenderer()
        self.pixal3d = Pixal3DReconstructionEngine(config=self.config)
        self.baker = PBRTextureBaker(texture_resolution=self.config.texture_resolution)

    def run(
        self,
        surfels: SurfelCloudTorch,
        transforms: TransformsDataset,
        output_dir: Union[str, Path],
        scene_id: str = "scene_001",
    ) -> Tuple[MeshManifest, Dict[str, Path]]:
        """Run full pipeline on provided surfels and camera trajectory."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        textures_dir = out_dir / "textures"
        textures_dir.mkdir(parents=True, exist_ok=True)

        # Move tensors to configured device
        device = self.config.device if (self.config.device == "cuda" and torch.cuda.is_available()) else "cpu"
        surfels = surfels.to(device)
        transforms = transforms.to(device)

        # -------------------------------------------------------------
        # Phase 2: Segmentation
        # -------------------------------------------------------------
        arch_masks, instance_clusters = self.segmenter.cluster_foreground_instances(
            surfels=surfels,
            transforms=transforms,
        )

        arch_indices = torch.nonzero(arch_masks["architecture"], as_tuple=True)[0]
        fg_indices = torch.nonzero(arch_masks["foreground"], as_tuple=True)[0]

        arch_surfels = surfels.slice(arch_indices)
        fg_surfels = surfels.slice(fg_indices)

        # -------------------------------------------------------------
        # Phase 3: Architectural Infill & Watertight Shell
        # -------------------------------------------------------------
        void_boxes = [c.bounding_box for c in instance_clusters]
        infill_result: InfillResult = self.infill_engine.process_architectural_infill(
            arch_surfels=arch_surfels,
            foreground_surfels=fg_surfels,
            void_bounds=void_boxes,
        )
        arch_mesh = infill_result.shell_mesh

        # Bake PBR textures for architectural shell
        unwrapped_arch, arch_tex = self.baker.bake_pbr_textures(
            mesh=arch_mesh,
            surfels=infill_result.infilled_surfels,
            base_color=(0.85, 0.85, 0.80),
            roughness=0.85,
            metallic=0.0,
        )
        arch_material = arch_tex.save(textures_dir, prefix="architecture")

        # -------------------------------------------------------------
        # Phase 4: Prototype De-duplication & Generative Completion
        # -------------------------------------------------------------
        object_nodes: List[MeshNode] = []
        object_meshes: Dict[str, trimesh.Trimesh] = {}

        if instance_clusters:
            prototypes = self.cluster_engine.cluster_prototypes(instance_clusters, surfels)

            for proto in prototypes:
                # Render 4 canonical views
                views = self.renderer.render_prototype_views(surfels, proto)

                # Generate 3D solid mesh via Pixal3D (or parametric fallback)
                proto_mesh, proto_mat_base = self.pixal3d.reconstruct_prototype(proto, views)

                # Bake PBR textures for prototype
                rep_surfels = surfels.slice(proto.representative_surfel_indices)
                unwrapped_proto, proto_tex = self.baker.bake_pbr_textures(
                    mesh=proto_mesh,
                    surfels=rep_surfels,
                    base_color=(0.35, 0.45, 0.65),
                    roughness=0.7,
                    metallic=0.1,
                )
                proto_material = proto_tex.save(textures_dir, prefix=proto.prototype_id)

                # Align and clone for each instance in the scene
                cloned_nodes = OBBAligner.align_and_clone_instances(
                    prototype=proto,
                    prototype_mesh=unwrapped_proto,
                    prototype_material=proto_material,
                    instances=instance_clusters,
                )

                for node in cloned_nodes:
                    object_nodes.append(node)
                    object_meshes[node.name] = unwrapped_proto

        # -------------------------------------------------------------
        # Phase 6: Scene Graph Assembly & Multi-Format Export
        # -------------------------------------------------------------
        assembler = SceneAssembler(scene_id=scene_id)
        manifest, exported_paths = assembler.assemble_and_export(
            architecture_mesh=unwrapped_arch,
            object_nodes=object_nodes,
            object_meshes=object_meshes,
            output_dir=out_dir,
            arch_material=arch_material,
        )

        # Clear GPU VRAM after run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return manifest, exported_paths
