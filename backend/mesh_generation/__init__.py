"""GlomeHomeTour: Automated 2DGS to Segmented PBR CAD/BIM Mesh Pipeline.

Converts trained 2D Gaussian Splatting radiance fields into watertight,
object-segmented 3D triangle meshes with baked PBR materials, ready for
CAD (AutoCAD, Revit, SketchUp) and DCC (Blender, Unreal Engine 5) export.
"""

from .types import (
    BoundingBox3D,
    InstanceCluster,
    MeshManifest,
    MeshNode,
    MeshPipelineConfig,
    NodeCategory,
    PBRMaterial,
    SurfelCloudTorch,
)
from .io_adapter import (
    CameraIntrinsicsTorch,
    KeyframePoseTorch,
    TransformsDataset,
    create_mock_surfel_cloud,
    create_mock_transforms,
    load_splats_ply,
    load_transforms_json,
    save_splats_ply,
)

__all__ = [
    "BoundingBox3D",
    "InstanceCluster",
    "MeshManifest",
    "MeshNode",
    "MeshPipelineConfig",
    "NodeCategory",
    "PBRMaterial",
    "SurfelCloudTorch",
    "CameraIntrinsicsTorch",
    "KeyframePoseTorch",
    "TransformsDataset",
    "create_mock_surfel_cloud",
    "create_mock_transforms",
    "load_splats_ply",
    "load_transforms_json",
    "save_splats_ply",
]

from .segmentation import (
    GeometricArchitectureFilter,
    KeyframeDetection,
    KeyframeSemanticResult,
    SemanticInstanceSegmenter,
)

__all__.extend([
    "GeometricArchitectureFilter",
    "KeyframeDetection",
    "KeyframeSemanticResult",
    "SemanticInstanceSegmenter",
])

from .planar_snapping import (
    FittedPlane,
    ManhattanPlanarRANSAC,
)
from .architectural_infill import (
    ArchitecturalInfillEngine,
    InfillResult,
)

__all__.extend([
    "FittedPlane",
    "ManhattanPlanarRANSAC",
    "ArchitecturalInfillEngine",
    "InfillResult",
])

from .prototype_cluster import (
    InstanceSignature,
    PrototypeClusterEngine,
    PrototypeGroup,
)
from .object_reconstruction import (
    CanonicalMultiView,
    CanonicalViewRenderer,
    OBBAligner,
    Pixal3DReconstructionEngine,
)

__all__.extend([
    "InstanceSignature",
    "PrototypeClusterEngine",
    "PrototypeGroup",
    "CanonicalMultiView",
    "CanonicalViewRenderer",
    "OBBAligner",
    "Pixal3DReconstructionEngine",
])

from .texture_baking import (
    BakedTextureSet,
    PBRTextureBaker,
    UVAtlasUnwrapper,
)

__all__.extend([
    "BakedTextureSet",
    "PBRTextureBaker",
    "UVAtlasUnwrapper",
])

from .exporters.cad_exporter import (
    DXFExporter,
    IFCExporter,
)
from .exporters.gltf_exporter import (
    GLTFExporter,
)
from .scene_assembler import (
    SceneAssembler,
)

__all__.extend([
    "DXFExporter",
    "IFCExporter",
    "GLTFExporter",
    "SceneAssembler",
])

from .kernels.surfel_projection import (
    project_surfels_frustum_torch,
)
from .pipeline import (
    MeshGenerationPipeline,
)

__all__.extend([
    "project_surfels_frustum_torch",
    "MeshGenerationPipeline",
])
