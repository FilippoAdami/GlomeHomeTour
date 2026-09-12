"""GlomeHomeTour Backend: 2DGS Material & Density Optimization Engine."""

from reconstruction.training.model import (
    Material2DGSModel,
    matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_tangent_frame,
)
from reconstruction.training.rasterizer_interface import (
    GBufferOutput,
    Base2DGSRasterizer,
    PyTorchFallbackRasterizer,
    get_default_rasterizer,
    is_hip_rasterizer_available,
)
from reconstruction.training.pbr_shader import (
    DeferredCookTorranceShader,
    SpecularResidualMLP,
)
from reconstruction.training.losses import (
    ReconstructionLoss,
    l1_loss,
    ssim_loss,
    normal_loss,
)
from reconstruction.training.density_control import (
    DensityControlConfig,
    TamingDensityController,
)
from reconstruction.training.planar_reflections import (
    MirrorPlane,
    PlanarMirrorDetector,
    render_mirror_reflection_pass,
)
from reconstruction.training.compressor import (
    Compressed2DGSBundle,
    LightGaussianCompressor,
)
from reconstruction.training.trainer import (
    Material2DGSTrainer,
    TrainerConfig,
    TrainingKeyframeData,
)
from reconstruction.training.dataset import (
    GSInputDataset,
    GSSceneData,
)

__all__ = [
    "Material2DGSModel",
    "matrix_to_quaternion",
    "quaternion_to_rotation_matrix",
    "quaternion_to_tangent_frame",
    "GBufferOutput",
    "Base2DGSRasterizer",
    "PyTorchFallbackRasterizer",
    "get_default_rasterizer",
    "is_hip_rasterizer_available",
    "DeferredCookTorranceShader",
    "SpecularResidualMLP",
    "ReconstructionLoss",
    "l1_loss",
    "ssim_loss",
    "normal_loss",
    "DensityControlConfig",
    "TamingDensityController",
    "MirrorPlane",
    "PlanarMirrorDetector",
    "render_mirror_reflection_pass",
    "Compressed2DGSBundle",
    "LightGaussianCompressor",
    "Material2DGSTrainer",
    "TrainerConfig",
    "TrainingKeyframeData",
    "GSInputDataset",
    "GSSceneData",
]
