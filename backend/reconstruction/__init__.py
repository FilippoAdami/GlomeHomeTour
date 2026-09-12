"""GlomeHomeTour Backend: Reconstruction Layer.

Exposes Depth Anything metric depth estimation, metric alignment,
surface normal derivation, and 2D Gaussian surfel initialization.
"""

from .depth_priors import (
    AlignedDepthResult,
    DepthPriorEstimator,
    GlobalDepthGraphOptimizer,
    GlobalDepthGraphResult,
    MetricDepthAligner,
    compute_surface_normals,
)
from .initialization import (
    SurfelCloud,
    SurfelCloudInitializer,
    build_orthonormal_tangent_frame,
)

__all__ = [
    "DepthPriorEstimator",
    "MetricDepthAligner",
    "GlobalDepthGraphOptimizer",
    "GlobalDepthGraphResult",
    "AlignedDepthResult",
    "compute_surface_normals",
    "SurfelCloud",
    "SurfelCloudInitializer",
    "build_orthonormal_tangent_frame",
]
