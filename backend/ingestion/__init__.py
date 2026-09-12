"""GlomeHomeTour Backend: Ingestion Layer.

Exposes package loading, schema contract verification, quality gating,
timestamp-to-VIO pose alignment, and hybrid SfM pose refinement.
"""

from .package_loader import (
    CameraIntrinsics,
    CapturePackage,
    Keyframe,
    PackageLoader,
    PackageValidationError,
    TrajectorySample,
    load_package,
)
from .pose_aligner import (
    PoseAligner,
    matrix_to_quaternion,
    opencv_to_opengl,
    opengl_to_opencv,
    quaternion_slerp,
    quaternion_to_matrix,
)
from .keyframe_selector import (
    DynamicKeyframeSelector,
    KeyframeSelectionResult,
)
from .quality_gate import (
    FrameQualityMetrics,
    QualityGate,
    QualityGateResult,
)
from .sfm_refinement import (
    HybridSfMRefiner,
    SfMRefinementResult,
)

__all__ = [
    "CameraIntrinsics",
    "CapturePackage",
    "Keyframe",
    "PackageLoader",
    "PackageValidationError",
    "TrajectorySample",
    "load_package",
    "PoseAligner",
    "quaternion_slerp",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "opengl_to_opencv",
    "opencv_to_opengl",
    "QualityGate",
    "QualityGateResult",
    "FrameQualityMetrics",
    "DynamicKeyframeSelector",
    "KeyframeSelectionResult",
    "HybridSfMRefiner",
    "SfMRefinementResult",
]
