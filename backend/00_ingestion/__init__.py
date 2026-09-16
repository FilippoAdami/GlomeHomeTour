"""GlomeHomeTour Backend: Ingestion Layer.

Exposes package loading, schema contract verification, quality gating, and
timestamp-to-VIO pose alignment.

Pose refinement now lives in stage 2 (``01_poses_refinment/``), which drives
COLMAP triangulation via ``convert_transforms_to_colmap.py``.
"""

from package_loader import (
    CameraIntrinsics,
    CapturePackage,
    Keyframe,
    PackageLoader,
    PackageValidationError,
    TrajectorySample,
    load_package,
)
from pose_aligner import (
    PoseAligner,
    matrix_to_quaternion,
    opencv_to_opengl,
    opengl_to_opencv,
    quaternion_slerp,
    quaternion_to_matrix,
)
from keyframe_selector import (
    DynamicKeyframeSelector,
    KeyframeSelectionResult,
    estimate_scene_depths,
)
from quality_gate import (
    FrameQualityMetrics,
    QualityGate,
    QualityGateResult,
    prune_redundant,
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
    "prune_redundant",
    "DynamicKeyframeSelector",
    "KeyframeSelectionResult",
    "estimate_scene_depths",
]
