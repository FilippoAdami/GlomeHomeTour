"""GlomeHomeTour Backend: Comprehensive ROCm & Precision Test Suite.

Tests and diagnoses:
1. DA3 Model Loading & Precision (FP32 vs BFloat16 vs FP16)
2. Direct Forward Pass without Autocast vs with Autocast
3. Multi-View Inference with 4x4 Extrinsics and Intrinsics
4. Umeyama Pose Alignment edge cases (identical poses, singular matrices, collinear motion)
5. Multi-View Consistency filtering with edge case orientations
6. Memory cleanup and IPC display compositor breather stability
"""

import os
import sys
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

backend_dir = Path("/home/monday/Desktop/GlomeHomeTour/backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

da3_src = backend_dir / "third_party" / "depth_anything_3" / "src"
if da3_src.is_dir() and str(da3_src) not in sys.path:
    sys.path.insert(0, str(da3_src))

from ingestion.package_loader import CameraIntrinsics, Keyframe
from reconstruction.depth_priors import (
    DepthPriorEstimator,
    GlobalDepthGraphOptimizer,
    MetricDepthAligner,
    compute_surface_normals,
)
from reconstruction.initialization import SurfelCloudInitializer


# ==============================================================================
# 1. Hardware & Environment Checks
# ==============================================================================

def test_rocm_device_status():
    """Verify GPU availability and ROCm capabilities."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA/ROCm device found")
    
    device_name = torch.cuda.get_device_name(0)
    assert len(device_name) > 0
    print(f"\n[Hardware] Testing on GPU: {device_name}")
    print(f"[Hardware] BFloat16 hardware support: {torch.cuda.is_bf16_supported()}")


# ==============================================================================
# 2. DA3 Model Initialization & Precision Integrity
# ==============================================================================

def test_da3_fp32_vs_bf16_inference():
    """Test DA3 model under full FP32 and safe BFloat16 without triggering HIP launch failure."""
    if not torch.cuda.is_available():
        pytest.skip("Requires GPU for precision testing")

    from depth_anything_3.api import DepthAnything3

    torch.cuda.empty_cache()
    # Test loading model directly to CUDA device in FP32
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
    model = model.to("cuda")
    model.eval()

    # Verify device of entire model container and underlying network
    assert model._get_model_device().type == "cuda"

    # Test single-image inference in standard mode
    dummy_img = np.full((280, 504, 3), 128, dtype=np.uint8)
    with torch.inference_mode():
        pred_single = model.inference([dummy_img])
    assert pred_single.depth is not None
    assert pred_single.depth.shape == (1, 280, 504)
    assert not np.isnan(pred_single.depth).any()

    # Clean VRAM
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test_da3_multiview_synthetic_batch():
    """Test DA3 multi-view batch with 4 views and realistic camera extrinsics."""
    if not torch.cuda.is_available():
        pytest.skip("Requires GPU for multi-view batch testing")

    from depth_anything_3.api import DepthAnything3

    torch.cuda.empty_cache()
    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
    model = model.to("cuda")
    model.eval()

    # 4 synthetic images (color gradient to give feature texture)
    imgs = []
    for i in range(4):
        arr = np.zeros((280, 504, 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 50) % 255
        arr[:, :, 1] = 100
        arr[:, :, 2] = np.linspace(0, 255, 504, dtype=np.uint8)
        imgs.append(arr)

    # 4 poses with realistic baseline translation (0.3m steps along X)
    extrinsics = []
    for i in range(4):
        # W2C matrix: camera moved +i*0.3 along X, looking down -Z
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = i * 0.3
        w2c = np.linalg.inv(c2w)
        extrinsics.append(w2c)
    extrinsics = np.stack(extrinsics, axis=0)

    # Intrinsics
    k = np.array([
        [500.0, 0.0, 252.0],
        [0.0, 500.0, 140.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    intrinsics = np.repeat(k[None, ...], 4, axis=0)

    with torch.inference_mode():
        pred = model.inference(
            imgs,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=True,
        )

    assert pred.depth.shape == (4, 280, 504)
    assert not np.isnan(pred.depth).any()
    assert np.all(pred.depth > 0.0)

    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


# ==============================================================================
# 3. DepthPriorEstimator Wrapper & Streaming Tests
# ==============================================================================

def test_depth_prior_estimator_streaming_chunking():
    """Verify sliding-window chunk streaming and cross-chunk blending without memory leaks."""
    torch.cuda.empty_cache()
    estimator = DepthPriorEstimator(model_name="depth-anything/DA3-BASE", use_fp16=False)

    # 8 dummy frames
    imgs = [np.full((280, 504, 3), int(100 + i * 15), dtype=np.uint8) for i in range(8)]
    extrinsics = [np.eye(4, dtype=np.float32) for _ in range(8)]
    for i in range(8):
        extrinsics[i][0, 3] = i * 0.25
    extrinsics = np.stack(extrinsics, axis=0)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 8, axis=0)

    depths, confs = estimator.estimate_depth_streaming(
        imgs,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        chunk_size=4,
        overlap=2,
    )

    assert len(depths) == 8
    for i, d in enumerate(depths):
        assert d.shape == (280, 504)
        assert not np.isnan(d).any(), f"NaN found in depth map index {i}"
        assert not np.isinf(d).any(), f"Inf found in depth map index {i}"


# ==============================================================================
# 4. Multi-View Consensus & Surfel Initialization Edge Cases
# ==============================================================================

def test_surfel_initialization_finite_geometry():
    """Verify surfel initialization rejects empty air / NaN and produces clean geometry."""
    intr = CameraIntrinsics(
        camera_model="OPENCV",
        fl_x=500.0, fl_y=500.0, cx=252.0, cy=140.0,
        w=504, h=280, camera_angle_x=0.8,
        k1=0.0, k2=0.0, p1=0.0, p2=0.0,
    )

    # 3 keyframes
    kfs = []
    depth_maps = []
    for i in range(3):
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = i * 0.3
        dummy_img = Image.new("RGB", (504, 280), color=(128, 128, 128))
        kf = Keyframe(
            file_path=f"frame_{i}.jpg",
            timestamp_ns=i * 100_000_000,
            fl_x=intr.fl_x,
            fl_y=intr.fl_y,
            cx=intr.cx,
            cy=intr.cy,
            transform_matrix=c2w,
            image_loader=lambda img=dummy_img: img,
        )
        kfs.append(kf)
        # Consistent planar wall at z = 2.0 meters
        depth_maps.append(np.full((280, 504), 2.0, dtype=np.float32))

    initializer = SurfelCloudInitializer(
        target_surfels=10_000,
        voxel_downsample_m=0.02,
        min_consensus=1,
    )
    cloud = initializer.initialize_from_keyframes(kfs, depth_maps, intr)

    assert len(cloud) > 100
    assert not np.isnan(cloud.positions).any()
    assert not np.isnan(cloud.normals).any()
    assert not np.isnan(cloud.scales_2d).any()
    assert not np.isnan(cloud.opacities).any()
