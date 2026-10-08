"""Unit tests for FastSAM2Generator."""

import numpy as np
import pytest
import torch

from fast_sam2_generator import FastSAM2Generator, CHECKPOINTS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_fast_sam2_generator_shapes_and_types():
    generator = FastSAM2Generator(
        model_variant="small",
        grid=8,
        min_area=100,
        compile_model=False,
    )
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # Put a white rectangle in the middle
    dummy_rgb[100:300, 150:450] = 255

    masks, metadata = generator.generate(dummy_rgb)
    assert isinstance(masks, np.ndarray)
    assert masks.dtype == np.uint8
    if len(masks) > 0:
        assert masks.shape[1:] == (480, 640)
        assert len(metadata) == len(masks)
        for meta in metadata:
            assert "bbox_xyxy" in meta
            assert "area_px" in meta
            assert "predicted_iou" in meta
            assert "stability" in meta
            assert meta["area_px"] >= 100
