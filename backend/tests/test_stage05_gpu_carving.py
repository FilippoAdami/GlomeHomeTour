import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_semantic_segmentation"))
from run_full_scene_pipeline import carve_overlapping_masks


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gpu_carving_batches_sync_and_rejected_masks_do_not_occlude_later_masks(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    masks = torch.tensor(
        [
            [[1, 1, 0, 0]],
            [[0, 1, 1, 0]],
            [[0, 0, 1, 1]],
        ],
        dtype=torch.bool,
        device=device,
    )

    carved, keep = carve_overlapping_masks(masks, min_pixels=2)

    assert keep.tolist() == [True, False, True]
    assert np.array_equal(carved, np.array([[[1, 1, 0, 0]], [[0, 0, 1, 1]]], dtype=np.uint8))


def test_gpu_carving_empty_input_preserves_spatial_shape():
    masks = torch.zeros((0, 3, 5), dtype=torch.bool)

    carved, keep = carve_overlapping_masks(masks, min_pixels=1)

    assert carved.shape == (0, 3, 5)
    assert carved.dtype == np.uint8
    assert keep.shape == (0,)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gpu_carving_keeps_occupancy_across_batches_without_rejected_masks_blocking(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    masks = torch.zeros((17, 1, 4), dtype=torch.bool, device=device)
    masks[0, 0] = torch.tensor([1, 1, 0, 0], device=device)
    masks[1, 0] = torch.tensor([0, 1, 1, 0], device=device)
    masks[16, 0] = torch.tensor([0, 0, 1, 1], device=device)

    carved, keep = carve_overlapping_masks(masks, min_pixels=2)

    assert keep[[0, 1, 16]].tolist() == [True, False, True]
    assert np.array_equal(carved, np.array([[[1, 1, 0, 0]], [[0, 0, 1, 1]]], dtype=np.uint8))
