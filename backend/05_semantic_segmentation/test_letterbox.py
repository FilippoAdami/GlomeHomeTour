"""Small coordinate and crop check for portrait letterbox inference."""

import torch

from fast_sam2_generator import FastSAM2Generator


def test_letterbox_coordinates() -> None:
    generator = FastSAM2Generator.__new__(FastSAM2Generator)
    generator.resize_mode = "letterbox"
    generator._image_rect = (0, 224, 1024, 576)
    corners = torch.tensor([[0, 0], [540, 960], [1080, 1920]], dtype=torch.float32)
    mapped = generator._transform_coords(corners, 1920, 1080)
    assert torch.equal(mapped, torch.tensor([[224, 0], [512, 512], [800, 1024]], dtype=torch.float32))


if __name__ == "__main__":
    test_letterbox_coordinates()
