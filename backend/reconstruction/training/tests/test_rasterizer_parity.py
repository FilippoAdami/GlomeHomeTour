"""HIP rasterizer vs pure-PyTorch oracle: forward and backward parity.

The PyTorch fallback composites with cumprod under plain autograd, so its
gradients are exact by construction. Any disagreement is a bug in the hand
written HIP kernels -- in particular the alpha gradient, which must carry the
occlusion term  dC/dalpha_j = T_j * (c_j - C_behind_j),  not just T_j * c_j.
"""

import math

import pytest
import torch

from reconstruction.training.model import Material2DGSModel
from reconstruction.training.rasterizer_interface import (
    HIP2DGSRasterizer,
    PyTorchFallbackRasterizer,
    bin_tiles,
    is_hip_rasterizer_available,
)

requires_hip = pytest.mark.skipif(
    not (is_hip_rasterizer_available() and torch.cuda.is_available()),
    reason="compiled HIP rasterizer / ROCm device unavailable",
)

H, W = 48, 64
INTRINSICS = (60.0, 60.0, W / 2.0, H / 2.0)


def _make_model(n: int, device: torch.device, seed: int = 0) -> Material2DGSModel:
    g = torch.Generator(device="cpu").manual_seed(seed)
    xyz = torch.empty((n, 3)).uniform_(-0.6, 0.6, generator=g)
    xyz[:, 2] = torch.empty(n).uniform_(1.0, 3.0, generator=g)  # +Z forward (OpenCV)
    rot = torch.randn((n, 4), generator=g)
    scaling = torch.full((n, 2), math.log(0.05))
    opacity = torch.empty((n, 1)).uniform_(0.5, 2.5, generator=g)   # logit -> 0.62..0.92
    albedo = torch.randn((n, 3), generator=g) * 0.5
    model = Material2DGSModel(
        xyz=xyz, rotation=rot, scaling=scaling, opacity=opacity,
        albedo=albedo, roughness=torch.zeros((n, 1)), metallic=torch.zeros((n, 1)),
    )
    return model.to(device)


def _loss(gbuffer, target):
    """A scalar touching every differentiable output channel."""
    return (
        (gbuffer.albedo - target).abs().mean()
        + 0.3 * gbuffer.normal.mean()
        + 0.2 * gbuffer.depth.mean()
    )


def test_bin_tiles_covers_every_touched_tile():
    """Brute-force check that the binning misses no tile a splat's bbox touches."""
    uv = torch.tensor([[8.0, 8.0], [40.0, 20.0], [-3.0, 30.0], [200.0, 200.0]])
    radius = torch.tensor([4.0, 20.0, 6.0, 5.0])
    depth = torch.tensor([3.0, 1.0, 2.0, 4.0])

    point_list, tile_ranges = bin_tiles(uv, radius, depth, (H, W), tile=16)
    ranges = tile_ranges.reshape(-1, 2)
    grid_w = (W + 15) // 16
    grid_h = (H + 15) // 16
    assert ranges.shape[0] == grid_w * grid_h

    for t in range(grid_w * grid_h):
        tx, ty = t % grid_w, t // grid_w
        x0, x1 = tx * 16, tx * 16 + 16
        y0, y1 = ty * 16, ty * 16 + 16
        expected = {
            i for i in range(uv.shape[0])
            if uv[i, 0] + radius[i] >= x0 and uv[i, 0] - radius[i] < x1
            and uv[i, 1] + radius[i] >= y0 and uv[i, 1] - radius[i] < y1
        }
        s, e = int(ranges[t, 0]), int(ranges[t, 1])
        got = set(point_list[s:e].tolist())
        assert expected <= got, f"tile {t}: missing {expected - got}"

        # Within a tile, primitives must be depth-ascending (front to back).
        d = depth[point_list[s:e].long()]
        assert torch.all(d[1:] >= d[:-1]), f"tile {t} not depth-sorted"


@requires_hip
def test_hip_forward_matches_pytorch_oracle():
    device = torch.device("cuda")
    model = _make_model(200, device)
    w2c = torch.eye(4, device=device)

    hip = HIP2DGSRasterizer()(model, w2c, INTRINSICS, (H, W))
    ref = PyTorchFallbackRasterizer()(model, w2c, INTRINSICS, (H, W))

    assert torch.allclose(hip.albedo, ref.albedo, atol=2e-3), \
        f"max colour delta {(hip.albedo - ref.albedo).abs().max():.4g}"
    assert torch.allclose(hip.alpha, ref.alpha, atol=2e-3)
    assert torch.allclose(hip.depth, ref.depth, atol=5e-3)
    assert hip.alpha.max() > 0.5, "test scene rendered empty"


@requires_hip
@pytest.mark.parametrize("param", ["_opacity", "_albedo", "_xyz"])
def test_hip_backward_matches_pytorch_oracle(param):
    device = torch.device("cuda")
    target = torch.rand((1, 3, H, W), device=device)
    w2c = torch.eye(4, device=device)

    grads = {}
    for name, rasterizer in (("hip", HIP2DGSRasterizer()), ("ref", PyTorchFallbackRasterizer())):
        model = _make_model(200, device)
        gbuffer = rasterizer(model, w2c, INTRINSICS, (H, W))
        _loss(gbuffer, target).backward()
        grads[name] = getattr(model, param).grad.clone()

    g_hip, g_ref = grads["hip"], grads["ref"]
    scale = g_ref.abs().max().clamp(min=1e-12)
    assert scale > 1e-8, f"oracle produced no {param} gradient"

    # Cosine similarity is the property that matters: a wrong dL_dalpha flips
    # the sign of the occlusion term and tanks this well below 1.
    cos = torch.nn.functional.cosine_similarity(
        g_hip.flatten(), g_ref.flatten(), dim=0
    ).item()
    assert cos > 0.97, f"{param} gradient direction diverges (cos={cos:.4f})"
    assert (g_hip - g_ref).abs().max() / scale < 0.1, \
        f"{param} gradient magnitude diverges (rel={((g_hip - g_ref).abs().max() / scale).item():.4g})"


@pytest.mark.parametrize("convention,expected", [("opengl", -1.0), ("opencv", 1.0)])
def test_camera_convention_is_explicit_not_guessed(convention, expected):
    """An explicit convention must ignore the point cloud's median depth.

    Indoors the median flips sign frame to frame (about half of the bedroom
    capture's 273 keyframes), so "auto" silently rendered the opposite wall.
    """
    from reconstruction.training.rasterizer_interface import _camera_sign

    ambiguous = torch.tensor([[0.0, 0.0, -5.0], [0.0, 0.0, 4.0], [0.0, 0.0, -0.01]])
    assert _camera_sign(convention, ambiguous) == expected
    assert _camera_sign("auto", ambiguous) == -1.0  # median is negative here
    with pytest.raises(ValueError):
        _camera_sign("nonsense", ambiguous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU-only regression")
def test_rotate_is_exact_past_the_matmul_cliff():
    """On gfx1200/ROCm 7.1 a (N,3)@(3,3) matmul zeroes every row >= 2**19."""
    from reconstruction.training.rasterizer_interface import _rotate

    n = (1 << 19) + 1024
    g = torch.Generator(device="cpu").manual_seed(0)
    vecs = torch.randn((n, 3), generator=g)
    rot = torch.randn((3, 3), generator=g)
    expected = (vecs.double() @ rot.double().T).float()

    got = _rotate(vecs.cuda(), rot.cuda()).cpu()
    assert torch.allclose(got, expected, atol=1e-3), \
        f"max err {(got - expected).abs().max().item():.3e} at n={n}"


@pytest.mark.parametrize("cls", [PyTorchFallbackRasterizer, HIP2DGSRasterizer])
def test_fully_culled_view_stays_differentiable(cls):
    """A frame with nothing in frustum must still backprop, not kill the run."""
    if cls is HIP2DGSRasterizer and not (is_hip_rasterizer_available() and torch.cuda.is_available()):
        pytest.skip("compiled HIP rasterizer / ROCm device unavailable")
    device = torch.device("cuda" if cls is HIP2DGSRasterizer else "cpu")

    model = _make_model(32, device)
    w2c = torch.eye(4, device=device)
    w2c[2, 3] = -1000.0  # push every splat behind the near plane

    gbuffer = cls(camera_convention="opencv")(
        model, extrinsics=w2c, intrinsics=INTRINSICS, image_size=(H, W)
    )
    assert float(gbuffer.alpha.sum()) == 0.0
    gbuffer.albedo.sum().backward()  # must not raise
