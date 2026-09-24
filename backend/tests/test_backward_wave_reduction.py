"""Per-splat parallel backward (Taming 3DGS) -- wavefront-reduction correctness.

The backward pass replaced 17 per-lane atomicAdds with a __shfl_down reduction across
each 32-wide wavefront, so a lane that skips a splat must still reach the reduction
carrying zero. The two ways that breaks are a dropped lane (gradient too small) and an
inactive lane's uninitialised accumulator leaking in (gradient too large, or landing on
a splat that contributed nothing). Both are checked here without a reference kernel.

Run: backend/.venv/bin/python -m pytest backend/tests/test_backward_wave_reduction.py -q
"""
import math
import sys
from pathlib import Path

import pytest
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1] / "03_2DGS_training"
sys.path.insert(0, str(TRAIN_DIR))

pytest.importorskip("diff_surfel_rasterization")
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402

RES = 128
FOV = math.radians(60.0)


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")


def _settings(device, res=RES):
    from utils.graphics_utils import getProjectionMatrix

    view = torch.eye(4, device=device)
    p = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FOV, fovY=FOV).transpose(0, 1).to(device)
    proj = view.unsqueeze(0).bmm(p.unsqueeze(0)).squeeze(0)
    return GaussianRasterizationSettings(
        image_height=res, image_width=res,
        tanfovx=math.tan(FOV / 2), tanfovy=math.tan(FOV / 2),
        bg=torch.zeros(3, device=device),  # zero bg: no T_final*bg term in the identity below
        scale_modifier=1.0, viewmatrix=view, projmatrix=proj,
        sh_degree=0, campos=torch.zeros(3, device=device),
        prefiltered=False, debug=False,
    )


def _render(means, scales, rots, opac, cols, res=RES):
    m2 = torch.zeros_like(means, requires_grad=True)
    return GaussianRasterizer(_settings(means.device, res))(
        means3D=means, means2D=m2, opacities=opac,
        colors_precomp=cols, scales=scales, rotations=rots,
    )


def test_color_gradient_sums_to_rendered_image():
    """Conservation law over every (pixel, splat) pair the reduction touches.

    rgb = sum_i T_i * alpha_i * c_i, so for L = rgb[0].sum() the gradient on splat i is
    dL/dc_i0 = sum_pixels T_i * alpha_i. Summing that over splats gives back L exactly.
    A lane dropped from the reduction makes the total fall short; a leaked accumulator
    makes it overshoot. Neither can cancel, since every term T_i*alpha_i is positive.
    """
    _cuda_or_skip()
    dev = "cuda"
    torch.manual_seed(0)
    n = 3000
    means = torch.randn(n, 3, device=dev) * 1.2
    means[:, 2] = means[:, 2].abs() + 2.0
    s0 = torch.rand(n, 1, device=dev) * 0.06 + 0.01
    scales = torch.cat([s0, s0 * (torch.rand(n, 1, device=dev) * 3.0 + 1.0)], 1)
    rots = torch.randn(n, 4, device=dev)
    rots = rots / rots.norm(dim=1, keepdim=True)
    opac = torch.rand(n, 1, device=dev) * 0.9 + 0.02
    cols = torch.ones(n, 3, device=dev, requires_grad=True)

    rgb, radii, _ = _render(means, scales, rots, opac, cols)
    loss = rgb[0].sum()
    loss.backward()

    got = cols.grad[:, 0].sum().item()
    want = loss.item()
    assert radii.gt(0).sum() > n // 4, "scene degenerate; almost nothing on screen"
    # fp32 summation order differs between the two sides, hence relative not exact.
    assert abs(got - want) / want < 1e-4, f"gradient mass {got} != rendered mass {want}"


def test_occluded_splat_gets_exactly_zero_gradient():
    """A splat past last_contributor is skipped by every lane; the reduction must give 0.

    This is the path where all 32 lanes break early. If an inactive lane carried garbage
    into __shfl_down, a splat that contributed to no pixel would pick up a gradient.
    The same splat pulled in front of the occluders must get one, which rules out the
    zero coming from frustum culling instead.
    """
    _cuda_or_skip()
    dev = "cuda"
    # Slabs must stay opaque out to the frame corners, not just on axis: G decays off
    # centre, so a surfel only just covering the frame leaves T ~1e-2 there and the probe
    # genuinely contributes. Scale 8 keeps G > 0.97 across the whole image; two of them
    # drop T to 1e-4 and trip the forward's done-threshold.
    depths = [2.0, 2.1]
    probe_behind, probe_front = 5.0, 1.5

    def run(probe_z):
        z = depths + [probe_z]
        means = torch.zeros(len(z), 3, device=dev)
        means[:, 2] = torch.tensor(z, device=dev)
        scales = torch.full((len(z), 2), 8.0, device=dev)
        rots = torch.zeros(len(z), 4, device=dev)
        rots[:, 1] = 1.0  # 180 deg about x: face the camera, clearing BACKFACE_CULL
        opac = torch.full((len(z), 1), 1.0, device=dev)
        cols = torch.ones(len(z), 3, device=dev, requires_grad=True)
        rgb, radii, _ = _render(means, scales, rots, opac, cols)
        rgb[0].sum().backward()
        return cols.grad[-1, 0].item(), radii[-1].item()

    g_behind, r_behind = run(probe_behind)
    g_front, r_front = run(probe_front)

    assert r_behind > 0, "occluded probe was frustum-culled; test proves nothing"
    assert g_behind == 0.0, f"fully occluded splat picked up gradient {g_behind}"
    assert r_front > 0 and g_front > 1e-3, f"unoccluded probe got no gradient ({g_front})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
