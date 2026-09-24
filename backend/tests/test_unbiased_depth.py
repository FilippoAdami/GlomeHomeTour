"""Unbiased Depth for 2DGS (arXiv 2503.06587) Eq. (9) -- HIP rasterizer channel check.

Builds a stack of axis-aligned surfels at known depths in front of an identity camera.
At the centre pixel every surfel is hit dead-on (rho = 0 => G = 1), so Eq. (9) reduces to
a closed form we can evaluate in Python and compare against the kernel output.

Run: backend/.venv/bin/python -m pytest backend/tests/test_unbiased_depth.py -q
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1] / "03_2DGS_training"
sys.path.insert(0, str(TRAIN_DIR))

pytest.importorskip("diff_surfel_rasterization")
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # noqa: E402

# Must match hip_rasterizer/auxiliary.h
O_THRESH = 0.6
EPS = 0.1

DEPTHS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
OPACITY = 0.13  # increment 0.23 -> O_i never lands on the 0.6 threshold


def _expected_unbiased_depth(depths, opacity):
    """Eq. (9) at a pixel where every splat is hit at its centre (G = 1).

    Evaluated in float32, because the kernel is: a float64 reference lands on the other
    side of the `<=` whenever an O_i sits exactly on the threshold. _assert_no_tie below
    keeps the scene off those knife edges regardless.
    """
    f = np.float32
    o_acc, surface = f(0), depths[0]
    for d in depths:
        if o_acc <= f(O_THRESH):
            surface = d
        o_acc = f(o_acc + f(f(min(0.99, opacity)) + f(EPS)))  # alpha_j*G_j + eps*G_j, G_j = 1
    return surface


def _assert_no_tie(depths, opacity, margin=1e-4):
    """Fail loudly if the scene puts an O_i on the threshold, where `<=` is a coin flip."""
    f = np.float32
    o_acc = f(0)
    for _ in depths:
        assert abs(float(o_acc) - O_THRESH) > margin or o_acc == f(0), (
            f"opacity={opacity} puts O_i={float(o_acc)} on the {O_THRESH} threshold; "
            "pick an opacity that avoids the tie"
        )
        o_acc = f(o_acc + f(f(min(0.99, opacity)) + f(EPS)))


def _expected_median_depth(depths, opacity):
    """Existing 2DGS median criterion, for contrast: first splat where T drops past 0.5."""
    t = 1.0
    surface = depths[0]
    for d in depths:
        if t > 0.5:
            surface = d
        t *= 1.0 - min(0.99, opacity)
    return surface


def _render(depths, opacity, device="cuda"):
    from utils.graphics_utils import getProjectionMatrix

    n = len(depths)
    fov = math.radians(60.0)
    tanfov = math.tan(fov * 0.5)
    res = 65  # odd, so pixel (32, 32) sits on the optical axis

    # Same composition as scene/cameras.py: both matrices transposed, then multiplied.
    view = torch.eye(4, device=device)
    p = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(device)
    proj = (view.unsqueeze(0).bmm(p.unsqueeze(0))).squeeze(0)

    settings = GaussianRasterizationSettings(
        image_height=res, image_width=res,
        tanfovx=tanfov, tanfovy=tanfov,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=proj,
        sh_degree=0, campos=torch.zeros(3, device=device),
        prefiltered=False, debug=False,
    )

    means3D = torch.zeros(n, 3, device=device)
    means3D[:, 2] = torch.tensor(depths, device=device)
    means3D.requires_grad_(True)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    # 20 cm surfels: wide enough to cover the centre pixel, small enough not to fill the frame
    scales = torch.full((n, 2), 0.2, device=device)
    # 180 deg about X, so the surfel normal is -Z and faces the camera.
    # BACKFACE_CULL in auxiliary.h drops surfels facing away outright.
    rotations = torch.zeros(n, 4, device=device)
    rotations[:, 1] = 1.0
    opacities = torch.full((n, 1), opacity, device=device)
    colors = torch.ones(n, 3, device=device)

    rgb, radii, allmap = GaussianRasterizer(settings)(
        means3D=means3D, means2D=means2D, opacities=opacities,
        colors_precomp=colors, scales=scales, rotations=rotations,
    )
    return allmap, means3D, res // 2


def _cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")


def test_allmap_has_unbiased_channel():
    _cuda_or_skip()
    allmap, _, _ = _render(DEPTHS, OPACITY)
    assert allmap.shape[0] == 8, f"expected 8 aux channels (7 + unbiased), got {allmap.shape[0]}"


def test_unbiased_depth_matches_equation_9():
    _cuda_or_skip()
    _assert_no_tie(DEPTHS, OPACITY)
    allmap, _, c = _render(DEPTHS, OPACITY)
    got = allmap[7, c, c].item()
    want = _expected_unbiased_depth(DEPTHS, OPACITY)
    assert abs(got - want) < 1e-3, f"unbiased depth {got} != Eq.(9) {want}"


def test_unbiased_differs_from_median():
    """Guards against the channel silently aliasing the median buffer."""
    _cuda_or_skip()
    allmap, _, c = _render(DEPTHS, OPACITY)
    want_med = _expected_median_depth(DEPTHS, OPACITY)
    want_unb = _expected_unbiased_depth(DEPTHS, OPACITY)
    assert want_med != want_unb, "test scene is not discriminating; pick another opacity"
    assert abs(allmap[5, c, c].item() - want_med) < 1e-3
    assert abs(allmap[7, c, c].item() - want_unb) < 1e-3


@pytest.mark.parametrize("opacity", [0.07, 0.13, 0.23, 0.45, 0.85])
def test_unbiased_depth_across_opacities(opacity):
    _cuda_or_skip()
    _assert_no_tie(DEPTHS, opacity)
    allmap, _, c = _render(DEPTHS, opacity)
    got = allmap[7, c, c].item()
    want = _expected_unbiased_depth(DEPTHS, opacity)
    assert abs(got - want) < 1e-3, f"opacity={opacity}: {got} != {want}"


def test_unbiased_depth_gradient_reaches_selected_splat():
    """dL/dz must land on the splat Eq. (9) selected, and on no other."""
    _cuda_or_skip()
    allmap, means3D, c = _render(DEPTHS, OPACITY)
    allmap[7, c, c].backward()
    grad_z = means3D.grad[:, 2]
    selected = DEPTHS.index(_expected_unbiased_depth(DEPTHS, OPACITY))
    assert grad_z[selected].abs() > 1e-4, "selected splat got no depth gradient"
    others = torch.cat([grad_z[:selected], grad_z[selected + 1:]])
    assert others.abs().max() < 1e-4, f"gradient leaked to non-selected splats: {grad_z}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
