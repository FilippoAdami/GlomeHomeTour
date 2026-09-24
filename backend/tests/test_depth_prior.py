"""Stage 4 per-pixel depth/normal priors (03_2DGS_training/utils/depth_prior.py).

The thing that actually breaks silently here is the reprojection convention: get
it wrong and every confidence comes out near zero, the prior loss is weighted to
nothing, and training looks fine while the walls drift. So the core test is a
synthetic scene where the right answer is known -- a plane seen from two offset
cameras must agree everywhere, and a depth map corrupted by a constant offset
must agree nowhere.
"""

import math

import numpy as np
import pytest
import torch

from utils.depth_prior import (
    UNVERIFIED_CONF,
    DepthPriors,
    build_confidence_cache,
    depth_convergence_loss,
    depth_prior_loss,
    normal_prior_loss,
    prior_weight,
    _intrinsics_from_fov,
)

H, W = 60, 40
FOVX = 2 * math.atan(W / (2 * 50.0))
FOVY = 2 * math.atan(H / (2 * 50.0))
PLANE_Z = 3.0


def _frame(tmp_path, name, cam_x, depth):
    """A camera translated along world +X, looking down +Z, plus its depth map."""
    np.save(tmp_path / f"{name}.npy", depth.astype(np.float32))
    w2c = np.eye(4)
    w2c[0, 3] = -cam_x          # camera at (cam_x, 0, 0), identity rotation
    return {"name": name, "depth_path": str(tmp_path / f"{name}.npy"), "w2c": w2c,
            "fovx": FOVX, "fovy": FOVY, "width": W, "height": H}


def _fronto_plane():
    """z-depth of the plane z = PLANE_Z. Constant, because depth maps are z-depth."""
    return np.full((H * 2, W * 2), PLANE_Z, dtype=np.float32)


def _build(tmp_path, frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = tmp_path / "priors"
    build_confidence_cache(frames, out, device=device, verbose=False)
    return {f["name"]: np.load(out / f"{f['name']}.npz")["conf"].astype(np.float32) / 255.0
            for f in frames}


def test_consistent_plane_scores_high_confidence(tmp_path):
    # Three views of the same plane, 0.3 m apart: inside MIN/MAX_BASELINE_M and
    # well inside MAX_VIEW_ANGLE_DEG, so every one is a usable neighbour.
    frames = [_frame(tmp_path, f"f{i}", 0.3 * i, _fronto_plane()) for i in range(3)]
    conf = _build(tmp_path, frames)
    # The middle view is the one fully covered by both neighbours; the outer two
    # lose a strip to the frustum edge, which is scored as unverified, not wrong.
    assert conf["f1"].mean() > 0.9


def test_offset_depth_map_scores_low_confidence(tmp_path):
    # f1's depth is wrong by 0.5 m -- far outside DEPTH_TOL_ABS_M + 3%. It must
    # disagree with both neighbours, and they must disagree with it.
    frames = [_frame(tmp_path, "f0", 0.0, _fronto_plane()),
              _frame(tmp_path, "f1", 0.3, _fronto_plane() + 0.5),
              _frame(tmp_path, "f2", 0.6, _fronto_plane())]
    conf = _build(tmp_path, frames)
    assert conf["f1"].mean() < 0.1
    # f0's only neighbours are the liar and f2; agreement with f2 alone caps it
    # well below the all-consistent case.
    assert conf["f0"].mean() < 0.9


def test_isolated_frame_is_unverified_not_zero(tmp_path):
    # A single frame has no neighbour in the baseline band. Zeroing its
    # confidence would silently drop the prior for any slow segment of a capture.
    conf = _build(tmp_path, [_frame(tmp_path, "solo", 0.0, _fronto_plane())])
    assert np.allclose(conf["solo"], 1.0)


def test_cache_is_half_resolution(tmp_path):
    frames = [_frame(tmp_path, f"f{i}", 0.3 * i, _fronto_plane()) for i in range(2)]
    conf = _build(tmp_path, frames)
    assert conf["f0"].shape == (H, W)      # inputs are (2H, 2W)


def test_plane_fit_normal_and_confidence_n_cached(tmp_path):
    # Fronto-parallel plane, camera looking down +Z with identity rotation: the
    # plane-fit normal should come out unit-length and pointing back at the
    # camera (world -Z), and its residual should be ~0 so C_n stays high.
    frames = [_frame(tmp_path, f"f{i}", 0.3 * i, _fronto_plane()) for i in range(3)]
    out = tmp_path / "priors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_confidence_cache(frames, out, device=device, verbose=False)
    z = np.load(out / "f1.npz")
    assert "normal" in z and "conf_n" in z
    normal = z["normal"].astype(np.float32)
    conf_n = z["conf_n"].astype(np.float32) / 255.0
    interior = normal[10:-10, 10:-10]
    np.testing.assert_allclose(np.linalg.norm(interior, axis=-1), 1.0, atol=1e-2)
    np.testing.assert_allclose(interior.mean(axis=(0, 1)), [0.0, 0.0, -1.0], atol=1e-2)
    assert conf_n[10:-10, 10:-10].mean() > 0.8


def test_get_normal_matches_get_resolution_and_is_unit(tmp_path):
    frames = [_frame(tmp_path, "f0", 0.0, _fronto_plane())]
    out = tmp_path / "priors"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_confidence_cache(frames, out, device=device, verbose=False)
    priors = DepthPriors(out)

    class _View:
        image_name = "f0"
        image_height = H * 2  # forces the bilinear-resample + renormalise path
        image_width = W * 2

    normal, conf_n = priors.get_normal(_View())
    assert normal.shape == (3, H * 2, W * 2)
    assert conf_n.shape == (1, H * 2, W * 2)
    norms = normal.norm(dim=0)[10:-10, 10:-10]
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-3, rtol=0)


def test_intrinsics_are_centred():
    fx, fy, cx, cy = _intrinsics_from_fov(FOVX, FOVY, W, H)
    assert cx == pytest.approx(W * 0.5)
    assert cy == pytest.approx(H * 0.5)
    assert fx == pytest.approx(50.0)


def test_prior_weight_decays_to_floor_then_holds():
    assert prior_weight(0.5, 0, 1000, 2500, 0.1) == pytest.approx(0.5)
    assert prior_weight(0.5, 1000, 1000, 2500, 0.1) == pytest.approx(0.5)
    assert prior_weight(0.5, 1750, 1000, 2500, 0.1) == pytest.approx(0.5 * 0.55)
    assert prior_weight(0.5, 2500, 1000, 2500, 0.1) == pytest.approx(0.05)
    assert prior_weight(0.5, 9999, 1000, 2500, 0.1) == pytest.approx(0.05)
    # Degenerate window must not divide by zero.
    assert prior_weight(0.5, 500, 1000, 1000, 0.1) == pytest.approx(0.5)


def test_losses_vanish_on_a_perfect_match_and_respect_confidence():
    d = torch.full((1, 8, 8), 2.0)
    conf = torch.ones_like(d)
    mask = torch.ones_like(d)
    assert depth_prior_loss(d, d.clone(), conf, mask).item() == pytest.approx(0.0)

    wrong = d + 0.4
    assert depth_prior_loss(wrong, d, conf, mask).item() == pytest.approx(0.4, abs=1e-5)
    # Zero confidence everywhere means no evidence, not a free pass on a
    # zero-weight average: the loss must be exactly zero, not NaN.
    out = depth_prior_loss(wrong, d, torch.zeros_like(conf), mask)
    assert torch.isfinite(out) and out.item() == 0.0

    n = torch.zeros(3, 8, 8)
    n[2] = 1.0
    assert normal_prior_loss(n, n.clone(), conf, mask).item() == pytest.approx(0.0, abs=1e-6)
    flipped = -n
    assert normal_prior_loss(n, flipped, conf, mask).item() == pytest.approx(2.0, abs=1e-5)


def test_depth_convergence_targets_median_without_backpropping_it():
    expected = torch.full((1, 4, 4), 2.0, requires_grad=True)
    median = torch.full((1, 4, 4), 2.5, requires_grad=True)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    out = depth_convergence_loss(expected, median, mask)
    assert out.item() == pytest.approx(0.5)
    out.backward()
    assert expected.grad.abs().sum() > 0
    assert median.grad is None       # detached: the rasteriser does not backprop it

    empty = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert depth_convergence_loss(expected, median, empty).item() == 0.0


def test_unverified_confidence_is_between_zero_and_one():
    assert 0.0 < UNVERIFIED_CONF < 1.0
