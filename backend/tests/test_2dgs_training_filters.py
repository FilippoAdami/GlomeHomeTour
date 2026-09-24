"""Unit tests for Stage 3 2DGS training filters:
1. Dynamic Sensor Saturation Masking and Masked Losses (L1 & SSIM)
2. Asymmetric Multi-View Homography Loss with Free-Space Veto
3. Observation-Count / View-Evidence Pruning (TIDI-GS style)
"""

import math
import pytest
import torch
import torch.nn as nn
from utils.loss_utils import (
    compute_saturation_mask,
    masked_l1_loss,
    masked_ssim,
    l1_loss,
    ssim,
)
from utils.multiview_loss import multiview_photometric_loss
from utils.point_utils import _rotate
from scene.gaussian_model import GaussianModel
from arguments import OptimizationParams


# ==============================================================================
# 1. Sensor Saturation Masking & Masked Photometric Losses
# ==============================================================================

def test_compute_saturation_mask_glare_vs_color():
    """Verify saturation mask flags overexposed bloom while preserving colored surfaces."""
    img = torch.zeros((3, 30, 30), dtype=torch.float32)

    # Region 1: Overexposed optical bloom / glare (near white, > 0.98, low chroma diff)
    img[:, 0:10, 0:10] = 1.0

    # Region 2: Bright saturated color (e.g. bright red: [1.0, 0.05, 0.05], high chroma diff)
    img[0, 10:20, 10:20] = 1.0
    img[1, 10:20, 10:20] = 0.05
    img[2, 10:20, 10:20] = 0.05

    # Region 3: Normal textured surface (< 0.98)
    img[:, 20:30, 20:30] = 0.75

    mask = compute_saturation_mask(img, min_threshold=0.98, percentile=99.8, max_chroma_diff=0.15)
    assert mask.shape == (30, 30)

    # Glare region must be masked
    assert mask[0:10, 0:10].all()

    # Bright colored region must NOT be masked (chroma difference is ~0.95 > 0.15)
    assert not mask[10:20, 10:20].any()

    # Normal region must NOT be masked
    assert not mask[20:30, 20:30].any()


def test_masked_l1_loss_ignores_saturated_patches():
    """Verify masked_l1_loss ignores masked errors and computes gradients only on valid pixels."""
    gt = torch.full((3, 20, 20), 0.5, dtype=torch.float32)
    # Prediction has a huge error strictly inside the top-left 5x5 patch
    pred = gt.clone()
    pred[:, 0:5, 0:5] += 0.5  # error = 0.5
    pred.requires_grad_(True)

    mask = torch.ones((20, 20), dtype=torch.bool)
    mask[0:5, 0:5] = False  # mask out the 5x5 patch

    loss_unmasked = masked_l1_loss(pred, gt, mask=None)
    assert loss_unmasked.item() > 0.0

    loss_masked = masked_l1_loss(pred, gt, mask=mask)
    assert torch.isclose(loss_masked, torch.tensor(0.0), atol=1e-6)

    # Backward pass should produce zero gradient on masked pixels
    loss_masked.backward()
    assert torch.all(pred.grad[:, 0:5, 0:5] == 0.0)


def test_masked_ssim_computation():
    """Verify masked_ssim behaves consistently and weights valid regions."""
    img1 = torch.rand((3, 32, 32), dtype=torch.float32)
    img2 = img1.clone()
    mask = torch.ones((32, 32), dtype=torch.bool)
    mask[0:8, 0:8] = False

    # Identical images give SSIM = 1.0
    val = masked_ssim(img1, img2, mask=mask)
    assert torch.isclose(val, torch.tensor(1.0), atol=1e-4)

    # Corrupted image only in masked region should yield high SSIM
    img2_corrupt = img1.clone()
    img2_corrupt[:, 0:8, 0:8] = 0.0
    val_masked = masked_ssim(img1, img2_corrupt, mask=mask)
    val_unmasked = masked_ssim(img1, img2_corrupt, mask=None)
    assert val_masked > val_unmasked


# ==============================================================================
# 2. Asymmetric Multi-View Homography Loss & Free-Space Veto
# ==============================================================================

class MockCamera:
    """Mock camera view for multiview_photometric_loss tests."""
    def __init__(self, R, T, image, name="cam", uid=0):
        self.image_name = name
        self.uid = uid
        self.image_width = image.shape[-1]
        self.image_height = image.shape[-2]
        self.FoVx = 2.0 * math.atan((self.image_width / 2.0) / 100.0)
        self.FoVy = 2.0 * math.atan((self.image_height / 2.0) / 100.0)
        self.original_image = image

        # World to Camera matrix (row-vector convention: x_cam = x_world @ W2C)
        W2C = torch.eye(4, dtype=torch.float32, device=image.device)
        W2C[:3, :3] = R.T
        W2C[3, :3] = T
        self.world_view_transform = W2C


def _smooth_texture(h, w, device):
    """Band-limited test texture.

    White noise is useless for testing an NCC: it has no spatial correlation, so
    the sub-pixel resampling of any warp decorrelates it and the NCC reads ~0 even
    for a perfect warp. Real images are band-limited; this stands in for that.
    """
    y = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(0)
    t = 0.5 + 0.25 * torch.sin(2 * math.pi * x / 13.0) * torch.cos(2 * math.pi * y / 17.0)
    return t.unsqueeze(0).repeat(3, 1, 1)


def test_asymmetric_veto_weighting():
    """Verify asymmetric veto weight penalizes worst mismatch instead of diluting."""
    H, W = 40, 40
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The loss is an NCC, which is invariant to affine intensity change: a
    # constant patch correlates with every other constant patch and is discarded
    # as textureless. So both agreement and disagreement have to be expressed as
    # *structure*, not as brightness.
    texture = _smooth_texture(H, W, device)

    # Reference camera looking at planar wall at z=2.0
    R_ref = torch.eye(3, device=device)
    T_ref = torch.zeros(3, device=device)
    ref_img = texture
    view = MockCamera(R_ref, T_ref, ref_img, name="cam_0", uid=0)

    # Neighbor 1: small baseline, identical structure (NCC ~ 1, loss ~ 0)
    T_nb1 = torch.tensor([0.05, 0.0, 0.0], device=device)
    nb1_img = ref_img.clone()
    nb1 = MockCamera(R_ref, T_nb1, nb1_img, name="cam_1", uid=1)

    # Neighbor 2: inverted structure (NCC ~ -1, loss ~ 1)
    T_nb2 = torch.tensor([0.1, 0.0, 0.0], device=device)
    nb2_img = 1.0 - texture
    nb2 = MockCamera(R_ref, T_nb2, nb2_img, name="cam_2", uid=2)

    # Synthetic planar surfel render outputs
    depth = torch.full((1, H, W), 2.0, device=device)
    normal = torch.zeros((3, H, W), device=device)
    normal[2, :, :] = -1.0  # Facing camera
    alpha = torch.ones((1, H, W), device=device)

    # Compute loss with veto_weight = 0.0 (standard average)
    loss_mean = multiview_photometric_loss(
        view, [nb1, nb2], depth, normal, alpha,
        num_samples=1000, veto_weight=0.0, saturation_threshold=0.0
    )

    # Compute loss with veto_weight = 0.5 (asymmetric consensus)
    loss_veto = multiview_photometric_loss(
        view, [nb1, nb2], depth, normal, alpha,
        num_samples=1000, veto_weight=0.5, saturation_threshold=0.0
    )

    # Veto loss must be strictly greater than mean loss when one neighbor disagrees
    assert loss_veto.item() > loss_mean.item()
    assert loss_veto.item() > 0.0


def test_multiview_loss_finite_gradients_with_degenerate_planes():
    """Surfel planes through the camera origin must not poison the gradients.

    q = n . X_cam is the plane offset, and it goes to zero for a surfel whose
    plane passes through the camera centre. `usable` drops those pixels from the
    loss *value*, so the loss stays finite and no loss-level guard can see a
    problem -- but the n_cam / q division had already produced inf for them, and
    0 * inf is NaN in backward. That silently corrupted xyz/scaling/rotation/
    opacity from iteration 2 onward and wrote a 63%-NaN PLY out of a run whose
    logs looked perfectly clean.
    """
    H, W = 40, 40
    device = "cuda" if torch.cuda.is_available() else "cpu"

    R_ref = torch.eye(3, device=device)
    view = MockCamera(R_ref, torch.zeros(3, device=device),
                      torch.rand((3, H, W), device=device), name="cam_0", uid=0)
    nb = MockCamera(R_ref, torch.tensor([0.05, 0.0, 0.0], device=device),
                    torch.rand((3, H, W), device=device), name="cam_1", uid=1)

    depth = torch.full((1, H, W), 2.0, device=device, requires_grad=True)
    # Normal along +x makes q == x_cam, which is exactly 0 on the principal-point
    # column and non-zero elsewhere: a few degenerate pixels among many usable
    # ones, which is what a real scene has. A fully degenerate plane would just
    # trip the early return and exercise nothing.
    normal = torch.zeros((3, H, W), device=device)
    normal[0, :, :] = 1.0
    normal.requires_grad_(True)
    alpha = torch.ones((1, H, W), device=device)

    loss = multiview_photometric_loss(
        view, [nb], depth, normal, alpha,
        num_samples=H * W, veto_weight=0.0, saturation_threshold=0.0
    )
    # The value was never the symptom -- it stayed finite throughout the bug.
    assert torch.isfinite(loss)

    loss.backward()
    assert depth.grad is not None and torch.isfinite(depth.grad).all(), \
        "non-finite depth gradient from a degenerate plane (q ~ 0)"
    assert normal.grad is not None and torch.isfinite(normal.grad).all(), \
        "non-finite normal gradient from a degenerate plane (q ~ 0)"


def test_multiview_loss_saturation_exclusion():
    """Verify saturated pixels are excluded from multiview homography sampling."""
    H, W = 40, 40
    device = "cuda" if torch.cuda.is_available() else "cpu"

    R = torch.eye(3, device=device)
    T = torch.zeros(3, device=device)

    # Image is completely blown out / saturated
    sat_img = torch.full((3, H, W), 1.0, device=device)
    view = MockCamera(R, T, sat_img, name="cam_sat", uid=0)
    nb = MockCamera(R, torch.tensor([0.05, 0.0, 0.0], device=device), sat_img, name="cam_nb", uid=1)

    depth = torch.full((1, H, W), 2.0, device=device)
    normal = torch.zeros((3, H, W), device=device)
    normal[2, :, :] = -1.0
    alpha = torch.ones((1, H, W), device=device)

    # With saturation_threshold=0.98, all solid pixels are excluded, returning zero loss
    loss = multiview_photometric_loss(
        view, [nb], depth, normal, alpha,
        num_samples=1000, saturation_threshold=0.98
    )
    assert torch.isclose(loss, torch.tensor(0.0, device=device))


# ==============================================================================
# 3. View-Evidence Floater Pruning (TIDI-GS style)
# ==============================================================================

@pytest.fixture
def mock_gaussians():
    """Create a minimal GaussianModel on CUDA or CPU with dummy parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gaussians = GaussianModel(sh_degree=0)
    N = 4

    gaussians._xyz = nn.Parameter(torch.zeros((N, 3), device=device))
    gaussians._features_dc = nn.Parameter(torch.zeros((N, 1, 3), device=device))
    gaussians._features_rest = nn.Parameter(torch.zeros((N, 0, 3), device=device))
    gaussians._scaling = nn.Parameter(torch.zeros((N, 2), device=device))
    gaussians._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N, device=device))
    # Opacity > 0.1 so standard opacity pruning does not trigger
    gaussians._opacity = nn.Parameter(torch.full((N, 1), 2.0, device=device))  # sigmoid(2.0) ~ 0.88
    gaussians.max_radii2D = torch.zeros(N, device=device)
    gaussians.xyz_gradient_accum = torch.zeros((N, 1), device=device)
    gaussians.denom = torch.ones((N, 1), device=device)

    gaussians.frustum_counter = torch.zeros((N, 1), dtype=torch.int32, device=device)
    gaussians.observation_counter = torch.zeros((N, 1), dtype=torch.int32, device=device)
    gaussians.seen_frustum_mask = torch.zeros((N, 64), dtype=torch.uint8, device=device)
    gaussians.seen_contrib_mask = torch.zeros((N, 64), dtype=torch.uint8, device=device)

    # Optimizer with required param groups
    opt_params = [
        {"params": [gaussians._xyz], "name": "xyz"},
        {"params": [gaussians._features_dc], "name": "f_dc"},
        {"params": [gaussians._features_rest], "name": "f_rest"},
        {"params": [gaussians._opacity], "name": "opacity"},
        {"params": [gaussians._scaling], "name": "scaling"},
        {"params": [gaussians._rotation], "name": "rotation"},
    ]
    gaussians.optimizer = torch.optim.Adam(opt_params, lr=1e-3)
    return gaussians


def test_view_evidence_pruning_culls_floaters(mock_gaussians):
    """Verify densify_and_prune culls primitives with high frustum count but low contribution."""
    g = mock_gaussians
    # Point 0: Floater: in frustum of 10 views, but only actively contributed in 1 view
    g.frustum_counter[0] = 10
    g.observation_counter[0] = 1

    # Point 1: Valid Surface: in frustum of 10 views, contributed in all 10 views
    g.frustum_counter[1] = 10
    g.observation_counter[1] = 10

    # Point 2: Boundary/Corner: in frustum of only 3 views (< 8), contributed in 1
    g.frustum_counter[2] = 3
    g.observation_counter[2] = 1

    # Point 3: Floater at threshold: in frustum of 8 views, contributed in 2 views (<= 2)
    g.frustum_counter[3] = 8
    g.observation_counter[3] = 2

    # Pruning with view_evidence_cull=False: all 4 points survive
    initial_xyz = g.get_xyz.clone()
    # Mock densify_and_clone/split to no-op
    g.densify_and_clone = lambda grads, max_grad, extent: None
    g.densify_and_split = lambda grads, max_grad, extent: None

    # Test with view evidence culling ACTIVE (frustum_min=8, min_obs=2)
    g.densify_and_prune(
        max_grad=100.0, min_opacity=0.01, extent=1.0, max_screen_size=None,
        view_evidence_cull=True, frustum_min=8, min_obs=2
    )

    # Point 0 and Point 3 must be pruned! Point 1 and Point 2 must survive!
    assert g.get_xyz.shape[0] == 2
    # Frustum counters of surviving points:
    # Point 1 had frustum=10, obs=10; Point 2 had frustum=3, obs=1
    surviving_frustums = set(g.frustum_counter.squeeze(-1).tolist())
    assert 10 in surviving_frustums
    assert 3 in surviving_frustums
    assert 8 not in surviving_frustums  # Point 3 pruned


def test_add_densification_stats_unique_camera_tracking(mock_gaussians):
    """Verify add_densification_stats tracks distinct cameras without overcounting."""
    g = mock_gaussians
    device = g.frustum_counter.device

    viewspace_points = torch.ones((4, 2), device=device, requires_grad=True)
    # Simulate non-zero viewspace gradient on point 0 and point 1
    loss = (viewspace_points[0]**2).sum() + (viewspace_points[1]**2).sum()
    loss.backward()

    update_filter = torch.tensor([True, True, True, False], device=device)

    # Camera 5 visits
    g.add_densification_stats(viewspace_points, update_filter, cam_id=5)

    # Points 0, 1, 2 in frustum -> count = 1
    assert g.frustum_counter[0].item() == 1
    assert g.frustum_counter[1].item() == 1
    assert g.frustum_counter[2].item() == 1
    assert g.frustum_counter[3].item() == 0

    # Points 0, 1 had gradient -> obs count = 1, Point 2 had no gradient -> obs count = 0
    assert g.observation_counter[0].item() == 1
    assert g.observation_counter[1].item() == 1
    assert g.observation_counter[2].item() == 0

    # Camera 5 visits AGAIN in the next epoch (same camera id = 5)
    # Must NOT increment counters again!
    g.add_densification_stats(viewspace_points, update_filter, cam_id=5)
    assert g.frustum_counter[0].item() == 1
    assert g.observation_counter[0].item() == 1

    # Camera 6 visits: new camera id -> increments counters to 2
    g.add_densification_stats(viewspace_points, update_filter, cam_id=6)
    assert g.frustum_counter[0].item() == 2
    assert g.observation_counter[0].item() == 2


def test_decoupled_pruning_without_densification(mock_gaussians):
    """Verify densify_and_prune with allow_densification=False prunes dead points without spawning new ones."""
    g = mock_gaussians
    device = g.frustum_counter.device

    # Point 0 has low opacity (< 0.05, logit -4.0 -> sigmoid(-4.0) ~ 0.018)
    g._opacity.data[0] = -4.0
    g._opacity.data[1] = 2.0
    g._opacity.data[2] = 2.0
    g._opacity.data[3] = 2.0

    # Give huge gradients that would trigger clone/split if densification were enabled
    g.xyz_gradient_accum = torch.full((4, 1), 100.0, device=device)
    g.denom = torch.ones((4, 1), device=device)

    # Pruning active, but densification disabled (e.g. Phase 1 fitting stage)
    g.densify_and_prune(
        max_grad=0.0002, min_opacity=0.05, extent=1.0, max_screen_size=None,
        allow_densification=False, view_evidence_cull=False
    )

    # Point 0 must be pruned; no new points cloned or split
    assert g.get_xyz.shape[0] == 3
    # Gradient accumulators must be cleanly reset
    assert torch.all(g.xyz_gradient_accum == 0.0)
    assert torch.all(g.denom == 0.0)



def test_rotate_matches_matmul_past_rocm_row_limit():
    """_rotate must equal `points @ M` well past the 2**19 row the gemm kernel stops at."""
    torch.manual_seed(0)
    points = torch.randn(600_000, 3)
    M = torch.randn(3, 3)

    out = _rotate(points, M)

    # Reference on a slice small enough that matmul is trustworthy even on gfx1200.
    assert torch.allclose(out[:1000], points[:1000] @ M, atol=1e-5)
    # The rows the buggy kernel drops must carry real values.
    assert out[524_288:].abs().sum() > 0
    assert torch.allclose(out[-1], points[-1] @ M, atol=1e-5)


def test_multiview_ncc_is_invariant_to_exposure_but_not_to_structure():
    """The NCC must ignore a brightness/contrast change and catch a structural one.

    This is the whole reason the loss is an NCC rather than a colour difference:
    the two views of a phone capture differ by auto-exposure, and only a
    structural mismatch means the geometry is wrong.
    """
    H, W = 40, 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texture = _smooth_texture(H, W, device)

    R = torch.eye(3, device=device)
    view = MockCamera(R, torch.zeros(3, device=device), texture, name="cam_0", uid=0)
    # Zero baseline, so the homography is the identity and the only thing under
    # test is the comparison itself. With a real baseline the mock neighbour
    # image would also have to be shifted by the induced disparity
    # (f * t / z = 2.5 px here) or the patches legitimately would not match, and
    # the test would be measuring the fixture rather than the loss.
    T_nb = torch.zeros(3, device=device)

    # Same structure, different exposure (affine in intensity).
    exposed = (texture * 0.6 + 0.3).clamp(0.0, 1.0)
    nb_exposed = MockCamera(R, T_nb, exposed, name="cam_1", uid=1)
    # Same brightness statistics, structure inverted (NCC -> -1).
    inverted = 1.0 - texture
    nb_inverted = MockCamera(R, T_nb, inverted, name="cam_2", uid=2)

    depth = torch.full((1, H, W), 2.0, device=device)
    normal = torch.zeros((3, H, W), device=device)
    normal[2, :, :] = -1.0
    alpha = torch.ones((1, H, W), device=device)

    kw = dict(num_samples=1000, veto_weight=0.0, saturation_threshold=0.0)
    loss_exposed = multiview_photometric_loss(view, [nb_exposed], depth, normal, alpha, **kw)
    loss_inverted = multiview_photometric_loss(view, [nb_inverted], depth, normal, alpha, **kw)

    assert loss_exposed.item() < 0.02, f"exposure change must not cost: {loss_exposed.item()}"
    assert loss_inverted.item() > 0.9, f"structural mismatch must cost: {loss_inverted.item()}"


def test_multiview_ncc_skips_textureless_patches():
    """A flat wall has no structure to correlate; the depth prior owns those pixels."""
    H, W = 40, 40
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flat = torch.full((3, H, W), 0.5, device=device)

    R = torch.eye(3, device=device)
    view = MockCamera(R, torch.zeros(3, device=device), flat, name="cam_0", uid=0)
    nb = MockCamera(R, torch.tensor([0.05, 0.0, 0.0], device=device),
                    torch.full((3, H, W), 0.9, device=device), name="cam_1", uid=1)

    depth = torch.full((1, H, W), 2.0, device=device)
    normal = torch.zeros((3, H, W), device=device)
    normal[2, :, :] = -1.0
    alpha = torch.ones((1, H, W), device=device)

    loss = multiview_photometric_loss(view, [nb], depth, normal, alpha,
                                      num_samples=1000, saturation_threshold=0.0)
    assert loss.item() == 0.0


def test_multiview_warp_lands_on_the_corresponding_pixel():
    """The plane-induced homography must match plain pinhole projection.

    Built without reference to the loss's internals: the neighbour image is
    synthesised by projecting the same fronto-parallel plane through elementary
    pinhole maths, so a correct homography has to score NCC ~ 1 against it and a
    mis-derived one cannot. Nothing else in the suite pins down the warp itself --
    the other tests would pass just as happily with the homography transposed.
    """
    H, W = 64, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texture = _smooth_texture(H, W, device)

    z = 2.0
    tx = 0.05
    R = torch.eye(3, device=device)
    view = MockCamera(R, torch.zeros(3, device=device), texture, name="cam_0", uid=0)
    fx = (W / 2.0) / math.tan(view.FoVx / 2.0)

    # MockCamera documents x_cam = x_world @ W2C, and the reference camera is the
    # identity, so world == reference camera frame. A point at (X, Y, z) there is
    # at (X + tx, Y, z) in the neighbour, i.e. the neighbour sees it shifted by
    # fx * tx / z pixels. Synthesise the neighbour image by that shift.
    shift = fx * tx / z
    xs = torch.arange(W, device=device, dtype=torch.float32) - shift
    grid_x = (xs / (W - 1) * 2.0 - 1.0).unsqueeze(0).repeat(H, 1)
    grid_y = (torch.arange(H, device=device, dtype=torch.float32) / (H - 1) * 2.0 - 1.0).unsqueeze(1).repeat(1, W)
    nb_img = torch.nn.functional.grid_sample(
        texture.unsqueeze(0), torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0),
        mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)
    nb = MockCamera(R, torch.tensor([tx, 0.0, 0.0], device=device), nb_img, name="cam_1", uid=1)

    depth = torch.full((1, H, W), z, device=device)
    normal = torch.zeros((3, H, W), device=device)
    normal[2, :, :] = -1.0
    alpha = torch.ones((1, H, W), device=device)

    loss = multiview_photometric_loss(view, [nb], depth, normal, alpha,
                                      num_samples=1000, veto_weight=0.0,
                                      saturation_threshold=0.0)
    assert loss.item() < 0.05, f"correct warp must score NCC ~ 1, got loss {loss.item()}"


def test_growth_target_is_linear_and_clamped():
    from train import growth_target

    assert growth_target(100, 1000, 2000, 200, 1200) == 1000   # before start
    assert growth_target(700, 1000, 2000, 200, 1200) == 1500   # halfway
    assert growth_target(9999, 1000, 2000, 200, 1200) == 2000  # clamped at budget


def test_densify_to_target_honours_the_budget_exactly(mock_gaussians):
    """k candidates -> exactly k new surfels, whether they clone or split."""
    g = mock_gaussians
    n0 = g.get_xyz.shape[0]
    grads = torch.tensor([[1.0], [0.5], [0.25], [0.125]], device=g._xyz.device)
    # Half the candidates oversized, so both the clone and the split path run.
    g._scaling = nn.Parameter(torch.tensor([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]],
                                           device=g._xyz.device))
    g.optimizer.param_groups[4]["params"][0] = g._scaling

    added = g.densify_to_target(grads, extent=1.0, target_count=n0 + 3)
    assert added == 3
    assert g.get_xyz.shape[0] == n0 + 3

    # Already at target: no growth, no error.
    assert g.densify_to_target(grads[:1].repeat(g.get_xyz.shape[0], 1), 1.0, n0) == 0


@pytest.mark.parametrize("width,height", [(540, 960), (1080, 1920)])
def test_depth_to_normal_is_not_degenerate_at_native_resolution(width, height):
    """surf_normal must survive full resolution.

    |dx x dy| scales with the metric gap between neighbouring pixels' 3D points,
    so it falls off as resolution rises. An absolute cutoff zeroed 97-99% of the
    normal map at 1080p -- the resolution the final stage trains at -- which turned
    the normal consistency loss into the constant 1, with no gradient.
    """
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix
    from utils.point_utils import depth_to_normal

    fovx, fovy = math.radians(47.0), math.radians(72.0)  # phone portrait capture
    w2c = torch.eye(4, device="cuda")
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    view = MiniCam(width, height, fovy, fovx, 0.01, 100.0, w2c, w2c @ proj)

    # A fronto-parallel wall 1.5 m away -- indoor capture distance -- where the
    # per-pixel 3D spacing is ~1 mm and |dx x dy| lands around 1e-6.
    depth = torch.full((1, height, width), 1.5, device="cuda")
    normal = depth_to_normal(view, depth)

    interior = normal[1:-1, 1:-1]
    lengths = interior.norm(dim=-1)
    assert (lengths > 0.9).float().mean() > 0.99


# ==============================================================================
# 4. Multi-Scale Scheduling, Opacity Recovery & Gradient Shock Damping
# ==============================================================================

def test_compute_epoch_schedule_calibrated_and_scaling():
    """Verify compute_epoch_schedule yields calibrated 5k/8k/10.5k for 200 cams and scales for huge scenes."""
    import sys
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_dir / "03_2DGS_training"))
    from step_train import compute_epoch_schedule

    # Case 1: Dense depth prior (use_depth=True) - fast, streamlined schedule, no opacity reset
    stages_200, params_200 = compute_epoch_schedule(200, use_depth=True)
    assert stages_200 == ((420, 3_000), (750, 5_000), (1, 6_500))
    assert params_200["total_iterations"] == 6_500
    assert params_200["densify_until_iter"] == 5_000
    assert params_200["opacity_reset_iter"] == -1
    assert params_200["prior_decay_from"] == 3_000
    assert params_200["prior_decay_until"] == 5_000

    stages_383, params_383 = compute_epoch_schedule(383, use_depth=True)
    assert stages_383[0][1] == round(8.0 * 383)
    assert stages_383[1][1] == stages_383[0][1] + round(6.0 * 383)
    assert stages_383[2][1] == stages_383[1][1] + round(4.0 * 383)
    assert params_383["opacity_reset_iter"] == -1

    # Case 2: Sparse SfM fallback (use_depth=False) - traditional 25/15/10 epochs with targeted reset
    stages_sfm_200, params_sfm_200 = compute_epoch_schedule(200, use_depth=False)
    assert stages_sfm_200 == ((420, 5_000), (750, 8_000), (1, 10_500))
    assert params_sfm_200["opacity_reset_iter"] == 6_000

    stages_sfm_650, params_sfm_650 = compute_epoch_schedule(650, use_depth=False)
    assert stages_sfm_650[0][1] == 25 * 650
    assert stages_sfm_650[1][1] == stages_sfm_650[0][1] + 15 * 650
    assert stages_sfm_650[2][1] == stages_sfm_650[1][1] + 10 * 650


def test_max_world_size_bloater_pruning(mock_gaussians):
    """Verify max_world_size (3.5 cm) prunes oversized surfels and prevents bloaters."""
    g = mock_gaussians
    # Set scales: surfels 0 and 1 are valid (1.5cm, 2.5cm); surfels 2 and 3 are bloaters (5cm, 20cm)
    scales_m = torch.tensor([
        [0.015, 0.015],
        [0.025, 0.020],
        [0.050, 0.040],
        [0.200, 0.150],
    ], dtype=torch.float32, device=g.get_xyz.device)
    g._scaling.data = torch.log(scales_m)

    # Densify and prune with max_world_size = 0.035 (3.5 cm)
    g.densify_and_prune(
        max_grad=0.0002, min_opacity=0.01, extent=4.0, max_screen_size=None,
        allow_densification=False, view_evidence_cull=False,
        max_world_size=0.035
    )

    # Only surfels 0 and 1 must survive
    assert g.get_xyz.shape[0] == 2, f"Expected 2 surfels, got {g.get_xyz.shape[0]}"
    assert (g.get_scaling <= 0.035).all(), "Surviving surfels must all be <= 3.5 cm"

    # Verify log-scale clamping physically bounds maximum scaling
    g._scaling.data.fill_(0.0)  # exp(0) = 1.0 meter (huge bloater)
    g._scaling.data.clamp_(max=math.log(0.035))
    assert (g.get_scaling <= 0.035001).all()


def test_opacity_recovery_window_prevents_premature_extinction(mock_gaussians):
    """Verify opacity recovery window (min_opacity=0.0) protects surfels until cull step."""
    g = mock_gaussians
    device = g.frustum_counter.device

    # Simulate post-reset state: opacities clamped to 0.01 (logit ~ -4.6)
    g._opacity.data.fill_(-4.6)
    assert (g.get_opacity < 0.05).all()

    # During recovery window: effective_opacity_cull is 0.0 -> all surfels survive!
    g.densify_and_prune(
        max_grad=0.0002, min_opacity=0.0, extent=1.0, max_screen_size=None,
        allow_densification=False, view_evidence_cull=False
    )
    assert g.get_xyz.shape[0] == 4, "Recovery window must not cull surfels at min_opacity=0.0"

    # End of recovery window (cull step): effective_opacity_cull = 0.05 -> low opacity surfels culled!
    g.densify_and_prune(
        max_grad=0.0002, min_opacity=0.05, extent=1.0, max_screen_size=None,
        allow_densification=False, view_evidence_cull=False
    )
    assert g.get_xyz.shape[0] == 0, "Cull step must purge surfels that failed to recover"

