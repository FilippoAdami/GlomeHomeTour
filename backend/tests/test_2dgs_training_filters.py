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


def test_asymmetric_veto_weighting():
    """Verify asymmetric veto weight penalizes worst mismatch instead of diluting."""
    H, W = 40, 40
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reference camera looking at planar wall at z=2.0
    R_ref = torch.eye(3, device=device)
    T_ref = torch.zeros(3, device=device)
    ref_img = torch.full((3, H, W), 0.5, device=device)
    view = MockCamera(R_ref, T_ref, ref_img, name="cam_0", uid=0)

    # Neighbor 1: small baseline, identical image (error ~ 0.0)
    T_nb1 = torch.tensor([0.05, 0.0, 0.0], device=device)
    nb1_img = ref_img.clone()
    nb1 = MockCamera(R_ref, T_nb1, nb1_img, name="cam_1", uid=1)

    # Neighbor 2: side view with completely contradictory texture (error ~ 0.5)
    T_nb2 = torch.tensor([0.1, 0.0, 0.0], device=device)
    nb2_img = torch.full((3, H, W), 1.0, device=device)
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
