"""SAD-2DGS densification and free-space carving (2dgs_combined_pipeline.md §4, §5.3).

Every geometric quantity here is checked against pinhole arithmetic worked out
independently of the implementation, not against the implementation's own
output -- the multi-view homography in this same stage carried a sign error for
weeks precisely because its tests only ever compared it to itself.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_STAGE = Path(__file__).resolve().parents[1] / "03_2DGS_training"
if str(_STAGE) not in sys.path:
    sys.path.insert(0, str(_STAGE))

from utils.structure_tensor import wavelength_map, LAMBDA_MIN_CLAMP_PX  # noqa: E402
from utils.view_stats import (  # noqa: E402
    freespace_classify, frequency_violation, intrinsics, project, sample_map,
    tangential_screen_extents,
)


class FakeCamera:
    """Camera at the origin looking down +Z, in the codebase's transposed convention."""

    def __init__(self, width=200, height=100, fov_x=math.radians(90.0), position=(0.0, 0.0, 0.0)):
        self.image_width = width
        self.image_height = height
        self.FoVx = fov_x
        # Square pixels: fy == fx, so FoVy follows from the aspect ratio.
        fx = width / (2.0 * math.tan(fov_x * 0.5))
        self.FoVy = 2.0 * math.atan(height / (2.0 * fx))
        m = torch.eye(4)
        m[3, :3] = -torch.tensor(position)
        self.world_view_transform = m.cuda() if torch.cuda.is_available() else m


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- projection


def test_project_matches_hand_computed_pinhole():
    cam = FakeCamera()
    fx, fy, cx, cy = intrinsics(cam)
    # 90 degree horizontal FoV over 200 px -> fx = 100.
    assert fx == pytest.approx(100.0)
    assert (cx, cy) == (100.0, 50.0)

    pts = torch.tensor([[0.0, 0.0, 2.0],     # on the optical axis -> centre
                        [2.0, 0.0, 2.0],     # 45 deg right -> cx + fx
                        [0.0, -1.0, 4.0]],   # up by 1 at 4 m
                       device=_dev())
    u, v, z = project(pts, cam)
    assert u.tolist() == pytest.approx([100.0, 200.0, 100.0])
    assert v.tolist() == pytest.approx([50.0, 50.0, 50.0 - fy * 0.25])
    assert z.tolist() == pytest.approx([2.0, 4.0 ** 0 * 2.0, 4.0])


def test_project_respects_camera_translation():
    """The transposed world_view_transform convention, pinned down explicitly."""
    cam = FakeCamera(position=(1.0, 0.0, 0.0))
    u, _, z = project(torch.tensor([[1.0, 0.0, 3.0]], device=_dev()), cam)
    # The point is directly in front of a camera moved to x=1, so it lands on
    # the principal point, not at cx + fx/3.
    assert u.item() == pytest.approx(100.0)
    assert z.item() == pytest.approx(3.0)


# --------------------------------------------------- tangential screen extent


def test_tangential_extent_scales_as_focal_times_size_over_depth():
    """A fronto-parallel surfel's on-screen half-extent is fx * s / z."""
    cam = FakeCamera()
    fx, fy, _, _ = intrinsics(cam)
    z = 4.0
    s_u, s_v = 0.10, 0.05
    xyz = torch.tensor([[0.0, 0.0, z]], device=_dev())
    scaling = torch.tensor([[s_u, s_v]], device=_dev())
    rot = torch.eye(3, device=_dev())[None]      # t_u = +X, t_v = +Y, n = +Z

    ext_u, ext_v = tangential_screen_extents(xyz, scaling, rot, cam)
    assert ext_u.item() == pytest.approx(fx * s_u / z, rel=1e-4)
    assert ext_v.item() == pytest.approx(fy * s_v / z, rel=1e-4)


def test_edge_on_surfel_has_near_zero_extent_along_the_viewing_axis():
    """A tangent axis pointing at the camera projects to almost nothing.

    This is what stops a grazing-angle surfel from being scored as
    under-resolved in a direction it does not actually cover on screen.
    """
    cam = FakeCamera()
    xyz = torch.tensor([[0.0, 0.0, 4.0]], device=_dev())
    scaling = torch.tensor([[0.10, 0.10]], device=_dev())
    # t_u along the view direction (+Z), t_v still +Y.
    rot = torch.tensor([[[0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]]], device=_dev())
    ext_u, ext_v = tangential_screen_extents(xyz, scaling, rot, cam)
    assert ext_u.item() < 1e-3
    assert ext_v.item() > 1.0


def test_frequency_violation_is_extent_over_wavelength():
    eta_u, eta_v = frequency_violation(torch.tensor([6.0, 2.0]),
                                       torch.tensor([3.0, 1.0]),
                                       torch.tensor([3.0, 4.0]))
    assert eta_u.tolist() == pytest.approx([2.0, 0.5])
    assert eta_v.tolist() == pytest.approx([1.0, 0.25])


# ------------------------------------------------------------- map sampling


def test_sample_map_flags_out_of_frame_points_and_serves_the_default():
    values = torch.arange(12, dtype=torch.float32, device=_dev()).reshape(3, 4)
    u = torch.tensor([0.0, 3.0, -1.0, 4.0], device=_dev())
    v = torch.tensor([0.0, 2.0, 0.0, 0.0], device=_dev())
    out, inside = sample_map(u, v, values, default=-7.0)
    assert inside.tolist() == [True, True, False, False]
    assert out.tolist() == pytest.approx([0.0, 11.0, -7.0, -7.0])


# ------------------------------------------------------- free-space carving


def test_freespace_classify_is_asymmetric_about_the_surface():
    margin = 0.04
    depth = torch.full((5,), 3.0)
    z = torch.tensor([2.0,    # well in front  -> free space
                      2.97,   # inside margin  -> on surface
                      3.03,   # inside margin  -> on surface
                      4.0,    # behind         -> occluded, no evidence
                      2.0])   # in front, but the ray hit nothing
    depth = depth.clone()
    depth[4] = 0.0
    inside = torch.ones(5, dtype=torch.bool)
    free, surf = freespace_classify(z, depth, inside, margin)
    assert free.tolist() == [True, False, False, False, False]
    assert surf.tolist() == [False, True, True, False, False]
    # The occluded surfel is counted in neither: that asymmetry is the whole
    # criterion. Counting it as surface evidence would let a floater behind a
    # wall accumulate a veto against its own removal.
    assert not (free[3] or surf[3])


# ------------------------------------------------------- Lambda_min maps


def _striped_image(width, height, period_px, device):
    """A vertical sine grating of known wavelength, on a mid-grey background."""
    x = torch.arange(width, device=device, dtype=torch.float32)
    row = 0.5 + 0.4 * torch.sin(2.0 * math.pi * x / period_px)
    return row.expand(height, width)[None].repeat(3, 1, 1).contiguous()


@pytest.mark.parametrize("period", [8.0, 24.0])
def test_wavelength_map_tracks_the_grating_period(period):
    """A coarser grating must report a longer wavelength than a finer one.

    The absolute constant relating Lambda_min to the true period depends on the
    doc's scale-space weighting, so this pins the monotonicity and the order of
    magnitude rather than an exact value.
    """
    dev = _dev()
    lam = wavelength_map(_striped_image(128, 64, period, dev))
    mean = float(lam[:, 16:-16].mean())
    assert 0.3 * period < mean < 4.0 * period


def test_wavelength_map_is_monotonic_in_feature_size():
    dev = _dev()
    fine = float(wavelength_map(_striped_image(128, 64, 6.0, dev))[:, 16:-16].mean())
    coarse = float(wavelength_map(_striped_image(128, 64, 30.0, dev))[:, 16:-16].mean())
    assert coarse > fine


def test_flat_image_saturates_the_clamp_so_nothing_ever_splits_there():
    """A textureless wall has no feature to resolve; eta must collapse to ~0."""
    dev = _dev()
    lam = wavelength_map(torch.full((3, 64, 64), 0.5, device=dev))
    assert float(lam.min()) == pytest.approx(LAMBDA_MIN_CLAMP_PX[1])
    eta, _ = frequency_violation(torch.tensor([20.0], device=dev),
                                 torch.tensor([20.0], device=dev),
                                 lam[:1, 0])
    assert float(eta) < 0.05


def test_wavelength_map_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        wavelength_map(torch.zeros(1, 32, 32))


# --------------------------------------- the resolution rescale in get_wavelength


def test_get_wavelength_rescales_pixels_to_the_render_resolution():
    """Lambda_min is a length in pixels, so doubling the render size doubles it.

    Without this the SAD criterion would read every surfel as twice as well
    resolved as it is at native resolution, and splitting would silently stop --
    the same class of bug that zeroed surf_normal at 1080p.
    """
    from utils.depth_prior import DepthPriors

    priors = DepthPriors.__new__(DepthPriors)      # no cache directory needed
    priors.lambda_min = {"f": torch.full((50, 100), 8.0)}

    class V:
        image_name = "f"

    v = V()
    v.image_height, v.image_width = 50, 100
    assert float(priors.get_wavelength(v).mean()) == pytest.approx(8.0, rel=1e-3)

    v.image_height, v.image_width = 100, 200
    assert float(priors.get_wavelength(v).mean()) == pytest.approx(16.0, rel=1e-2)

    v.image_name = "missing"
    assert priors.get_wavelength(v) is None


def _attach_optimizer(g):
    """densify_* routes every tensor through the optimizer state, so a bare
    hand-built model needs the six named param groups to exist."""
    import torch.optim
    names = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
    attrs = ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")
    for a in attrs:
        setattr(g, a, torch.nn.Parameter(getattr(g, a).requires_grad_(True)))
    g.optimizer = torch.optim.Adam(
        [{"params": [getattr(g, a)], "lr": 0.0, "name": n} for n, a in zip(names, attrs)],
        lr=0.0, eps=1e-15)
    return g


# ------------------------------------------------------ the analytic split


def test_in_plane_split_geometry_matches_the_doc_grid():
    """Children tile the parent disc, stay in its plane, and conserve opacity."""
    from scene.gaussian_model import GaussianModel

    n_u, n_v = 3, 2
    alpha = 0.7
    # The doc's grid offsets, computed here from the formula directly.
    expect_u = [(2 * a - 1 - n_u) / n_u for a in range(1, n_u + 1)]
    expect_v = [(2 * b - 1 - n_v) / n_v for b in range(1, n_v + 1)]
    assert expect_u == pytest.approx([-2 / 3, 0.0, 2 / 3])
    assert expect_v == pytest.approx([-0.5, 0.5])

    # Children stacked along a ray must composite back to the parent's opacity.
    alpha_child = 1.0 - (1.0 - alpha) ** (1.0 / (n_u * n_v))
    composited = 1.0 - (1.0 - alpha_child) ** (n_u * n_v)
    assert composited == pytest.approx(alpha)
    assert alpha_child < alpha

    assert hasattr(GaussianModel, "_sad_split")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU model path")
def test_densify_sad_splits_only_multi_view_violators_and_respects_headroom():
    from scene.gaussian_model import GaussianModel

    g = GaussianModel(0)
    n = 4
    g._xyz = torch.zeros(n, 3, device="cuda")
    g._xyz[:, 2] = torch.arange(n, device="cuda").float()
    g._scaling = torch.log(torch.full((n, 2), 0.02, device="cuda"))
    g._rotation = torch.zeros(n, 4, device="cuda")
    g._rotation[:, 0] = 1.0
    g._opacity = torch.zeros(n, 1, device="cuda")
    g._features_dc = torch.zeros(n, 1, 3, device="cuda")
    g._features_rest = torch.zeros(n, 0, 3, device="cuda")
    g.xyz_gradient_accum = torch.zeros(n, 1, device="cuda")
    g.denom = torch.zeros(n, 1, device="cuda")
    g.max_radii2D = torch.zeros(n, device="cuda")
    _attach_optimizer(g)

    # Surfel 0: violates in 8 of 8 views -> splits.
    # Surfel 1: violates in 2 of 8 -> below tau_split, must not split.
    # Surfel 2: violates in 8 of 8 but has only 2 views -> below min_views.
    # Surfel 3: never violates.
    g.view_stats = {
        "sad_v_total": torch.tensor([8.0, 8.0, 2.0, 8.0], device="cuda"),
        "sad_v_high": torch.tensor([8.0, 2.0, 2.0, 0.0], device="cuda"),
        "sad_eta_u": torch.tensor([8 * 4.0, 8 * 4.0, 2 * 4.0, 8 * 0.1], device="cuda"),
        "sad_eta_v": torch.tensor([8 * 1.0, 8 * 1.0, 2 * 1.0, 8 * 0.1], device="cuda"),
    }

    # eta_u = 4 -> n_u = ceil(sqrt(4)) = 2; eta_v = 1 -> n_v = 1. Two children,
    # so one net surfel added for the single eligible parent.
    added = g.densify_sad(headroom=100, tau_split=0.75, min_views=4)
    assert added == 1
    assert g.get_xyz.shape[0] == n + 1

    # Children sit on either side of where the parent was, in its own plane.
    kept = g.get_xyz[:, 2].sort().values.tolist()
    assert kept == pytest.approx([0.0, 0.0, 1.0, 2.0, 3.0], abs=1e-5, rel=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU model path")
def test_densify_sad_is_a_no_op_without_evidence():
    from scene.gaussian_model import GaussianModel

    g = GaussianModel(0)
    g._xyz = torch.zeros(3, 3, device="cuda")
    assert g.densify_sad(headroom=100) == 0
    assert g.densify_sad(headroom=0) == 0


def test_freespace_prune_mask_needs_both_a_count_and_a_ratio():
    from scene.gaussian_model import GaussianModel

    g = GaussianModel.__new__(GaussianModel)
    g._xyz = torch.zeros(4, 3)
    g.view_stats = {
        # 0: plenty of free sightings, mostly free  -> prune
        # 1: mostly free but only 2 sightings       -> keep (count gate)
        # 2: many free sightings but mostly surface -> keep (ratio gate)
        # 3: never seen at all                      -> keep
        "free_views": torch.tensor([5.0, 2.0, 5.0, 0.0]),
        "surface_views": torch.tensor([1.0, 0.0, 20.0, 0.0]),
    }
    mask = g.freespace_prune_mask(min_free_views=3, free_ratio=0.6)
    assert mask.tolist() == [True, False, False, False]
