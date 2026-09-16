"""Unit tests for Stage 4: Taming-2DGS Density Control & Normal-Aware Voxel Pruning."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from density_control import DensityControlConfig, TamingDensityController
from model import Material2DGSModel


def _make_dummy_model(
    positions: np.ndarray,
    normals: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    opacities: Optional[np.ndarray] = None,
) -> Material2DGSModel:
    n = len(positions)
    pos_t = torch.from_numpy(positions.astype(np.float32))

    if normals is None:
        normals_np = np.zeros((n, 3), dtype=np.float32)
        normals_np[:, 2] = 1.0
    else:
        normals_np = normals.astype(np.float32)

    # Construct orthonormal tangent frames to derive proper rotation quaternions
    ref = np.zeros_like(normals_np)
    near_z = np.abs(normals_np[:, 2]) > 0.9
    ref[near_z, 0] = 1.0
    ref[~near_z, 2] = 1.0
    u = np.cross(normals_np, ref)
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-6)
    v = np.cross(normals_np, u)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    r_mat = np.stack([u, v, normals_np], axis=-1)

    from model import matrix_to_quaternion
    quats = matrix_to_quaternion(torch.from_numpy(r_mat.astype(np.float32)))

    if scales is None:
        log_scales = torch.full((n, 2), float(np.log(0.02)), dtype=torch.float32)
    else:
        log_scales = torch.log(torch.clamp(torch.from_numpy(scales.astype(np.float32)), min=1e-6))

    if opacities is None:
        logit_op = torch.zeros((n, 1), dtype=torch.float32)  # sigmoid(0) = 0.5
    else:
        op_clamped = np.clip(opacities, 1e-4, 1.0 - 1e-4)
        logit_op = torch.from_numpy(np.log(op_clamped / (1.0 - op_clamped)).astype(np.float32))
        if logit_op.ndim == 1:
            logit_op = logit_op.unsqueeze(-1)

    return Material2DGSModel(
        xyz=pos_t,
        rotation=quats,
        scaling=log_scales,
        opacity=logit_op,
        albedo=torch.zeros((n, 3), dtype=torch.float32),
        roughness=torch.zeros((n, 1), dtype=torch.float32),
        metallic=torch.zeros((n, 1), dtype=torch.float32),
    )


def test_normal_aware_voxel_pruning_coplanar_vs_opposite():
    """Verify that coplanar duplicates in the same voxel are collapsed,

    while opposite-facing surfels on a thin wall are preserved.
    """
    controller = TamingDensityController(DensityControlConfig(
        enable_voxel_pruning=True,
        voxel_size_m=0.015,  # 1.5 cm
    ))

    # Point 0 and Point 1: same cell, same normal -> duplicate, should collapse to 1
    # Point 2: same cell, opposite normal (-Z) -> opposite side of thin wall, should be PRESERVED
    positions = np.array([
        [0.005, 0.005, 0.005],   # Point 0: +Z normal, opacity 0.8
        [0.007, 0.006, 0.004],   # Point 1: +Z normal, opacity 0.4 (lower, should be pruned)
        [0.006, 0.005, 0.006],   # Point 2: -Z normal, opacity 0.7 (opposite face, must be kept)
    ], dtype=np.float32)

    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)

    opacities = np.array([0.8, 0.4, 0.7], dtype=np.float32)

    model = _make_dummy_model(positions, normals=normals, opacities=opacities)
    keep_mask = controller.normal_aware_voxel_pruning(model)

    assert keep_mask[0].item() is True, "Point 0 (highest opacity +Z) must be kept"
    assert keep_mask[1].item() is False, "Point 1 (duplicate +Z) must be pruned"
    assert keep_mask[2].item() is True, "Point 2 (opposite -Z on thin wall) must be kept"


def test_clone_and_split_with_budget_capping():
    """Verify cloning, splitting, and strict adherence to max_primitives budget cap."""
    cfg = DensityControlConfig(
        max_primitives=6,
        growth_rate=1.0,
        grad_threshold=0.001,
        scale_split_threshold=0.04,
        min_opacity_prune=0.01,
        enable_voxel_pruning=False,
    )
    controller = TamingDensityController(cfg)

    # 3 Gaussians:
    # 0: high gradient, small scale (0.02) -> Clone candidate (+1 primitive)
    # 1: high gradient, large scale (0.05) -> Split candidate (+2 children, -1 parent = +1 primitive)
    # 2: low gradient -> neither
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)

    scales = np.array([
        [0.02, 0.02],  # small
        [0.05, 0.05],  # large
        [0.02, 0.02],
    ], dtype=np.float32)

    model = _make_dummy_model(positions, scales=scales)
    controller.init_buffers(3, model.xyz.device)

    # Fake gradients: primitive 0 and 1 have high gradients
    controller.grad_accum[0] = 0.01
    controller.denom[0] = 1
    controller.grad_accum[1] = 0.01
    controller.denom[1] = 1
    controller.grad_accum[2] = 0.0001
    controller.denom[2] = 1

    cloned, split, pruned = controller.densify_and_prune(model)

    assert cloned == 1, "Expected 1 clone"
    assert split == 1, "Expected 1 split"
    assert model.num_gaussians == 5, f"Expected 3 + 1 (clone) + 2 (split) - 1 (pruned parent) = 5, got {model.num_gaussians}"
    assert model.num_gaussians <= cfg.max_primitives


def test_opacity_reset():
    """Verify periodic opacity reset lowers high opacities to eliminate floaters."""
    controller = TamingDensityController(DensityControlConfig(reset_opacity_val=0.1))
    model = _make_dummy_model(
        np.zeros((5, 3)),
        opacities=np.array([0.9, 0.8, 0.05, 0.95, 0.08]),
    )

    controller.reset_opacities(model)
    op = model.opacity.squeeze(-1)

    assert (op <= 0.105).all(), "All opacities above reset value should be clamped down"
    assert op[2] < 0.1, "Opacities already below reset value should remain low"


def test_population_grows_with_realistic_tiny_gradients():
    """Regression: densification must outpace opacity pruning.

    Real positional gradients sit far below the old 2e-4 absolute floor, so
    almost nothing was densified while opacity pruning kept firing and the
    scene decayed. Selection is now rank-based against a growth schedule.
    """
    n = 1000
    rng = np.random.default_rng(0)
    positions = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    opacities = np.full(n, 0.5, dtype=np.float32)
    opacities[:50] = 0.01  # 5% below the prune threshold, as at the end of a real run

    model = _make_dummy_model(positions, scales=np.full((n, 2), 0.02, dtype=np.float32), opacities=opacities)
    cfg = DensityControlConfig(max_primitives=1500, growth_rate=0.3, enable_voxel_pruning=False)
    controller = TamingDensityController(cfg)
    controller.init_buffers(n, model.xyz.device)
    controller.grad_accum[:] = torch.from_numpy(rng.uniform(1e-7, 1e-5, size=(n, 1)).astype(np.float32))
    controller.denom[:] = 1

    cloned, split, pruned = controller.densify_and_prune(model)

    assert cloned + split == 300, f"Expected ceil(1000 * 0.3) densified, got {cloned + split}"
    assert model.num_gaussians == 1000 + 300 - 50, f"Expected net +250, got {model.num_gaussians}"

    # With intervals left in the schedule, the remaining budget is spread out
    # instead of being spent at once.
    before = model.num_gaussians
    controller.grad_accum[:] = 1e-5
    controller.denom[:] = 1
    cloned, split, _ = controller.densify_and_prune(model, remaining_intervals=5)
    assert cloned + split == math.ceil((cfg.max_primitives - before) / 5)

    # And the budget caps growth rather than the gradient distribution.
    controller.grad_accum[:] = 1e-5
    controller.denom[:] = 1
    controller.densify_and_prune(model)
    assert model.num_gaussians <= cfg.max_primitives


def test_adam_moments_survive_densification():
    """Kept primitives must keep their Adam moments; new ones start at zero."""
    n = 20
    positions = np.linspace(-1.0, 1.0, n * 3).reshape(n, 3).astype(np.float32)
    model = _make_dummy_model(positions, scales=np.full((n, 2), 0.02, dtype=np.float32))
    opt = torch.optim.Adam([{"params": [model._xyz], "lr": 1e-3, "name": "_xyz"}], eps=1e-15)

    model._xyz.grad = torch.full_like(model._xyz, 0.1)
    opt.step()
    moments_before = opt.state[model._xyz]["exp_avg"].clone()
    assert (moments_before != 0).any()

    cfg = DensityControlConfig(max_primitives=1000, growth_rate=0.25, enable_voxel_pruning=False)
    controller = TamingDensityController(cfg)
    controller.init_buffers(n, model.xyz.device)
    controller.grad_accum[:] = torch.arange(n, dtype=torch.float32).unsqueeze(-1) * 1e-6
    controller.denom[:] = 1

    controller.densify_and_prune(model, optimizer=opt)

    state = opt.state[model._xyz]
    assert state["exp_avg"].shape == model._xyz.shape, "Adam state must be resized with the parameter"
    # Split children are appended last and must start cold.
    assert (state["exp_avg"][-2:] == 0).all(), "New primitives must start with zero momentum"
    assert (state["exp_avg"][:5] == moments_before[:5]).all(), "Survivors must keep their momentum"

    opt.step()  # must not blow up on the resized state
