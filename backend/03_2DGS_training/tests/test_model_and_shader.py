"""Unit tests for Phase 5 Stage 1 & Stage 2: 2DGS Material Model, PBR Shader, and Losses."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model import (
    Material2DGSModel,
    matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_tangent_frame,
)
from rasterizer_interface import (
    GBufferOutput,
    PyTorchFallbackRasterizer,
)
from pbr_shader import (
    DeferredCookTorranceShader,
    SpecularResidualMLP,
)
from losses import (
    ReconstructionLoss,
    l1_loss,
    ssim_loss,
    normal_loss,
)


@dataclass
class MockSurfelCloud:
    """Mock SurfelCloud structure for self-contained testing without upstream dependencies."""
    positions: np.ndarray
    normals: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    scales_2d: np.ndarray
    colors_rgb: np.ndarray
    sh_degree_0: np.ndarray
    opacities: np.ndarray


def create_synthetic_surfel_cloud(n: int = 50, seed: int = 42) -> MockSurfelCloud:
    """Generate mock surfel cloud data with valid orthonormal frames."""
    rng = np.random.RandomState(seed)

    positions = rng.randn(n, 3).astype(np.float32)
    # Normals: random unit vectors
    raw_normals = rng.randn(n, 3).astype(np.float32)
    norm = np.linalg.norm(raw_normals, axis=-1, keepdims=True)
    normals = raw_normals / np.maximum(norm, 1e-6)

    # Tangents: orthonormal basis
    ref = np.zeros_like(normals)
    near_z = np.abs(normals[:, 2]) > 0.9
    ref[near_z, 0] = 1.0
    ref[~near_z, 2] = 1.0

    u = np.cross(normals, ref)
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-6)
    v = np.cross(normals, u)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)

    scales = rng.uniform(0.01, 0.05, size=(n, 2)).astype(np.float32)
    colors = rng.uniform(0.1, 0.9, size=(n, 3)).astype(np.float32)
    opacities = rng.uniform(0.3, 0.8, size=(n,)).astype(np.float32)
    sh0 = (colors * 0.28209479177387814).astype(np.float32)

    return MockSurfelCloud(
        positions=positions,
        normals=normals.astype(np.float32),
        tangent_u=u.astype(np.float32),
        tangent_v=v.astype(np.float32),
        scales_2d=scales,
        colors_rgb=colors,
        sh_degree_0=sh0,
        opacities=opacities,
    )


# =========================================================================
# Stage 1: Model & Parameter Activation Tests
# =========================================================================

def test_quaternion_matrix_conversions():
    """Verify matrix_to_quaternion and quaternion_to_rotation_matrix roundtrips."""
    # Identity rotation
    eye = torch.eye(3).unsqueeze(0)
    q_eye = matrix_to_quaternion(eye)
    assert torch.allclose(q_eye, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1e-5)
    r_rec = quaternion_to_rotation_matrix(q_eye)
    assert torch.allclose(r_rec, eye, atol=1e-5)

    # Arbitrary 90 deg rotation around Y
    angle = math.pi / 2.0
    r_y = torch.tensor([
        [math.cos(angle), 0.0, math.sin(angle)],
        [0.0, 1.0, 0.0],
        [-math.sin(angle), 0.0, math.cos(angle)],
    ], dtype=torch.float32).unsqueeze(0)

    q_y = matrix_to_quaternion(r_y)
    r_y_rec = quaternion_to_rotation_matrix(q_y)
    assert torch.allclose(r_y, r_y_rec, atol=1e-4)

    # Tangent frame decomposition matches columns of R
    u, v, n = quaternion_to_tangent_frame(q_y)
    assert torch.allclose(u, r_y_rec[0, :, 0], atol=1e-5)
    assert torch.allclose(v, r_y_rec[0, :, 1], atol=1e-5)
    assert torch.allclose(n, r_y_rec[0, :, 2], atol=1e-5)


def test_material_2dgs_model_from_surfel_cloud():
    """Verify initializing Material2DGSModel from SurfelCloud."""
    surfels = create_synthetic_surfel_cloud(n=64)
    model = Material2DGSModel.from_surfel_cloud(surfels, default_roughness=0.45, default_metallic=0.1)

    assert model.num_gaussians == 64
    assert len(model) == 64

    # Parameter shape checks
    assert model._xyz.shape == (64, 3)
    assert model._rotation.shape == (64, 4)
    assert model._scaling.shape == (64, 2)
    assert model._opacity.shape == (64, 1)
    assert model._albedo.shape == (64, 3)
    assert model._roughness.shape == (64, 1)
    assert model._metallic.shape == (64, 1)

    # Activation ranges
    assert (model.opacity >= 0.0).all() and (model.opacity <= 1.0).all()
    assert (model.scaling > 0.0).all()
    assert (model.albedo >= 0.0).all() and (model.albedo <= 1.0).all()
    assert (model.roughness >= 0.04).all() and (model.roughness <= 1.0).all()
    assert (model.metallic >= 0.0).all() and (model.metallic <= 1.0).all()

    # Quaternions are normalized
    q_norms = torch.linalg.norm(model.rotation, dim=-1)
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5)

    # Check that model normals align with surfel cloud normals
    cos_sim = (model.normals * torch.from_numpy(surfels.normals)).sum(dim=-1)
    assert (cos_sim > 0.95).all()

    # Check orthonormal basis (u . v = 0, u . n = 0, v . n = 0)
    u, v, n = model.tangent_u, model.tangent_v, model.normals
    assert torch.allclose((u * v).sum(dim=-1), torch.zeros(64), atol=1e-4)
    assert torch.allclose((u * n).sum(dim=-1), torch.zeros(64), atol=1e-4)
    assert torch.allclose((v * n).sum(dim=-1), torch.zeros(64), atol=1e-4)


def test_material_2dgs_gradient_flow():
    """Verify backprop gradients flow to all model parameters."""
    surfels = create_synthetic_surfel_cloud(n=10)
    model = Material2DGSModel.from_surfel_cloud(surfels)

    view_dirs = torch.tensor([[0.0, 0.0, 1.0]], device=model.xyz.device).expand(10, 3)
    loss = (
        model.xyz.sum()
        + model.rotation.sum()
        + model.scaling.sum()
        + model.opacity.sum()
        + model.albedo.sum()
        + model.get_radiance(view_dirs).sum()
        + model.roughness.sum()
        + model.metallic.sum()
        + model.normals.sum()
    )
    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"
        assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf gradient"


# =========================================================================
# Stage 2: G-Buffer, PBR Shader & Loss Tests
# =========================================================================

def test_gbuffer_output_contract():
    """Verify GBufferOutput dataclass validation and device/dtype methods."""
    b, h, w = 2, 32, 48
    gbuffer = GBufferOutput(
        albedo=torch.rand(b, 3, h, w),
        normal=F.normalize(torch.randn(b, 3, h, w), dim=1),
        roughness=torch.rand(b, 1, h, w) * 0.96 + 0.04,
        metallic=torch.rand(b, 1, h, w),
        depth=torch.rand(b, 1, h, w) * 5.0 + 0.5,
    )

    assert gbuffer.batch_size == b
    assert gbuffer.height == h
    assert gbuffer.width == w

    # Clone & Detach
    clone = gbuffer.clone()
    assert torch.equal(clone.albedo, gbuffer.albedo)
    detached = gbuffer.detach()
    assert not detached.albedo.requires_grad

    # Validation errors on invalid shapes
    with pytest.raises(ValueError):
        GBufferOutput(
            albedo=torch.rand(b, 4, h, w),  # Should be 3 channels
            normal=torch.randn(b, 3, h, w),
            roughness=torch.rand(b, 1, h, w),
            metallic=torch.rand(b, 1, h, w),
            depth=torch.rand(b, 1, h, w),
        )


def test_deferred_cook_torrance_shader():
    """Verify PBR Cook-Torrance shader execution, physical bounds, and gradient flow."""
    shader = DeferredCookTorranceShader()

    b, h, w = 2, 16, 16
    albedo = torch.full((b, 3, h, w), 0.8, requires_grad=True)
    normal = torch.zeros((b, 3, h, w), requires_grad=True)
    normal.data[:, 2, :, :] = 1.0  # Facing +Z
    roughness = torch.full((b, 1, h, w), 0.3, requires_grad=True)
    metallic = torch.full((b, 1, h, w), 0.0, requires_grad=True)
    depth = torch.full((b, 1, h, w), 2.0)

    gbuffer = GBufferOutput(
        albedo=albedo,
        normal=normal,
        roughness=roughness,
        metallic=metallic,
        depth=depth,
    )

    rendered = shader(gbuffer)

    # Shape and values
    assert rendered.shape == (b, 3, h, w)
    assert not torch.isnan(rendered).any()
    assert not torch.isinf(rendered).any()
    assert (rendered >= 0.0).all()

    # Test metallic behavior: metallic = 1.0 reflects colored specular highlights
    gbuffer_metal = GBufferOutput(
        albedo=torch.tensor([[[[1.0]], [[0.5]], [[0.0]]]]).expand(1, 3, 4, 4),  # Gold/orange
        normal=torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]]).expand(1, 3, 4, 4),
        roughness=torch.tensor([[[[0.05]]]]).expand(1, 1, 4, 4),
        metallic=torch.tensor([[[[1.0]]]]).expand(1, 1, 4, 4),
        depth=torch.tensor([[[[1.0]]]]).expand(1, 1, 4, 4),
    )
    rendered_metal = shader(gbuffer_metal)
    # Blue channel should be lowest due to orange albedo base reflectivity
    assert rendered_metal[0, 0].mean() > rendered_metal[0, 2].mean()

    # Backpropagation gradient flow
    loss = rendered.sum()
    loss.backward()
    assert albedo.grad is not None and not torch.isnan(albedo.grad).any()
    assert roughness.grad is not None and not torch.isnan(roughness.grad).any()
    assert metallic.grad is not None and not torch.isnan(metallic.grad).any()


def test_specular_residual_mlp():
    """Verify specular residual MLP integration."""
    shader = DeferredCookTorranceShader(enable_specular_residuals=True)
    assert shader.residual_mlp is not None

    b, h, w = 1, 8, 8
    gbuffer = GBufferOutput(
        albedo=torch.rand(b, 3, h, w),
        normal=F.normalize(torch.randn(b, 3, h, w), dim=1),
        roughness=torch.full((b, 1, h, w), 0.2),
        metallic=torch.full((b, 1, h, w), 0.5),
        depth=torch.ones((b, 1, h, w)),
    )

    out = shader(gbuffer)
    assert out.shape == (b, 3, h, w)
    assert not torch.isnan(out).any()


def test_losses_l1_ssim_normal():
    """Verify individual and combined reconstruction losses."""
    b, h, w = 1, 32, 32
    img1 = torch.rand(b, 3, h, w, requires_grad=True)
    img2 = img1.clone().detach()

    # Identical images -> L1 = 0, SSIM loss = 0
    assert torch.isclose(l1_loss(img1, img2), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(ssim_loss(img1, img2), torch.tensor(0.0), atol=1e-5)

    # Different images
    img3 = torch.clamp(img1 + 0.3, 0.0, 1.0)
    assert l1_loss(img1, img3) > 0.1
    assert ssim_loss(img1, img3) > 0.05

    # Normal consistency loss
    norm1 = torch.zeros(b, 3, h, w)
    norm1[:, 2, :, :] = 1.0  # pointing +Z
    norm2 = norm1.clone()
    assert torch.isclose(normal_loss(norm1, norm2), torch.tensor(0.0), atol=1e-6)

    # Opposite normals -> 1 - (-1) = 2.0
    norm_opposite = -norm1
    assert torch.isclose(normal_loss(norm1, norm_opposite), torch.tensor(2.0), atol=1e-5)

    # Orthogonal normals -> 1 - 0 = 1.0
    norm_ortho = torch.zeros_like(norm1)
    norm_ortho[:, 0, :, :] = 1.0
    assert torch.isclose(normal_loss(norm1, norm_ortho), torch.tensor(1.0), atol=1e-5)

    # Composite ReconstructionLoss
    criterion = ReconstructionLoss(lambda_ssim=0.2, gamma_normal=0.05)
    loss, metrics = criterion(
        pred_rgb=img1,
        gt_rgb=img3,
        pred_normal=norm1,
        gt_normal=norm2,
    )
    assert "loss_total" in metrics
    assert "loss_l1" in metrics
    assert "loss_ssim" in metrics
    assert "loss_normal" in metrics
    assert loss > 0.0

    # Backprop through composite loss
    loss.backward()
    assert img1.grad is not None
    assert not torch.isnan(img1.grad).any()


def test_pytorch_fallback_rasterizer_and_end_to_end_loop():
    """Verify fallback rasterizer generates valid GBuffer and can optimize parameters."""
    # Place a single planar surfel right in front of camera
    surfels = MockSurfelCloud(
        positions=np.array([[0.0, 0.0, -1.5]], dtype=np.float32),  # In front of camera at z = -1.5m
        normals=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),    # Facing camera
        tangent_u=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        tangent_v=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        scales_2d=np.array([[0.5, 0.5]], dtype=np.float32),
        colors_rgb=np.array([[0.8, 0.2, 0.2]], dtype=np.float32),
        sh_degree_0=np.array([[0.2, 0.05, 0.05]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
    )

    model = Material2DGSModel.from_surfel_cloud(surfels)
    rasterizer = PyTorchFallbackRasterizer()
    shader = DeferredCookTorranceShader()
    loss_fn = ReconstructionLoss(lambda_ssim=0.2, gamma_normal=0.05)

    # Identity camera extrinsics (camera at origin)
    w2c = torch.eye(4, dtype=torch.float32)
    # Simple pinhole intrinsics: 32x32 image
    intrinsics = [32.0, 32.0, 16.0, 16.0]

    # Forward pass: Rasterize
    gbuffer = rasterizer(model, extrinsics=w2c, intrinsics=intrinsics, image_size=(32, 32))
    assert gbuffer.albedo.shape == (1, 3, 32, 32)
    assert gbuffer.normal.shape == (1, 3, 32, 32)
    assert not torch.isnan(gbuffer.albedo).any()

    # Center pixel should have high red albedo
    center_albedo = gbuffer.albedo[0, :, 16, 16]
    assert center_albedo[0] > 0.3  # Red channel dominant

    # Forward pass: Shader
    rendered = shader(gbuffer)
    assert rendered.shape == (1, 3, 32, 32)

    # Compute loss against target
    target_rgb = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    target_normal = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    target_normal[:, 2, :, :] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    initial_loss, _ = loss_fn(rendered, target_rgb, gbuffer.normal, target_normal)

    # Optimize for 3 steps
    for _ in range(3):
        optimizer.zero_grad()
        gb = rasterizer(model, extrinsics=w2c, intrinsics=intrinsics, image_size=(32, 32))
        img = shader(gb)
        loss, _ = loss_fn(img, target_rgb, gb.normal, target_normal)
        loss.backward()
        optimizer.step()

    # Final loss should be lower than initial loss
    with torch.no_grad():
        gb_final = rasterizer(model, extrinsics=w2c, intrinsics=intrinsics, image_size=(32, 32))
        img_final = shader(gb_final)
        final_loss, _ = loss_fn(img_final, target_rgb, gb_final.normal, target_normal)

    assert final_loss < initial_loss, f"Optimization failed: initial {initial_loss} -> final {final_loss}"


def test_fp16_preservation_and_zero_division_guards():
    """Verify FP16 compatibility and zero-division protection against degenerate cases."""
    shader = DeferredCookTorranceShader()

    b, h, w = 1, 8, 8
    # Degenerate inputs: normal length = 0, roughness = 0, metallic = 0, view_dir collinear
    gbuffer_zero = GBufferOutput(
        albedo=torch.zeros((b, 3, h, w), dtype=torch.float32),
        normal=torch.zeros((b, 3, h, w), dtype=torch.float32),  # Norm = 0 guard check
        roughness=torch.zeros((b, 1, h, w), dtype=torch.float32),
        metallic=torch.zeros((b, 1, h, w), dtype=torch.float32),
        depth=torch.zeros((b, 1, h, w), dtype=torch.float32),
    )

    out = shader(gbuffer_zero)
    assert not torch.isnan(out).any(), "Shader output has NaNs on degenerate input"
    assert not torch.isinf(out).any(), "Shader output has Infs on degenerate input"

    # FP16 evaluation on GPU or CPU
    gbuffer_fp16 = gbuffer_zero.to(dtype=torch.float16)
    out_fp16 = shader(gbuffer_fp16)
    assert out_fp16.dtype == torch.float16
    assert not torch.isnan(out_fp16).any()
