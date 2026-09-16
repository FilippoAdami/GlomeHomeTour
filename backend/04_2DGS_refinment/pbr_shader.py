"""GlomeHomeTour Backend: Deferred Cook-Torrance PBR Shader.

Implements DeferredCookTorranceShader performing microfacet Cook-Torrance BRDF
shading with directional sun lighting and Split-Sum Image-Based Lighting (IBL).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rasterizer_interface import GBufferOutput


class SpecularResidualMLP(nn.Module):
    """Lightweight 2-layer MLP predicting high-frequency view-dependent specular residuals."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        # Input: view_dir (3) + normal (3) + roughness (1) + metallic (1) = 8
        self.net = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),  # Residuals bounded in [-1, 1], scaled by 0.2
        )

    def forward(
        self,
        view_dirs: torch.Tensor,
        normals: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
    ) -> torch.Tensor:
        """Args:

        view_dirs: (B, 3, H, W)
        normals: (B, 3, H, W)
        roughness: (B, 1, H, W)
        metallic: (B, 1, H, W)
        Returns:
            residual: (B, 3, H, W) in range [-0.2, 0.2]
        """
        b, _, h, w = view_dirs.shape
        features = torch.cat([view_dirs, normals, roughness, metallic], dim=1)  # (B, 8, H, W)
        # Reshape to (B*H*W, 8)
        feat_flat = features.permute(0, 2, 3, 1).reshape(-1, 8)
        res_flat = self.net(feat_flat) * 0.2
        return res_flat.reshape(b, h, w, 3).permute(0, 3, 1, 2)


class DeferredCookTorranceShader(nn.Module):
    """Deferred Cook-Torrance PBR Shader with Split-Sum IBL and directional illumination.

    Evaluates:
    - Microfacet GGX Normal Distribution Function D(H)
    - Schlick Fresnel approximation F(V, H) parameterized by base reflectivity F0
    - Smith-GGX geometric attenuation G(L, V, H)
    - Energy-conserving Lambertian diffuse
    - Epic Games Split-Sum IBL ambient approximation
    """

    def __init__(
        self,
        default_light_dir: Sequence[float] = (0.57735, 0.57735, 0.57735),
        default_light_color: Sequence[float] = (1.0, 1.0, 1.0),
        ambient_sky_color: Sequence[float] = (0.6, 0.65, 0.7),
        ambient_ground_color: Sequence[float] = (0.25, 0.22, 0.2),
        ambient_intensity: float = 0.6,
        direct_intensity: float = 1.0,
        enable_specular_residuals: bool = False,
    ) -> None:
        super().__init__()

        # Register lighting parameters as buffers or learnable modules
        l_dir = torch.tensor(default_light_dir, dtype=torch.float32)
        l_dir = F.normalize(l_dir, dim=-1, eps=1e-6)
        self.register_buffer("light_dir", l_dir.view(1, 3, 1, 1))

        l_col = torch.tensor(default_light_color, dtype=torch.float32) * direct_intensity
        self.register_buffer("light_color", l_col.view(1, 3, 1, 1))

        sky_col = torch.tensor(ambient_sky_color, dtype=torch.float32) * ambient_intensity
        self.register_buffer("ambient_sky", sky_col.view(1, 3, 1, 1))

        gnd_col = torch.tensor(ambient_ground_color, dtype=torch.float32) * ambient_intensity
        self.register_buffer("ambient_ground", gnd_col.view(1, 3, 1, 1))

        self.enable_specular_residuals = enable_specular_residuals
        if enable_specular_residuals:
            self.residual_mlp = SpecularResidualMLP()
        else:
            self.residual_mlp = None

    def forward(
        self,
        gbuffer: GBufferOutput,
        view_dirs: Optional[torch.Tensor] = None,
        light_dir: Optional[torch.Tensor] = None,
        light_color: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate deferred Cook-Torrance shading over rasterized G-Buffer.

        Args:
            gbuffer: GBufferOutput containing albedo, normal, roughness, metallic, depth.
            view_dirs: (B, 3, H, W) unit vectors pointing from surface toward camera.
                       If None, defaults to [0, 0, 1] (+Z in camera frame).
            light_dir: Optional (B, 3, 1, 1) or (3,) directional light direction.
            light_color: Optional (B, 3, 1, 1) or (3,) directional light color/radiance.

        Returns:
            rendered_rgb: (B, 3, H, W) shaded RGB image in [0, inf).
        """
        device = gbuffer.device
        dtype = gbuffer.dtype
        b, _, h, w = gbuffer.albedo.shape

        # Surface attributes
        albedo = torch.clamp(gbuffer.albedo, min=0.0, max=1.0)
        roughness = torch.clamp(gbuffer.roughness, min=0.04, max=1.0)
        metallic = torch.clamp(gbuffer.metallic, min=0.0, max=1.0)

        # Normals: ensure unit length with zero-division guard
        n_len = torch.linalg.norm(gbuffer.normal, dim=1, keepdim=True)
        normals = gbuffer.normal / torch.clamp(n_len, min=1e-6)

        # View directions V
        if view_dirs is None:
            v_dir = torch.zeros((b, 3, h, w), dtype=dtype, device=device)
            v_dir[:, 2, :, :] = 1.0  # Default camera forward direction
            view_dirs = v_dir
        else:
            v_len = torch.linalg.norm(view_dirs, dim=1, keepdim=True)
            view_dirs = view_dirs / torch.clamp(v_len, min=1e-6)

        # Direct Light direction L
        if light_dir is None:
            l_dir = self.light_dir.to(device=device, dtype=dtype)
        else:
            if light_dir.ndim == 1:
                light_dir = light_dir.view(1, 3, 1, 1)
            l_len = torch.linalg.norm(light_dir, dim=1, keepdim=True)
            l_dir = light_dir / torch.clamp(l_len, min=1e-6)

        if light_color is None:
            l_col = self.light_color.to(device=device, dtype=dtype)
        else:
            if light_color.ndim == 1:
                light_color = light_color.view(1, 3, 1, 1)
            l_col = light_color.to(device=device, dtype=dtype)

        # Half-vector H = normalize(L + V)
        h_vec = l_dir + view_dirs
        h_len = torch.linalg.norm(h_vec, dim=1, keepdim=True)
        h_dir = h_vec / torch.clamp(h_len, min=1e-6)

        # Dot products with zero-division / bounds clamping
        n_dot_l = torch.clamp(torch.sum(normals * l_dir, dim=1, keepdim=True), min=0.0, max=1.0)
        n_dot_v = torch.clamp(torch.sum(normals * view_dirs, dim=1, keepdim=True), min=1e-4, max=1.0)
        n_dot_h = torch.clamp(torch.sum(normals * h_dir, dim=1, keepdim=True), min=0.0, max=1.0)
        v_dot_h = torch.clamp(torch.sum(view_dirs * h_dir, dim=1, keepdim=True), min=0.0, max=1.0)

        # Linear roughness alpha = roughness^2 (Disney parameterization)
        alpha = roughness * roughness
        alpha_sq = alpha * alpha

        # 1. Normal Distribution Function D(H) - Trowbridge-Reitz GGX
        # D = alpha^2 / (pi * ((N.H)^2 * (alpha^2 - 1) + 1)^2)
        denom_d = (n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0)
        denom_d = math.pi * denom_d * denom_d
        d_term = alpha_sq / torch.clamp(denom_d, min=1e-6)

        # 2. Fresnel-Schlick F(V, H)
        # F0 = lerp(0.04, albedo, metallic)
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic
        f_term = f0 + (1.0 - f0) * torch.pow(torch.clamp(1.0 - v_dot_h, min=0.0, max=1.0), 5.0)

        # 3. Geometric Attenuation G(L, V, H) - Smith GGX
        # k = (roughness + 1)^2 / 8
        k = (roughness + 1.0) * (roughness + 1.0) / 8.0
        g1_l = n_dot_l / torch.clamp(n_dot_l * (1.0 - k) + k, min=1e-6)
        g1_v = n_dot_v / torch.clamp(n_dot_v * (1.0 - k) + k, min=1e-6)
        g_term = g1_l * g1_v

        # 4. Cook-Torrance Specular BRDF:
        # f_spec = (D * F * G) / (4 * (N.L) * (N.V))
        denom_spec = 4.0 * n_dot_l * n_dot_v
        specular_brdf = (d_term * f_term * g_term) / torch.clamp(denom_spec, min=1e-6)

        # 5. Lambertian Diffuse BRDF (Energy Conserving):
        # kd = (1 - F) * (1 - metallic)
        kd = (1.0 - f_term) * (1.0 - metallic)
        diffuse_brdf = kd * albedo / math.pi

        # Direct illumination
        direct_irr = (diffuse_brdf * math.pi + specular_brdf) * l_col * n_dot_l

        # 6. Split-Sum Image-Based Lighting (IBL) Environment Approximation
        # Reflection vector R = 2 * (N.V) * N - V
        r_dir = 2.0 * n_dot_v * normals - view_dirs
        r_len = torch.linalg.norm(r_dir, dim=1, keepdim=True)
        r_dir = r_dir / torch.clamp(r_len, min=1e-6)

        # Environment hemisphere gradient (sky vs ground)
        sky = self.ambient_sky.to(device=device, dtype=dtype)
        gnd = self.ambient_ground.to(device=device, dtype=dtype)

        # Diffuse irradiance from ambient hemisphere
        hemi_diffuse = 0.5 + 0.5 * normals[:, 1:2, :, :]
        hemi_diffuse = torch.clamp(hemi_diffuse, min=0.0, max=1.0)
        ambient_irr = sky * hemi_diffuse + gnd * (1.0 - hemi_diffuse)
        diffuse_ibl = (1.0 - f0) * (1.0 - metallic) * albedo * ambient_irr

        # Specular IBL (Karis polynomial environment BRDF approximation)
        # scale = (1 - roughness) * (1 - (1 - N.V)^5)
        # bias = roughness * 0.04
        env_scale = (1.0 - roughness) * (1.0 - torch.pow(torch.clamp(1.0 - n_dot_v, min=0.0, max=1.0), 5.0))
        env_bias = roughness * 0.04
        env_brdf = f0 * env_scale + env_bias

        hemi_specular = 0.5 + 0.5 * r_dir[:, 1:2, :, :]
        hemi_specular = torch.clamp(hemi_specular, min=0.0, max=1.0)
        ambient_spec_radiance = sky * hemi_specular + gnd * (1.0 - hemi_specular)
        specular_ibl = ambient_spec_radiance * env_brdf

        # Total combined shaded output
        total_rgb = direct_irr + diffuse_ibl + specular_ibl

        # Optional specular residual module (high frequency glints / reflections)
        if self.enable_specular_residuals and self.residual_mlp is not None:
            residual = self.residual_mlp(view_dirs, normals, roughness, metallic)
            total_rgb = total_rgb + residual

        return torch.clamp(total_rgb, min=0.0)
