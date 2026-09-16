"""GlomeHomeTour Backend: 2DGS Material & Density Optimization Model.

Implements Material2DGSModel representing 2D planar Gaussian surfel primitives
with physical PBR attributes (Albedo, Roughness, Metallic, Normals, Opacity).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrices to normalized (w, x, y, z) quaternions.

    Args:
        matrix: (..., 3, 3) rotation matrices where columns are (u, v, n) basis vectors.

    Returns:
        (..., 4) unit quaternions in (w, x, y, z) convention.
    """
    orig_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    n_matrices = m.shape[0]

    # Diagonal and trace
    r00, r01, r02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    r10, r11, r12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    r20, r21, r22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]

    # Shepperd's algorithm: 4 candidates for 4*q_i^2
    # Branch 0: trace = r00 + r11 + r22
    d0 = 1.0 + r00 + r11 + r22
    # Branch 1:
    d1 = 1.0 + r00 - r11 - r22
    # Branch 2:
    d2 = 1.0 - r00 + r11 - r22
    # Branch 3:
    d3 = 1.0 - r00 - r11 + r22

    candidates = torch.stack([d0, d1, d2, d3], dim=-1)  # (N, 4)
    max_idx = torch.argmax(candidates, dim=-1)  # (N,)

    quats = torch.zeros((n_matrices, 4), dtype=matrix.dtype, device=matrix.device)

    # Branch 0: w is largest
    b0 = max_idx == 0
    if b0.any():
        s = 2.0 * torch.sqrt(torch.clamp(d0[b0], min=1e-6))
        quats[b0, 0] = 0.25 * s
        quats[b0, 1] = (r21[b0] - r12[b0]) / s
        quats[b0, 2] = (r02[b0] - r20[b0]) / s
        quats[b0, 3] = (r10[b0] - r01[b0]) / s

    # Branch 1: x is largest
    b1 = max_idx == 1
    if b1.any():
        s = 2.0 * torch.sqrt(torch.clamp(d1[b1], min=1e-6))
        quats[b1, 0] = (r21[b1] - r12[b1]) / s
        quats[b1, 1] = 0.25 * s
        quats[b1, 2] = (r01[b1] + r10[b1]) / s
        quats[b1, 3] = (r02[b1] + r20[b1]) / s

    # Branch 2: y is largest
    b2 = max_idx == 2
    if b2.any():
        s = 2.0 * torch.sqrt(torch.clamp(d2[b2], min=1e-6))
        quats[b2, 0] = (r02[b2] - r20[b2]) / s
        quats[b2, 1] = (r01[b2] + r10[b2]) / s
        quats[b2, 2] = 0.25 * s
        quats[b2, 3] = (r12[b2] + r21[b2]) / s

    # Branch 3: z is largest
    b3 = max_idx == 3
    if b3.any():
        s = 2.0 * torch.sqrt(torch.clamp(d3[b3], min=1e-6))
        quats[b3, 0] = (r10[b3] - r01[b3]) / s
        quats[b3, 1] = (r02[b3] + r20[b3]) / s
        quats[b3, 2] = (r12[b3] + r21[b3]) / s
        quats[b3, 3] = 0.25 * s

    # Enforce positive w for canonical hemisphere
    sign = torch.where(quats[:, 0:1] < 0.0, -1.0, 1.0)
    quats = quats * sign

    # Guard normalization
    norm = torch.linalg.norm(quats, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=1e-6)
    quats = quats / norm

    return quats.reshape(*orig_shape, 4)


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) unit quaternions to 3x3 rotation matrices.

    Args:
        quaternions: (..., 4) unit quaternions (w, x, y, z).

    Returns:
        (..., 3, 3) orthogonal rotation matrices R = [u, v, n].
    """
    q = F.normalize(quaternions, dim=-1, eps=1e-6)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)

    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)

    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    # Stack along rows then columns
    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def quaternion_to_tangent_frame(quaternions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract orthonormal tangent basis (u, v, n) directly from (w, x, y, z) quaternions.

    Args:
        quaternions: (..., 4) unit quaternions.

    Returns:
        tangent_u: (..., 3) unit tangent vector (column 0).
        tangent_v: (..., 3) unit tangent vector (column 1).
        normals: (..., 3) unit normal vector (column 2).
    """
    q = F.normalize(quaternions, dim=-1, eps=1e-6)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    u_x = 1.0 - 2.0 * (y * y + z * z)
    u_y = 2.0 * (x * y + w * z)
    u_z = 2.0 * (x * z - w * y)
    tangent_u = F.normalize(torch.stack([u_x, u_y, u_z], dim=-1), dim=-1, eps=1e-6)

    v_x = 2.0 * (x * y - w * z)
    v_y = 1.0 - 2.0 * (x * x + z * z)
    v_z = 2.0 * (y * z + w * x)
    tangent_v = F.normalize(torch.stack([v_x, v_y, v_z], dim=-1), dim=-1, eps=1e-6)

    n_x = 2.0 * (x * z + w * y)
    n_y = 2.0 * (y * z - w * x)
    n_z = 1.0 - 2.0 * (x * x + y * y)
    normals = F.normalize(torch.stack([n_x, n_y, n_z], dim=-1), dim=-1, eps=1e-6)

    return tangent_u, tangent_v, normals


class Material2DGSModel(nn.Module):
    """2D Gaussian Splatting scene model with physically based deferred materials.

    Stores 2D planar surfel primitives parameterized by:
    - _xyz: (N, 3) 3D centers in world space (meters).
    - _rotation: (N, 4) unit quaternions (w, x, y, z) orienting the surfel frame (u, v, n).
    - _scaling: (N, 2) log-space planar radii (ln sigma_u, ln sigma_v).
    - _opacity: (N, 1) logit-space primitive opacities.
    - _albedo: (N, 3) logit-space view-independent SH-DC radiance in [0, 1].
      Named "albedo" for historical reasons and checkpoint compatibility; with
      the direct-radiance rasterizer it IS the rendered colour, not a BRDF
      parameter. See the `radiance` alias.
    - _roughness: (N, 1) logit-space microfacet roughness values in [0.04, 1.0].
    - _metallic: (N, 1) logit-space metallic factors in [0, 1].
    """

    def __init__(
        self,
        xyz: Optional[torch.Tensor] = None,
        rotation: Optional[torch.Tensor] = None,
        scaling: Optional[torch.Tensor] = None,
        opacity: Optional[torch.Tensor] = None,
        albedo: Optional[torch.Tensor] = None,
        roughness: Optional[torch.Tensor] = None,
        metallic: Optional[torch.Tensor] = None,
        features_rest: Optional[torch.Tensor] = None,
        sh_degree: int = 1,
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        n_rest = 3 * ((sh_degree + 1) ** 2 - 1) if sh_degree > 0 else 0

        # Tracks which primitives survive from the trusted initial surfel
        # cloud vs. were cloned/split during training. Used to exclude
        # depth-prior-anchored geometry (walls/floor/ceiling) from opacity
        # sparsity regularization, which should only pressure grown splats.
        self.register_buffer(
            "is_original",
            torch.ones(
                (xyz.shape[0] if xyz is not None else 0,),
                dtype=torch.bool,
                device=xyz.device if xyz is not None else "cpu",
            ),
        )

        if xyz is None:
            # Initialize empty parameter tensors
            self._xyz = nn.Parameter(torch.empty((0, 3), dtype=torch.float32))
            self._rotation = nn.Parameter(torch.empty((0, 4), dtype=torch.float32))
            self._scaling = nn.Parameter(torch.empty((0, 2), dtype=torch.float32))
            self._opacity = nn.Parameter(torch.empty((0, 1), dtype=torch.float32))
            self._albedo = nn.Parameter(torch.empty((0, 3), dtype=torch.float32))
            self._roughness = nn.Parameter(torch.empty((0, 1), dtype=torch.float32))
            self._metallic = nn.Parameter(torch.empty((0, 1), dtype=torch.float32))
            self._features_rest = nn.Parameter(torch.empty((0, n_rest), dtype=torch.float32))
        else:
            self._xyz = nn.Parameter(xyz.float())
            self._rotation = nn.Parameter(rotation.float() if rotation is not None else torch.zeros((xyz.shape[0], 4)))
            self._scaling = nn.Parameter(scaling.float() if scaling is not None else torch.zeros((xyz.shape[0], 2)))
            self._opacity = nn.Parameter(opacity.float() if opacity is not None else torch.zeros((xyz.shape[0], 1)))
            self._albedo = nn.Parameter(albedo.float() if albedo is not None else torch.zeros((xyz.shape[0], 3)))
            self._roughness = nn.Parameter(roughness.float() if roughness is not None else torch.zeros((xyz.shape[0], 1)))
            self._metallic = nn.Parameter(metallic.float() if metallic is not None else torch.zeros((xyz.shape[0], 1)))
            if features_rest is not None:
                self._features_rest = nn.Parameter(features_rest.float())
            elif n_rest > 0:
                self._features_rest = nn.Parameter(torch.zeros((xyz.shape[0], n_rest), dtype=torch.float32, device=xyz.device))
            else:
                self._features_rest = nn.Parameter(torch.empty((xyz.shape[0], 0), dtype=torch.float32, device=xyz.device))

    # --- Activation Property Accessors ---

    @property
    def xyz(self) -> torch.Tensor:
        """Surfel centers (N, 3) in world coordinates."""
        return self._xyz

    @property
    def opacity(self) -> torch.Tensor:
        """Evaluated opacities in [0, 1], shape (N, 1)."""
        return torch.sigmoid(self._opacity)

    @property
    def scaling(self) -> torch.Tensor:
        """Evaluated 2D planar radii (sigma_u, sigma_v) in meters, shape (N, 2)."""
        return torch.clamp(torch.exp(self._scaling), min=1e-6)

    @property
    def rotation(self) -> torch.Tensor:
        """Unit quaternions (w, x, y, z), shape (N, 4)."""
        return F.normalize(self._rotation, dim=-1, eps=1e-6)

    @property
    def albedo(self) -> torch.Tensor:
        """Evaluated diffuse base colors in [0, 1], shape (N, 3)."""
        return torch.sigmoid(self._albedo)

    @property
    def features_rest(self) -> torch.Tensor:
        """Higher-order Spherical Harmonics coefficients, shape (N, n_rest)."""
        return self._features_rest

    @property
    def radiance(self) -> torch.Tensor:
        """Alias of `albedo`: view-independent SH-DC radiance in [0, 1], (N, 3)."""
        return self.albedo

    def get_radiance(self, view_dirs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute view-dependent RGB radiance in [0, 1].

        Args:
            view_dirs: (N, 3) normalized unit vectors pointing from surfels toward camera origin.

        Returns:
            (N, 3) RGB radiance in [0, 1].
        """
        base_color = torch.sigmoid(self._albedo)
        if self._features_rest.numel() == 0 or view_dirs is None or self.sh_degree == 0:
            return base_color

        # Spherical harmonics degree 1 basis functions
        # d = (x, y, z)
        x = view_dirs[:, 0:1]
        y = view_dirs[:, 1:2]
        z = view_dirs[:, 2:3]

        c1 = 0.4886025119029199
        y0 = -c1 * y
        y1 = c1 * z
        y2 = -c1 * x

        sh_rest = self._features_rest.view(-1, 3, 3)
        sh_contrib = (
            y0 * sh_rest[:, :, 0] +
            y1 * sh_rest[:, :, 1] +
            y2 * sh_rest[:, :, 2]
        )
        return torch.clamp(base_color + sh_contrib, 0.0, 1.0)

    @property
    def roughness(self) -> torch.Tensor:
        """Evaluated microfacet roughness in [0.04, 1.0], shape (N, 1)."""
        return torch.clamp(torch.sigmoid(self._roughness), min=0.04, max=1.0)

    @property
    def metallic(self) -> torch.Tensor:
        """Evaluated metallic factors in [0, 1], shape (N, 1)."""
        return torch.sigmoid(self._metallic)

    @property
    def normals(self) -> torch.Tensor:
        """Surface unit normal vectors (column 2 of frame), shape (N, 3)."""
        _, _, n = quaternion_to_tangent_frame(self._rotation)
        return n

    @property
    def tangent_u(self) -> torch.Tensor:
        """Unit tangent vectors along u-axis (column 0 of frame), shape (N, 3)."""
        u, _, _ = quaternion_to_tangent_frame(self._rotation)
        return u

    @property
    def tangent_v(self) -> torch.Tensor:
        """Unit tangent vectors along v-axis (column 1 of frame), shape (N, 3)."""
        _, v, _ = quaternion_to_tangent_frame(self._rotation)
        return v

    def get_rotation_matrices(self) -> torch.Tensor:
        """Return (N, 3, 3) orthonormal basis matrices R = [u, v, n]."""
        return quaternion_to_rotation_matrix(self._rotation)

    @property
    def num_gaussians(self) -> int:
        """Total number of Gaussian surfel primitives in the model."""
        return self._xyz.shape[0]

    def __len__(self) -> int:
        return self.num_gaussians

    @classmethod
    def from_surfel_cloud(
        cls,
        surfel_cloud: Any,
        default_roughness: float = 0.5,
        default_metallic: float = 0.05,
        init_opacity: Optional[float] = 0.90,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Material2DGSModel":
        """Construct a Material2DGSModel initialized from a SurfelCloud instance.

        Args:
            surfel_cloud: SurfelCloud or duck-typed object containing:
                - positions: (N, 3)
                - normals: (N, 3)
                - tangent_u: (N, 3) [optional]
                - tangent_v: (N, 3) [optional]
                - scales_2d: (N, 2)
                - colors_rgb: (N, 3)
                - opacities: (N,) or (N, 1)
            default_roughness: Initial scalar roughness in [0.04, 1.0].
            default_metallic: Initial scalar metallic factor in [0, 1].
            init_opacity: Override every surfel opacity with this value. A depth
                prior gives real surfaces, so start them opaque and let pruning
                remove what does not belong; None keeps the cloud's own values.
            device: Target torch device (CPU, CUDA, ROCm).
            dtype: Floating point precision (default torch.float32).

        Returns:
            Initialized Material2DGSModel.
        """
        def _to_tensor(val: Any) -> torch.Tensor:
            if isinstance(val, torch.Tensor):
                return val.to(device=device, dtype=dtype)
            arr = np.asarray(val)
            return torch.from_numpy(arr).to(device=device, dtype=dtype)

        pos = _to_tensor(surfel_cloud.positions)
        n_points = pos.shape[0]

        # Normals and tangent vectors
        normals = _to_tensor(surfel_cloud.normals)
        normals = F.normalize(normals, dim=-1, eps=1e-6)

        if hasattr(surfel_cloud, "tangent_u") and surfel_cloud.tangent_u is not None and hasattr(surfel_cloud, "tangent_v") and surfel_cloud.tangent_v is not None:
            t_u = _to_tensor(surfel_cloud.tangent_u)
            t_v = _to_tensor(surfel_cloud.tangent_v)
            t_u = F.normalize(t_u, dim=-1, eps=1e-6)
            t_v = F.normalize(t_v, dim=-1, eps=1e-6)
        else:
            # Build orthogonal tangent frame from normals directly
            near_z = torch.abs(normals[:, 2]) > 0.9
            ref_axis = torch.zeros_like(normals)
            ref_axis[near_z, 0] = 1.0
            ref_axis[~near_z, 2] = 1.0

            u = torch.cross(normals, ref_axis, dim=-1)
            t_u = F.normalize(u, dim=-1, eps=1e-6)
            v = torch.cross(normals, t_u, dim=-1)
            t_v = F.normalize(v, dim=-1, eps=1e-6)

        # Orthonormal basis matrix R = [u, v, n] as columns
        rot_matrices = torch.stack([t_u, t_v, normals], dim=-1)  # (N, 3, 3)
        quats = matrix_to_quaternion(rot_matrices)

        # Scales: log space
        scales_2d = _to_tensor(surfel_cloud.scales_2d)
        log_scales = torch.log(torch.clamp(scales_2d, min=1e-6))

        # Opacities: logit space
        opacities = _to_tensor(surfel_cloud.opacities)
        if opacities.ndim == 1:
            opacities = opacities.unsqueeze(-1)
        if init_opacity is not None:
            opacities = torch.full_like(opacities, float(init_opacity))
        clamped_opacities = torch.clamp(opacities, min=1e-4, max=1.0 - 1e-4)
        logit_opacity = torch.logit(clamped_opacities, eps=1e-6)

        # Albedo colors: logit space
        colors = _to_tensor(surfel_cloud.colors_rgb)
        clamped_colors = torch.clamp(colors, min=1e-4, max=1.0 - 1e-4)
        logit_albedo = torch.logit(clamped_colors, eps=1e-6)

        # Roughness: logit space (default 0.5 -> logit 0.0)
        clamped_roughness = float(np.clip(default_roughness, 0.04, 0.999))
        init_roughness_t = torch.full((n_points, 1), clamped_roughness, dtype=dtype, device=device)
        logit_roughness = torch.logit(init_roughness_t, eps=1e-6)

        # Metallic: logit space (default 0.05 -> logit ~ -2.94)
        clamped_metallic = float(np.clip(default_metallic, 1e-4, 1.0 - 1e-4))
        init_metallic_t = torch.full((n_points, 1), clamped_metallic, dtype=dtype, device=device)
        logit_metallic = torch.logit(init_metallic_t, eps=1e-6)

        model = cls(
            xyz=pos,
            rotation=quats,
            scaling=log_scales,
            opacity=logit_opacity,
            albedo=logit_albedo,
            roughness=logit_roughness,
            metallic=logit_metallic,
        )
        return model

    @classmethod
    def from_ply(
        cls,
        ply_path: Union[str, Path],
        max_surfels: Optional[int] = None,
        voxel_downsample_m: Optional[float] = None,
        default_roughness: float = 0.5,
        default_metallic: float = 0.05,
        init_opacity: Optional[float] = 0.90,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "Material2DGSModel":
        """Construct a Material2DGSModel loaded directly from a binary PLY surfel file with optional voxel downsampling."""
        from initialization import SurfelCloud
        surfel_cloud = SurfelCloud.from_ply(
            ply_path=ply_path,
            max_surfels=max_surfels,
            voxel_downsample_m=voxel_downsample_m,
        )
        return cls.from_surfel_cloud(
            surfel_cloud=surfel_cloud,
            default_roughness=default_roughness,
            default_metallic=default_metallic,
            init_opacity=init_opacity,
            device=device,
            dtype=dtype,
        )

    def to_surfel_cloud(self) -> Any:
        """Export current model state back to a SurfelCloud instance."""
        from initialization import SurfelCloud, SH_C0

        pos_np = self.xyz.detach().cpu().numpy().astype(np.float32)
        norm_np = self.normals.detach().cpu().numpy().astype(np.float32)
        tu_np = self.tangent_u.detach().cpu().numpy().astype(np.float32)
        tv_np = self.tangent_v.detach().cpu().numpy().astype(np.float32)
        scales_np = self.scaling.detach().cpu().numpy().astype(np.float32)
        colors_np = self.albedo.detach().cpu().numpy().astype(np.float32)
        opacities_np = self.opacity.squeeze(-1).detach().cpu().numpy().astype(np.float32)
        sh_np = (colors_np * SH_C0).astype(np.float32)

        return SurfelCloud(
            positions=pos_np,
            normals=norm_np,
            tangent_u=tu_np,
            tangent_v=tv_np,
            scales_2d=scales_np,
            colors_rgb=colors_np,
            sh_degree_0=sh_np,
            opacities=opacities_np,
        )
