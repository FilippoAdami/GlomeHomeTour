"""GlomeHomeTour Backend: 2DGS Rasterizer Interface & PyTorch Fallback Oracle.

Provides the GBufferOutput schema and pure-PyTorch differentiable rasterizer
for testing, oracle validation, and CPU/GPU deferred shading.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from reconstruction.training.model import Material2DGSModel


@dataclass
class GBufferOutput:
    """Rasterized screen-space 4-channel G-Buffer tuple with metric depth.

    Attributes:
        albedo: (B, 3, H, W) float tensor of diffuse base colors in [0, 1].
        normal: (B, 3, H, W) float tensor of surface unit normals in camera or world space.
        roughness: (B, 1, H, W) float tensor of microfacet roughness in [0.04, 1.0].
        metallic: (B, 1, H, W) float tensor of metallic factors in [0, 1].
        depth: (B, 1, H, W) float tensor of metric depth in meters.
        alpha: (B, 1, H, W) accumulated opacity (coverage) in [0, 1], or None.

    Note: with the direct-radiance rasterizer, `albedo` carries view-independent
    SH-DC radiance and is the rendered image; roughness/metallic are filled with
    constants and kept only for the legacy deferred-PBR consumers.
    """
    albedo: torch.Tensor
    normal: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    depth: torch.Tensor
    alpha: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        """Validate tensor shapes, dimensions, and non-null status."""
        for name in ("albedo", "normal", "roughness", "metallic", "depth"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Field '{name}' must be a torch.Tensor, got {type(tensor)}")

        if self.albedo.ndim != 4 or self.albedo.shape[1] != 3:
            raise ValueError(f"albedo must have shape (B, 3, H, W), got {self.albedo.shape}")
        if self.normal.ndim != 4 or self.normal.shape[1] != 3:
            raise ValueError(f"normal must have shape (B, 3, H, W), got {self.normal.shape}")
        if self.roughness.ndim != 4 or self.roughness.shape[1] != 1:
            raise ValueError(f"roughness must have shape (B, 1, H, W), got {self.roughness.shape}")
        if self.metallic.ndim != 4 or self.metallic.shape[1] != 1:
            raise ValueError(f"metallic must have shape (B, 1, H, W), got {self.metallic.shape}")
        if self.depth.ndim != 4 or self.depth.shape[1] != 1:
            raise ValueError(f"depth must have shape (B, 1, H, W), got {self.depth.shape}")

        b, _, h, w = self.albedo.shape
        if self.normal.shape != (b, 3, h, w):
            raise ValueError(f"normal shape mismatch: expected ({b}, 3, {h}, {w}), got {self.normal.shape}")
        if self.roughness.shape != (b, 1, h, w):
            raise ValueError(f"roughness shape mismatch: expected ({b}, 1, {h}, {w}), got {self.roughness.shape}")
        if self.metallic.shape != (b, 1, h, w):
            raise ValueError(f"metallic shape mismatch: expected ({b}, 1, {h}, {w}), got {self.metallic.shape}")
        if self.depth.shape != (b, 1, h, w):
            raise ValueError(f"depth shape mismatch: expected ({b}, 1, {h}, {w}), got {self.depth.shape}")
        if self.alpha is not None and self.alpha.shape != (b, 1, h, w):
            raise ValueError(f"alpha shape mismatch: expected ({b}, 1, {h}, {w}), got {self.alpha.shape}")

    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> "GBufferOutput":
        """Move all buffers to specified device and/or dtype."""
        return GBufferOutput(
            albedo=self.albedo.to(device=device, dtype=dtype),
            normal=self.normal.to(device=device, dtype=dtype),
            roughness=self.roughness.to(device=device, dtype=dtype),
            metallic=self.metallic.to(device=device, dtype=dtype),
            depth=self.depth.to(device=device, dtype=dtype),
            alpha=None if self.alpha is None else self.alpha.to(device=device, dtype=dtype),
        )

    def detach(self) -> "GBufferOutput":
        """Detach all tensors from computation graph."""
        return GBufferOutput(
            albedo=self.albedo.detach(),
            normal=self.normal.detach(),
            roughness=self.roughness.detach(),
            metallic=self.metallic.detach(),
            depth=self.depth.detach(),
            alpha=None if self.alpha is None else self.alpha.detach(),
        )

    def clone(self) -> "GBufferOutput":
        """Clone all tensors."""
        return GBufferOutput(
            albedo=self.albedo.clone(),
            normal=self.normal.clone(),
            roughness=self.roughness.clone(),
            metallic=self.metallic.clone(),
            depth=self.depth.clone(),
            alpha=None if self.alpha is None else self.alpha.clone(),
        )

    @property
    def batch_size(self) -> int:
        return self.albedo.shape[0]

    @property
    def height(self) -> int:
        return self.albedo.shape[2]

    @property
    def width(self) -> int:
        return self.albedo.shape[3]

    @property
    def device(self) -> torch.device:
        return self.albedo.device

    @property
    def dtype(self) -> torch.dtype:
        return self.albedo.dtype


def _camera_sign(convention: str, pts_cam: torch.Tensor) -> float:
    """Return -1.0 for a -Z-forward (OpenGL/ARCore) camera, +1.0 for OpenCV.

    "auto" infers it from the median camera-space z, which is only reliable when
    the scene sits clearly in front of the camera. Inside a closed room it is
    not: about half of this project's ARCore keyframes flip the wrong way, which
    silently renders the opposite wall. Pass the convention explicitly for real
    captures -- `transforms.json` here is ARCore c2w, i.e. "opengl".
    """
    if convention == "opengl":
        return -1.0
    if convention == "opencv":
        return 1.0
    if convention != "auto":
        raise ValueError(f"camera_convention must be 'auto', 'opengl' or 'opencv', got {convention!r}")
    return -1.0 if pts_cam[:, 2].detach().median().item() < 0 else 1.0


def _unpack_intrinsics(intrinsics: Any) -> Tuple[float, float, float, float]:
    """Normalize CameraIntrinsics / (3,3) matrix / (4,) sequence to (fx, fy, cx, cy)."""
    if hasattr(intrinsics, "fl_x"):
        return (float(intrinsics.fl_x), float(intrinsics.fl_y),
                float(intrinsics.cx), float(intrinsics.cy))
    if isinstance(intrinsics, torch.Tensor) and intrinsics.shape == (3, 3):
        return (float(intrinsics[0, 0]), float(intrinsics[1, 1]),
                float(intrinsics[0, 2]), float(intrinsics[1, 2]))
    if isinstance(intrinsics, (list, tuple, torch.Tensor)) and len(intrinsics) == 4:
        return tuple(float(v) for v in intrinsics)  # type: ignore[return-value]
    raise ValueError(f"Unrecognized intrinsics format: {intrinsics}")


def _empty_gbuffer(b: int, h: int, w: int, dtype: torch.dtype, device: torch.device) -> GBufferOutput:
    """Background G-Buffer for an empty model or a fully culled view."""
    return GBufferOutput(
        albedo=torch.zeros((b, 3, h, w), dtype=dtype, device=device),
        normal=torch.zeros((b, 3, h, w), dtype=dtype, device=device),
        roughness=torch.ones((b, 1, h, w), dtype=dtype, device=device),
        metallic=torch.zeros((b, 1, h, w), dtype=dtype, device=device),
        depth=torch.zeros((b, 1, h, w), dtype=dtype, device=device),
        alpha=torch.zeros((b, 1, h, w), dtype=dtype, device=device),
    )


class Base2DGSRasterizer(nn.Module):
    """Abstract base class for 2DGS differentiable rasterizers."""

    def forward(
        self,
        model: Material2DGSModel,
        extrinsics: torch.Tensor,
        intrinsics: Any,
        image_size: Tuple[int, int],
        **kwargs: Any,
    ) -> GBufferOutput:
        raise NotImplementedError


# Attempt to import compiled ROCm/HIP custom kernel
_HIP_RASTERIZER_AVAILABLE = False
try:
    import rasterizer_hip  # type: ignore
    _HIP_RASTERIZER_AVAILABLE = True
except ImportError:
    _HIP_RASTERIZER_AVAILABLE = False


def is_hip_rasterizer_available() -> bool:
    """Return True if hand-tuned ROCm/HIP rasterizer extension is compiled and available."""
    return _HIP_RASTERIZER_AVAILABLE


TILE_SIZE = 16
MAX_SCREEN_RADIUS = 128.0


def bin_tiles(
    proj_uv: torch.Tensor,
    radius: torch.Tensor,
    depth: torch.Tensor,
    image_size: Tuple[int, int],
    tile: int = TILE_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bin splats into screen tiles, ordered by (tile_id, depth).

    Replaces the old "every tile walks every splat" scheme: each splat only
    lands in the tiles its 3-sigma bounding box touches.

    Returns:
        point_list: (T,) int32 splat indices, grouped by tile, depth-ascending.
        tile_ranges: (num_tiles * 2,) int32 [start, end) pairs into point_list.
    """
    h, w = image_size
    device = proj_uv.device
    grid_w = (w + tile - 1) // tile
    grid_h = (h + tile - 1) // tile
    num_tiles = grid_w * grid_h

    px = proj_uv[:, 0].detach()
    py = proj_uv[:, 1].detach()
    r = radius.detach()

    x0 = ((px - r) / tile).floor().clamp(0, grid_w).to(torch.int64)
    x1 = ((px + r) / tile).floor().add(1).clamp(0, grid_w).to(torch.int64)
    y0 = ((py - r) / tile).floor().clamp(0, grid_h).to(torch.int64)
    y1 = ((py + r) / tile).floor().add(1).clamp(0, grid_h).to(torch.int64)

    nx = (x1 - x0).clamp(min=0)
    ny = (y1 - y0).clamp(min=0)
    counts = nx * ny
    total = int(counts.sum().item())

    empty_ranges = torch.zeros(num_tiles * 2, dtype=torch.int32, device=device)
    if total == 0:
        return torch.zeros(0, dtype=torch.int32, device=device), empty_ranges

    gid = torch.repeat_interleave(torch.arange(counts.numel(), device=device), counts)
    starts_per_g = torch.cumsum(counts, 0) - counts
    k = torch.arange(total, device=device) - starts_per_g[gid]
    tile_id = (y0[gid] + torch.div(k, nx[gid], rounding_mode="floor")) * grid_w + (x0[gid] + k % nx[gid])

    # Lexicographic (tile_id, depth): sort by depth, then stable-sort by tile.
    o1 = torch.argsort(depth.detach()[gid])
    tid1 = tile_id[o1]
    o2 = torch.argsort(tid1, stable=True)
    point_list = gid[o1[o2]].to(torch.int32)
    sorted_tid = tid1[o2]

    tiles = torch.arange(num_tiles, device=device, dtype=sorted_tid.dtype)
    starts = torch.searchsorted(sorted_tid, tiles)
    ends = torch.searchsorted(sorted_tid, tiles, right=True)
    tile_ranges = torch.stack([starts, ends], dim=-1).reshape(-1).to(torch.int32)
    return point_list.contiguous(), tile_ranges.contiguous()


def _flip_normals_to_camera(n_cam: torch.Tensor, pts_cam: torch.Tensor) -> torch.Tensor:
    """Orient camera-space normals toward the camera origin.

    Convention-agnostic (works for both +Z- and -Z-forward): pts_cam is the
    vector camera -> point, so a normal facing us has a negative dot with it.
    Surfel tangent frames carry an arbitrary sign; depth-prior normals don't.
    """
    dot = (n_cam * pts_cam).sum(dim=-1, keepdim=True)
    return n_cam * torch.where(dot > 0, -1.0, 1.0)


def _zero_grad_scalar(model: Material2DGSModel) -> torch.Tensor:
    """A 0.0 that is still wired to the model parameters, for empty renders."""
    return model.albedo.sum() * 0.0


def _rotate(vecs: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Return ``vecs @ r.T`` for (N, 3) vectors and a (3, 3) matrix.

    Written column-wise on purpose. On gfx1200 / ROCm 7.1 a float32 (N, 3) @ (3, 3)
    matmul silently leaves every output row from index 524,288 (2**19) onward as
    zeros -- the BLAS kernel launches a grid that only covers the first 2**19 rows.
    Points past the cliff come back at the camera origin, get frustum-culled, and
    the frame renders empty. The column form is exact at any N (and cheaper than a
    BLAS call for k=3). Do not "simplify" this back to matmul.
    """
    return vecs[:, 0:1] * r[:, 0] + vecs[:, 1:2] * r[:, 1] + vecs[:, 2:3] * r[:, 2]


class _HIPRasterizerFunction(torch.autograd.Function):
    """PyTorch autograd bridge to the compiled C++/HIP 2DGS rasterizer extension."""

    @staticmethod
    def forward(
        ctx: Any,
        depths: torch.Tensor,
        normals_cam: torch.Tensor,
        inv_covs: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        proj_uv: torch.Tensor,
        point_list: torch.Tensor,
        tile_ranges: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out_color, out_normal, out_depth, out_alpha, n_contrib = rasterizer_hip.rasterize_forward(
            depths, normals_cam, inv_covs, opacities, colors, proj_uv, point_list, tile_ranges, H, W
        )
        ctx.save_for_backward(
            depths, normals_cam, inv_covs, opacities, colors, proj_uv,
            point_list, tile_ranges, out_alpha, n_contrib,
        )
        ctx.H = H
        ctx.W = W
        return out_color, out_normal, out_depth, out_alpha

    @staticmethod
    def backward(
        ctx: Any,
        dL_dcolor: torch.Tensor,
        dL_dnormal: torch.Tensor,
        dL_ddepth: torch.Tensor,
        dL_dalpha: torch.Tensor,
    ) -> Tuple[Any, ...]:
        (
            depths, normals_cam, inv_covs, opacities, colors, proj_uv,
            point_list, tile_ranges, out_alpha, n_contrib,
        ) = ctx.saved_tensors
        grads = rasterizer_hip.rasterize_backward(
            dL_dcolor.contiguous(),
            dL_dnormal.contiguous(),
            dL_ddepth.contiguous(),
            dL_dalpha.contiguous(),
            out_alpha,
            n_contrib,
            depths, normals_cam, inv_covs, opacities, colors, proj_uv,
            point_list, tile_ranges, ctx.H, ctx.W,
        )
        dL_dproj_uv, dL_dinv_cov, dL_dopacity, dL_dcolor_out, dL_dnormal_out, dL_ddepth_out = grads
        return (
            dL_ddepth_out,   # depths
            dL_dnormal_out,  # normals_cam
            dL_dinv_cov,     # inv_covs
            dL_dopacity,     # opacities
            dL_dcolor_out,   # colors
            dL_dproj_uv,     # proj_uv
            None,            # point_list
            None,            # tile_ranges
            None,            # H
            None,            # W
        )


class HIP2DGSRasterizer(Base2DGSRasterizer):
    """High-performance hand-tuned ROCm/HIP 2DGS rasterizer.

    Executes 16x16 tiled rasterization with 32-wide wavefronts and LDS shared
    memory caching directly on AMD RDNA4 (Radeon RX 9060 XT). Splats carry
    direct SH-DC radiance in `albedo`; there is no deferred BRDF pass.
    """

    def __init__(
        self,
        near_plane: float = 0.05,
        far_plane: float = 50.0,
        low_pass_filter_sigma: float = 0.3,
        camera_convention: str = "auto",
        enable_backface_culling: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.low_pass_filter_sigma = low_pass_filter_sigma
        self.camera_convention = camera_convention
        self.enable_backface_culling = enable_backface_culling

    def forward(
        self,
        model: Material2DGSModel,
        extrinsics: torch.Tensor,
        intrinsics: Any,
        image_size: Tuple[int, int],
        **kwargs: Any,
    ) -> GBufferOutput:
        h, w = image_size
        device = model.xyz.device
        dtype = model.xyz.dtype

        if device.type == "cpu":
            if getattr(self, "_cpu_fallback", None) is None:
                self._cpu_fallback = PyTorchFallbackRasterizer(
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                    low_pass_filter_sigma=self.low_pass_filter_sigma,
                    camera_convention=self.camera_convention,
                )
            return self._cpu_fallback(model, extrinsics, intrinsics, image_size, **kwargs)

        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)
        b = extrinsics.shape[0]

        fx, fy, cx, cy = _unpack_intrinsics(intrinsics)

        num_pts = model.num_gaussians
        if num_pts == 0:
            return _empty_gbuffer(b, h, w, dtype, device)

        pts_world = model.xyz
        scales = model.scaling
        opacities = model.opacity
        radiance = model.albedo
        t_u_world = model.tangent_u
        t_v_world = model.tangent_v
        normals_world = model.normals

        batch_color = []
        batch_normal = []
        batch_depth = []
        batch_alpha = []

        for b_idx in range(b):
            w2c = extrinsics[b_idx].to(device=device, dtype=dtype)
            r_cw = w2c[:3, :3]
            t_cw = w2c[:3, 3]

            c_pos = -torch.matmul(r_cw.T, t_cw)
            view_dirs_world = F.normalize(c_pos.unsqueeze(0) - pts_world, dim=-1, eps=1e-6)
            radiance = model.get_radiance(view_dirs_world)

            pts_cam = _rotate(pts_world, r_cw) + t_cw

            sign_y = _camera_sign(self.camera_convention, pts_cam)
            depth_cam = sign_y * pts_cam[:, 2] if sign_y < 0 else pts_cam[:, 2]
            valid_mask = (depth_cam >= self.near_plane) & (depth_cam <= self.far_plane)
            proj_x = fx * (pts_cam[:, 0] / torch.clamp(depth_cam, min=1e-6)) + cx
            proj_y = sign_y * fy * (pts_cam[:, 1] / torch.clamp(depth_cam, min=1e-6)) + cy

            n_cam = F.normalize(_rotate(normals_world, r_cw), dim=-1, eps=1e-6)
            if self.enable_backface_culling:
                # FasterGS backface culling: view ray is pts_cam (camera -> surfel).
                # Normal pointing away from camera has (n_cam * pts_cam) >= 0. Front-facing has < 0.
                dot_view = (n_cam * pts_cam).sum(dim=-1)
                valid_mask = valid_mask & (dot_view < 0.0)

            n_cam = _flip_normals_to_camera(n_cam, pts_cam)
            tu_cam = _rotate(t_u_world, r_cw)
            tv_cam = _rotate(t_v_world, r_cw)

            in_bounds = (
                (proj_x >= -32) & (proj_x < w + 32) &
                (proj_y >= -32) & (proj_y < h + 32) &
                valid_mask
            )

            if not in_bounds.any():
                # Stay attached to the graph: a detached zero render makes
                # loss.backward() raise "does not require grad" and kills the run
                # over one unlucky view. Zero gradient, but a differentiable zero.
                zero = _zero_grad_scalar(model)
                batch_color.append(torch.zeros((3, h, w), dtype=dtype, device=device) + zero)
                batch_normal.append(torch.zeros((3, h, w), dtype=dtype, device=device) + zero)
                batch_depth.append(torch.zeros((1, h, w), dtype=dtype, device=device) + zero)
                batch_alpha.append(torch.zeros((1, h, w), dtype=dtype, device=device) + zero)
                continue

            v_idx = torch.nonzero(in_bounds).squeeze(-1)

            p_d = depth_cam[v_idx]
            inv_d = 1.0 / torch.clamp(p_d, min=1e-6)
            inv_d2 = inv_d * inv_d

            h_u = scales[v_idx, 0:1] * tu_cam[v_idx]
            h_v = scales[v_idx, 1:2] * tv_cam[v_idx]

            su_x = fx * h_u[:, 0] * inv_d - fx * pts_cam[v_idx, 0] * h_u[:, 2] * inv_d2
            su_y = sign_y * fy * h_u[:, 1] * inv_d - sign_y * fy * pts_cam[v_idx, 1] * h_u[:, 2] * inv_d2
            sv_x = fx * h_v[:, 0] * inv_d - fx * pts_cam[v_idx, 0] * h_v[:, 2] * inv_d2
            sv_y = sign_y * fy * h_v[:, 1] * inv_d - sign_y * fy * pts_cam[v_idx, 1] * h_v[:, 2] * inv_d2

            sigma_filter_sq = self.low_pass_filter_sigma ** 2
            cov_xx = su_x * su_x + sv_x * sv_x + sigma_filter_sq
            cov_yy = su_y * su_y + sv_y * sv_y + sigma_filter_sq
            cov_xy = su_x * su_y + sv_x * sv_y

            det = torch.clamp(cov_xx * cov_yy - cov_xy * cov_xy, min=1e-6)
            inv_covs = torch.stack([cov_yy / det, cov_xx / det, -cov_xy / det], dim=-1)

            proj_uv = torch.stack([proj_x[v_idx], proj_y[v_idx]], dim=-1)

            radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(torch.maximum(cov_xx, cov_yy), min=1e-4)))
            radius = torch.clamp(radius, min=1.0, max=MAX_SCREEN_RADIUS)

            point_list, tile_ranges = bin_tiles(proj_uv, radius, p_d, (h, w))

            out_col, out_norm, out_dep, out_alpha = _HIPRasterizerFunction.apply(
                p_d.contiguous(),
                n_cam[v_idx].contiguous(),
                inv_covs.contiguous(),
                opacities[v_idx].contiguous(),
                radiance[v_idx].contiguous(),
                proj_uv.contiguous(),
                point_list,
                tile_ranges,
                h,
                w,
            )

            batch_color.append(out_col)
            batch_normal.append(out_norm / torch.clamp(torch.linalg.norm(out_norm, dim=0, keepdim=True), min=1e-6))
            batch_depth.append(out_dep / torch.clamp(out_alpha, min=1e-3))
            batch_alpha.append(out_alpha)

        color = torch.stack(batch_color, dim=0)
        return GBufferOutput(
            albedo=color,
            normal=torch.stack(batch_normal, dim=0),
            roughness=torch.ones((b, 1, h, w), dtype=dtype, device=device),
            metallic=torch.zeros((b, 1, h, w), dtype=dtype, device=device),
            depth=torch.stack(batch_depth, dim=0),
            alpha=torch.stack(batch_alpha, dim=0),
        )


def get_default_rasterizer(**kwargs: Any) -> Base2DGSRasterizer:
    """Return HIP2DGSRasterizer if compiled HIP kernel is available; else PyTorchFallbackRasterizer."""
    if is_hip_rasterizer_available():
        return HIP2DGSRasterizer(**kwargs)
    return PyTorchFallbackRasterizer(**kwargs)



from torch.utils.checkpoint import checkpoint


def _render_screen_block(
    bx0: int,
    bx1: int,
    by0: int,
    by1: int,
    p_x: torch.Tensor,
    p_y: torch.Tensor,
    p_d: torch.Tensor,
    p_op: torch.Tensor,
    p_alb: torch.Tensor,
    p_norm: torch.Tensor,
    p_rou: torch.Tensor,
    p_met: torch.Tensor,
    inv_cov_xx: torch.Tensor,
    inv_cov_yy: torch.Tensor,
    inv_cov_xy: torch.Tensor,
    tile_size: int = 32,
    max_surf_chunk: int = 256,
) -> torch.Tensor:
    """Render a spatial 2D block of pixels with front-to-back alpha compositing.

    Evaluated with activation checkpointing to ensure strictly bounded VRAM
    footprint (<1 GB) across all viewport resolutions.
    """
    bh, bw = by1 - by0, bx1 - bx0
    dtype = p_x.dtype
    dev = p_x.device

    out_albedo = torch.zeros((3, bh, bw), dtype=dtype, device=dev)
    out_normal = torch.zeros((3, bh, bw), dtype=dtype, device=dev)
    out_roughness = torch.ones((1, bh, bw), dtype=dtype, device=dev)
    out_metallic = torch.zeros((1, bh, bw), dtype=dtype, device=dev)
    out_depth = torch.zeros((1, bh, bw), dtype=dtype, device=dev)
    out_alpha = torch.zeros((1, bh, bw), dtype=dtype, device=dev)

    for y0 in range(by0, by1, tile_size):
        y1 = min(y0 + tile_size, by1)
        for x0 in range(bx0, bx1, tile_size):
            x1 = min(x0 + tile_size, bx1)
            th, tw = y1 - y0, x1 - x0
            ly0, ly1 = y0 - by0, y1 - by0
            lx0, lx1 = x0 - bx0, x1 - bx0

            ty, tx = torch.meshgrid(
                torch.arange(y0, y1, dtype=dtype, device=dev),
                torch.arange(x0, x1, dtype=dtype, device=dev),
                indexing="ij",
            )
            px = tx.reshape(-1, 1)
            py = ty.reshape(-1, 1)
            nP = px.shape[0]

            t_alb = torch.zeros((nP, 3), dtype=dtype, device=dev)
            t_norm = torch.zeros((nP, 3), dtype=dtype, device=dev)
            t_rou = torch.zeros((nP, 1), dtype=dtype, device=dev)
            t_met = torch.zeros((nP, 1), dtype=dtype, device=dev)
            t_dep = torch.zeros((nP, 1), dtype=dtype, device=dev)
            t_trans = torch.ones((nP, 1), dtype=dtype, device=dev)

            n_surf = p_x.shape[0]
            for s_start in range(0, n_surf, max_surf_chunk):
                s_end = min(s_start + max_surf_chunk, n_surf)
                cur_px = p_x[s_start:s_end].unsqueeze(0)
                cur_py = p_y[s_start:s_end].unsqueeze(0)
                cur_ixx = inv_cov_xx[s_start:s_end].unsqueeze(0)
                cur_iyy = inv_cov_yy[s_start:s_end].unsqueeze(0)
                cur_ixy = inv_cov_xy[s_start:s_end].unsqueeze(0)
                cur_op = p_op[s_start:s_end].squeeze(-1).unsqueeze(0)

                dx = px - cur_px
                dy = py - cur_py
                d2 = dx * dx * cur_ixx + 2.0 * dx * dy * cur_ixy + dy * dy * cur_iyy

                alpha = cur_op * torch.exp(-0.5 * torch.clamp(d2, min=0.0))
                alpha = torch.where(d2 <= 9.0, alpha, torch.zeros_like(alpha))
                alpha = torch.clamp(alpha, min=0.0, max=0.99)

                one_minus_a = torch.clamp(1.0 - alpha, min=1e-6)
                prefix_ones = torch.ones((nP, 1), dtype=dtype, device=dev)
                cum_t = torch.cumprod(torch.cat([prefix_ones, one_minus_a[:, :-1]], dim=1), dim=1)
                weights = alpha * (t_trans * cum_t)

                t_alb = t_alb + torch.matmul(weights, p_alb[s_start:s_end])
                t_norm = t_norm + torch.matmul(weights, p_norm[s_start:s_end])
                t_rou = t_rou + torch.matmul(weights, p_rou[s_start:s_end])
                t_met = t_met + torch.matmul(weights, p_met[s_start:s_end])
                t_dep = t_dep + torch.matmul(weights, p_d[s_start:s_end].unsqueeze(-1))

                t_trans = t_trans * torch.prod(one_minus_a, dim=1, keepdim=True)
                if (t_trans < 1e-4).all():
                    break

            out_albedo[:, ly0:ly1, lx0:lx1] = t_alb.reshape(th, tw, 3).permute(2, 0, 1)
            out_normal[:, ly0:ly1, lx0:lx1] = t_norm.reshape(th, tw, 3).permute(2, 0, 1)
            out_roughness[:, ly0:ly1, lx0:lx1] = (t_rou + t_trans * 1.0).reshape(th, tw, 1).permute(2, 0, 1)
            out_metallic[:, ly0:ly1, lx0:lx1] = t_met.reshape(th, tw, 1).permute(2, 0, 1)
            out_depth[:, ly0:ly1, lx0:lx1] = t_dep.reshape(th, tw, 1).permute(2, 0, 1)
            out_alpha[:, ly0:ly1, lx0:lx1] = (1.0 - t_trans).reshape(th, tw, 1).permute(2, 0, 1)

    return torch.cat([out_albedo, out_normal, out_roughness, out_metallic, out_depth, out_alpha], dim=0)


class PyTorchFallbackRasterizer(Base2DGSRasterizer):
    """Reference pure-PyTorch 2DGS rasterizer oracle.

    Executes analytical 2D Gaussian projection and differentiable front-to-back
    alpha compositing directly in PyTorch. Used for bring-up, unit tests, and
    cross-validation against the custom ROCm/HIP kernel.
    """

    def __init__(
        self,
        near_plane: float = 0.05,
        far_plane: float = 50.0,
        low_pass_filter_sigma: float = 0.3,
        max_tile_size: int = 64,
        camera_convention: str = "auto",
        enable_backface_culling: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.low_pass_filter_sigma = low_pass_filter_sigma
        self.max_tile_size = max_tile_size
        self.camera_convention = camera_convention
        self.enable_backface_culling = enable_backface_culling

    def forward(
        self,
        model: Material2DGSModel,
        extrinsics: torch.Tensor,
        intrinsics: Any,
        image_size: Tuple[int, int],
        **kwargs: Any,
    ) -> GBufferOutput:
        """Render G-Buffer from Material2DGSModel.

        Args:
            model: Trained or initialized Material2DGSModel.
            extrinsics: (4, 4) or (B, 4, 4) world-to-camera matrix.
            intrinsics: CameraIntrinsics or (4,) [fx, fy, cx, cy] or (3, 3) matrix.
            image_size: (H, W) viewport dimensions.

        Returns:
            GBufferOutput containing rendered albedo, normal, roughness, metallic, and depth.
        """
        h, w = image_size
        device = model.xyz.device
        dtype = model.xyz.dtype

        # Normalize extrinsics shape: (B, 4, 4)
        if extrinsics.ndim == 2:
            extrinsics = extrinsics.unsqueeze(0)
        b = extrinsics.shape[0]

        fx, fy, cx, cy = _unpack_intrinsics(intrinsics)

        if model.num_gaussians == 0:
            return _empty_gbuffer(b, h, w, dtype, device)

        # Retrieve evaluated model properties
        pts_world = model.xyz          # (N, 3)
        scales = model.scaling         # (N, 2)
        opacities = model.opacity      # (N, 1)
        albedo = model.albedo          # (N, 3)
        roughness = model.roughness    # (N, 1)
        metallic = model.metallic      # (N, 1)
        t_u_world = model.tangent_u    # (N, 3)
        t_v_world = model.tangent_v    # (N, 3)
        normals_world = model.normals  # (N, 3)

        batch_albedo = []
        batch_normal = []
        batch_roughness = []
        batch_metallic = []
        batch_depth = []
        batch_alpha = []

        for b_idx in range(b):
            w2c = extrinsics[b_idx].to(device=device, dtype=dtype)
            r_cw = w2c[:3, :3]
            t_cw = w2c[:3, 3]

            c_pos = -torch.matmul(r_cw.T, t_cw)
            view_dirs_world = F.normalize(c_pos.unsqueeze(0) - pts_world, dim=-1, eps=1e-6)
            albedo = model.get_radiance(view_dirs_world)

            # Transform points to camera space
            pts_cam = _rotate(pts_world, r_cw) + t_cw  # (N, 3)

            # Determine depth convention (+Z forward vs -Z forward)
            sign_y = _camera_sign(self.camera_convention, pts_cam)
            if sign_y < 0:
                # OpenGL / ARCore convention (-Z forward)
                depth_cam = -pts_cam[:, 2]
                valid_mask = (depth_cam >= self.near_plane) & (depth_cam <= self.far_plane)
                proj_x = fx * (pts_cam[:, 0] / torch.clamp(depth_cam, min=1e-6)) + cx
                proj_y = -fy * (pts_cam[:, 1] / torch.clamp(depth_cam, min=1e-6)) + cy
            else:
                # OpenCV convention (+Z forward)
                depth_cam = pts_cam[:, 2]
                valid_mask = (depth_cam >= self.near_plane) & (depth_cam <= self.far_plane)
                proj_x = fx * (pts_cam[:, 0] / torch.clamp(depth_cam, min=1e-6)) + cx
                proj_y = fy * (pts_cam[:, 1] / torch.clamp(depth_cam, min=1e-6)) + cy

            # Surface normals and tangents in camera space
            n_cam = F.normalize(_rotate(normals_world, r_cw), dim=-1, eps=1e-6)
            if self.enable_backface_culling:
                # FasterGS backface culling: view ray is pts_cam (camera -> surfel).
                # Normal pointing away from camera has (n_cam * pts_cam) >= 0. Front-facing has < 0.
                dot_view = (n_cam * pts_cam).sum(dim=-1)
                valid_mask = valid_mask & (dot_view < 0.0)

            n_cam = _flip_normals_to_camera(n_cam, pts_cam)
            tu_cam = _rotate(t_u_world, r_cw)
            tv_cam = _rotate(t_v_world, r_cw)

            # Cull out-of-frustum points
            in_bounds = (
                (proj_x >= -32) & (proj_x < w + 32) &
                (proj_y >= -32) & (proj_y < h + 32) &
                valid_mask
            )

            if not in_bounds.any():
                zero = _zero_grad_scalar(model)
                batch_albedo.append(torch.zeros((3, h, w), dtype=dtype, device=device) + zero)
                batch_normal.append(torch.zeros((3, h, w), dtype=dtype, device=device) + zero)
                batch_roughness.append(torch.ones((1, h, w), dtype=dtype, device=device))
                batch_metallic.append(torch.zeros((1, h, w), dtype=dtype, device=device))
                batch_depth.append(torch.zeros((1, h, w), dtype=dtype, device=device) + zero)
                batch_alpha.append(torch.zeros((1, h, w), dtype=dtype, device=device) + zero)
                continue

            indices = torch.nonzero(in_bounds).squeeze(-1)
            p_x = proj_x[indices]
            p_y = proj_y[indices]
            p_d = depth_cam[indices]
            p_scale = scales[indices]
            p_op = opacities[indices]
            p_alb = albedo[indices]
            p_rou = roughness[indices]
            p_met = metallic[indices]
            p_norm = n_cam[indices]
            p_tu = tu_cam[indices]
            p_tv = tv_cam[indices]
            p_pts_cam = pts_cam[indices]

            # Jacobian J of pinhole projection at (x_c, y_c, z_c)
            # Screen tangent vectors: s_u = J * (sigma_u * tu), s_v = J * (sigma_v * tv)
            inv_d = 1.0 / torch.clamp(p_d, min=1e-6)
            inv_d2 = inv_d * inv_d

            # Tangent semi-axes in camera space
            h_u = p_scale[:, 0:1] * p_tu  # (M, 3)
            h_v = p_scale[:, 1:2] * p_tv  # (M, 3)

            # J * h:
            # J = [[fx/d, 0, -fx*xc/d^2], [0, fy/d, -fy*yc/d^2]]
            su_x = fx * h_u[:, 0] * inv_d - fx * p_pts_cam[:, 0] * h_u[:, 2] * inv_d2
            su_y = sign_y * fy * h_u[:, 1] * inv_d - sign_y * fy * p_pts_cam[:, 1] * h_u[:, 2] * inv_d2

            sv_x = fx * h_v[:, 0] * inv_d - fx * p_pts_cam[:, 0] * h_v[:, 2] * inv_d2
            sv_y = sign_y * fy * h_v[:, 1] * inv_d - sign_y * fy * p_pts_cam[:, 1] * h_v[:, 2] * inv_d2

            # Screen 2D Covariance Matrix Sigma = S_u S_u^T + S_v S_v^T + low_pass * I
            sigma_filter_sq = self.low_pass_filter_sigma ** 2
            cov_xx = su_x * su_x + sv_x * sv_x + sigma_filter_sq
            cov_yy = su_y * su_y + sv_y * sv_y + sigma_filter_sq
            cov_xy = su_x * su_y + sv_x * sv_y

            det = cov_xx * cov_yy - cov_xy * cov_xy
            det = torch.clamp(det, min=1e-6)

            # Inverse covariance components
            inv_cov_xx = cov_yy / det
            inv_cov_yy = cov_xx / det
            inv_cov_xy = -cov_xy / det

            # 3-sigma bounding radius in screen pixels
            radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(torch.maximum(cov_xx, cov_yy), min=1e-4)))
            radius = torch.clamp(radius, min=1.0, max=float(max(h, w)))

            # Sort primitives front-to-back (ascending depth)
            sort_order = torch.argsort(p_d)
            p_x = p_x[sort_order]
            p_y = p_y[sort_order]
            p_d = p_d[sort_order]
            p_op = p_op[sort_order]
            p_alb = p_alb[sort_order]
            p_rou = p_rou[sort_order]
            p_met = p_met[sort_order]
            p_norm = p_norm[sort_order]
            inv_cov_xx = inv_cov_xx[sort_order]
            inv_cov_yy = inv_cov_yy[sort_order]
            inv_cov_xy = inv_cov_xy[sort_order]
            radius = radius[sort_order]

            # Precompute 2D surfel screen bounding boxes
            bbox_x0 = p_x - radius
            bbox_x1 = p_x + radius
            bbox_y0 = p_y - radius
            bbox_y1 = p_y + radius

            # Block size for spatial checkpointing (balances kernel call overhead and VRAM usage)
            block_size = 128
            block_rows = []

            for by0 in range(0, h, block_size):
                by1 = min(by0 + block_size, h)
                block_cols = []
                for bx0 in range(0, w, block_size):
                    bx1 = min(bx0 + block_size, w)
                    bh, bw = by1 - by0, bx1 - bx0

                    overlap = (
                        (bbox_x1 >= bx0) & (bbox_x0 <= bx1) &
                        (bbox_y1 >= by0) & (bbox_y0 <= by1)
                    )
                    b_idx = torch.nonzero(overlap).squeeze(-1)
                    if b_idx.numel() == 0:
                        empty_b = torch.zeros((10, bh, bw), dtype=dtype, device=device)
                        empty_b[6:7] = 1.0  # default roughness
                        block_cols.append(empty_b)
                        continue

                    b_res = checkpoint(
                        _render_screen_block,
                        bx0, bx1, by0, by1,
                        p_x[b_idx], p_y[b_idx], p_d[b_idx],
                        p_op[b_idx], p_alb[b_idx], p_norm[b_idx],
                        p_rou[b_idx], p_met[b_idx],
                        inv_cov_xx[b_idx], inv_cov_yy[b_idx], inv_cov_xy[b_idx],
                        32, 256,
                        use_reentrant=False,
                    )
                    block_cols.append(b_res)
                block_rows.append(torch.cat(block_cols, dim=2))
            full_img = torch.cat(block_rows, dim=1)

            out_albedo = full_img[0:3]
            out_normal = full_img[3:6]
            out_roughness = full_img[6:7]
            out_metallic = full_img[7:8]
            out_depth = full_img[8:9]
            out_alpha = full_img[9:10]

            # Expected depth: the kernel accumulates alpha-weighted depth.
            out_depth = out_depth / torch.clamp(out_alpha, min=1e-3)

            # Normalize rendered normals
            norm_len = torch.linalg.norm(out_normal, dim=0, keepdim=True)
            norm_len = torch.clamp(norm_len, min=1e-6)
            out_normal = out_normal / norm_len

            out_roughness = torch.clamp(out_roughness, min=0.04, max=1.0)
            out_metallic = torch.clamp(out_metallic, min=0.0, max=1.0)

            batch_albedo.append(out_albedo)
            batch_normal.append(out_normal)
            batch_roughness.append(out_roughness)
            batch_metallic.append(out_metallic)
            batch_depth.append(out_depth)
            batch_alpha.append(out_alpha)

        return GBufferOutput(
            albedo=torch.stack(batch_albedo, dim=0),
            normal=torch.stack(batch_normal, dim=0),
            roughness=torch.stack(batch_roughness, dim=0),
            metallic=torch.stack(batch_metallic, dim=0),
            depth=torch.stack(batch_depth, dim=0),
            alpha=torch.stack(batch_alpha, dim=0),
        )


