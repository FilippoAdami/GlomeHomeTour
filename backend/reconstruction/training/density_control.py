"""GlomeHomeTour Backend: 2DGS Taming Density Control & Primitive Capping.

Implements budget-bounded densification (splitting, cloning, pruning, opacity resets)
and normal-aware spatial voxel grid filtering, capped at DensityControlConfig.max_primitives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reconstruction.training.model import Material2DGSModel


@dataclass
class DensityControlConfig:
    """Configuration parameters for Taming-2DGS density control."""
    max_primitives: int = 1_200_000           # Strict hard budget cap (scales dynamically up to 1.4M)
    growth_rate: float = 0.3                  # Per-interval ceiling on growth, as a fraction of the population
    grad_threshold: float = 0.0               # Floor on avg positional gradient; only excludes untouched primitives
    scale_split_threshold: float = 0.008      # Scale threshold (meters) for splitting vs cloning (8mm)
    max_scale_prune_m: Optional[float] = 0.04 # Prune splats whose planar radius exceeds 4 cm (kills free-space floaters)
    min_opacity_prune: float = 0.05           # Opacity threshold below which surfels are pruned
    max_aspect_ratio: float = 10.0            # Max allowable aspect ratio sigma_u / sigma_v
    min_aspect_ratio: float = 0.1             # Min allowable aspect ratio
    densify_interval: int = 200               # Execute densify & prune every N steps
    opacity_reset_interval: int = 100_000     # Opacity resets disabled by default to avoid opacity collapse
    reset_opacity_val: float = 0.1            # Opacity reset value
    enable_voxel_pruning: bool = False        # Normal-aware voxel dedup
    voxel_size_m: float = 0.015               # 1.5 cm spatial voxel grid resolution
    normal_bins: int = 8                      # 8 directional normal octants for thin-wall preservation

    @classmethod
    def from_initial_surfels(
        cls,
        num_init_surfels: int,
        growth_multiplier: float = 2.5,
        max_hard_cap: int = 1_400_000,
        scale_split_threshold: float = 0.008,
        **kwargs: Any,
    ) -> "DensityControlConfig":
        """Compute scene-proportional budget scaled by initial surface area."""
        proportional_budget = int(round(num_init_surfels * growth_multiplier))
        capped_budget = min(max(proportional_budget, num_init_surfels), max_hard_cap)
        return cls(
            max_primitives=capped_budget,
            scale_split_threshold=scale_split_threshold,
            **kwargs,
        )


class TamingDensityController:
    """Monitors positional gradients and orchestrates budget-capped densification,

    splitting, cloning, and normal-aware spatial voxel pruning.
    """

    def __init__(
        self,
        config: Optional[DensityControlConfig] = None,
    ) -> None:
        self.config = config or DensityControlConfig()

        # Positional gradient accumulation buffers
        self.grad_accum = torch.empty(0, dtype=torch.float32)
        self.denom = torch.empty(0, dtype=torch.int32)

    def init_buffers(self, num_gaussians: int, device: torch.device) -> None:
        """Initialize gradient accumulation buffers matching the model count."""
        self.grad_accum = torch.zeros((num_gaussians, 1), dtype=torch.float32, device=device)
        self.denom = torch.zeros((num_gaussians, 1), dtype=torch.int32, device=device)

    def accumulate_gradients(self, model: Material2DGSModel) -> None:
        """Accumulate L2 positional gradients from the current step."""
        if model._xyz.grad is None:
            return

        device = model._xyz.device
        if self.grad_accum.shape[0] != model.num_gaussians or self.grad_accum.device != device:
            self.init_buffers(model.num_gaussians, device)

        # L2 norm of spatial gradient nabla_p L: (N, 1)
        p_grad = torch.linalg.norm(model._xyz.grad.detach()[:, :3], dim=-1, keepdim=True)
        self.grad_accum += p_grad
        # Count only steps where the primitive was actually visible, else a
        # surfel seen in 3 of 273 views averages ~0 and never densifies.
        self.denom += (p_grad > 0).to(self.denom.dtype)

    def reset_buffers(self, num_gaussians: int, device: torch.device) -> None:
        """Clear gradient accumulation buffers."""
        self.init_buffers(num_gaussians, device)

    def reset_opacities(self, model: Material2DGSModel, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Reset primitive opacities to eliminate invisible floaters.

        Not called by Material2DGSTrainer any more: with a depth-prior init the
        surfels start on real surfaces, and clamping every opacity to 0.1 left
        the scene 91% transparent. Kept for manual floater surgery.
        """
        reset_logit = float(np.log(self.config.reset_opacity_val / (1.0 - self.config.reset_opacity_val)))
        with torch.no_grad():
            # Only clamp down opacities that are higher than reset value
            cur_op = torch.sigmoid(model._opacity)
            new_op = torch.clamp(cur_op, max=self.config.reset_opacity_val)
            new_logit = torch.logit(torch.clamp(new_op, 1e-4, 1.0 - 1e-4), eps=1e-6)
            model._opacity.copy_(new_logit)

    def normal_aware_voxel_pruning(self, model: Material2DGSModel) -> torch.Tensor:
        """Identify redundant coplanar surfels within 1.5 cm voxels with matching normals.

        Preserves opposite-facing surfaces (such as thin drywall or doors)
        by combining 3D voxel index with an 8-octant normal direction hash.
        Returns a boolean mask of kept surfels.
        """
        n = model.num_gaussians
        if n == 0 or not self.config.enable_voxel_pruning:
            return torch.ones(n, dtype=torch.bool, device=model.xyz.device)

        device = model.xyz.device
        pos = model.xyz.detach()
        normals = model.normals.detach()
        opacities = model.opacity.detach().squeeze(-1)

        # 1. Spatial grid coordinates: (N, 3)
        v_size = self.config.voxel_size_m
        voxel_idx = torch.floor(pos / v_size).to(dtype=torch.int64)

        # 2. Normal octant direction: 3 bits (sign of nx, ny, nz) in [0, 7]
        sign_x = (normals[:, 0] >= 0).to(torch.int64)
        sign_y = (normals[:, 1] >= 0).to(torch.int64)
        sign_z = (normals[:, 2] >= 0).to(torch.int64)
        octant = sign_x + (sign_y << 1) + (sign_z << 2)

        # Construct 64-bit spatial-normal hash
        # prime multipliers for spatial coordinates
        p1, p2, p3, p4 = 73856093, 19349663, 83492791, 2654435761
        spatial_hash = (voxel_idx[:, 0] * p1) ^ (voxel_idx[:, 1] * p2) ^ (voxel_idx[:, 2] * p3)
        combined_hash = spatial_hash ^ (octant * p4)

        # Sort by hash, then by opacity descending so the highest-opacity surfel comes first
        # Convert to CPU for robust unique hashing
        hash_np = combined_hash.cpu().numpy()
        op_np = opacities.cpu().numpy()

        # Sort order: primary hash, secondary -opacity
        sort_keys = np.lexsort((-op_np, hash_np))
        sorted_hashes = hash_np[sort_keys]

        # Find first occurrence of each unique hash
        _, unique_first_indices = np.unique(sorted_hashes, return_index=True)
        keep_indices = sort_keys[unique_first_indices]

        keep_mask = torch.zeros(n, dtype=torch.bool, device=device)
        keep_mask[torch.from_numpy(keep_indices).to(device=device)] = True
        return keep_mask

    def densify_and_prune(
        self,
        model: Material2DGSModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        iteration: int = 0,
        max_depth_ceiling: Optional[float] = None,
        remaining_intervals: int = 1,
    ) -> Tuple[int, int, int]:
        """Perform budget-aware cloning, splitting, and pruning.

        Args:
            model: Material2DGSModel being trained.
            optimizer: PyTorch optimizer containing model parameters.
            iteration: Current training iteration step.
            max_depth_ceiling: Optional room depth boundary (meters).
            remaining_intervals: Densification calls left in the schedule, including
                this one. The remaining budget is spread evenly over them.

        Returns:
            (num_cloned, num_split, num_pruned)
        """
        device = model._xyz.device
        num_orig = model.num_gaussians
        if num_orig == 0:
            return 0, 0, 0

        # Compute average positional gradients
        safe_denom = torch.clamp(self.denom, min=1)
        avg_grads = (self.grad_accum / safe_denom).squeeze(-1)  # (N,)

        # Budget-scheduled densification (Taming 3DGS): rank by accumulated
        # positional gradient and densify exactly as many primitives as the
        # growth schedule allows. A world-space gradient threshold starved this
        # -- only 0.6% of primitives cleared 2e-4, so opacity pruning outpaced
        # densification and the scene decayed 148k -> 88k over a 7k run.
        # Clone and split are both net +1 (a split parent is pruned), so k
        # selected candidates grow the population by k.
        # Skip primitives that this same call is about to prune: splitting one
        # just spawns two children that die at the next interval.
        viable = (self.denom.squeeze(-1) > 0) & (
            model.opacity.detach().squeeze(-1) >= self.config.min_opacity_prune
        )
        avg_grads = torch.where(viable, avg_grads, torch.zeros_like(avg_grads))
        # Progressive ramp to the budget (Taming 3DGS): spread what is left of
        # the budget evenly over the remaining intervals rather than saturating
        # in the first few, which spends the whole run on primitives placed
        # while the geometry was still coarse.
        avail_budget = max(0, self.config.max_primitives - num_orig)
        n_add = min(
            int(math.ceil(avail_budget / max(1, remaining_intervals))),
            int(math.ceil(num_orig * self.config.growth_rate)),
            avail_budget,
            num_orig,
        )

        clone_mask = torch.zeros(num_orig, dtype=torch.bool, device=device)
        split_mask = torch.zeros(num_orig, dtype=torch.bool, device=device)
        if n_add > 0:
            cand = torch.topk(avg_grads, k=n_add).indices
            cand = cand[avg_grads[cand] > self.config.grad_threshold]
            max_scales = torch.max(model.scaling.detach(), dim=-1).values  # (N,)
            is_large = max_scales[cand] >= self.config.scale_split_threshold
            split_mask[cand[is_large]] = True
            clone_mask[cand[~is_large]] = True

        # 2. Extract clone primitives
        num_cloned = int(clone_mask.sum().item())
        cloned_xyz = None
        cloned_rot = None
        cloned_scale = None
        cloned_op = None
        cloned_alb = None
        cloned_rou = None
        cloned_met = None
        cloned_rest = None

        if num_cloned > 0:
            c_idx = torch.nonzero(clone_mask).squeeze(-1)
            # Slightly jitter cloned position along tangent_u
            tu = model.tangent_u[c_idx]
            jitter = tu * (model.scaling[c_idx, 0:1] * 0.5)
            cloned_xyz = model._xyz[c_idx] + jitter
            cloned_rot = model._rotation[c_idx].clone()
            cloned_scale = model._scaling[c_idx].clone()
            cloned_op = model._opacity[c_idx].clone()
            cloned_alb = model._albedo[c_idx].clone()
            cloned_rou = model._roughness[c_idx].clone()
            cloned_met = model._metallic[c_idx].clone()
            if hasattr(model, "_features_rest") and model._features_rest is not None:
                cloned_rest = model._features_rest[c_idx].clone()

        # 3. Extract split primitives
        num_split = int(split_mask.sum().item())
        split_xyz_list = []
        split_scale_list = []
        split_rot_list = []
        split_op_list = []
        split_alb_list = []
        split_rou_list = []
        split_met_list = []
        split_rest_list = []

        if num_split > 0:
            s_idx = torch.nonzero(split_mask).squeeze(-1)
            tu = model.tangent_u[s_idx]
            tv = model.tangent_v[s_idx]
            sc = model.scaling[s_idx]

            # Create 2 sub-surfels offset along tangent axes and scaled down by factor 1.6
            offset = tu * (sc[:, 0:1] * 0.4)
            scaled_down_log = model._scaling[s_idx] - float(np.log(1.6))
            has_rest = hasattr(model, "_features_rest") and model._features_rest is not None

            # Child 1: +offset
            split_xyz_list.append(model._xyz[s_idx] + offset)
            split_scale_list.append(scaled_down_log)
            split_rot_list.append(model._rotation[s_idx].clone())
            split_op_list.append(model._opacity[s_idx].clone())
            split_alb_list.append(model._albedo[s_idx].clone())
            split_rou_list.append(model._roughness[s_idx].clone())
            split_met_list.append(model._metallic[s_idx].clone())
            if has_rest:
                split_rest_list.append(model._features_rest[s_idx].clone())

            # Child 2: -offset
            split_xyz_list.append(model._xyz[s_idx] - offset)
            split_scale_list.append(scaled_down_log)
            split_rot_list.append(model._rotation[s_idx].clone())
            split_op_list.append(model._opacity[s_idx].clone())
            split_alb_list.append(model._albedo[s_idx].clone())
            split_rou_list.append(model._roughness[s_idx].clone())
            split_met_list.append(model._metallic[s_idx].clone())
            if has_rest:
                split_rest_list.append(model._features_rest[s_idx].clone())

        # 4. Pruning condition:
        # - Low opacity
        # - Aspect ratio extreme
        # - Split parents are pruned (replaced by children)
        # - Depth outside ceiling
        # - Normal-aware spatial voxel collision
        cur_opacity = model.opacity.squeeze(-1)
        cur_scales = model.scaling
        aspect_ratio = cur_scales[:, 0] / torch.clamp(cur_scales[:, 1], min=1e-6)

        prune_mask = cur_opacity < self.config.min_opacity_prune
        if self.config.max_scale_prune_m is not None:
            prune_mask = prune_mask | (torch.max(cur_scales, dim=-1).values > self.config.max_scale_prune_m)
        prune_mask = prune_mask | (aspect_ratio > self.config.max_aspect_ratio)
        prune_mask = prune_mask | (aspect_ratio < self.config.min_aspect_ratio)
        prune_mask = prune_mask | split_mask  # Original split parents get replaced

        if max_depth_ceiling is not None:
            depth_radial = torch.linalg.norm(model._xyz.detach(), dim=-1)
            prune_mask = prune_mask | (depth_radial > max_depth_ceiling)

        # Normal-aware spatial voxel deduplication
        if self.config.enable_voxel_pruning:
            voxel_keep = self.normal_aware_voxel_pruning(model)
            prune_mask = prune_mask | (~voxel_keep)

        # Kept original primitives
        keep_mask = ~prune_mask
        num_pruned = int(prune_mask.sum().item())

        # Provenance of each new row: index into the old tensors, -1 for
        # freshly created primitives. Used to carry Adam moments across.
        src_idx = torch.arange(num_orig, device=device)[keep_mask]

        # Combine kept originals + cloned + split children
        new_xyz_parts = [model._xyz[keep_mask]]
        new_rot_parts = [model._rotation[keep_mask]]
        new_scale_parts = [model._scaling[keep_mask]]
        new_op_parts = [model._opacity[keep_mask]]
        new_alb_parts = [model._albedo[keep_mask]]
        new_rou_parts = [model._roughness[keep_mask]]
        new_met_parts = [model._metallic[keep_mask]]
        new_rest_parts = [model._features_rest[keep_mask]] if hasattr(model, "_features_rest") and model._features_rest is not None else []

        if cloned_xyz is not None:
            src_idx = torch.cat([src_idx, torch.full((num_cloned,), -1, dtype=src_idx.dtype, device=device)])
            new_xyz_parts.append(cloned_xyz)
            new_rot_parts.append(cloned_rot)
            new_scale_parts.append(cloned_scale)
            new_op_parts.append(cloned_op)
            new_alb_parts.append(cloned_alb)
            new_rou_parts.append(cloned_rou)
            new_met_parts.append(cloned_met)
            if cloned_rest is not None:
                new_rest_parts.append(cloned_rest)

        if split_xyz_list:
            for s_idx_split, (s_xyz, s_sc, s_rot, s_op, s_alb, s_rou, s_met) in enumerate(zip(
                split_xyz_list, split_scale_list, split_rot_list, split_op_list, split_alb_list, split_rou_list, split_met_list
            )):
                src_idx = torch.cat([src_idx, torch.full((s_xyz.shape[0],), -1, dtype=src_idx.dtype, device=device)])
                new_xyz_parts.append(s_xyz)
                new_rot_parts.append(s_rot)
                new_scale_parts.append(s_sc)
                new_op_parts.append(s_op)
                new_alb_parts.append(s_alb)
                new_rou_parts.append(s_rou)
                new_met_parts.append(s_met)
                if split_rest_list:
                    new_rest_parts.append(split_rest_list[s_idx_split])

        # Re-assign parameters in model
        new_xyz = torch.cat(new_xyz_parts, dim=0)
        new_rot = torch.cat(new_rot_parts, dim=0)
        new_scale = torch.cat(new_scale_parts, dim=0)
        new_op = torch.cat(new_op_parts, dim=0)
        new_alb = torch.cat(new_alb_parts, dim=0)
        new_rou = torch.cat(new_rou_parts, dim=0)
        new_met = torch.cat(new_met_parts, dim=0)
        new_rest = torch.cat(new_rest_parts, dim=0) if new_rest_parts else None

        # Enforce budget cap
        if new_xyz.shape[0] > self.config.max_primitives:
            perm = torch.randperm(new_xyz.shape[0], device=device)[:self.config.max_primitives]
            new_xyz = new_xyz[perm]
            new_rot = new_rot[perm]
            new_scale = new_scale[perm]
            new_op = new_op[perm]
            new_alb = new_alb[perm]
            new_rou = new_rou[perm]
            new_met = new_met[perm]
            if new_rest is not None:
                new_rest = new_rest[perm]
            src_idx = src_idx[perm]

        self._replace_model_parameters(
            model,
            new_xyz,
            new_rot,
            new_scale,
            new_op,
            new_alb,
            new_rou,
            new_met,
            new_rest=new_rest,
            optimizer=optimizer,
            src_idx=src_idx,
        )

        # Reset accumulation buffers
        self.reset_buffers(model.num_gaussians, device)
        return num_cloned, num_split, num_pruned

    def _replace_model_parameters(
        self,
        model: Material2DGSModel,
        new_xyz: torch.Tensor,
        new_rot: torch.Tensor,
        new_scale: torch.Tensor,
        new_op: torch.Tensor,
        new_alb: torch.Tensor,
        new_rou: torch.Tensor,
        new_met: torch.Tensor,
        new_rest: Optional[torch.Tensor] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        src_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Replace model parameter tensors and update optimizer states if provided."""
        # Replace parameters
        model._xyz = nn.Parameter(new_xyz)
        model._rotation = nn.Parameter(new_rot)
        model._scaling = nn.Parameter(new_scale)
        model._opacity = nn.Parameter(new_op)
        model._albedo = nn.Parameter(new_alb)
        model._roughness = nn.Parameter(new_rou)
        model._metallic = nn.Parameter(new_met)
        if new_rest is not None:
            model._features_rest = nn.Parameter(new_rest)

        # Update optimizer state dict if an optimizer is bound
        if optimizer is not None:
            # Reconstruct optimizer param groups
            # Cleanly reset optimizer parameter list to point to new tensors
            param_map = {
                "_xyz": model._xyz,
                "_rotation": model._rotation,
                "_scaling": model._scaling,
                "_opacity": model._opacity,
                "_albedo": model._albedo,
                "_roughness": model._roughness,
                "_metallic": model._metallic,
            }
            if hasattr(model, "_features_rest") and model._features_rest is not None:
                param_map["_features_rest"] = model._features_rest

            for group in optimizer.param_groups:
                name = group.get("name", None)
                if name in param_map:
                    # Carry Adam moments over to the surviving primitives.
                    # Dropping them re-warms up momentum at every densification
                    # interval, which at ~20 intervals is most of the run.
                    old_param = group["params"][0]
                    state = optimizer.state.pop(old_param, None)
                    new_param = param_map[name]
                    if state is not None and src_idx is not None and "exp_avg" in state:
                        keep = src_idx >= 0
                        moved = {}
                        for key in ("exp_avg", "exp_avg_sq"):
                            buf = torch.zeros_like(new_param.detach())
                            buf[keep] = state[key][src_idx[keep]]
                            moved[key] = buf
                        state.update(moved)
                        optimizer.state[new_param] = state
                    group["params"] = [new_param]
                else:
                    # Generic fallback: match by parameter index
                    group["params"] = list(model.parameters())
                    optimizer.state.clear()
                    break
