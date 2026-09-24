#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


def _quat_aligning_z_to(normals: torch.Tensor) -> torch.Tensor:
    """(w,x,y,z) quaternion rotating local +Z onto each (unit) normal row."""
    n = torch.nn.functional.normalize(normals, dim=-1)
    z_axis = torch.zeros_like(n)
    z_axis[:, 2] = 1.0
    dot = (z_axis * n).sum(-1)
    cross = torch.linalg.cross(z_axis, n, dim=-1)
    q = torch.cat([(1.0 + dot)[:, None], cross], dim=-1)
    # +Z antiparallel to n: cross collapses to 0 too, so the general formula
    # gives a zero quaternion. 180 degrees about any axis orthogonal to +Z works.
    opposite = dot < (-1.0 + 1e-6)
    if opposite.any():
        q[opposite] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=n.device)
    return torch.nn.functional.normalize(q, dim=-1)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.frustum_counter = torch.empty(0)
        self.observation_counter = torch.empty(0)
        self.seen_frustum_mask = torch.empty(0)
        self.seen_contrib_mask = torch.empty(0)
        # Per-surfel, per-view evidence accumulated over training, all (N,)
        # float tensors kept in one dict so prune/densify only has to resize one
        # thing. Reset on restore rather than checkpointed: every entry is
        # evidence gathered from rendering, and a resolution stage re-renders
        # every view within a few hundred iterations anyway.
        #   sad_eta_u/v  -- summed frequency violation per tangent axis (§4.1)
        #   sad_v_high   -- views reporting eta > 1
        #   sad_v_total  -- views this surfel was visible in
        #   free_views   -- views seeing it strictly inside free space (§5.3)
        #   surface_views-- views seeing it on the surface (§5.3)
        self.view_stats = {}
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        # .float()/.cuda() preserve_format, so a transposed source array arrives strided
        # (1, N) and stays that way in the parameter -- uncoalesced for every kernel that
        # reads xyz, and wrong for any that indexes it as flat row-major (see sparse_adam).
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda().contiguous()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # Honour the scales/opacity the depth prior (step 4) already solved for.
        # Falling through to the kNN/0.1 defaults threw that away and started
        # every surfel at twice the cull threshold; the kNN over ~2.7M points is
        # also the slow part of startup, so this skips it entirely.
        if getattr(pcd, "scales", None) is not None:
            scales_m = torch.tensor(np.asarray(pcd.scales), dtype=torch.float32, device="cuda")
            scales = torch.log(torch.clamp_min(scales_m, 1e-7))
        else:
            #dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            # --- Bypass AMD HIP simple-knn with CPU KDTree ---
            import scipy.spatial
            points_np = np.asarray(pcd.points)
            tree = scipy.spatial.cKDTree(points_np)
            # Query the 4 nearest points (k=4 because the first is the point itself)
            dists, _ = tree.query(points_np, k=4)
            # Calculate mean squared distance of the 3 actual neighbors
            dist2_np = np.mean(dists[:, 1:]**2, axis=1)
            dist2 = torch.clamp_min(torch.tensor(dist2_np, dtype=torch.float32, device="cuda"), 0.0000001)
            # -------------------------------------------------
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)

        # Honour step 4's per-surfel normals too: the rasterizer reads a
        # surfel's world-space normal off column 2 of its rotation matrix
        # (forward.cu: normal = R[:, 2]), so a quaternion that maps local +Z
        # onto the depth prior's (nx,ny,nz) starts training oriented flat
        # against the real surface instead of facing a random direction.
        # Tangent-plane spin around that normal is left random: the cloud
        # only carries |scale_u|, |scale_v| magnitudes, no tangent direction
        # to match it to. COLMAP's fallback cloud (storePly) writes all-zero
        # normals, so those points/fallback path still get a random quaternion.
        normals_np = np.asarray(pcd.normals) if getattr(pcd, "normals", None) is not None else np.empty((0, 3))
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        if normals_np.size:
            normals_t = torch.tensor(normals_np, dtype=torch.float32, device="cuda")
            valid = normals_t.norm(dim=-1) > 1e-6
            if valid.any():
                rots[valid] = _quat_aligning_z_to(normals_t[valid])

        if getattr(pcd, "opacities", None) is not None:
            op = torch.tensor(np.asarray(pcd.opacities), dtype=torch.float32, device="cuda")
            opacities = self.inverse_opacity_activation(torch.clamp(op, 0.01, 0.99))
        else:
            opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # .contiguous() on every one: several of these are built via transpose/repeat and
        # .float()/.cuda() preserve_format, so they otherwise reach the optimiser strided
        # (1, N). No-op when already packed; see the note on fused_point_cloud above.
        self._xyz = nn.Parameter(fused_point_cloud.contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rots.contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.frustum_counter = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        self.observation_counter = torch.zeros((self.get_xyz.shape[0], 1), dtype=torch.int32, device="cuda")
        self.seen_frustum_mask = torch.zeros((self.get_xyz.shape[0], 64), dtype=torch.uint8, device="cuda")
        self.seen_contrib_mask = torch.zeros((self.get_xyz.shape[0], 64), dtype=torch.uint8, device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if getattr(training_args, "sparse_adam", False):
            # Taming 3DGS: skips splats not visible in the current view. Same state layout
            # as Adam, so prune/cat/replace below are unaffected.
            from utils.sparse_adam import SparseGaussianAdam
            self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.frustum_counter.numel() > 0:
            self.frustum_counter = self.frustum_counter[valid_points_mask]
            self.observation_counter = self.observation_counter[valid_points_mask]
        if self.seen_frustum_mask.numel() > 0:
            self.seen_frustum_mask = self.seen_frustum_mask[valid_points_mask]
            self.seen_contrib_mask = self.seen_contrib_mask[valid_points_mask]
        for k, v in self.view_stats.items():
            self.view_stats[k] = v[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self.frustum_counter.numel() > 0:
            extension_frustum = torch.zeros((new_xyz.shape[0], 1), dtype=torch.int32, device=self.frustum_counter.device)
            extension_obs = torch.zeros((new_xyz.shape[0], 1), dtype=torch.int32, device=self.observation_counter.device)
            self.frustum_counter = torch.cat([self.frustum_counter, extension_frustum], dim=0)
            self.observation_counter = torch.cat([self.observation_counter, extension_obs], dim=0)
        if self.seen_frustum_mask.numel() > 0:
            n_bytes = self.seen_frustum_mask.shape[1]
            extension_mask = torch.zeros((new_xyz.shape[0], n_bytes), dtype=torch.uint8, device=self.seen_frustum_mask.device)
            self.seen_frustum_mask = torch.cat([self.seen_frustum_mask, extension_mask], dim=0)
            self.seen_contrib_mask = torch.cat([self.seen_contrib_mask, extension_mask], dim=0)
        # New surfels start with no view evidence. In particular a freshly split
        # child inherits none of its parent's free-space count: the parent was
        # judged at its old size and position, and re-earning the verdict costs
        # 3 views.
        for k, v in self.view_stats.items():
            self.view_stats[k] = torch.cat(
                [v, torch.zeros(new_xyz.shape[0], dtype=v.dtype, device=v.device)], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        self.split_selected(selected_pts_mask, N)

    def split_selected(self, selected_pts_mask, N=2):
        """Replace each selected surfel with `N` smaller ones. Net +(N-1) each."""
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        stds = torch.nan_to_num(stds, nan=0.01, posinf=0.1, neginf=0.01).clamp(1e-6, 0.5)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        self.clone_selected(selected_pts_mask)

    def clone_selected(self, selected_pts_mask):
        """Duplicate each selected surfel in place. Net +1 each."""
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_to_target(self, grads, extent, target_count, N=2):
        """Grow towards exactly `target_count` surfels, best-scoring candidates first.

        Threshold densification asks "who is above 0.0002?", so the count added
        depends on the absolute scale of the photometric gradient -- which moves
        with scene size, exposure, resolution and how good the initialisation was.
        That is why the threshold needed hand-tuning per phase, and why it does not
        transfer to another property.

        Budgeted densification asks "who are the best `k`?", where `k` is whatever
        the schedule says is still owed. The ranking is the same gradient; only the
        cut moves. The result is a surfel count that follows the schedule rather
        than the gradient's absolute scale, on any scene.

        Every candidate nets exactly +1 surfel, whether it clones (+1) or splits
        into two and drops the original (+2-1), so `k` candidates means `k` new
        surfels and the budget is honoured exactly.
        """
        headroom = int(target_count) - self.get_xyz.shape[0]
        if headroom <= 0:
            return 0

        score = torch.norm(grads, dim=-1)
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
        # A surfel no view has accumulated gradient for is not a densification
        # candidate; without this, topk pads the selection with zero-score surfels
        # and spends budget duplicating surfels nothing asked to refine.
        eligible = int((score > 0).sum())
        k = min(headroom, eligible)
        if k <= 0:
            return 0

        chosen = torch.zeros_like(score, dtype=torch.bool)
        chosen[torch.topk(score, k).indices] = True
        # Same split-vs-clone rule as the threshold path: a surfel already larger
        # than the local density target is under-resolved and should be subdivided;
        # a small one is under-represented and should be duplicated.
        too_big = self.get_scaling.max(dim=1).values > self.percent_dense * extent

        # Clone first: it only appends, so the split mask below still lines up with
        # the surfels it was computed for. split_selected prunes its originals and
        # reindexes, which would invalidate a mask built before it ran.
        clone_mask = chosen & ~too_big
        split_mask = chosen & too_big
        if clone_mask.any():
            self.clone_selected(clone_mask)
        if split_mask.any():
            padded = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=split_mask.device)
            padded[:split_mask.shape[0]] = split_mask
            self.split_selected(padded, N)
        return k

    # ------------------------------------------------------------------
    # SAD-2DGS (2dgs_combined_pipeline.md §4) and free-space carving (§5.3)
    # ------------------------------------------------------------------

    def _view_stat(self, name):
        """The named (N,) counter, allocated zeroed on first use."""
        v = self.view_stats.get(name)
        n = self.get_xyz.shape[0]
        if v is None or v.shape[0] != n:
            v = torch.zeros(n, device=self.get_xyz.device)
            self.view_stats[name] = v
        return v

    @torch.no_grad()
    def accumulate_sad_stats(self, viewpoint_camera, lambda_min, visibility_filter):
        """§4.1: accumulate per-axis frequency violation and the multi-view counters.

        `lambda_min` is a (1, H, W) map at the *render* resolution -- see
        DepthPriors.get_wavelength for why that rescale is not optional.
        """
        from utils.view_stats import (frequency_violation, project, sample_map,
                                      tangential_screen_extents)

        rots = build_rotation(self._rotation)
        ext_u, ext_v = tangential_screen_extents(
            self.get_xyz, self.get_scaling, rots, viewpoint_camera)
        u, v, z = project(self.get_xyz, viewpoint_camera)
        # LAMBDA_MIN_CLAMP_PX's ceiling as the out-of-frame default: a surfel
        # projecting outside the image has no local wavelength, and a large one
        # makes eta ~ 0, which is the "no evidence to split" answer.
        lam, inside = sample_map(u, v, lambda_min, default=1000.0)

        eta_u, eta_v = frequency_violation(ext_u, ext_v, lam)
        seen = visibility_filter & inside & (z > 0)
        if not seen.any():
            return
        eta_max = torch.maximum(eta_u, eta_v)

        self._view_stat("sad_eta_u")[seen] += eta_u[seen]
        self._view_stat("sad_eta_v")[seen] += eta_v[seen]
        self._view_stat("sad_v_high")[seen] += (eta_max[seen] > 1.0).float()
        self._view_stat("sad_v_total")[seen] += 1.0

    @torch.no_grad()
    def accumulate_freespace_stats(self, viewpoint_camera, unbiased_depth, margin):
        """§5.3: count views seeing each surfel in free space vs on the surface."""
        from utils.view_stats import freespace_classify, project, sample_map

        u, v, z = project(self.get_xyz, viewpoint_camera)
        depth_at, inside = sample_map(u, v, unbiased_depth, default=0.0)
        free, on_surface = freespace_classify(z, depth_at, inside, margin)
        self._view_stat("free_views")[free] += 1.0
        self._view_stat("surface_views")[on_surface] += 1.0

    def freespace_prune_mask(self, min_free_views, free_ratio):
        """§5.3 prune condition, or None if no evidence has been gathered yet."""
        free = self.view_stats.get("free_views")
        surf = self.view_stats.get("surface_views")
        if free is None or surf is None or free.shape[0] != self.get_xyz.shape[0]:
            return None
        ratio = free / (free + surf + 1e-6)
        return (free >= min_free_views) & (ratio > free_ratio)

    @torch.no_grad()
    def densify_sad(self, headroom, tau_split=0.75, min_views=4, max_factor=4):
        """§4.2 in-plane analytic split. Returns the number of surfels added.

        Splits strictly inside the parent's tangent plane: children sit on a
        regular n_u x n_v grid across the parent disc, keep the parent's
        rotation (hence its normal), and divide its scales. The stock 2DGS
        split instead jitters children by a Gaussian sample and shrinks by a
        fixed 1/(0.8N), which both moves the surface and leaves overlap -- the
        two things the planar manifold is trying not to do.

        `headroom` caps the total added so this stays inside the surfel budget
        the schedule hands down. Candidates are ranked by how badly they
        violate, so a tight budget is spent on the worst offenders first.
        """
        if headroom <= 0:
            return 0
        v_total = self.view_stats.get("sad_v_total")
        if v_total is None or v_total.shape[0] != self.get_xyz.shape[0]:
            return 0

        v_high = self.view_stats["sad_v_high"]
        enough = v_total >= min_views
        ratio = v_high / v_total.clamp_min(1.0)
        # Multi-view gating: a surfel is only under-resolved if most of the
        # views that can see it say so. One near, oblique view saying "too big"
        # is a grazing-angle artefact, not a missing detail.
        candidates = enough & (ratio >= tau_split)
        if not candidates.any():
            return 0

        mean_u = self.view_stats["sad_eta_u"] / v_total.clamp_min(1.0)
        mean_v = self.view_stats["sad_eta_v"] / v_total.clamp_min(1.0)
        # Concave exponent p = 0.5 (§8): linear p = 1 explodes the primitive
        # count on fine repeating patterns like wallpaper or tiling.
        n_u = mean_u.sqrt().ceil().clamp(1, max_factor)
        n_v = mean_v.sqrt().ceil().clamp(1, max_factor)
        children = (n_u * n_v).long()
        candidates &= children > 1
        if not candidates.any():
            return 0

        # Each candidate costs (children - 1) net surfels. Take the worst
        # violators, in order, until the budget is spent -- searchsorted on the
        # running cost rather than a loop.
        order = torch.argsort(torch.where(candidates, ratio * torch.maximum(mean_u, mean_v),
                                          torch.zeros_like(ratio)), descending=True)
        order = order[candidates[order]]
        cost = (children[order] - 1).cumsum(0)
        take = int(torch.searchsorted(cost, torch.tensor(headroom, device=cost.device),
                                      right=True).item())
        if take == 0:
            return 0
        chosen = order[:take]
        added = int(cost[take - 1].item())

        self._sad_split(chosen, n_u[chosen].long(), n_v[chosen].long())
        return added

    def _sad_split(self, idx, n_u, n_v):
        """Replace surfels `idx` with their n_u x n_v in-plane children."""
        rots = build_rotation(self._rotation[idx])
        t_u, t_v = rots[:, :, 0], rots[:, :, 1]
        s_u = self.get_scaling[idx, 0:1]
        s_v = self.get_scaling[idx, 1:2]

        # One flat list of (parent, a, b) triples. The grids differ per parent,
        # so this is built by repeat_interleave on the per-parent child counts
        # rather than by a fixed reshape.
        counts = n_u * n_v
        parent = torch.repeat_interleave(torch.arange(idx.shape[0], device=idx.device), counts)
        # Child ordinal within its own parent: arange(total) minus the parent's
        # start offset.
        offsets = torch.cumsum(counts, 0) - counts
        within = torch.arange(int(counts.sum().item()), device=idx.device) - offsets[parent]
        nu_p, nv_p = n_u[parent], n_v[parent]
        a = within % nu_p           # 0 .. n_u-1
        b = within // nu_p          # 0 .. n_v-1

        # Doc's grid, with a, b 1-based: ((2a - 1 - n) / n) * s * t.
        off_u = ((2.0 * (a + 1) - 1.0 - nu_p) / nu_p).unsqueeze(-1)
        off_v = ((2.0 * (b + 1) - 1.0 - nv_p) / nv_p).unsqueeze(-1)
        new_xyz = (self.get_xyz[idx][parent]
                   + off_u * s_u[parent] * t_u[parent]
                   + off_v * s_v[parent] * t_v[parent])

        new_scaling = self.scaling_inverse_activation(
            torch.cat([s_u[parent] / nu_p.unsqueeze(-1).float(),
                       s_v[parent] / nv_p.unsqueeze(-1).float()], dim=-1).clamp_min(1e-7))
        new_rotation = self._rotation[idx][parent]
        new_features_dc = self._features_dc[idx][parent]
        new_features_rest = self._features_rest[idx][parent]

        # alpha_child = 1 - (1 - alpha)^(1 / (n_u n_v)): stacking the children
        # back along a ray reproduces the parent's opacity exactly, so a split
        # does not brighten or darken the surface it happened on.
        alpha = self.get_opacity[idx][parent].clamp(1e-6, 1 - 1e-6)
        alpha_child = 1.0 - (1.0 - alpha).pow(1.0 / counts[parent].unsqueeze(-1).float())
        new_opacity = inverse_sigmoid(alpha_child.clamp(1e-6, 1 - 1e-6))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                   new_opacity, new_scaling, new_rotation)

        # Drop the parents. The mask has to be sized for the post-append cloud,
        # with the newly appended children marked keep.
        prune_filter = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device=idx.device)
        prune_filter[idx] = True
        self.prune_points(prune_filter)

    def reset_sad_stats(self):
        """Clear the §4 counters. Called after every densification pass, since
        the surfels that survive it are a different size than the views voted on."""
        for k in ("sad_eta_u", "sad_eta_v", "sad_v_high", "sad_v_total"):
            if k in self.view_stats:
                self.view_stats[k].zero_()

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_gaussians=None,
                          allow_densification=True, view_evidence_cull=False,
                          frustum_min=8, min_obs=2, target_count=None,
                          max_world_size=None, sad=None, freespace=None):
        sad_added = 0
        if allow_densification:
            grads = self.xyz_gradient_accum / self.denom
            grads[~torch.isfinite(grads)] = 0.0

            if target_count is not None:
                # SAD first: its candidates are justified by image structure, so
                # they get first claim on the budget. Whatever the target curve
                # still owes afterwards is filled by the gradient ranking, which
                # also covers every frame with no Lambda_min map and the case
                # where the wavelength maps turn out to be calibrated such that
                # nothing violates. Without that fallback a miscalibrated §0.6
                # would silently stop densification altogether.
                if sad is not None:
                    sad_added = self.densify_sad(
                        int(target_count) - self.get_xyz.shape[0],
                        tau_split=sad["tau_split"], min_views=sad["min_views"])
                    self.reset_sad_stats()
                    # SAD may have added surfels, so grads (computed before SAD)
                    # is now shorter than self.get_xyz.  Pad with zeros so that
                    # densify_to_target sees consistently-sized tensors.  We pad
                    # rather than recompute because densification_postfix resets
                    # the accumulators to zero — recomputing would lose gradient
                    # information for all pre-existing surfels.
                    if sad_added > 0:
                        pad = torch.zeros(
                            self.get_xyz.shape[0] - grads.shape[0], grads.shape[1],
                            dtype=grads.dtype, device=grads.device)
                        grads = torch.cat([grads, pad], dim=0)
                self.densify_to_target(grads, extent, target_count)
            else:
                self.densify_and_clone(grads, max_grad, extent)
                self.densify_and_split(grads, max_grad, extent)
        else:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.get_xyz.device)
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.get_xyz.device)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

        if max_world_size is not None:
            big_points_ws = self.get_scaling.max(dim=1).values > max_world_size
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        elif max_screen_size:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        if view_evidence_cull and self.frustum_counter.numel() > 0:
            floater_mask = (self.frustum_counter.squeeze(-1) >= frustum_min) & (self.observation_counter.squeeze(-1) <= min_obs)
            prune_mask = torch.logical_or(prune_mask, floater_mask)

        # §5.3. Complements the view-evidence cull above rather than repeating
        # it: that one catches surfels no view ever composites (occluded junk),
        # this one catches surfels that composite happily into every head-on
        # view but sit in the empty air a side view can see through.
        self.last_freespace_pruned = 0
        if freespace is not None:
            fs_mask = self.freespace_prune_mask(freespace["min_free_views"],
                                                freespace["free_ratio"])
            if fs_mask is not None:
                self.last_freespace_pruned = int((fs_mask & ~prune_mask).sum())
                prune_mask = torch.logical_or(prune_mask, fs_mask)

        self.prune_points(prune_mask)

        # Taming3DGS-style hard cap: once over budget, drop the lowest-opacity
        # (least useful) surfels rather than letting the model grow unbounded.
        if max_gaussians is not None and self.get_xyz.shape[0] > max_gaussians:
            excess = self.get_xyz.shape[0] - max_gaussians
            order = torch.argsort(self.get_opacity.squeeze(-1))
            budget_prune_mask = torch.zeros(self.get_xyz.shape[0], dtype=torch.bool, device="cuda")
            budget_prune_mask[order[:excess]] = True
            self.prune_points(budget_prune_mask)

        self.last_sad_added = sad_added
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, cam_id=None):
        grad_source = viewspace_point_tensor.grad if viewspace_point_tensor.grad is not None else torch.zeros_like(viewspace_point_tensor)
        grads = torch.norm(grad_source[update_filter], dim=-1, keepdim=True)
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        self.xyz_gradient_accum[update_filter] += grads
        self.denom[update_filter] += 1

        if self.frustum_counter.numel() > 0:
            dev = self.frustum_counter.device
            if cam_id is not None:
                byte_idx = cam_id // 8
                bit_val = 1 << (cam_id % 8)
                if byte_idx >= self.seen_frustum_mask.shape[1]:
                    pad_bytes = max(byte_idx + 1 - self.seen_frustum_mask.shape[1], 16)
                    pad = torch.zeros((self.seen_frustum_mask.shape[0], pad_bytes), dtype=torch.uint8, device=dev)
                    self.seen_frustum_mask = torch.cat([self.seen_frustum_mask, pad], dim=1)
                    self.seen_contrib_mask = torch.cat([self.seen_contrib_mask, pad], dim=1)

                is_new_frustum = (self.seen_frustum_mask[update_filter, byte_idx] & bit_val) == 0
                self.seen_frustum_mask[update_filter, byte_idx] |= bit_val
                self.frustum_counter[update_filter] += is_new_frustum.unsqueeze(-1).int()

                contrib_mask = (grads.squeeze(-1) > 1e-6)
                if contrib_mask.any():
                    idx_in_all = update_filter.nonzero(as_tuple=False).squeeze(-1)
                    contrib_indices = idx_in_all[contrib_mask]
                    is_new_contrib = (self.seen_contrib_mask[contrib_indices, byte_idx] & bit_val) == 0
                    self.seen_contrib_mask[contrib_indices, byte_idx] |= bit_val
                    self.observation_counter[contrib_indices] += is_new_contrib.unsqueeze(-1).int()
            else:
                self.frustum_counter[update_filter] += 1
                contrib_mask = (grads.squeeze(-1) > 1e-6)
                if contrib_mask.any():
                    idx_in_all = update_filter.nonzero(as_tuple=False).squeeze(-1)
                    contrib_indices = idx_in_all[contrib_mask]
                    self.observation_counter[contrib_indices] += 1