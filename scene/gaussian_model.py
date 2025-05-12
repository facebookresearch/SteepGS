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
# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, eigh_in_batch
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
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
        self.xyz_gradient_norm_accum = torch.empty(0)
        # self.xyz_gradient3d_norm_accum = torch.empty(0)
        self.xyz_grad_estimator = 'mean'
        self.xyz_grad_ema_ratio = 0.3
        self.xyz_gradient_accum = torch.empty(0)
        # self.cov_feat_gradient_accum = torch.empty(0)
        self.S_estimator = 'partial'
        self.xyz_splitting_mat_accum = torch.empty(0)
        if self.S_estimator == 'quasi_nt':
            self.xyz_prev = torch.empty(0)
            self.xyz_dG_prev = torch.empty(0)
            self.xyz_d2G_prev = torch.empty(0)
            self.xyz_dG_diff_prev = torch.empty(0)
            self.xyz_diff_prev = torch.empty(0)
        self.denom = torch.empty(0)
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
            self.xyz_gradient_norm_accum,
            self.xyz_gradient_accum,
            self.xyz_splitting_mat_accum,
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
        xyz_gradient_norm_accum,
        xyz_gradient_accum, 
        xyz_splitting_mat_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_norm_accum = xyz_gradient_norm_accum
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_splitting_mat_accum = xyz_splitting_mat_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
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
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_matrix(self, scaling_modifier = 1):
        L = build_scaling_rotation(scaling_modifier * self.get_scaling, self._rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return actual_covariance

    def get_covariance_inv_matrix(self, scaling_modifier = 1):
        L = build_scaling_rotation(scaling_modifier / (self.get_scaling + 1e-8), self._rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return actual_covariance

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_grad_estimator = training_args.xyz_grad_estimator
        self.xyz_grad_ema_ratio = training_args.xyz_grad_ema_ratio
        self.S_estimator = training_args.S_estimator
        self.xyz_grad_estimator = training_args.xyz_grad_estimator
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_norm_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.xyz_gradient3d_norm_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        # self.cov_feat_gradient_accum = torch.zeros((self.get_xyz.shape[0], self.get_opacity.shape[1] + self.get_scaling.shape[1] + \
        #                                             self.get_rotation.shape[1] + self._features_dc.reshape(self.get_xyz.shape[0], -1).shape[1] + \
        #                                             self._features_rest.reshape(self.get_xyz.shape[0], -1).shape[1]), device='cuda')
        self.xyz_splitting_mat_accum = torch.zeros((self.get_xyz.shape[0], 3, 3), device="cuda")
        if self.S_estimator == 'quasi_nt':
            self.xyz_prev = self.get_xyz.clone()
            self.xyz_dG_prev = torch.randn((self.get_xyz.shape[0], 3), device='cuda')
            self.xyz_d2G_prev = 0.1 * torch.eye(3).unsqueeze(0).repeat(self.get_xyz.shape[0], 1, 1).to(device="cuda")
            self.xyz_dG_diff_prev = 1e-6 * torch.ones((self.get_xyz.shape[0], 3), device='cuda')
            self.xyz_diff_prev = 1e-6 * torch.ones((self.get_xyz.shape[0], 3), device='cuda')
        
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

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

    def get_current_learning_rate(self, name):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == name:
                lr = param_group['lr']
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
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
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

        self.xyz_gradient_norm_accum = self.xyz_gradient_norm_accum[valid_points_mask]
        # self.xyz_gradient3d_norm_accum = self.xyz_gradient3d_norm_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.cov_feat_gradient_accum = self.cov_feat_gradient_accum[valid_points_mask]
        self.xyz_splitting_mat_accum = self.xyz_splitting_mat_accum[valid_points_mask]

        if self.S_estimator == 'quasi_nt':
            self.xyz_prev = self.get_xyz.clone()
            self.xyz_dG_prev = self.xyz_dG_prev[valid_points_mask]
            self.xyz_d2G_prev = self.xyz_d2G_prev[valid_points_mask]
            self.xyz_dG_diff_prev = self.xyz_dG_diff_prev[valid_points_mask]
            self.xyz_diff_prev = self.xyz_diff_prev[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

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

        self.xyz_gradient_norm_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.xyz_splitting_mat_accum = torch.zeros((self.get_xyz.shape[0], 3, 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.S_estimator == 'quasi_nt':
            self.xyz_prev = self.get_xyz.clone()
            self.xyz_dG_prev = torch.concat([self.xyz_dG_prev, torch.zeros_like(new_xyz)], dim=0)
            self.xyz_d2G_prev = torch.concat([self.xyz_d2G_prev, 0.1 * torch.eye(3).unsqueeze(0).repeat(new_xyz.shape[0], 1, 1).to(device="cuda")], dim=0)
            self.xyz_dG_diff_prev = torch.concat([self.xyz_dG_diff_prev, 1e-6 * torch.ones_like(new_xyz)], dim=0)
            self.xyz_diff_prev = torch.concat([self.xyz_diff_prev, 1e-6 * torch.ones_like(new_xyz)], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
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

        return int(selected_pts_mask.sum().item())

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

        return int(selected_pts_mask.sum().item())


    def densify_and_prune(self, densify_strategy, max_grad, S_threshold, min_opacity, extent, max_screen_size):
        total_points_before = self.get_xyz.shape[0]

        if 'adc' in densify_strategy:

            grad_norms = self.xyz_gradient_norm_accum / self.denom
            grad_norms[grad_norms.isnan()] = 0.0
            
            if hasattr(self, 'visualize_densify_hook') and self.visualize_densify_hook is not None:
                self.visualize_densify_hook({
                    'selected': torch.norm(grad_norms, dim=-1) >= max_grad,
                })

            xyz_grads = self.xyz_gradient_accum / self.denom
            xyz_grads[xyz_grads.isnan()] = 0.0

            splitting_mats = self.xyz_splitting_mat_accum / self.denom[..., None]
            splitting_mats[splitting_mats.isnan()] = 0.0

            num_clone_points = self.densify_and_clone(grad_norms, max_grad, extent)
            num_split_points = self.densify_and_split(grad_norms, max_grad, extent)

            with torch.no_grad():
                xyz_grad_norms = torch.norm(xyz_grads, dim=-1)

                S_eigvals, S_eigvecs = eigh_in_batch(splitting_mats, least_k=1)
                S_eigvals, S_eigvecs = S_eigvals.squeeze(-1), S_eigvecs.squeeze(-1)

            num_stationary_points = int(torch.sum(xyz_grad_norms <= 1e-6).item())
            num_saddle_points = int(torch.sum(torch.logical_and(xyz_grad_norms <= 1e-6, S_eigvals < S_threshold)).item())


        elif 'steepest' in densify_strategy:

            viewspace_grad_norms = self.xyz_gradient_norm_accum / self.denom
            viewspace_grad_norms[viewspace_grad_norms.isnan()] = 0.0

            if self.xyz_grad_estimator == 'mean':
                grads = self.xyz_gradient_accum / self.denom
            elif self.xyz_grad_estimator == 'ema':
                grads = self.xyz_gradient_accum
            grads[grads.isnan()] = 0.0
            grad_norms = torch.norm(grads, dim=-1)

            splitting_mats = self.xyz_splitting_mat_accum / self.denom[..., None]
            splitting_mats[splitting_mats.isnan()] = 0.0

            # min_grad = 1e-6
            min_grad = 1e-4
            out_dict = self.densify_and_split_steepest(
                densify_strategy,
                viewspace_grad_norms, 
                grads,
                splitting_mats,
                max_grad,
                min_grad,
                S_threshold,
                extent,
            )

            num_added_points = out_dict['num_added_points']
            num_clone_points = out_dict['num_clone_points']
            num_split_points = out_dict['num_split_points']
            num_stationary_points = out_dict['num_stationary_points']
            num_saddle_points = out_dict['num_saddle_points']

        else:
            raise ValueError(f'Unknown densification strategy: {densify_strategy}')

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        num_pruned_points = int(torch.sum(prune_mask).item())

        torch.cuda.empty_cache()

        return dict(
            total_points_before=total_points_before,
            total_points_after=self.get_xyz.shape[0],
            num_added_points=num_clone_points + num_split_points,
            num_clone_points=num_clone_points,
            num_split_points=num_split_points,
            num_stationary_points=num_stationary_points,
            num_saddle_points=num_saddle_points,
            num_pruned_points=num_pruned_points,
        )

    
    def densify_and_split_steepest(self, options, viewspace_grad_norms, xyz_grads, splitting_mats, grad_var_threshold, grad_threshold, S_threshold, scene_extent):

        # set N=2 optimally
        N = 2

        assert S_threshold < 0., "S_threshold must be a negative number."

        with torch.no_grad():
            grad_norms = torch.norm(xyz_grads, dim=-1)

            S_eigvals, S_eigvecs = eigh_in_batch(splitting_mats, least_k=1)
            S_eigvals, S_eigvecs = S_eigvals.squeeze(-1), S_eigvecs.squeeze(-1)

        optimized_pts_mask = self.denom[..., 0] >= 1
        stationary_pts_mask = torch.logical_and(grad_norms <= grad_threshold, optimized_pts_mask)
        uncertain_pts_mask = (viewspace_grad_norms.squeeze(-1) >= grad_var_threshold)
        saddle_pts_mask = torch.logical_and(stationary_pts_mask, S_eigvals < S_threshold)


        filters = []
        if not 'no_uncertain' in options:
            filters.append(uncertain_pts_mask)
        if not 'no_eig_cond' in options and not 'no_saddle' in options:
            filters.append(S_eigvals < S_threshold)
        if 'stationary' in options and not 'no_saddle' in options:
            filters.append(stationary_pts_mask)
    
        selected_pts_mask = filters[0]
        for mask in filters[1:]:
            selected_pts_mask = torch.logical_and(selected_pts_mask, mask)

        if hasattr(self, 'visualize_densify_hook') and self.visualize_densify_hook is not None:
            self.visualize_densify_hook({'selected': selected_pts_mask})


        split_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if 'no_eig_upd' in options:
            stds = self.get_scaling[split_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            offsets_split = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[split_pts_mask]).repeat(N,1,1)
            offsets_split = torch.bmm(rots, offsets_split.unsqueeze(-1)).squeeze(-1)

            if 'zero_upd' in options:
                offsets_split = torch.zeros_like(offsets_split)
        else:
            offsets_split = S_eigvecs[split_pts_mask].repeat(N,1)

            # sample len
            stds = self.get_scaling[split_pts_mask].repeat(N,1)
            means = torch.zeros((stds.size(0), 3),device="cuda")
            split_step = torch.normal(mean=means, std=stds)
            length = torch.norm(split_step, dim=-1, keepdim=True)


            offsets_split[0::2, ...] = offsets_split[0::2, ...] * length[0::2, ...]
            offsets_split[1::2, ...] = offsets_split[1::2, ...] * length[1::2, ...] * -1.

        
        clone_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if 'no_eig_upd' in options:
            offsets_clone = torch.zeros_like(S_eigvecs[clone_pts_mask])
        else:
            offsets_clone = S_eigvecs[clone_pts_mask].repeat(N,1)

            # learning rate
            lr_xyz = self.get_current_learning_rate('xyz')
            # lr_xyz = 0.001
            offsets_clone = offsets_clone * lr_xyz

        ### Clone
        new_xyz = self._xyz[clone_pts_mask] + offsets_clone[offsets_clone.shape[0] // 2:, ...]
        self._xyz[clone_pts_mask] = self._xyz[clone_pts_mask] + offsets_clone[:offsets_clone.shape[0] // 2, ...]
        new_features_dc = self._features_dc[clone_pts_mask]
        new_features_rest = self._features_rest[clone_pts_mask]
        new_scaling = self._scaling[clone_pts_mask]
        new_rotation = self._rotation[clone_pts_mask]
        if 'no_div_opacity' not in options:
            new_opacities = self._opacity[clone_pts_mask]
        else:
            new_opacities = self.inverse_opacity_activation(self.get_opacity[clone_pts_mask] * 0.5)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


        ### Split
        split_pts_mask = torch.concat([split_pts_mask, torch.zeros((new_xyz.shape[0],), dtype=bool, device='cuda')])
        
        new_xyz = offsets_split + self._xyz[split_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[split_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[split_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[split_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[split_pts_mask].repeat(N,1,1)

        if 'no_div_opacity' not in options:
            new_opacity = self._opacity[split_pts_mask].repeat(N,1)
        else:
            new_opacity = self.inverse_opacity_activation(self.get_opacity[split_pts_mask].repeat(N,1) * 0.5)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((split_pts_mask, torch.zeros(N * split_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)



        return dict(
            num_added_points=selected_pts_mask.sum().item(),
            num_clone_points=selected_pts_mask.sum().item() - split_pts_mask.sum().item(),
            num_split_points=split_pts_mask.sum().item(),
            num_stationary_points=stationary_pts_mask.sum().item(),
            num_saddle_points=saddle_pts_mask.sum().item(),
            num_uncertain_points=uncertain_pts_mask.sum().item(),
            num_optimized_points = optimized_pts_mask.sum().item()
        )

    def add_densification_stats(self, viewspace_point_tensor, splitting_mats, update_filter):
        self.xyz_gradient_norm_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        if self.xyz_grad_estimator == 'mean':
            self.xyz_gradient_accum[update_filter] += self.get_xyz.grad[update_filter]

        elif self.xyz_grad_estimator == 'ema':
            self.xyz_gradient_accum[update_filter] = self.xyz_grad_ema_ratio * self.get_xyz.grad[update_filter] + (1. - self.xyz_grad_ema_ratio) * self.xyz_gradient_accum[update_filter]

        if self.S_estimator == 'partial' or self.S_estimator == 'approx':
            ## Partial estimator
            self.xyz_splitting_mat_accum[update_filter] += splitting_mats.grad[update_filter]

        elif self.S_estimator == 'inv_cov':
            dL_dG = splitting_mats.grad[update_filter][:, 0, 0]
            self.xyz_splitting_mat_accum[update_filter] += dL_dG[:, None, None] * self.get_covariance_inv_matrix()[update_filter]

        elif self.S_estimator == 'quasi_nt':
            raise NotImplementedError("QuasiNetwon estimator is not supported in this implementation.")

        self.denom[update_filter] += 1

