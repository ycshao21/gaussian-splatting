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

from scene.base_gaussian_model import BaseGaussianModel


class GaussianModelLimitSplits(BaseGaussianModel):

    def __init__(self, sh_degree : int):
        super().__init__(sh_degree)

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


    def merge_io_mask(self, mask, inside_outside_mask):
        if inside_outside_mask.shape[0]>=mask.shape[0]:
            inside_outside_mask = inside_outside_mask[:mask.shape[0]]
            ret = torch.logical_and(mask, inside_outside_mask)
        else:
            n = int(mask.shape[0] - inside_outside_mask.shape[0])
            inside_outside_mask = torch.cat((inside_outside_mask, torch.zeros(n, device="cuda", dtype=bool)))
            ret = torch.logical_and(mask, inside_outside_mask)
        return ret


    def prune_points(self, mask, inside_outside_mask):
        mask = self.merge_io_mask(mask, inside_outside_mask)
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


    def densify_and_split(self, grads, grad_threshold, scene_extent, inside_outside_mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        selected_pts_mask = self.merge_io_mask(selected_pts_mask, inside_outside_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
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
        self.prune_points(prune_filter, inside_outside_mask)


    def densify_and_clone(self, grads, grad_threshold, scene_extent, inside_outside_mask):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        selected_pts_mask = self.merge_io_mask(selected_pts_mask, inside_outside_mask)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def get_inside_outside_mask(self, circles_xyzs, circles_rs, max_split_times, split_times):
        padded_grad = torch.zeros(self._xyz.shape[0], device="cuda")
        selected_pts_mask = torch.where(padded_grad >= 0, True, False)

        inside_mask = torch.logical_and(selected_pts_mask, torch.tensor(False, device="cuda"))
        for i in range(len(circles_xyzs)):
            inside_mask = torch.logical_or(inside_mask,
                                           torch.where(
                                               torch.norm(
                                                   self._xyz - circles_xyzs[i], dim=-1
                                               ) <= circles_rs[i], True, False)
                                           )
        outside_mask = torch.logical_not(inside_mask)
        if split_times >= max_split_times["inside"]:
            inside_mask = torch.logical_and(inside_mask, torch.tensor(False, device="cuda"))
        if split_times >= max_split_times["outside"]:
            outside_mask = torch.logical_and(outside_mask, torch.tensor(False, device="cuda"))

        return inside_mask, outside_mask

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, circles_xyzs, circles_rs, max_split_times, split_times):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        inside_mask, outside_mask = self.get_inside_outside_mask(circles_xyzs, circles_rs, max_split_times, split_times)
        
        self.densify_and_clone(grads, max_grad, extent, outside_mask)
        self.densify_and_split(grads, max_grad, extent, torch.logical_or(inside_mask, outside_mask))

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        inside_mask, outside_mask = self.get_inside_outside_mask(circles_xyzs, circles_rs, max_split_times, split_times)
        self.prune_points(prune_mask, torch.logical_or(inside_mask, outside_mask))

        torch.cuda.empty_cache()