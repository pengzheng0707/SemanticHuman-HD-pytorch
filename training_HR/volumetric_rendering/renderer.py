# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Modified by Zijian Dong for AG3D: Learning to Generate 3D Avatars from 2D Image Collections


"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors, normal, depth, and density using the volume rendering equation.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from training.deformers import snarf_deformer
from torch_utils.ops.grid_sample_gradfix import grid_sample as grid_sample_gradfix

label_num = 6

def predefined_bbox(j):
    if j == 0:
        xyz_min = np.array([-0.0901, 0.2876, -0.0891])
        xyz_max = np.array([0.0916, 0.5555+0.04, 0.1390])
        xyz_min -= np.array([0.05, 0.05, 0.05])
        xyz_max += np.array([0.05, 0.05, 0.05])
    elif j == 1:
        xyz_min = np.array([-0.1752, 0.0208, -0.1198]) # combine 12 and 9
        xyz_max = np.array([0.1724, 0.2876, 0.1391])
    elif j == 2:
        xyz_min = np.array([-0.1569, -0.1144, -0.1095])
        xyz_max = np.array([0.1531, 0.0208, 0.1674])
    elif j == 3:
        xyz_min = np.array([-0.1888, -0.3147, -0.1224])
        xyz_max = np.array([0.1852, -0.1144, 0.1679])
    elif j == 4:
        xyz_min = np.array([0.1724, 0.1450, -0.0750])
        xyz_max = np.array([0.4321, 0.2758, 0.0406])
    elif j == 5:
        xyz_min = np.array([0.4321, 0.1721, -0.0753])
        xyz_max = np.array([0.6813, 0.2668, 0.0064])
    elif j == 6:
        xyz_min = np.array([0.6813, 0.1882, -0.1180])
        xyz_max = np.array([0.8731, 0.2445, 0.0461])
    elif j == 7:
        xyz_min = np.array([-0.4289, 0.1426, -0.0785])
        xyz_max = np.array([-0.1752, 0.2754, 0.0460])
    elif j == 8:
        xyz_min = np.array([-0.6842, 0.1705, -0.0780])
        xyz_max = np.array([-0.4289, 0.2659, 0.0059])
    elif j == 9:
        xyz_min = np.array([-0.8720, 0.1839, -0.1195])
        xyz_max = np.array([-0.6842, 0.2420, 0.0465])
    elif j == 10:
        xyz_min = np.array([0, -0.6899, -0.0849])
        xyz_max = np.array([0.1893, -0.3147, 0.1335])
    elif j == 11:
        xyz_min = np.array([0.0268, -1.0879, -0.0891])
        xyz_max = np.array([0.1570, -0.6899, 0.0691])
    elif j == 12:
        xyz_min = np.array([0.0625, -1.1591-0.04, -0.0876])
        xyz_max = np.array([0.1600, -1.0879+0.02, 0.1669])
    elif j == 13:
        xyz_min = np.array([-0.1935, -0.6964, -0.0883])
        xyz_max = np.array([0, -0.3147, 0.1299])
    elif j == 14:
        xyz_min = np.array([-0.1611, -1.0948, -0.0911])
        xyz_max = np.array([-0.0301, -0.6964, 0.0649])
    elif j == 15:
        xyz_min = np.array([-0.1614, -1.1618-0.04, -0.0882])
        xyz_max = np.array([-0.0632, -1.0948+0.02, 0.1680])
    else:
        xyz_min = xyz_max = cur_index = net_index = None

    return xyz_min, xyz_max

def semantic_box_from_joints(index, device):
    if index == 0:
        joint_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    elif index == 1:
        joint_index = [0]
    elif index == 2:
        joint_index = [1,2,3,4,5,6,7,8,9,10,13]
    elif index == 3:
        joint_index = [1,2,3,4,5,6,7,8,9,10,13]
    elif index == 4:
        joint_index = [3,10,11,12,13,14,15]
    elif index == 5:
        joint_index = [0]
    elif index == 6:
        joint_index = [11,12,14,15]
    elif index == 7:
        joint_index = [11,12,14,15]
    elif index == 8:
        joint_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    min_list = []
    max_list = []
    for j in joint_index:
        cur_min, cur_max = predefined_bbox(j)
        min_list.append(cur_min)
        max_list.append(cur_max)
    real_min = np.concatenate(min_list)
    real_min = np.min(real_min, axis=0)
    real_max = np.concatenate(max_list)
    real_max = np.max(real_max, axis=0)
    return torch.from_numpy(real_min).to(device), torch.from_numpy(real_max).to(device)

# def semantic_box(index, device):
#     if index == 0:
#         box_min = torch.tensor([-1.1,-1.1,-0.25],device=device)
#         box_max = torch.tensor([1.1,1.1,0.3],device=device)
#     elif index == 1:
#         box_min = torch.tensor([-0.5,0.4,-0.25],device=device)
#         box_max = torch.tensor([0.5,1.1,0.3],device=device)
#     elif index == 2:
#         box_min = torch.tensor([-1.1,-0.6,-0.25],device=device)
#         box_max = torch.tensor([1.1,1.1,0.3],device=device)
#     elif index == 3:
#         box_min = torch.tensor([-1.1,-0.6,-0.25],device=device)
#         box_max = torch.tensor([1.1,1.1,0.3],device=device)
#     elif index == 4:
#         box_min = torch.tensor([-1.1,-1.1,-0.25],device=device)
#         box_max = torch.tensor([1.1,0.2,0.3],device=device)
#     elif index == 5:
#         box_min = torch.tensor([-0.5,0.4,-0.25],device=device)
#         box_max = torch.tensor([0.5,1.1,0.3],device=device)
#     elif index == 6:
#         box_min = torch.tensor([-1.1,-1.1,-0.25],device=device)
#         box_max = torch.tensor([1.1,1.-0.3,0.3],device=device)
#     elif index == 7:
#         box_min = torch.tensor([-1.1,-1.1,-0.25],device=device)
#         box_max = torch.tensor([1.1,1.-0.3,0.3],device=device)
#     elif index == 8:
#         box_min = torch.tensor([-1.1,-1.1,-0.25],device=device)
#         box_max = torch.tensor([1.1,1.1,0.3],device=device)
#     return box_min.unsqueeze(0), box_max.unsqueeze(0)
def semantic_box(index, device):
    box_min = torch.tensor([-1.1, -1.1, -1.1],device=device)
    box_max = torch.tensor([1.1, 1.1, 1.1],device=device)
    return box_min.unsqueeze(0), box_max.unsqueeze(0)

def semantic_box_reg(index, device):
    box_min = torch.tensor([-1.1, -1.1, -1.1],device=device)*100
    box_max = torch.tensor([1.1, 1.1, 1.1],device=device)*100
    return box_min.unsqueeze(0), box_max.unsqueeze(0)

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes_full(plane_axes, decoder, plane_features, coordinates, deformer, sdf_activation, semantic_beta, mask=None, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    coordinates.requires_grad = True
    device = plane_features.device
    plane_features = plane_features.view(N, n_planes, label_num, C//label_num, H, W)
    out_list = {}
    for ib in range(N):
        coordinates_ib = coordinates[ib]
        out = {}
        rgbs = []
        semantics = []
        sigmas = []
        for i in range(label_num):
            local_out_rgb = torch.zeros((M, 3), device=device)
            local_out_semantic = torch.zeros((M, 1), device=device)
            local_out_sigma = torch.zeros((M, 1), device=device)
            local_plane = plane_features[:,:,i]
            local_min, local_max = semantic_box_reg(i, device)
            local_mask = (coordinates_ib <= local_max).sum(-1) + (coordinates_ib >= local_min).sum(-1)
            local_mask = (local_mask == 6)
            local_coordinates = coordinates_ib[local_mask]
            # local_coordinates = (local_coordinates - local_min) / (local_max- local_min) *2 -1
            local_coordinates = (2/box_warp) * local_coordinates # TODO: add specific box bounds

            projected_coordinates = project_onto_planes(plane_axes, local_coordinates.unsqueeze(0)).unsqueeze(1)
            
            output_features =  grid_sample_gradfix(local_plane[ib], projected_coordinates.float()).squeeze(2)

            local_feature = output_features.permute(2, 0, 1)
            # local_out = decoder(local_feature, None)
            local_out = decoder[i](local_feature, None)
            local_out_rgb[local_mask] = local_out['rgb']
            local_out_sigma[local_mask] = local_out['sigma']
            local_out_semantic[local_mask] = local_out['semantic']
            semantics.append(local_out_semantic)
            rgbs.append(local_out_rgb)
            sigmas.append(local_out_sigma)

        smpl_sdf = deformer.deformer.query_sdf(coordinates_ib[None].clone())[0] #(N,1)
        smpl_grad = sdf_gradient(coordinates_ib, smpl_sdf) #(N,1)
        # tmp_semantic = torch.softmax(torch.stack(semantics,1),1)
        # tmp_semantic = torch.softmax(torch.stack(sigmas,1),1)
        tmp_local_delta_sdf = torch.stack(sigmas,1) #(N,k,1)  
        tmp_delta_sdf = torch.sum(tmp_local_delta_sdf, 1)
        out['sdf'] = smpl_sdf.detach() + tmp_delta_sdf
        tmp_local_sigma = sdf_activation(tmp_local_delta_sdf+smpl_sdf.unsqueeze(-1).detach())
        out['sigma'] = torch.sum(tmp_local_sigma, 1)

        # tmp_semantic = torch.softmax(tmp_local_sigma,1) #(N,k,1)  
        tmp_semantic = tmp_local_sigma / (torch.sum(tmp_local_sigma, 1, keepdim=True)+1e-8) #(N,k,1)  
        # tmp_semantic = torch.sigmoid(tmp_local_sigma*semantic_beta) #(N,k,1)  

        tmp_rgb = torch.stack(rgbs,1) #(N,k,3)
        tmp_rgb = torch.sigmoid(torch.sum(tmp_rgb * tmp_semantic,dim=-2))*(1 + 2*0.001) - 0.001 #(N,3)
        out['rgb'] = torch.cat([tmp_rgb, tmp_semantic.squeeze(-1)], -1) #(N,k+3)
        # out['rgb'] = torch.stack(semantics,1)
        # out['sigma'] = torch.sum(torch.cat(sigmas,1), 1, keepdim=True)
        out['grad'] = sdf_gradient(coordinates_ib, tmp_delta_sdf)+smpl_grad.detach()
        out['grad_cano'] = out['grad'].clone()
        # out['grad'] = sdf_gradient(coordinates_ib, out['sigma'])
        for key in out:
            if key not in out_list:
                out_list[key] = []
            out_list[key].append(out[key])

    for key in out_list:
        out[key] = torch.stack(out_list[key], dim=0)
    return out

def sample_from_planes(plane_axes, decoder, plane_features, coordinates, mask, deformer, sdf_activation, semantic_beta, mode='bilinear', padding_mode='zeros', box_warp=None, is_normal=True):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    plane_features = plane_features.view(N, n_planes, label_num, C//label_num, H, W)
    N, M, _ = coordinates.shape
    # pts = coordinates.view(-1,3).cpu().numpy()
    # np.save('pts.npy',pts)
    device = plane_features.device
    out_list = {}
    for ib in range(N):
        mask_ib = mask[[ib]]

        coordinates_ib = coordinates[[ib]][mask_ib]
        coordinates_ib.requires_grad = True
        local_M = coordinates_ib.shape[-2]
        out = {}
        semantics = []
        sigmas = []
        rgbs = []
        for i in range(label_num):
            local_out_rgb = torch.zeros((local_M, 3), device=device)
            local_out_semantic = torch.zeros((local_M, 1), device=device)
            # local_out_sigma = torch.zeros((local_M, 1), device=device)
            local_out_sigma = torch.ones((local_M, 1), device=device)*-100
            local_planes = plane_features[:,:,i]
            local_min, local_max = semantic_box(i, device)
            local_mask = (coordinates_ib <= local_max).sum(-1) + (coordinates_ib >= local_min).sum(-1)
            local_mask = (local_mask == 6)
            local_coordinates = coordinates_ib[local_mask]
            local_coordinates = (2/box_warp) * local_coordinates # TODO: add specific box bounds
            # local_coordinates = (local_coordinates - local_min) / (local_max- local_min) *2 -1
            # print(torch.max(local_coordinates[...,0]))
            # print(torch.min(local_coordinates[...,0]))
            # print(torch.max(local_coordinates[...,1]))
            # print(torch.min(local_coordinates[...,1]))
            # print(torch.max(local_coordinates[...,2]))
            # print(torch.min(local_coordinates[...,2]))

            projected_coordinates = project_onto_planes(plane_axes, local_coordinates.unsqueeze(0)).unsqueeze(1).float()
            output_features = grid_sample_gradfix(local_planes[ib], projected_coordinates).squeeze(2) #(plane_features[ib], projected_coordinates).squeeze(2)
            # output_features = output_features.view(output_features.shape[0], 9, output_features.shape[1]//9, output_features.shape[2])
            local_feature = output_features.permute(2, 0, 1)
            # local_out = decoder(local_feature, None)
            local_out = decoder[i](local_feature, None)
            local_out_rgb[local_mask] = local_out['rgb']
            local_out_sigma[local_mask] = local_out['sigma']
            local_out_semantic[local_mask] = local_out['semantic']
            # if i != 0:
            semantics.append(local_out_semantic)
            rgbs.append(local_out_rgb)
            sigmas.append(local_out_sigma)

        smpl_sdf = deformer.deformer.query_sdf(coordinates_ib[None].clone())[0] #(N,1)
        smpl_grad = sdf_gradient(coordinates_ib, smpl_sdf) #(N,1)
        # tmp_semantic = torch.softmax(torch.stack(semantics,1),1)
        # tmp_semantic = torch.softmax(torch.stack(sigmas,1),1)
        tmp_local_delta_sdf = torch.stack(sigmas,1) #(N,k,1)  
        tmp_delta_sdf = torch.sum(tmp_local_delta_sdf, 1)
        out['sdf'] = smpl_sdf.detach() + tmp_delta_sdf
        tmp_local_sigma = sdf_activation(tmp_local_delta_sdf+smpl_sdf.unsqueeze(-1).detach())
        out['sigma'] = torch.sum(tmp_local_sigma, 1)

        # tmp_semantic = torch.softmax(tmp_local_sigma,1) #(N,k,1)  
        tmp_semantic = tmp_local_sigma / (torch.sum(tmp_local_sigma, 1, keepdim=True)+1e-8) #(N,k,1)  
        # tmp_semantic = torch.softmax(tmp_local_delta_sdf,1) #(N,k,1) 
        # tmp_semantic = torch.sigmoid(tmp_local_sigma*semantic_beta) #(N,k,1)   

        tmp_rgb = torch.stack(rgbs,1) #(N,k,3)
        tmp_rgb = torch.sigmoid(torch.sum(tmp_rgb * tmp_semantic,dim=-2))*(1 + 2*0.001) - 0.001 #(N,3)
        out['rgb'] = torch.cat([tmp_rgb, tmp_semantic.squeeze(-1)], -1) #(N,k+3)
        # out['rgb'] = torch.stack(semantics,1)
        # out['sigma'] = torch.sum(torch.cat(sigmas,1), 1, keepdim=True)
        if is_normal:
            out['grad_cano'] = sdf_gradient(coordinates_ib, tmp_delta_sdf)
            out['grad'] = out['grad_cano']+smpl_grad.detach()
            # out['grad'] = sdf_gradient(coordinates_ib, out['sigma'])
        for key in out:
            if key not in out_list:
                out_list[key] = []
            out_list[key].append(out[key])

    for key in out_list:
        out[key] = torch.cat(out_list[key], dim=0)

    return out

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

class ImportanceRenderer(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()
        self.deformer = snarf_deformer.SNARFDeformer('neutral')
        self.use_sdf = True

        if self.use_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1), requires_grad=True)
            # self.sigmoid_beta = 0.01
        self.semantic_beta = nn.Parameter(torch.ones(1), requires_grad=True)
    def forward(self, planes, decoder, ray_origins, ray_directions, super_depth_image, smpl_params, rendering_options, is_test=False):
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        self.is_normal = rendering_options['is_normal']
    
        # Create stratified depth samples
        if super_depth_image is not None:
            # depths_coarse = self.sample_stratified(ray_origins, -0.1, 0.1, 12, rendering_options['disparity_space_sampling'])
            depths_coarse = super_depth_image.unsqueeze(-1)
        else:
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'], is_test)
            depths_coarse = depths_coarse - ray_origins[:,:,-1].unsqueeze(-1).unsqueeze(-1)

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, smpl_params=smpl_params)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        
        if self.is_normal:
            grads_coarse = out['grad']
            grads_coarse = grads_coarse.reshape(batch_size, num_rays, samples_per_ray, grads_coarse.shape[-1])
            grad_cano_coarse = out['grad_cano']
        else:
            grads_coarse = None

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        sdf_coarse = out['sdf']

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        
        if N_importance > 0 and super_depth_image is None:
            
            _, _, weights, _ = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, grads_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options, smpl_params=smpl_params)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            
            if self.is_normal:
                
                grads_fine = out['grad']
                grad_cano_fine = out['grad_cano']
                grads_fine = grads_fine.reshape(batch_size, num_rays, samples_per_ray, grads_fine.shape[-1])
            else:
                grads_fine = None
                
            sdf_fine= out['sdf']

            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            
            all_depths, all_colors, all_densities, all_grads = self.unify_samples(depths_coarse, colors_coarse, densities_coarse, grads_coarse,
                                                                  depths_fine, colors_fine, densities_fine, grads_fine)
            
            if self.is_normal:
                grad_cano_final = torch.cat([grad_cano_coarse, grad_cano_fine], dim=1).unsqueeze(2)
            else:
                grad_cano_final = None
                
            sdf_final = torch.cat([sdf_coarse, sdf_fine], dim=1).unsqueeze(2)

            # Aggregate
            rgb_final, depth_final, weights, grads_final = self.ray_marcher(all_colors, all_densities, all_depths, all_grads, rendering_options)
 
        else:
            rgb_final, depth_final, weights, grads_final = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, grads_coarse, rendering_options)
            grad_cano_final = grad_cano_coarse.unsqueeze(2)
            sdf_final = sdf_coarse.unsqueeze(2)
        
        return rgb_final, depth_final, weights.sum(2), grads_final, grad_cano_final, sdf_final

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options, smpl_params=None):
        
        if smpl_params is None:

            with torch.enable_grad():
                out = sample_from_planes_full(self.plane_axes, decoder, planes, sample_coordinates, self.deformer, self.sdf_activation, self.semantic_beta, padding_mode='zeros', box_warp=options['box_warp'])

            with torch.enable_grad():
                sample_coordinates.requires_grad = True
                # smpl_sdf = self.deformer.deformer.query_sdf(sample_coordinates.reshape(1, -1, 3)).reshape( out['sigma'].shape[:-1]).unsqueeze(-1)
                # smpl_sdf = self.deformer.deformer.query_sdf(sample_coordinates.reshape(1, -1, 3)).reshape( out['sigma'].shape)
                # smpl_grad = sdf_gradient(sample_coordinates, smpl_sdf)

            # out['sigma'] = out['sigma'] + smpl_sdf.detach()
            # out['grad'] = out['grad'] + smpl_grad.detach()

            # if options['is_sdf']:
            #     out['sdf'] = out['sigma'].clone()
            #     # out['sigma'] = self.sdf_activation(out['sdf'])
            #     out['sigma'] = torch.sum(self.sdf_activation(out['sdf']),dim=-1,keepdim=True)
            
            # normal direction fix for camera coordinate system
            # out['grad_cano'] = out['grad'].clone()
            out['grad'][...,0] = -out['grad'][...,0]
            # out['grad'][...,-1] = -out['grad'][...,-1]
            # normalize gradients in deformed space to render normal map
            out['grad'] = torch.nn.functional.normalize(out['grad'], dim=-1)

            for key in out:
                out[key] = out[key].reshape(sample_coordinates.shape[0], sample_coordinates.shape[1], -1)

            return out

        self.deformer.prepare_deformer(smpl_params)

        def f_model(x, valid):
            
            b, n, k, _ = x.shape
            x = x.flatten(1,2)
            valid = valid.flatten(1,2)

            with torch.enable_grad():
                out = sample_from_planes(self.plane_axes, decoder, planes, x, valid, self.deformer, self.sdf_activation, self.semantic_beta, padding_mode='zeros', box_warp=options['box_warp'], is_normal=self.is_normal)

            rgb_pad = torch.zeros( (b,n*k,out['rgb'].shape[-1]),device=x.device)
            sdf_pad = torch.ones( (b,n*k,out['sdf'].shape[-1]),device=x.device)*-100
            sigma_pad = torch.zeros( (b,n*k,out['sigma'].shape[-1]),device=x.device)
            
            if self.is_normal:
                grad_pad = torch.zeros( (b,n*k,x.shape[-1]),device=x.device)
                grad_pred_pad = torch.zeros( (b,n*k,x.shape[-1]),device=x.device)
            else:
                grad_pad = None
                grad_pred_pad = None

            rgb_pad[valid] = out['rgb']

            # x_valid = x[valid][None].clone()
            # with torch.enable_grad():
            #     x_valid.requires_grad = True
            #     smpl_sdf = self.deformer.deformer.query_sdf(x_valid)[0]
            #     if self.is_normal:
            #         smpl_grad = sdf_gradient(x_valid, smpl_sdf)

            sigma_pad[valid] = out['sigma']
            sdf_pad[valid] = out['sdf']
            # sigma_pad[valid] = smpl_sdf.detach() + out['sigma']
            if self.is_normal:
                grad_pred_pad[valid] = out['grad_cano']
                grad_pad[valid] = out['grad']
            
            if self.is_normal:
                return rgb_pad.reshape(b,n,k,out['rgb'].shape[-1]), \
                        sigma_pad.reshape(b,n,k,out['sigma'].shape[-1]), \
                        sdf_pad.reshape(b,n,k,out['sdf'].shape[-1]), \
                        grad_pad.reshape(b,n,k,x.shape[-1]),\
                        grad_pred_pad.reshape(b,n,k,x.shape[-1])
            else:
                return rgb_pad.reshape(b,n,k,out['rgb'].shape[-1]), \
                        sigma_pad.reshape(b,n,k,out['sigma'].shape[-1]), \
                        sdf_pad.reshape(b,n,k,out['sdf'].shape[-1]), \
                        grad_pad, grad_pred_pad

        out = {}
        out['rgb'], out['sigma'], out['sdf'], out['grad'], out['grad_cano'], _ = self.deformer(sample_coordinates, f_model, is_normal=self.is_normal)

        # if options['is_sdf']:
        #     out['sdf'] = out['sigma'].clone()
        #     # out['sigma'] = self.sdf_activation(out['sdf'])
        #     out['sigma'] = torch.sum(self.sdf_activation(out['sdf']),dim=-1,keepdim=True)
        
        if self.is_normal:
        # normal direction fix for camera coordinate system
            # out['grad'][...,-1] = -out['grad'][...,-1]
            out['grad'][...,0] = -out['grad'][...,0]
        # normalize gradients in deformed space to render normal map
            out['grad'] = torch.nn.functional.normalize(out['grad'], dim=-1)

        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']
 
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, grads1, depths2, colors2, densities2, grads2):
        
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)
        if self.is_normal:
            all_grads = torch.cat([grads1, grads2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        if self.is_normal:
            all_grads = torch.gather(all_grads, -2, indices.expand(-1, -1, -1, all_grads.shape[-1]))
        else:
            all_grads = None

        return all_depths, all_colors, all_densities, all_grads

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False, is_test=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(0,
                                    1,
                                    depth_resolution,
                                    device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
            depth_delta = 1/(depth_resolution - 1)
            if not is_test:
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1./(1./ray_start * (1. - depths_coarse) + 1./ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1,2,0,3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                if not is_test:
                    depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                depths_coarse = torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device).reshape(1, 1, depth_resolution, 1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start)/(depth_resolution - 1)
                if not is_test:
                    depths_coarse += torch.rand_like(depths_coarse) * depth_delta
        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                             N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                                   # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

    def sdf_activation(self, input):
        alpha = 1.0 / self.get_beta()
        sigma = torch.sigmoid(input * alpha) * alpha
        return sigma
    
    def get_beta(self):
        if self.sigmoid_beta >= 0:
            return (self.sigmoid_beta + 1e-4)
        else:
            return (-self.sigmoid_beta + 1e-4)


def sdf_gradient(x, sdf):
    
    d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    gradients = torch.autograd.grad(
        outputs=sdf,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        only_inputs=True)[0]
    
    return gradients


@torch.jit.script
def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    image = image.view(N, C, IH * IW)
    scaled = ((optical + 1) / 2) * (IW-1)

    with torch.no_grad():
        scaled_f = torch.floor(scaled).type(torch.int64).detach()
        scaled_c = scaled_f + 1

    offset = scaled - scaled_f
    # offset = offset * offset * (3.0 - 2.0 * offset)

    with torch.no_grad():

        oob_index_c = torch.logical_or( (scaled_c > IW-1).any(-1),  (scaled_c < 0).any(-1))
        oob_index_f = torch.logical_or( (scaled_f > IW-1).any(-1),  (scaled_f < 0).any(-1))
        oob_index = torch.logical_or(oob_index_c, oob_index_f).squeeze(2).repeat(1,C,1)

        scaled_c.clamp_(0,IH-1)
        scaled_f.clamp_(0,IH-1)

        index_0 = scaled_c[...,1] * IW + scaled_c[...,0]
        index_2 = scaled_f[...,1] * IW + scaled_f[...,0]

        index_1 = scaled_f[...,1] * IW + scaled_c[...,0]
        index_3 = scaled_c[...,1] * IW + scaled_f[...,0]

    f0 = torch.gather(image, 2, index_0.view(N, 1, H * W).expand(-1, C, -1))
    f2 = torch.gather(image, 2, index_2.view(N, 1, H * W).expand(-1, C, -1))
    f1 = torch.gather(image, 2, index_1.view(N, 1, H * W).expand(-1, C, -1))
    f3 = torch.gather(image, 2, index_3.view(N, 1, H * W).expand(-1, C, -1))

    f03 = (f0 - f3) * offset[..., 0] + f3
    f12 = (f1 - f2) * offset[..., 0] + f2
    f0312 = (f03 - f12) * offset[..., 1] + f12

    f0312[oob_index] = 0

    return f0312