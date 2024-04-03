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

"""Generator architectures from the paper
"AG3D: Learning to Generate 3D Avatars from 2D Image Collections"

Code adapted from
"Efficient Geometry-aware 3D Generative Adversarial Networks."""

import torch
import utils.legacy as legacy
from torch_utils import misc
from torch_utils import persistence
# from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training_HR.networks_stylegan2 import SemanticGenerator as StyleGAN2Backbone
from training_HR.volumetric_rendering.renderer import ImportanceRenderer
from training_HR.volumetric_rendering.ray_sampler import RaySampler
from training_HR.Generator_HR import triplane_HR
from training_HR.volumetric_rendering.renderer_HR import SemanticRenderer
import dnnlib
import numpy as np
from tqdm import tqdm
import trimesh
from training.discriminator import filtered_resizing
from torch_utils.ops import upfirdn2d


def cat_neibor(input,size):
    H = input.shape[-1]
    tmp = input.repeat(1,11,1,1)
    depth_idx = 0
    for i in range(3):
        for j in range(3):
            tmp[:,depth_idx,1:H-1,1:H-1] = input[:,:,i:H-2+i,j:H-2+j]
            depth_idx += 1
    for k in range(1):
        tmp[:,depth_idx,1:H-1,1:H-1] = input[:,:,1:H-1,1:H-1]-0.0167*(k+1)
        depth_idx += 1
    for k in range(1):
        tmp[:,depth_idx,1:H-1,1:H-1] = input[:,:,1:H-1,1:H-1]+0.0167*(k+1)
        depth_idx += 1
    tmp,_ = torch.sort(tmp,1)
    # tmp = filtered_resizing(tmp, size//2, f=None,)
    tmp = upfirdn2d.upsample2d(tmp, upfirdn2d.setup_filter([1,3,3,1]).to(tmp.device))
    tmp = upfirdn2d.upsample2d(tmp, upfirdn2d.setup_filter([1,3,3,1]).to(tmp.device))
    # tmp = filtered_resizing(tmp, size, f=None)
    return tmp
# def cat_neibor(input,size):
#     H = input.shape[-1]
#     tmp = input.repeat(1,9,1,1)
#     depth_idx = 0
#     for i in range(3):
#         if i == 1:
#             continue
#         tmp[:,depth_idx,1:H-1,] = input[:,:,i:H-2+i]
#         depth_idx += 1
#     for j in range(3):
#         tmp[:,depth_idx,:,1:H-1] = input[:,:,:,j:H-2+j]
#         depth_idx += 1
#     for k in range(2):
#         tmp[:,depth_idx,1:H-1,1:H-1] = input[:,:,1:H-1,1:H-1]-0.01*(k+1)
#         depth_idx += 1
#     for k in range(2):
#         tmp[:,depth_idx,1:H-1,1:H-1] = input[:,:,1:H-1,1:H-1]+0.01*(k+1)
#         depth_idx += 1
#     tmp,_ = torch.sort(tmp,1)
#     # tmp = filtered_resizing(tmp, size//2, f=None,)
#     tmp = upfirdn2d.upsample2d(tmp, upfirdn2d.setup_filter([1,3,3,1]).to(tmp.device))
#     tmp = upfirdn2d.upsample2d(tmp, upfirdn2d.setup_filter([1,3,3,1]).to(tmp.device))
#     # tmp = upfirdn2d.upsample2d(tmp, upfirdn2d.setup_filter([1,3,3,1]).to(tmp.device))
#     # tmp = filtered_resizing(tmp, size, f=None)
#     return tmp
up2x = torch.nn.Upsample(scale_factor=2,mode='bilinear')
up4x = torch.nn.Upsample(scale_factor=4,mode='bilinear')

def get_semantic_mask(input,origin_size):
    num_channel = input.shape[-1]
    input = input.reshape(input.shape[0], origin_size, origin_size, input.shape[2]).permute(0,3,1,2)
    # input = filtered_resizing(input, origin_size*4, f=None)
    input = upfirdn2d.upsample2d(input, upfirdn2d.setup_filter([1,3,3,1]).to(input.device))
    input = upfirdn2d.upsample2d(input, upfirdn2d.setup_filter([1,3,3,1]).to(input.device))
    input = ( input > -0.999)
    input = input.permute(0,2,3,1)
    mask = input.reshape(input.shape[0], -1, num_channel)
    return mask

def upsample(input, origin_size):
    num_channel = input.shape[-1]
    input = input.reshape(input.shape[0], origin_size, origin_size, input.shape[2]).permute(0,3,1,2)
    # input = up2x(input)
    # input = up4x(input)
    # input = filtered_resizing(input, origin_size*4, f=None,filter_mode='nearest')
    # input = upfirdn2d.upsample2d(input, upfirdn2d.setup_filter([1,3,3,1]).to(input.device))
    # input = upfirdn2d.upsample2d(input, upfirdn2d.setup_filter([1,3,3,1]).to(input.device))
    input = cat_neibor(input,origin_size*4)
    input = input.permute(0,2,3,1)
    input = input.reshape(input.shape[0], -1, 11)
    return input

@persistence.persistent_class
class AG3DGenerator(torch.nn.Module):
    
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sigmoid_beta,
        sr_num_fp16_res     = 0,
        is_sr_module = False,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        # with dnnlib.util.open_url('/home/zhengpeng/AG3Dforzp/network-snapshot_000388.pkl') as f:
        #     G_LR = legacy.load_network_pkl(f)
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        img_resolution = img_resolution//2
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        # misc.copy_params_and_buffers(G_LR['G'].renderer, self.renderer, require_all=True)
        self.renderer_HR = SemanticRenderer(sigmoid_beta.detach())
        self.ray_sampler = RaySampler()
        # self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=8*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.backbone = StyleGAN2Backbone(z_dim, c_dim, w_dim, img_resolution=256, img_channels=16*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        # misc.copy_params_and_buffers(G_LR['G'].backbone, self.backbone, require_all=True)
        # self.decoder = OSGDecoder(8, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 1})
        # self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.decoder = torch.nn.ModuleList()
        for i in range(6):
            self.decoder.append(OSGDecoder(16, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 4}))
            # self.decoder.append(OSGDecoder(16, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 1}))
        # misc.copy_params_and_buffers(G_LR['G'].decoder, self.decoder.requires_grad_(False), require_all=True)
        # self.decoder = decoder
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.triplane_SR = triplane_HR(z_dim, c_dim, w_dim, img_resolution=512, img_channels=16*3, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.decoder_HR = torch.nn.ModuleList()
        for i in range(6):
            self.decoder_HR.append(OSGDecoder(16, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 4}))
        self.neural_rendering_resolution = img_resolution//2
        
        self.rendering_kwargs = rendering_kwargs

        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas).detach()

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_params=None, cano=False, is_test=False, **synthesis_kwargs):
        
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)        

        if not cano:
            smpl_params = {
            'betas': c[:,101:111],
            'body_pose': c[:,32:101],
            'global_orient': c[:,29:32],
            'transl': c[:,26:29],
            'scale':c[:, 25]
            }
        else:
            smpl_params = None
   
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            neural_rendering_resolution = neural_rendering_resolution // 2
            self.neural_rendering_resolution = neural_rendering_resolution
    
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution, patch_params=patch_params)
        ray_origins_HR, ray_directions_HR = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution*4, patch_params=patch_params)
        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if True:
            with torch.no_grad():
                if use_cached_backbone and self._last_planes is not None:
                    planes = self._last_planes
                else:
                    planes, xs = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
                if cache_backbone:
                    self._last_planes = planes
                feature_samples, depth_samples, weights_samples, grad_samples_raw, grad_cano_samples, sdf_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, None, smpl_params, self.rendering_kwargs, is_test) # channels last
            feature_samples.detach_()
            depth_samples.detach_()
        else:
            if use_cached_backbone and self._last_planes is not None:
                planes = self._last_planes
            else:
                planes, xs = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            if cache_backbone:
                self._last_planes = planes
            feature_samples, depth_samples, weights_samples, grad_samples_raw, grad_cano_samples, sdf_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, None, smpl_params, self.rendering_kwargs, is_test) # channels last
            # feature_samples.detach_()
            depth_samples.detach_()

        plane_HR = self.triplane_SR.synthesis(ws, xs, planes, update_emas=update_emas, **synthesis_kwargs)
        # plane_HR = planes
        # num_semantic = semantics.shape[-1]
        # weights_samples = upsample(weights_samples, neural_rendering_resolution)
        # all_depths = upsample(all_depths, neural_rendering_resolution)
        # all_pts = upsample(all_pts, neural_rendering_resolution)
        # # all_valids = upsample(all_valids, neural_rendering_resolution)
        # semantics = upsample(semantics, neural_rendering_resolution)
        # depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, neural_rendering_resolution, neural_rendering_resolution)
        super_depth_image = upsample(depth_samples, neural_rendering_resolution)
        semantic_mask = get_semantic_mask(feature_samples[...,3:], neural_rendering_resolution)
        # _, semantic_indices = torch.sort(semantics, dim=-1, descending=True)
        # semantic_masks = []
        # for i in range(num_semantic):
        #     semantic_mask = (semantic_indices == i)
        #     semantic_masks.append(semantic_mask)

        # _, indices = torch.sort(weights_samples, dim=-2, descending=True)
        # indices = indices[...,:6,:]
        # all_depths = torch.gather(all_depths, -2, indices)
        # # all_depths = torch.gather(all_depths, -2, indices)
        # _, depth_indices = torch.sort(all_depths, dim=-2)
        # all_depths = torch.gather(all_depths, -2, depth_indices)
        # real_indices = torch.gather(indices, -2, depth_indices)
        # all_pts = torch.gather(all_pts, -2, real_indices.expand(-1, -1, -1, all_pts.shape[-1]))
        # # all_valids = torch.gather(all_valids, -2, real_indices.expand(-1, -1, -1, all_valids.shape[-1]))
        # all_depths = upsample(all_depths, neural_rendering_resolution)
        # all_pts = upsample(all_pts, neural_rendering_resolution)
        # # all_valids = upsample(all_valids, neural_rendering_resolution)
        # # semantics = upsample(semantics, neural_rendering_resolution)
        # all_valids = all_pts[...,-1:] > 0.9
        # all_pts = all_pts[...,:-1]
        # all_valids = torch.tensor([True],dtype=torch.bool,device=all_pts.device).view(1,1,1,1).repeat(all_pts[...,:1].shape)

        # all_depths = all_depths.view(all_depths.shape[0], neural_rendering_resolution, neural_rendering_resolution, all_depths.shape[2]).permute(0,3,1,2)
        # all_depths = up2x(all_depths)
        # all_depths = all_depths.permute(0,2,3,1)
        # all_depths = all_depths.view(all_depths.shape[0], -1, all_depths.shape[3], 1)
        feature_HR, depth_samples_HR, weights_samples, grad_samples, grad_cano_samples = self.renderer_HR(plane_HR, self.decoder_HR, ray_origins_HR, ray_directions_HR, super_depth_image, semantic_mask, smpl_params, self.rendering_kwargs) # channels last
        # feature_HR, depth_HR, weights_HR, grad_HR, grad_cano_HR, sdf_HR = self.renderer_HR(plane_HR, self.decoder, ray_origins, ray_directions, all_depths, smpl_params, self.rendering_kwargs) # channels last
        # Reshape into 'raw' neural-rendered image
        # H = W = neural_rendering_resolution
        H = neural_rendering_resolution*4
        # W = H//2
        W = H
        
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H//4, W//4)
    
        depth_image_HR = depth_samples_HR.permute(0, 2, 1).reshape(N, 1, H, W)
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H//4, W//4)
        if self.rendering_kwargs['is_normal']:
            grad_image = grad_samples.permute(0, 2, 1).reshape(N, grad_samples.shape[-1], H, W)
            grad_image_raw = grad_samples_raw.permute(0, 2, 1).reshape(N, grad_samples_raw.shape[-1], H//4, W//4)
        else:
            grad_image = None
            grad_image_raw = None

        # Run superresolution to get final image
        #############################################
        # rgb_image = feature_image[:, :self.img_channels]
        rgb_image = feature_image
        sr_image = feature_HR.permute(0, 2, 1).reshape(N, feature_HR.shape[-1], H, W)
        # sr_image = self.superresolution(sr_image.clone(), feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image':            sr_image[:,:3], 
                'image_raw':        rgb_image[:,:3], 
                # 'image_seg':         rgb_image[:,3:], 
                'image_seg':        sr_image[:,3:], 
                'seg_raw':          rgb_image[:,3:],
                'image_normal':     grad_image,
                'normal_raw':       grad_image_raw,
                'image_depth':      depth_image_HR,
                'depth_raw':        depth_image, 
                'grad_cano':        grad_cano_samples}
                
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        with torch.no_grad():
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            planes, xs = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
            return self.renderer_HR.run_model(planes, self.decoder, coordinates, directions, None, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, smpl_params=None, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        with torch.no_grad():
            planes, xs = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        plane_HR = self.triplane_SR.synthesis(ws, xs, planes, update_emas=update_emas, **synthesis_kwargs)
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs,smpl_params=smpl_params)
        return self.renderer_HR.run_model(plane_HR, self.decoder_HR, coordinates, directions, None, self.rendering_kwargs,smpl_params=smpl_params)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, patch_params=None, cano=False, **synthesis_kwargs):
        # Render a batch of generated images.
        with torch.no_grad():
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, patch_params=patch_params, cano=cano, **synthesis_kwargs)
    
    def get_mesh(self, z, c, ws=None, voxel_resolution=256, truncation_psi=1, truncation_cutoff=None, update_emas=False, canonical=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        device = z.device

        if not canonical:
            smpl_params = {
                'betas': c[:,101:111],
                'body_pose': c[:,32:101],
                'global_orient': c[:,29:32],
                'transl': c[:,26:29],
                'scale':c[:, 25]
                }
        else:
            smpl_params = {
                'betas': c[:,101:]*0,
                'body_pose': c[:,32:101]*0,
                'global_orient': c[:,29:32]*0,
                'transl': c[:,26:29]*0,
                'scale': c[:, 25]*0+1
                }
            smpl_params['transl'][:,1] = 0.3

        if ws is None:
            ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes, xs = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        

        smpl_outputs = self.renderer.deformer.body_model(betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"],
                                       scale = smpl_params["scale"])
        
        face = self.renderer.deformer.smpl_faces
        smpl_verts = smpl_outputs.vertices.float()[0]

        scale = 1.1  # Scale of the padded bbox regarding the tight one.
        verts = smpl_verts.data.cpu().numpy()
        gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
        gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
        gt_scale = (gt_bbox[1] - gt_bbox[0]).max()


        samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, smpl_verts=smpl_verts)
        samples = samples.to(device)

        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)

        head = 0
        max_batch = int(1e7)
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.sample_mixed(samples[:, head:head+max_batch], samples[:, head:head+max_batch], ws, truncation_psi=truncation_psi, noise_mode='const', smpl_params=smpl_params)['sdf']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        from utils.shape_utils import convert_sdf_samples_to_ply
        mesh = convert_sdf_samples_to_ply(sigmas, [0, 0, 0], 1, level=0)

        verts = mesh.vertices
        verts = (verts / voxel_resolution - 0.5) * scale
        verts = verts * gt_scale + gt_center

        verts_torch = torch.from_numpy(verts).float().to(device).unsqueeze(0)

        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
  
        weights = self.renderer.deformer.deformer.query_weights(verts_torch).clamp(0,1)[0]
       
        mesh.visual.vertex_colors[:,:3] = weights2colors(weights.data.cpu().numpy()*0.999)*255
        return mesh, weights

from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 32
        # self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
    
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        if sampled_features.ndim == 2:
            x = sampled_features
        else:
            x = sampled_features.flatten(0,1)

        x = self.net(x)
        # rgb = x[..., 1:] # Uses sigmoid clamping from MipNeRF
        # rgb = torch.sigmoid(x[..., 1:]*10)*(1 + 2*0.001)  # Uses sigmoid clamping from MipNeRF
        # rgb = torch.sigmoid(x[..., 2:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        rgb = x[..., 2:] # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        semantic = x[..., 1:2]
        return {'rgb': rgb, 'sigma': sigma, 'semantic': semantic}


def create_samples(N=256, smpl_verts=None, cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    verts = smpl_verts.data.cpu().numpy()
    scale = 1.1
    gt_bbox = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()
    
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples = (samples / N - 0.5) * scale
    samples = samples * gt_scale + gt_center

    num_samples = N ** 3

    return samples.unsqueeze(0), None, None



def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors
