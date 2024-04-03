# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


"""Discriminator architectures from the paper
"AG3D: Learning to Generate 3D Avatars from 2D Image Collections"

Code adapted from
"Efficient Geometry-aware 3D Generative Adversarial Networks."""

import numpy as np
import torch
from torch_utils import persistence
from torch_utils.ops import upfirdn2d
from training.networks_stylegan2 import DiscriminatorBlock, MappingNetwork, DiscriminatorEpilogue
from training.cnerf import VolumeRenderDiscriminator, ComponentDualBranchDiscriminator
from training.cnerf import Discriminator

@persistence.persistent_class
class SingleDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        sr_upsample_factor  = 1,        # Ignored for SingleDiscriminator
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        resolution_scale    = 1,
        **kwargs
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, resolution_scale=resolution_scale, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        img = img['image']

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

def filtered_resizing(image_orig_tensor, size, f, filter_mode='antialiased'):
    if filter_mode == 'antialiased':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
    elif filter_mode == 'classic':
        ada_filtered_64 = upfirdn2d.upsample2d(image_orig_tensor, f, up=2)
        ada_filtered_64 = torch.nn.functional.interpolate(ada_filtered_64, size=(size * 2 + 2, size * 2 + 2), mode='bilinear', align_corners=False)
        ada_filtered_64 = upfirdn2d.downsample2d(ada_filtered_64, f, down=2, flip_filter=True, padding=-1)
    elif filter_mode == 'none':
        ada_filtered_64 = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False)
    elif type(filter_mode) == float:
        assert 0 < filter_mode < 1

        filtered = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=True)
        aliased  = torch.nn.functional.interpolate(image_orig_tensor, size=(size, size), mode='bilinear', align_corners=False, antialias=False)
        ada_filtered_64 = (1 - filter_mode) * aliased + (filter_mode) * filtered
        
    return ada_filtered_64
#----------------------------------------------------------------------------

@persistence.persistent_class
class AG3DDiscriminator(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # kwargs['img_resolution'] = kwargs['img_resolution']//4
        # self.D_seg = DualDiscriminator(resolution_scale=1, **kwargs)
        # self.D_seg = Discriminator()
        # self.D_seg = VolumeRenderDiscriminator()
        kwargs_image = kwargs.copy()
        kwargs_image['img_resolution'] = kwargs_image['img_resolution']
        # kwargs_normal['img_resolution'] = (kwargs_normal['img_resolution']//2, kwargs_normal['img_resolution']//4)
        kwargs_image['img_channels'] = 3
        self.D_image = SingleDiscriminator(resolution_scale=1, **kwargs_image)

        kwargs_sr = kwargs.copy()
        kwargs_sr['img_resolution'] = kwargs_sr['img_resolution']
        # kwargs_normal['img_resolution'] = (kwargs_normal['img_resolution']//2, kwargs_normal['img_resolution']//4)
        # kwargs_sr['img_channels'] = 3
        kwargs_sr['img_channels'] = 6
        self.D_sr = SingleDiscriminator(resolution_scale=1, **kwargs_sr)

        # kwargs_img = kwargs.copy()
        # kwargs_img['img_resolution'] = kwargs_img['img_resolution']//2
        # self.D_image = SingleDiscriminator(resolution_scale=1, **kwargs_img)
        kwargs_lr = kwargs.copy()
        kwargs_lr['img_resolution'] = kwargs_lr['img_resolution']//4
        # kwargs_normal['img_resolution'] = (kwargs_normal['img_resolution']//2, kwargs_normal['img_resolution']//4)
        kwargs_lr['img_channels'] = 3
        # self.D_lr= SingleDiscriminator(resolution_scale=1, **kwargs_lr)

        kwargs_normal = kwargs.copy()
        kwargs_normal['img_resolution'] = kwargs_normal['img_resolution']
        # kwargs_normal['img_resolution'] = (kwargs_normal['img_resolution']//2, kwargs_normal['img_resolution']//4)
        kwargs_normal['img_channels'] = 3
        self.D_normal= SingleDiscriminator(resolution_scale=1, **kwargs_normal)
        # self.D_img = SingleDiscriminator(resolution_scale=1, **kwargs_normal)
        kwargs_face = kwargs.copy()
        # kwargs_face['img_resolution'] = (kwargs_face['img_resolution']//8, kwargs_face['img_resolution']//8)
        kwargs_face['img_resolution'] = kwargs_face['img_resolution']//8
        self.D_face_image= SingleDiscriminator(**kwargs_face)

        kwargs_face_normal = kwargs.copy()
        kwargs_face_normal['img_resolution'] = kwargs_face_normal['img_resolution']//8
        # kwargs_face_normal['img_resolution'] = (kwargs_face_normal['img_resolution']//16, kwargs_face_normal['img_resolution']//16)
        kwargs_face_normal['img_channels'] = 3
        self.D_face_normal= SingleDiscriminator(**kwargs_face_normal)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
    def forward(self, img, c, update_emas=False, normal_gan=False, face_gan=False, **block_kwargs):
        if True:
            img_sr = torch.cat([filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter),img['image']],1)
            img_1024 = img['image']
            # x_sr = self.D_sr({'image':img_sr}, c, update_emas=False, **block_kwargs)
            x_img = self.D_image({'image':img_1024}, c, update_emas=False, **block_kwargs)
            # results = {'sr': x_sr}
            results = {'image': x_img}
            # results['image'] = x_img
        # img_color = {'image': img['image'],
        #              'image_raw': img['image_raw']}
        # img_raw = {'image': img['image_raw']}
        # img_color = {'image': img['image'],
        #              'image_seg': filtered_resizing(img['image_seg'], size=img['image_raw'].shape[-1], f=self.resample_filter)}
        # x_image = self.D_image(img_raw, c, update_emas=False, **block_kwargs)
        # x_seg = self.D_seg(img_color, c, update_emas=False, **block_kwargs)
        # x_seg = self.D_seg(img['image_raw'], img['image_seg'])
        # img_img = {'image': img['image_raw']}
        # x_img = self.D_img(img_img, c, update_emas=False, **block_kwargs)
        # results = {'full': x_image}
        # results = {'full': x_seg}
        # results['image'] = self.D_image({'image':img['image']}, c, update_emas=False, **block_kwargs)
        
        # results['img'] = x_img
        # results['seg'] = x_seg
        if normal_gan:
            img_normal = {'image': img['image_normal']}
            results['normal'] = self.D_normal(img_normal, c, update_emas=False, **block_kwargs)

        if face_gan:
            img_face = {'image': img['image_face']}
            results['face'] = self.D_face_image(img_face, c, update_emas=False, **block_kwargs)

            if normal_gan:
                img_face_normal = {'image': img['normal_face']}
                results['normal_face'] = self.D_face_normal(img_face_normal, c, update_emas=False, **block_kwargs)
        # if True:
        #     # img_lr = {'image': filtered_resizing(img['image_raw'], size=img['image_raw'].shape[-1]//2, f=self.resample_filter),
        #     #           'image_seg': filtered_resizing(img['image_seg'], size=img['image_seg'].shape[-1]//2, f=self.resample_filter)}
        #     # img_lr = {'image':img['image_raw']}
        #     img_lr = {'image': filtered_resizing(img['image'], size=img['image_raw'].shape[-1], f=self.resample_filter)}
        #     results['lr'] = self.D_lr(img_lr, c, update_emas=False, **block_kwargs)

        return results
#----------------------------------------------------------------------------
# DualDiscriminator is from EG3D
@persistence.persistent_class
class DualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        disc_c_noise        = 0,        # Corrupt camera parameters with X std dev of noise before disc. pose conditioning.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        is_sr_module        = False,
        resolution_scale    = 1
    ):
        super().__init__()
        # img_channels *= 2
        self.c_dim = c_dim
        ###################
        img_resolution = img_resolution //4
        # img_channels = 3
        # img_channels = 12
        img_channels = 6
        # img_channels = 13
        ###################
        self.img_resolution = img_resolution
        # self.resolution_scale = resolution_scale
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.is_sr_module = is_sr_module
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        # common_kwargs = dict(img_channels=9, architecture=architecture, conv_clamp=conv_clamp)
        # cur_layer_idx = 0
        # for res in self.block_resolutions:
        #     in_channels = channels_dict[res] if res < img_resolution else 0
        #     tmp_channels = channels_dict[res]
        #     out_channels = channels_dict[res // 2]
        #     use_fp16 = (res >= fp16_resolution)
        #     block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
        #         first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
        #     setattr(self, f'seg_b{res}', block)
        #     cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, resolution_scale= resolution_scale, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))
        self.disc_c_noise = disc_c_noise
        

    def forward(self, img, c, update_emas=False, **block_kwargs):
        
        # image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-2], f=self.resample_filter)
        # img = torch.cat([img['image'], image_raw], 1)                    
        # seg = img['image_seg']
        # img = img['image']  
        img = img['image_seg']
        # img = torch.cat([img['image'], img['image_seg']], 1) 
        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        # x_seg = None
        # for res in self.block_resolutions:
        #     block = getattr(self, f'seg_b{res}')
        #     x_seg, seg = block(x_seg, seg, **block_kwargs)
        # x = x + x_seg
        # img = img + seg
        cmap = None
        if self.c_dim > 0:
            if self.disc_c_noise > 0: c += torch.randn_like(c) * c.std(0) * self.disc_c_noise
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DummyDualDiscriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        resolution_scale    = 1
    ):
        super().__init__()
        img_channels *= 1

        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, resolution_scale=resolution_scale, **epilogue_kwargs, **common_kwargs)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter([1,3,3,1]))

        self.raw_fade = 1

    def forward(self, img, c, update_emas=False, **block_kwargs):
        self.raw_fade = max(0, self.raw_fade - 1/(500000/32))

        # image_raw = filtered_resizing(img['image_raw'], size=img['image'].shape[-1], f=self.resample_filter) * self.raw_fade
        img = img['image_raw'] # torch.cat([img['image'], image_raw], 1)

        _ = update_emas # unused
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, img, cmap)
        return x

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
