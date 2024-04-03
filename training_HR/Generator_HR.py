import torch
import numpy as np

from training_HR.networks_stylegan2 import *

@persistence.persistent_class
class SuperResolution(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        ##########################
        img_resolution = img_resolution*2
        ##########################
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        # block_kwargs['resample_filter'] = None
        for res in self.block_resolutions[-2:]:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, x, img, **block_kwargs):
        block_ws = []
        img = img.reshape(img.shape[0], img.shape[1]*img.shape[2],img.shape[3],img.shape[4])
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions[-2:]:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        # x = img = None
        for res, cur_ws in zip(self.block_resolutions[-2:], block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, SR_flag=True,**block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class triplane_HR(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis_net = torch.nn.ModuleList()
        for i in range(6):
            self.synthesis_net.append(SuperResolution(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs))
        self.num_ws = self.synthesis_net[0].num_ws


    def synthesis(self, ws, x, planes, update_emas, **synthesis_kwargs):
        imgs = []
        for i in range(len(self.synthesis_net)):
            branch_ws = ws[:, i*28+14 : i*28+14+self.num_ws]
            # img = self.synthesis_net[i](branch_ws.detach(), x[i].detach().requires_grad_(True), planes[:,:,i*16:i*16+16].detach().requires_grad_(True), update_emas=update_emas, **synthesis_kwargs)
            # img = self.synthesis_net[i](branch_ws, x[i].detach().requires_grad_(True), planes[:,:,i*16:i*16+16].detach().requires_grad_(True), update_emas=update_emas, **synthesis_kwargs)
            img = self.synthesis_net[i](branch_ws, x[i].requires_grad_(True), planes[:,:,i*16:i*16+16].requires_grad_(True), update_emas=update_emas, **synthesis_kwargs)
            imgs.append(img.view(img.shape[0], 3, img.shape[1]//3,img.shape[2],img.shape[3]))
        return torch.cat(imgs,2)
    def forward(self, z, c, x, planes, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, x, planes, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------