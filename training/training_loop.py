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

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
import wandb
import utils.legacy as legacy
from metrics import metric_main
from utils.camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section
import dill
import wandb
import shutil
from training.patch_utils import sample_patch_params, extract_patches, linear_schedule
from training.zp import *


#----------------------------------------------------------------------------
def mask_to_png(sample_seg):
    back_seg = -1 + np.ones_like(sample_seg)[...,-1:]*0.01
    sample_seg = np.concatenate([back_seg,sample_seg],-1)
    sample_seg = np.argmax(sample_seg, axis=-1)
    return sample_seg

#----------------------------------------------------------------------------
def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 8)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 8)

    # No labels => show random subset of training samples.
    if True:
    # if not training_set.has_labels:
        
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])
        
        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, normals, segs, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(segs), np.stack(normals), np.stack(labels)

#----------------------------------------------------------------------------


def save_seg_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    # img = (img - lo) * (255 / (hi - lo))
    # img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img_small = img[:8,:,:8,:,:]
    img_small = img_small.reshape([8 * H, 8 * W, C])
 
    # wandb.log({title: [wandb.Image(img_small, caption=fname.split("/")[-1])]})
    img = img.reshape([gh * H, gw * W, C])
    img = mask_to_png(img)
    img = PIL.Image.fromarray(img.astype(np.uint8))
    cmap = np.load('cmap.npy')
    img.putpalette(cmap.tolist())
    img.save(fname)
    
    return img_small

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img_small = img[:8,:,:8,:,:]
    img_small = img_small.reshape([8 * H, 8 * W, C])
 
    # wandb.log({title: [wandb.Image(img_small, caption=fname.split("/")[-1])]})
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
    
    
    return img_small

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    testing_set_kwargs      = {},
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    restart_every           = -1,       # Time interval in seconds to exit code
    is_load_noise           =True,
):
    # Initialize.
    start_time = time.time()
    is_load_noise = is_load_noise
    is_normal = loss_kwargs['is_normal']
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    __RESTART__ = torch.tensor(0., device=device)       # will be broadcasted to exit loop
    __CUR_NIMG__ = torch.tensor(0, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __AUGMENT_P__ = torch.tensor(augment_p, dtype=torch.float, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)
    best_fid = 9999
    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    
    if rank == 0:
        print('Constructing networks...')

    
    G_kwargs.c_dim = D_kwargs.c_dim = training_set.label_dim
    
    common_kwargs = dict(img_resolution=G_kwargs.rendering_kwargs['image_resolution'], img_channels=training_set.num_channels)
    #Generator network gan模型中的生成部分模型
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    dataset_label_std = torch.tensor(training_set.get_label_std()).to(device)
    G.register_buffer('dataset_label_std', dataset_label_std)
    common_kwargs['img_channels'] = 3
    # common_kwargs['img_channels'] = 9
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    
    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):       # find last pkl file
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)

    
    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            
            resume_data = legacy.load_network_pkl(f)
            
            
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, is_load_noise, require_all=False )
        
        # if ckpt_pkl is not None:            # Load ticks
        #     __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
        #     __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
        #     __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
        #     __AUGMENT_P__ = resume_data['progress'].get('augment_p', torch.tensor(0.)).to(device)
        #     __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
        #     if ckpt_pkl is not None:
        #         best_fid = resume_data['progress']['best_fid']       # only needed for rank == 0    # Load ticks
        __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
        __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
        __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
        __AUGMENT_P__ = resume_data['progress'].get('augment_p', torch.tensor(0.)).to(device)
        __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
        best_fid = resume_data['progress']['best_fid']       # only needed for rank == 0


    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, imgs, segs, normals, labels = setup_snapshot_image_grid(training_set=training_set)

        seg_real = save_seg_grid(segs, os.path.join(run_dir, 'reals_seg.png'), drange=[0,255], grid_size=grid_size)
        normal_real = save_image_grid(normals, os.path.join(run_dir, 'reals_normal.png'), drange=[0,255], grid_size=grid_size)
        img_real = save_image_grid(imgs, os.path.join(run_dir, 'reals_img.png'), drange=[0,255], grid_size=grid_size)
        # normal_real = save_image_grid(segs[:,:1], os.path.join(run_dir, 'reals_mask.png'), drange=[-1,1], grid_size=grid_size)

        # z noise
        # wandb.log({"real_image": [wandb.Image(seg_real), wandb.Image(normal_real)]})
        # grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_z = [mixing_noise(batch_gpu, G.z_dim, prob=0.9, device=device) for i in range(labels.shape[0] // batch_gpu)]
        
        # label parameters (batch, label_size)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
        
    if num_gpus > 1:  # broadcast loaded states to all
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.broadcast(__AUGMENT_P__, 0)
        torch.distributed.broadcast(__PL_MEAN__, 0)
        torch.distributed.barrier()  # ensure all processes received this info
        
    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    augment_p = __AUGMENT_P__
    if augment_pipe is not None:
        augment_pipe.p.copy_(augment_p)
    if hasattr(loss, 'pl_mean'):
        loss.pl_mean.copy_(__PL_MEAN__)
 
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            
            phase_real_img, phase_real_normal, phase_real_seg, phase_real_c = next(training_set_iterator)
            
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_normal = (phase_real_normal.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_seg = phase_real_seg.to(device).to(torch.float32).split(batch_gpu)
            # phase_real_seg = (phase_real_seg.to(device).to(torch.float32)*2 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            
            #############################################################
            # all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            # all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_z = [[mixing_noise(batch_gpu, G.z_dim, prob=0.9, device=device) for i in range(batch_size // batch_gpu)] for j in range(len(phases))]
            #############################################################
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            # for real_img, real_seg, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_normal, phase_real_c, phase_gen_z, phase_gen_c):
            for real_img, real_seg, real_normal, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_seg, phase_real_normal, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_seg=real_seg, real_normal=real_normal, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg, cur_tick=cur_tick)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1):
        # if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Check for restart.
        if (rank == 0) and (restart_every > 0) and (time.time() - start_time > restart_every):
            print('Restart job...')
            __RESTART__ = torch.tensor(1., device=device)
        if num_gpus > 1:
            torch.distributed.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f'Process {rank} leaving...')
            if num_gpus > 1:
                torch.distributed.barrier()

        # Save image snapshot.
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
        # if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_nimg % image_snapshot_ticks == 0):
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_nimg % (image_snapshot_ticks*100) == 0 or (cur_nimg % image_snapshot_ticks == 0 and cur_nimg < 1000)):
            with torch.no_grad():
               out = [G_ema(z=z, c=c,noise_mode='const') for z, c in zip(grid_z, grid_c)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            image_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            if is_normal:
                image_normal = torch.cat([o['image_normal'].cpu() for o in out]).numpy()
            images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            images_seg = torch.cat([o['image_seg'].cpu() for o in out]).numpy()
            
            # img_fake = save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg:09d}.png'), drange=[-1,1], grid_size=grid_size)
            # img_raw = save_image_grid(image_raw, os.path.join(run_dir, f'fakes{cur_nimg:09d}_raw.png'), drange=[-1,1], grid_size=grid_size)
            img_fake = save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg:09d}.png'), drange=[-1,1], grid_size=grid_size)
            img_raw = save_image_grid(image_raw, os.path.join(run_dir, f'fakes{cur_nimg:09d}_raw.png'), drange=[-1,1], grid_size=grid_size)
            images_seg = save_seg_grid(images_seg, os.path.join(run_dir, f'fakes{cur_nimg:09d}_seg.png'), drange=[-1,1], grid_size=grid_size)
            img_depth = save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg:09d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            if is_normal:
                img_normal = save_image_grid(image_normal, os.path.join(run_dir, f'fakes{cur_nimg:09d}_normal.png'), drange=[-1,1], grid_size=grid_size)
            # img_fake = save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
            # img_raw = save_image_grid(image_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
            # img_depth = save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
            # if is_normal:
            #     img_normal = save_image_grid(image_normal, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_normal.png'), drange=[-1,1], grid_size=grid_size)
            # if is_normal:
            #     wandb.log({"ouput_images": [wandb.Image(img_fake),wandb.Image(img_raw), wandb.Image(img_depth), wandb.Image(img_normal)]})
            #     del img_normal
            # else:
            #     wandb.log({"ouput_images": [wandb.Image(img_fake),wandb.Image(img_raw), wandb.Image(img_depth)]})
            
            del img_fake
            del img_depth
            del images_seg
            del img_raw
            del out
            if is_normal:
                del img_normal
        # Save Checkpoint if needed
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % (network_snapshot_ticks*100) == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    try:
                        value = copy.deepcopy(value).eval().requires_grad_(False)
                        if num_gpus > 1:
                            misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                            for param in misc.params_and_buffers(value):
                                torch.distributed.broadcast(param, src=0)
                        snapshot_data[key] = value.cpu()
                    except:
                        snapshot_data[key] = value
                del value # conserve memory
            # save as tensors to avoid error for multi GPU
            snapshot_data['progress'] = {
                'cur_nimg': torch.LongTensor([cur_nimg]),
                'cur_tick': torch.LongTensor([cur_tick]),
                'batch_idx': torch.LongTensor([batch_idx]),
                'best_fid': best_fid,
            }

            if augment_pipe is not None:
                snapshot_data['progress']['augment_p'] = augment_pipe.p.cpu()
            if hasattr(loss, 'pl_mean'):
                snapshot_data['progress']['pl_mean'] = loss.pl_mean.cpu()

            if cur_tick % (network_snapshot_ticks) == 0:
                
                snapshot_pkl = misc.get_ckpt_path(run_dir)
                if rank == 0:
                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)
                    shutil.copyfile(snapshot_pkl, snapshot_pkl.replace('snapshot', f'snapshot_{cur_nimg//1000:06d}'))
            
        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None) and \
                    not (phase.start_event.cuda_event == 0 and phase.end_event.cuda_event == 0):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()
        
        output = {}
        for name, value in stats_dict.items():
            output[name] = value.mean
            
        if rank == 0:
            wandb.log(output)
        
        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Emtpy cache
        torch.cuda.empty_cache()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
