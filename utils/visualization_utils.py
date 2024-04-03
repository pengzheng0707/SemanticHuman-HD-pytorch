'''
Helper functions for visualizing results of AG3D.
'''

import torch
import numpy as np
import cv2
import trimesh
import imageio
from imageio import imwrite
from utils.mesh_renderer import render_trimesh
from torch_utils.misc import crop_face
from training.zp import mixing_noise
import PIL.Image
import os

def tensor2seg(tensor, path):
    tensor = tensor.permute(0,2,3,1)[0]
    tensor = tensor.cpu().numpy()
    tensor = mask_to_png(tensor)
    img = PIL.Image.fromarray(tensor.astype(np.uint8))
    cmap = np.load('new_cmap.npy')
    img.putpalette(cmap.tolist())
    img.save(path)

def mask_to_png(sample_seg):
    back_seg = -1 + np.ones_like(sample_seg)[...,-1:]*0.01
    sample_seg = np.concatenate([back_seg,sample_seg],-1)
    sample_seg = np.argmax(sample_seg, axis=-1)
    return sample_seg

def rectify_pose(pose, rot):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_rot = cv2.Rodrigues(rot)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_rot.dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    
    return pose


def rectify_pose2(pose, root_abs):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.
    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_abs = cv2.Rodrigues(root_abs)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = np.linalg.inv(R_abs).dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    
    return pose

def calculate_rotation(rotation_angle, spacing):
    """
    
    Args:
      rotation_angle: 0-360 
      spacing: 
    Returns:
      angle_list: list of rotation angles
    """
    angle_list1 = np.array(range(0, rotation_angle, spacing))
    angle_list2 = np.array(range(rotation_angle, -rotation_angle, -spacing))
    angle_list3 = np.array(range(-rotation_angle, 1, spacing))
    angle_all = np.concatenate([angle_list1, angle_list2, angle_list3])
    
    return angle_all


def gen_samples(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, _i, cano=False):
    
    rendering = []

    if cano:
        
        c[:,-82:-10] = 0
        c[:,29] = np.pi
        c[:,26:29] = 0
        c[:,25] = 1
        c[:,27] = 0.15
        
    output = G(z=z, c=c, truncation_psi=truncation)
    
    if is_img:
        img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        # rendering.append(img)   
        imageio.imsave(save_path, img)
    
    if is_img_raw:
        
        img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        # img_raw = cv2.resize(img_raw, (res,res))
        # rendering.append(img_raw)
        imageio.imsave(save_path.replace('.png','_raw.png'), img_raw)
    
    if is_normal:
        
        img_normal = (output['image_normal'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        # img_normal = cv2.resize(img_normal, (res,res))
        # rendering.append(img_normal)
        imageio.imsave(save_path.replace('.png','_normal.png'), img_normal)
        
    if is_mesh:
        
        mesh = G.get_mesh(z, c, voxel_resolution=300,truncation_psi=truncation)[0]
        mesh_verts = mesh.vertices
        mesh_faces = mesh.faces
        mesh_new = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
        img_mesh = render_trimesh(mesh_new)[:,:,:3]
        img_mesh = cv2.resize(img_mesh, (res,res))
        rendering.append(img_mesh)                

    tensor2seg(output['image_seg'], save_path.replace('.png', '_seg.png'))

    torch.save(c,save_path.replace('.png','_c.pt'))
    torch.save(z,save_path.replace('.png','_z.pt'))

def gen_pose(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, cano=False):
    
    rendering = []
    if cano:
        
        c[:,-82:-10] = 0
        c[:,29] = np.pi
        c[:,26:29] = 0
        c[:,25] = 1
        c[:,27] = 0.15
        
    output = G.synthesis(ws=z, c=c, is_test=True)
    
    if is_img:
        img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        rendering.append(img)   
    
    if is_img_raw:
        
        img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        imageio.imsave(save_path.replace('.png','_raw.png'), img_raw)
    
    if is_normal:
        
        img_normal = (output['image_normal'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        imageio.imsave(save_path.replace('.png','_normal.png'), img_normal)
        
    if is_mesh:
        
        mesh = G.get_mesh(z, c, voxel_resolution=300,truncation_psi=truncation)[0]
        mesh_verts = mesh.vertices
        mesh_faces = mesh.faces
        mesh_new = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
        img_mesh = render_trimesh(mesh_new)[:,:,:3]
        img_mesh = cv2.resize(img_mesh, (res,res))
        rendering.append(img_mesh)                

    rendering = np.concatenate(rendering, axis=1)
    imageio.imsave(save_path, rendering)

    tensor2seg(output['seg_raw'], save_path.replace('.png', '_seg.png'))
    torch.save(c,save_path.replace('.png','_c.pt'))
    torch.save(z,save_path.replace('.png','_z.pt'))

def gen_semantic(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, cano=False, new_c_map=None, idx=0):
    rendering = []
    new_z = mixing_noise(1, G.z_dim, prob=0, device=z[0].device)
    new_z_map = G.mapping(new_z, c, truncation_psi=truncation)
    if True:
        new_z_map = G.mapping(new_z, new_c_map, truncation_psi=truncation)
    i = 1   # semantic index, 1 is for tops
    if idx != 0:
        z[:,28*i:28*(i+1),:] = new_z_map[:,28*i:28*(i+1),:]
    if cano:
        
        c[:,-82:-10] = 0
        c[:,29] = np.pi
        c[:,26:29] = 0
        c[:,25] = 1
        c[:,27] = 0.15
        
    output = G.synthesis(ws=z, c=c)
    if is_img:
        img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        rendering.append(img)   
    
    if is_img_raw:
        
        img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        img_raw = cv2.resize(img_raw, (res,res))
        rendering.append(img_raw)
    
    if is_normal:
        
        img_normal = (output['image_normal'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
        imageio.imsave(save_path.replace('.png','_normal.png'), img_normal)
        
    if is_mesh:
        
        mesh = G.get_mesh(z, c, voxel_resolution=300,truncation_psi=truncation)[0]
        mesh_verts = mesh.vertices
        mesh_faces = mesh.faces
        mesh_new = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
        img_mesh = render_trimesh(mesh_new)[:,:,:3]
        img_mesh = cv2.resize(img_mesh, (res,res))
        rendering.append(img_mesh)                
    
    rendering = np.concatenate(rendering, axis=1)
    imageio.imsave(save_path, rendering)

def gen_novel_view(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, angles):
    rot = c[:,29:32].data.cpu().numpy()[0]
    if is_mesh:
        mesh = G.get_mesh(z, c, voxel_resolution=300,truncation_psi=truncation)[0]
        mesh_verts = mesh.vertices
        mesh_faces = mesh.faces
    
    images = []
    n_view = len(angles)
    
    for k in range(n_view):
        
        rendering = []
        angle = np.pi * angles[k] / 180
        
        c_new = c.clone()
        rot_mat = cv2.Rodrigues(np.array([0, angle, 0]))[0]
        c_new = c.clone()
        c_new[:,29:32] = torch.from_numpy(rectify_pose(rot.copy(), np.array([0,-angle,0]))).to(c_new.device)

        output = G.synthesis(ws=z, c=c_new)
        
        if is_img:
            
            img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            rendering.append(img)
        
        if is_img_raw:
            
            img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            img_raw = cv2.resize(img_raw, (res,res))[:,res//4:-res//4,:3]
            rendering.append(img_raw)
            
        if is_normal:
            
            img_normal = (output['image_normal'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            imageio.imsave(save_path.replace('.mp4',f'_{k}_normal.png'), img_normal)
            
        if is_mesh:
            mesh_new = trimesh.Trimesh(vertices=np.einsum('ij,nj->ni',rot_mat,mesh_verts), faces=mesh_faces)    
            img_mesh = render_trimesh(mesh_new)[:,:,:3]
            rendering.append(img_mesh)


        all = np.concatenate(rendering, axis=1)
        imageio.imsave(save_path.replace('.mp4',f'_{k}.png'), all)
        tensor2seg(output['image_seg'], save_path.replace('.mp4',f'seg_{k}.png'))

def gen_video(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, angles):
    
    w = G.mapping(z, c, truncation_psi=truncation)

    rot = c[:,29:32].data.cpu().numpy()[0]
    if is_mesh:
        mesh = G.get_mesh(z, c, voxel_resolution=300,truncation_psi=truncation)[0]
        mesh_verts = mesh.vertices
        mesh_faces = mesh.faces
    
    images = []
    n_view = len(angles)
    
    for k in range(n_view):
        
        rendering = []
        angle = np.pi * angles[k] / 180
        
        c_new = c.clone()
        rot_mat = cv2.Rodrigues(np.array([0, angle, 0]))[0]
        c_new = c.clone()
        c_new[:,29:32] = torch.from_numpy(rectify_pose(rot.copy(), np.array([0,-angle,0]))).to(c_new.device)

        output = G.synthesis(ws=w, c=c_new)
        
        if is_img:
            
            img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            rendering.append(img)
        
        if is_img_raw:
            
            img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            img_raw = cv2.resize(img_raw, (res,res))[:,res//4:-res//4,:3]
            rendering.append(img_raw)
            
        if is_normal:
            
            img_normal = (output['image_normal'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            rendering.append(img_normal)
            
        if is_mesh:
            mesh_new = trimesh.Trimesh(vertices=np.einsum('ij,nj->ni',rot_mat,mesh_verts), faces=mesh_faces)    
            img_mesh = render_trimesh(mesh_new)[:,:,:3]
            rendering.append(img_mesh)

        all = np.concatenate(rendering, axis=1)
        images.append(all)

    imageio.mimwrite(save_path, images)
    torch.save(c,save_path.replace('.mp4','_c.pt'))
    torch.save(z,save_path.replace('.mp4','_z.pt'))

def gen_interp(G, z1, z2, c1, c2, truncation, res, is_img, is_img_raw, is_normal, is_mesh, save_path, n_steps, device):
    
    i = 2

    w1, w2 = G.mapping(z1, c1, truncation_psi=truncation), G.mapping(z2, c2, truncation_psi=truncation)
    cs = torch.lerp(c1, c1, torch.linspace(0, 1, n_steps, device=device)[:,None].expand(-1,c1.shape[-1]))
    ws = torch.lerp(w1, w2, torch.linspace(0, 1, n_steps, device=device)[:,None,None].expand(-1,168,512))
    # zs = torch.lerp(z1, z2, torch.linspace(0, 1, n_steps, device=device)[:,None].expand(-1,z1.shape[-1]))
    images = []
    normals = []
    segs = []
    for k in range(n_steps):
        w_new = w1.clone()
        # w_new[:,28*i:28*(i+1),:] = ws[[k]][:,28*i:28*(i+1),:]
        w_new = ws[[k]]
        rendering = []
        output = G.synthesis(ws=w_new, c=cs[k].unsqueeze(0))
    
        if is_img:
            img = (output['image'][:,:,:,256:768] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            images.append(img)
        
        if is_img_raw:
            
            img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            imageio.imsave(save_path.replace('.png','_raw.png'), img_raw)
        
        if is_normal:
            
            img_normal = (output['image_normal'][:,:,:,256:768] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            normals.append(img_normal)

            
        if is_mesh:
            
            mesh = G.get_mesh(zs[k], cs[[k]], ws=ws[[k]], voxel_resolution=300,truncation_psi=truncation)[0]
            img_mesh = render_trimesh(mesh)[:,:,:3]
            mesh_verts = mesh.vertices
            mesh_faces = mesh.faces
            mesh_new = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces)
            img_mesh = render_trimesh(mesh_new)[:,:,:3]
            img_mesh = cv2.resize(img_mesh, (res,res))
            rendering.append(img_mesh)                
    
        if True:
            tensor = output['image_seg'][:,:,:,256:768].permute(0,2,3,1)[0]
            tensor = tensor.cpu().numpy()
            segs.append(tensor)

    images = np.concatenate(images, axis=1)
    imageio.imsave(save_path, images)

    normals = np.concatenate(normals, axis=1)
    imageio.imsave(save_path.replace('.png', '_normal.png'), normals)

    segs = np.concatenate(segs, axis=1)
    segs = mask_to_png(segs)
    img = PIL.Image.fromarray(segs.astype(np.uint8))
    cmap = np.load('new_cmap.npy')
    img.putpalette(cmap.tolist())
    img.save(save_path.replace('.png','_seg.png'))

def amss_to_smpl_param(f):
    
    smpl_params_all = []
    smplx_to_smpl = list(range(66)) + [72, 73, 74, 117, 118, 119]  # SMPLH to SMPL
    
    
    smpl_params_all = np.zeros( (f['poses'].shape[0], 86) )
    smpl_params_all[:,0] = 1
    
    if f['poses'].shape[-1] == 72:
        smpl_params_all[:, 4:76] = f['poses']
        
        
    elif f['poses'].shape[-1] == 156:
        smpl_params_all[:, 4:76] = f['poses'][:,smplx_to_smpl]
        
    root_abs = smpl_params_all[0, 4:7].copy()
    for i in range(smpl_params_all.shape[0]):
        smpl_params_all[i, 4:7] = rectify_pose2(smpl_params_all[i, 4:7], root_abs)

    smpl_params_all = torch.tensor(smpl_params_all).float().cuda()
    smpl_params_all = smpl_params_all[::3]   
    
    return smpl_params_all     
    

def gen_anim(G, z, c, truncation,res, is_img, is_img_raw, is_normal, is_mesh, save_path, motion_path):
    
    f = np.load(motion_path)
    smpl_params_all = amss_to_smpl_param(f)
    w = G.mapping(z, c, truncation_psi=truncation)
    images = []
    n_frame= min(100*3, smpl_params_all.shape[0])
    spacing = 3
    
    for k in range(np.int(n_frame/spacing*0.5),np.int(n_frame/spacing)):
    # for k in range(np.int(n_frame/spacing)):
        
        rendering = []
        
        c_new = c.clone()
        c_new[0,25:111] = smpl_params_all[k*spacing].clone()
        angle = c_new[:,29:32].data.cpu().numpy()[0]
        angle = rectify_pose(angle.copy(), np.array([0,-np.pi,0]))
        angle = rectify_pose(angle.copy(), np.array([0,0,-np.pi]))
        c_new[:,29:32] = torch.from_numpy(angle).to(c.device)
        c_new[:,25] *= 0.9
        output = G.synthesis(ws=w, c=c_new)
        
        if is_img:
            
            img = (output['image'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            rendering.append(img)
        
        if is_img_raw:
            
            img_raw = (output['image_raw'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            rendering.append(img_raw)
        
        if is_normal:
            
            normal = (output['image_normal'] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
            rendering.append(normal)
            
        if is_mesh:
            mesh = G.get_mesh(z, c_new, ws=w, voxel_resolution=300,truncation_psi=truncation)[0]
            mesh_verts = mesh.vertices
            mesh_faces = mesh.faces
            mesh_new = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces) 
            img_mesh = render_trimesh(mesh_new)[:,:,:3]
            rendering.append(img_mesh)

        all = np.concatenate(rendering, axis=1)
        images.append(all)

    imageio.mimwrite(save_path, images)
    
    torch.save(c,save_path.replace('.mp4','_c.pt'))
    torch.save(z,save_path.replace('.mp4','_z.pt'))
    
    
    
        
        

    
    

    
    
    
    
    