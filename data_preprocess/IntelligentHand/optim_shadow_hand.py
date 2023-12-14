import os
import numpy as np
import torch
import tqdm
from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_matrix
from models.shadow_hand_builder import ShadowHandBuilder
from copy import deepcopy
from utils.loss import SR_Loss
import trimesh

def to_cpu(t):
    return t.detach().cpu()
    

def estimate_qpos(init_pose, tgt_pc):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sr_builder = ShadowHandBuilder(device=device,
                                   mjcf_path="assets/mjcf/shadow_hand_vis.xml"
                                )
    tgt_pc = tgt_pc.to(device)
    
    '''Set up param'''
    init_rotation = init_pose['global_rotation'].to(device) # [3,3]
    global_rotation = torch.zeros(3).to(device)
    global_rotation.requires_grad_(True)
    
    init_transl = init_pose['global_transl'].to(device)
    global_transl = torch.zeros(3).to(device)
    global_transl.requires_grad_(True)
    
    init_qpos = init_pose['qpos'].to(device)
    qpos = init_qpos
    qpos.requires_grad_(True)
    
    param = []
    param.append({"params": [global_rotation, global_transl, qpos]})
    # param.append({"params": [global_transl]})
    
    '''Setup loss'''
    sr_loss = SR_Loss(sr_builder)
    sr_loss.setup_tgt(tgt_pc)
    
    
    '''Setup optimizer'''
    optimizer = torch.optim.Adam(param, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    proc_bar = tqdm.tqdm(range(500))

    for i, _ in enumerate(proc_bar):
        optimizer.zero_grad()
        
        rot = axis_angle_to_matrix(global_rotation) @ init_rotation
        transl = init_transl + global_transl
        ret_dict = sr_builder.get_hand_model(rot, transl, qpos, without_arm=False)
        sr_meshes = ret_dict['meshes']
        sr_points = torch.concat(ret_dict['sampled_pts'])
        sr_pts_normals = torch.concat(ret_dict['sampled_pts_normal'])
        loss = sr_loss.pc_diff_loss(sr_points, sr_pts_normals)
        
        # proc_bar.set_description(f"transl: {transl_loss.item():.5f}, recon: {recon_loss.item():.5f}")
        proc_bar.set_description(f"loss: {loss.item():.5f}")
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    print(param)
    
    meshes = trimesh.Trimesh(vertices=to_cpu(ret_dict['meshes'].verts_packed()), 
                             faces=to_cpu(ret_dict['meshes'].faces_packed()))
    
    return meshes
    

        


if __name__ == "__main__":
    data_dir = "/remote-home/share/yumeng/our_data/single_frame/"
    out_dir = '/remote-home/share/yumeng/our_data/single_frame/optimized_results'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data = torch.load(os.path.join(data_dir, "init_pose.pt"))
    pc = trimesh.load(os.path.join(data_dir, "pc_transform_to_world/0_cropped.ply"))
    pc = torch.tensor(pc.vertices).float()
    meshes = estimate_qpos(data, pc)
    meshes.export(os.path.join(out_dir, "./hand.ply"))

    
