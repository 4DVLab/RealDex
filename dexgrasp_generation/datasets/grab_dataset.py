import sys
sys.path.append(".")
from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
import time
from PIL import Image
import json
import trimesh
from torch.utils.data._utils.collate import default_collate
import time

import pytorch3d
from pytorch3d.transforms import Transform3d, axis_angle_to_matrix
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from network.models.loss import contact_map_of_m_to_n
from utils.grab_hand_model import HandModel
from manotorch.manolayer import ManoLayer, MANOOutput
from tqdm import tqdm, trange
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def transform_mat(ori, transl):
    rotmat = axis_angle_to_matrix(ori)
    t = Transform3d().rotate(rotmat).translate(transl)
    return t.get_matrix()

def feature_to_color(feature_values):
    colormap = plt.get_cmap('rainbow')  # Choose a colormap (e.g., 'viridis')
    colors = colormap(feature_values)[:, :3] # Apply the colormap and extract RGB values
    print(np.max(colors), np.min(colors))
    
    colors *= 255
    return colors
    
def export_data(obj_pc, hand_pc, contact_map, idx=0):
    print(torch.max(contact_map), torch.min(contact_map))
    colors = feature_to_color(contact_map.numpy())
    obj_cloud = trimesh.PointCloud(vertices=obj_pc, colors=colors)
    hand_cloud = trimesh.PointCloud(vertices=hand_pc)
    obj_cloud.export(f"obj_{idx}.ply")
    hand_cloud.export(f"hand_{idx}.ply")
    
class GRABDataset(Dataset):
    def __init__(self, cfg, mode='train', need_process=False):
        
        self.cfg = cfg
        self.mode = mode

        dataset_cfg = cfg["dataset"]
        self.dataset_cfg = dataset_cfg
        root_path = dataset_cfg["root_path"]
        baseDir = dataset_cfg["dataset_dir"]
        baseDir = os.path.join(root_path, baseDir)
        self.ds_path = os.path.join(baseDir, mode)
        self.num_obj_points = dataset_cfg["num_obj_points"]
        self.num_hand_points = dataset_cfg["num_hand_points"]
        
        
        # load frame name
        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        
        self.frame_names =np.asarray([file.split('/')[-1] for file in frame_names])
        self.frame_sbjs = np.asarray([file.split('/')[-2] for file in frame_names])
        self.frame_objs = np.asarray([name.split('_')[0] for name in self.frame_names])
        
        # get mano model
        self.mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False)
        self.mano_faces = self.mano_layer.get_mano_closed_faces().unsqueeze(0)
        
        # load data
        
        self.ds = {}
        self.ds['rhand_data'] = torch.load(os.path.join(self.ds_path, 'rhand_data.pt'))
        self.ds['object_data'] = torch.load(os.path.join(self.ds_path, 'object_data.pt'))
        self.ds['plane_data'] = torch.load(os.path.join(self.ds_path, 'plane_data.pt'))
        
        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(baseDir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(baseDir, 'sbj_info.npy'), allow_pickle=True).item()

        ## Hand vtemps and betas
        self.tools_path = os.path.join(baseDir, 'tools')
        if len(self.sbj_info) == 0:
            self.__load_tools__()
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
            
        if need_process:
            frame_num = len(self.frame_names)
            batch_size = 8
            self.ds['rhand_data']['points'] = torch.zeros(frame_num, self.num_hand_points, 3)
            # self.ds['rhand_data']['observed_hand_pc'] = torch.zeros(frame_num, self.num_hand_points, 3)
            
            for i in trange(frame_num // batch_size):
                start_id = i * batch_size
                end_id = (i+1) * batch_size
                end_id = frame_num if end_id > frame_num else end_id
                batch_id = list(range(start_id, end_id))
                self.__process_hand_data__(batched_idx=batch_id)
            torch.save( self.ds['rhand_data'], os.path.join(self.ds_path, 'rhand_data.pt'))
    
    def __process_hand_data__(self, batched_idx):
        # sample points on rhand
        rhand_verts = self.ds['rhand_data']['verts'][batched_idx].cuda()
        rhand_faces = self.mano_faces.expand(len(batched_idx), -1, -1).cuda()
        gt_hand_mesh = Meshes(verts=rhand_verts, faces=rhand_faces)
        gt_hand_pc = sample_points_from_meshes(
            gt_hand_mesh,
            num_samples=self.num_hand_points
        ).type(torch.float32) # torch.tensor: [NH, 3]
        self.ds['rhand_data']['points'][batched_idx] = gt_hand_pc.cpu()
        
    
    def __load_tools__(self):
        sub_path = os.path.join(self.tools_path, 'subject_meshes')
        self.sbj_info = {}
        for sid in self.sbjs:
            temp_name = sid + "_rhand.ply"
            beta_name = sid + "_rhand_betas.npy"
            for relpath, dirs, files in os.walk(sub_path):
                if temp_name in files:
                    temp_path = os.path.join(sub_path, relpath, temp_name)
                    beta_path = os.path.join(sub_path, relpath, beta_name)
                    
                    rh_vtemp = trimesh.load_mesh(temp_path).vertices
                    rh_betas = np.load(beta_path)
                    
                    self.sbj_info[sid] = {'rh_vtemp': rh_vtemp, 'rh_betas': rh_betas}
        
    def __len__(self):
        
        return len(self.frame_names)

    def __getitem__(self, idx):
        obj_data = self.ds['object_data']
        obj_transl = obj_data['transl'][idx]
        obj_pc = obj_data['verts'][idx] #[3000, 3]
        
        if self.dataset_cfg['fps']:
            obj_pc = pytorch3d.ops.sample_farthest_points(obj_pc.unsqueeze(0), K=self.num_obj_points)[0][0]  # [NO, 3]
            
        # obj_rotation = axis_angle_to_matrix(self.ds['object_data']['global_orient'][idx])
        
        sbj_name = self.frame_sbjs[idx]
        hand_beta = self.sbj_betas[sbj_name]
        
        
        # hand mano param
        rh_data = self.ds['rhand_data']
        rh_rotation_mat = axis_angle_to_matrix(rh_data['global_orient'][idx])
        
        
        global_translation = rh_data['transl'][idx]
        hand_pose = rh_data['hand_pose'][idx]
        hand_points = rh_data['points'][idx]
        
        obj_pc = obj_pc - obj_transl
        hand_points = hand_points - obj_transl
        global_translation = global_translation - obj_transl
        
        
        
        """return data should be same as dex dataset"""
        if self.cfg["network_type"] == "ipdf":
            # global_rotation_mat = plane_pose[:3, :3] @ global_rotation_mat
            obj_gt_rotation = rh_rotation_mat.T  # so that obj_gt_rotation @ obj_pc is what we want

            ret_dict = {
                "obj_pc": obj_pc,
                "obj_gt_rotation": obj_gt_rotation,
                "world_frame_hand_rotation_mat": rh_rotation_mat,
            }
        elif self.cfg["network_type"] == "glow":  # TODO: 2: glow
            canon_obj_pc = obj_pc @ rh_rotation_mat
            hand_rotation_mat = np.eye(3)
            hand_translation = global_translation @ rh_rotation_mat
            ret_dict = {
                "obj_pc": obj_pc,
                "canon_obj_pc": canon_obj_pc,
                "hand_pos": hand_pose,
                "canon_rotation": hand_rotation_mat,
                "canon_translation": hand_translation,
            }
        elif self.cfg["network_type"] == "cm_net":  # TODO: 2: Contact Map
            contact_map = contact_map_of_m_to_n(obj_pc, hand_points)  # [NO]
            
            # Canonicalize pc
            obj_pc = obj_pc @ rh_rotation_mat # R^-1 X
            hand_translation = global_translation @ rh_rotation_mat
    
            gt_hand_pc = (hand_points-global_translation) @ rh_rotation_mat  + hand_translation
            

            ret_dict = {
                "canon_obj_pc": obj_pc,
                "gt_hand_pc": gt_hand_pc,
                "contact_map": contact_map,
                "observed_hand_pc": gt_hand_pc
            }

            if self.dataset_cfg["perturb"]:
                # pert_hand_translation = hand_translation + torch.randn(3) * 0.03
                pert_pc = gt_hand_pc + torch.randn(gt_hand_pc.size()) * 0.03
                
                ret_dict["observed_hand_pc"] = pert_pc
        else:
            print("WARNING: entered undefined dataset type!")
            ret_dict = {
                "obj_pc": obj_pc,
                "hand_qpos": hand_pose,
                "world_frame_hand_rotation_mat": rh_rotation_mat,
                "world_frame_hand_translation": global_translation
            }
        ret_dict["obj_scale"] = 1
        ret_dict["beta"] = hand_beta
        return ret_dict
    
class GRABMeshData(Dataset):
    def __init__(self, cfg, mode='train', splits=None):
        self.cfg = cfg
        self.mode = mode

        dataset_cfg = cfg["dataset"]
        self.dataset_cfg = dataset_cfg
        
        root_path = dataset_cfg["root_path"]
        baseDir = dataset_cfg["dataset_dir"]
        baseDir = os.path.join(root_path, baseDir)
        self.obj_info = np.load(os.path.join(baseDir, 'obj_info.npy'), allow_pickle=True).item()
        self.all_class = self.obj_info.keys()
        if splits is None:
            self.splits = {'test': ['mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'],
                            'val': ['apple', 'toothbrush', 'elephant', 'hand'],
                            'train': []}
            for key in self.all_class:
                if key not in self.splits['test'] or key not in self.splits['val']:
                    self.splits['train'].append(key)
        else:
            self.splits = splits
        self.obj_list = self.splits[mode]
    def get_categories(self):
        
        return self.obj_list
        
    def __len__(self):
        return len(self.obj_list)
        
    def __getitem__(self, index):
        obj_class = self.obj_list[index]
        obj = self.obj_info[obj_class]
        obj_pc = torch.tensor(obj['verts_sample'])
        # print(obj['verts_sample'].shape)
        if self.dataset_cfg['fps']:
            obj_pc = pytorch3d.ops.sample_farthest_points(obj_pc.unsqueeze(0), K=self.dataset_cfg["num_obj_points"])[0][0]  # [NO, 3]
        ret_dict = {
                "obj_pc": obj_pc,
            }
        return ret_dict
        

if __name__ == '__main__':
    cfg = {
        'dataset': {
            'root_path': "/remote-home/share/yumeng/GRAB-Unidexgrasp-data/",
            'dataset_dir': "grab",
            'hand_global_trans': [0, -0.7, 0.2],
            'hand_global_rotation_xyz': [-1.57, 0, 3.14],
            'num_obj_points': 1024,
            'num_hand_points': 1024,
            'perturb': True,
            'fps': True,
        },
        
        'network_type':'cm_net'
        
    }
    # dataset = GRABDataset(cfg)
    # ret_dict = dataset[0]
    # export_data(ret_dict['canon_obj_pc'], ret_dict['gt_hand_pc'], ret_dict['contact_map'])
    dataset = MeshData(cfg)
    for i in [0, 1]:
        dataset[i]