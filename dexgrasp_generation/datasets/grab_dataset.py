import sys
from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
from utils import utils
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def transform_mat(ori, transl):
    rotmat = axis_angle_to_matrix(ori)
    t = Transform3d().rotate(rotmat).translate(transl)
    return t.get_matrix()
    

class GRABDataset(Dataset):
    def __init__(self, baseDir, mode='train'):
        self.ds_path = os.path.join(baseDir, mode)
        
        # load frame name
        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([file.split('/')[-1] for file in frame_names])
        self.frame_sbjs = np.asarray([file.split('/')[-2] for file in frame_names])
        self.frame_objs = np.asarray([name.split('_')[0] for name in self.frame_names])
        
        # get mano model
        self.mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False)
        self.mano_faces = self.mano_layer.get_mano_closed_faces()
        
        # load data
        
        self.ds = []
        
        

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(baseDir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(baseDir, 'sbj_info.npy'), allow_pickle=True).item()

        ## Hand vtemps and betas
        self.tools_path = os.path.join(baseDir, 'tools')
        if len(self.sbj_info) == 0:
            self.__load_tools__()
        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
            
    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]) for k in data.keys}
        return data_torch
    
    def load_disk(self,idx):

        if isinstance(idx, int):
            return self._np2torch(self.frame_names[idx])

        frame_names = self.frame_names[idx]
        from_disk = []
        for f in frame_names:
            from_disk.append(self._np2torch(f))
        from_disk = default_collate(from_disk)
        return from_disk
    
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
                
        obj_pc = self.ds['object_data']['points'][idx] #[3000, 3]
        
        obj_cmap = self.ds['object_data']['contact'][idx] # [3000, 1]
        obj_cmap = obj_cmap > 0

        sbj_name = self.frame_sbjs[idx]
        hand_beta = self.sbj_betas[sbj_name]
        
        # object data
        obj_data = self.ds['object_data'] 
        
        
        # hand mano param
        rh_data = self.ds['rhand_data']
        global_rotation_mat = axis_angle_to_matrix(rh_data['global_orient'])
        global_translation = rh_data['transl']
        hand_pose = rh_data['fullpose']
        
        # table data
        plane_data = self.ds['plane_data']
        plane_ori = plane_data['global_orient']
        plane_transl = plane_data['transl']
        
        """return data should be same as dex dataset"""
        if self.cfg["network_type"] == "ipdf":
            plane_pose = plane_data['transform']
            global_rotation_mat = plane_pose[:3, :3] @ global_rotation_mat
            obj_gt_rotation = global_rotation_mat.T  # so that obj_gt_rotation @ obj_pc is what we want
            # place the table horizontally
            obj_pc = obj_pc @ plane_pose[:3, :3].T + plane_pose[:3, 3]

            ret_dict = {
                "obj_pc": obj_pc,
                "obj_gt_rotation": obj_gt_rotation,
                "world_frame_hand_rotation_mat": global_rotation_mat,
            }
        elif self.cfg["network_type"] == "glow":  # TODO: 2: glow
            canon_obj_pc = obj_pc @ global_rotation_mat
            hand_rotation_mat = np.eye(3)
            hand_translation = global_translation @ global_rotation_mat
            ret_dict = {
                "obj_pc": obj_pc,
                "canon_obj_pc": canon_obj_pc,
                "hand_pos": hand_pose,
                "canon_rotation": hand_rotation_mat,
                "canon_translation": hand_translation,
            }
        elif self.cfg["network_type"] == "cm_net":  # TODO: 2: Contact Map
            # Canonicalize pc
            obj_pc = obj_pc @ global_rotation_mat
            hand_rotation_mat = np.eye(3)
            hand_translation = global_translation @ global_rotation_mat

            # gt_hand_mesh = Meshes(verts=rh_data['verts'], faces=)
            # gt_hand_pc = sample_points_from_meshes(
            #     gt_hand_mesh,
            #     num_samples=self.num_hand_points
            # ).type(torch.float32).squeeze()  # torch.tensor: [NH, 3]
            
            gt_hand_pc = rh_data['verts']
            contact_map = contact_map_of_m_to_n(obj_pc, gt_hand_pc)  # [NO]

            ret_dict = {
                "canon_obj_pc": obj_pc,
                "gt_hand_pc": gt_hand_pc,
                "contact_map": contact_map,
                "observed_hand_pc": gt_hand_pc
            }

            if self.dataset_cfg["perturb"]:
                pert_hand_translation = hand_translation + np.random.randn(3) * 0.03
                pert_pos = hand_pose + np.random.randn(len(hand_pose)) * 0.1
                
                pert_hand_model: MANOOutput = self.mano_layer(pert_pos, hand_beta)
                
                ret_dict["observed_hand_pc"] = pert_hand_model.verts
        else:
            print("WARNING: entered undefined dataset type!")
            ret_dict = {
                "obj_pc": obj_pc,
                "hand_qpos": hand_pose,
                "world_frame_hand_rotation_mat": global_rotation_mat,
                "world_frame_hand_translation": global_translation
            }
        ret_dict["obj_scale"] = 1
        return ret_dict
    
                