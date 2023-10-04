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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class GRABDataset(Dataset):
    def __init__(self, baseDir, mode='train'):
        self.ds_path = os.path.join(baseDir, mode)
        
        # load frame name
        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([file.split('/')[-1] for file in frame_names])
        self.frame_sbjs = np.asarray([file.split('/')[-2] for file in frame_names])
        self.frame_objs = np.asarray([name.split('_')[0] for name in self.frame_names])
        
        # load data
        self.ds = {}
        self.ds['rhand_data'] = torch.load(os.path.join(self.ds_path, 'rhand_data.pt'))
        self.ds['next_frame_data'] = torch.load(os.path.join(self.ds_path, 'next_frame_data.pt'))
        self.ds['object_data'] = torch.load(os.path.join(self.ds_path, 'object_data.pt'))

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(baseDir, 'obj_info.npy'), allow_pickle=True).item()
        # self.sbj_info = np.load(os.path.join(baseDir, 'sbj_info.npy'), allow_pickle=True).item()

        ## Hand vtemps and betas
        self.tools_path = os.path.join(baseDir, 'tools')
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
        beta = self.sbj_betas[sbj_name]
        
        # hand mano param
        rh_data = self.ds['rhand_data']
        hand_param = torch.cat([beta, rh_data['transl'][idx], rh_data['global_orient'][idx], rh_data['fullpose'][idx]], dim=-1).float() # [61]
        
        # next frame
        nf_data = self.ds['next_frame_data']
        next_frame_hand = torch.cat([beta, nf_data['transl'][idx], nf_data['global_orient'][idx], nf_data['fullpose'][idx]], dim=-1).float() # [61]
        
        return (obj_pc, hand_param, next_frame_hand, obj_cmap)
    
                