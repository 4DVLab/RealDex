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

from torch.utils.data._utils.collate import default_collate
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class GRABDataset(Dataset):
    def __init__(self, baseDir, mode='train', load_on_ram=True):
        self.ds_path = os.path.join(baseDir, mode)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%mode))

        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([file.split('/')[-1] for file in frame_names])
        self.frame_sbjs = np.asarray([file.split('/')[-2] for file in frame_names])
        self.frame_objs = np.asarray([name.split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(baseDir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(baseDir, 'sbj_info.npy'), allow_pickle=True).item()

        ## Hand vtemps and betas

        self.sbj_vtemp = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_vtemp'] for sbj in self.sbjs]))
        self.sbj_betas = torch.from_numpy(np.asarray([self.sbj_info[sbj]['rh_betas'] for sbj in self.sbjs]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)

        self.load_on_ram = False
        if load_on_ram:
            self.ds = self[:]
            self.load_on_ram = True
            
    def _np2torch(self,ds_path):
        data = np.load(ds_path, allow_pickle=True)
        data_torch = {k:torch.tensor(data[k]) for k in data.files}
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

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]
        # return len(self.frame_names)

    def __getitem__(self, idx):

        data_out = {k: self.ds[k][idx] for k in self.ds.keys()}
        if not self.only_params:
            if not self.load_on_ram:
                form_disk = self.load_disk(idx)
                data_out.update(form_disk)
                
        obj_name = self.frame_objs[idx]
        obj_pc = data_out['object_data']['points'].float().T #[3, 3000]
        obj_pc = torch.stack([obj_pc, torch.ones(1, obj_pc.shape[1])]) # [4, 3000]
        
        obj_cmap = data_out['object_data']['contact'] # [3000, 10]
        obj_cmap = obj_cmap > 0

        # hand mano param
        rh_data = data_out['rhand_data']
        hand_param = torch.cat([rh_data['transl'], rh_data['global_orient'], rh_data['fullpose']], dim=0).float() # [61]
        
        # next frame
        nf_data = data_out['next_frame_data']
        next_frame_hand = torch.cat([nf_data['transl'], nf_data['global_orient'], nf_data['fullpose']], dim=0).float() # [61]
        
        
        return (obj_pc, hand_param, next_frame_hand, obj_cmap)
