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
from datasets.utils import read_annotation


#

import numpy as np
import torch
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
import os

import time
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

class GRABDataset(Dataset):
    def __init__(self, baseDir, mode='train'):
        seqName = None
        anno = read_annotation(baseDir, seqName, id, mode)
        
        self.ds_path = os.path.join(baseDir, mode)
        self.ds = self._np2torch(os.path.join(self.ds_path,'grabnet_%s.npz'%mode))

        frame_names = np.load(os.path.join(self.ds_path, 'frame_names.npz'))['frame_names']
        self.frame_names =np.asarray([file.split('/')[-1] for file in frame_names])
        self.frame_sbjs = np.asarray([file.split('/')[-2] for file in frame_names])
        self.frame_objs = np.asarray([name.split('_')[0] for name in self.frame_names])

        self.sbjs = np.unique(self.frame_sbjs)
        self.obj_info = np.load(os.path.join(baseDir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(baseDir, 'sbj_info.npy'), allow_pickle=True).item()

        ## bps_torch data

        bps_fname = os.path.join(baseDir, 'bps.npz')
        self.bps = torch.from_numpy(np.load(bps_fname)['basis']).to(dtype)
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
        return data_out
        
        

class obman(Dataset):
    def __init__(self, img_root='/hand-object-3/download/dataset/ObMan/obman',
                 obj_root='/hand-object-3/download/dataset',
                 mode="train", vis=False, batch_size=160):

        self.mode = mode
        self.obj_pc_path = '/hand-object-3/download/dataset/ObMan/obman/processed/obj_pc_{}.npy'.format(mode)
        self.obj_cmap_path = '/hand-object-3/download/dataset/ObMan/obman/processed/obj_cmap_contactdb_{}.npy'.format(mode)
        self.hand_param_path = '/hand-object-3/download/dataset/ObMan/obman/processed/hand_param_{}.npy'.format(mode)
        self.__load_dataset__()

        self.dataset_size = self.all_obj_pc.shape[0]

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000
        self.batch_size = batch_size

    def __load_dataset__(self):
        print('loading dataset start')
        self.all_obj_pc = np.load(self.obj_pc_path)  # [S, 4, 3000]
        self.all_obj_cmap = np.load(self.obj_cmap_path)  # [S, 3000, 10]
        self.all_hand_param = np.load(self.hand_param_path)
        print('loading dataset finish')


    def __len__(self):
        return self.dataset_size - (self.dataset_size % self.batch_size)  # in case of unmatched mano batch size

    def __getitem__(self, idx):
        # obj_pc
        obj_pc = torch.tensor(self.all_obj_pc[idx], dtype=torch.float32)  # [4, 3000]

        # obj cmap contactdb
        obj_cmap = torch.tensor(self.all_obj_cmap[idx])  # [3000, 10]
        obj_cmap = obj_cmap > 0

        # hand mano param
        hand_param = torch.tensor(self.all_hand_param[idx], dtype=torch.float32)  # [61]

        return (obj_pc, hand_param, obj_cmap)

