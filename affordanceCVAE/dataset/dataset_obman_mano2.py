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

