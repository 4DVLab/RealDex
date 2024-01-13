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

class IntelligentHand(Dataset):
    def __init__(self, base_dir, mode="train", load_on_ram = True, vis=False):

        self.mode = mode
        self.base_dir = base_dir
        self.file_list = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                full_path = os.path.join(root, file)
                self.file_list.append(full_path)
                
        if load_on_ram:
            self.__load_dataset__()

        self.sample_nPoint = 3000
        self.batch_size = batch_size

    def __load_dataset__(self):
        print('loading dataset start')
        for file in self.file_list:
            npz_data = np.load(file)
            
            
        print('loading dataset finish')


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # obj_pc
        obj_pc = torch.tensor(self.all_obj_pc[idx], dtype=torch.float32)  # [4, 3000]

        # obj cmap contactdb
        obj_cmap = torch.tensor(self.all_obj_cmap[idx])  # [3000, 10]
        obj_cmap = obj_cmap > 0

        # hand mano param
        hand_param = torch.tensor(self.all_hand_param[idx], dtype=torch.float32)  # [61]

        return (obj_pc, hand_param, obj_cmap)

if __name__ == '__main__':
    base_dir = "/storage/group/4dvlab/yumeng/IntelligentHand/collected_data/"
