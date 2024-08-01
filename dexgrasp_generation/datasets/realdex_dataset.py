import torch
from torch.utils.data import Dataset

import numpy as np

import transforms3d
import pytorch3d
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_axis_angle

import glob
import json

import sys
import os
from os.path import join as pjoin
from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from network.models.loss import contact_map_of_m_to_n


def split_data(split_type='object'):
    if split_type == 'object':
        train = ['blue_magnet_toy', 'body_lotion', 'crisps', 'dust_cleaning_sprayer', 
                    'laundry_detergent', 'toilet_cleaning_sprayer']
        val = ['goji_jar', 'small_sprayer', 'yogurt']
        # test = ['duck_toy', 'cosmetics', 'sprayer']
        test = ['duck_toy', 'cosmetics', 'sprayer', 'goji_jar', 'small_sprayer', 'yogurt']
        
        split_dict = {'train': train,
                      'val': val,
                      'test': test}
    return split_dict

def str_to_ascii_tensor(s):
    # Convert string to a list of ASCII values
    ascii_values = [ord(c) for c in s]
    # Create a tensor from the list of ASCII values
    tensor = torch.tensor(ascii_values, dtype=torch.int8)
    return tensor

def ascii_tensor_to_str(tensor):
    # Convert each integer in the tensor to the corresponding character
    # and then join them into a single string
    return ''.join(chr(c) for c in tensor)

class RealDexDataset(Dataset):
    def __init__(self, cfg, mode):
        super(RealDexDataset, self).__init__()

        self.cfg = cfg
        self.mode = mode

        dataset_cfg = cfg["dataset"]

        self.root_path = dataset_cfg["root_path"]
        self.dataset_cfg = dataset_cfg
        self.num_obj_points = dataset_cfg["num_obj_points"]
        self.num_hand_points = dataset_cfg["num_hand_points"]
        # if originally self.categories is None, it means we want all categories
        self.categories = dataset_cfg.get("categories", None)
 
        self.object_name_list = split_data('object')[mode]
        self.file_list = self.get_file_list(self.root_path,
                                            self.object_name_list)
        print(f"Total Sequence Num: {len(self.file_list)}.")
        
        self._load_data()

    def __len__(self):
        return self.data['qpos'].shape[0]

    def __getitem__(self, item):
        """
        :param item: int
        :return:
        """
        qpos = self.data['qpos'][item]
        hand_transl = self.data['hand_transl'][item]
        hand_orient = self.data['hand_orient'][item]
        obj_pc = self.data['object_points'][item]
        obj_name = self.data['object_names'][item]
        object_orient = self.data['object_orient'][item]
        object_transl = self.data['object_transl'][item]
        
        
        if self.cfg["network_type"] == "affordance_cvae":
            ret_dict = {
                "obj_pc": obj_pc,
                "hand_qpos": qpos,
                "rotation": hand_orient,
                "translation": hand_transl,
                "object_orient": object_orient,
                "object_transl": object_transl,
                # "object_name": obj_name
            }
            
        ret_dict["obj_scale"] = 1
        return ret_dict, obj_name


    def get_file_list(self, root_dir, obj_list):
        """
        :param root_dir: e.g. "./data/"
                dataset_dir: "RealDex"
        :return: root_dir + model_name + **.npz
        """
        file_list = []
        for obj in obj_list:
            file_dir_path = pjoin(root_dir, self.dataset_cfg["dataset_dir"], obj)
            files = list(filter(lambda file: file[0] != "." and (file.endswith(".npz")),
                                    os.listdir(file_dir_path)))
            files = list(map(lambda file: pjoin(file_dir_path, file),
                                files))
            # print(obj, len(files))
            file_list += files 

        return file_list
    
    def _load_data(self):
        
        self.data = {
            'qpos': [],
            'hand_transl': [],
            'hand_orient': [],
            'object_transl': [],
            'object_orient': []
            # 'object_points': []
        }
        
        obj_list = []
        obj_name_list = []
        for file in tqdm(self.file_list):
            seq_data = np.load(file, allow_pickle=True)
            seq_len = seq_data['qpos'].shape[0]
            obj_pc = torch.tensor(seq_data['object_points']).unsqueeze(0)
            obj_pc = obj_pc.expand([seq_len, -1, -1])
            obj_list.append(obj_pc)
            
            obj_name = file.split('/')[-2]
            obj_name_list += [obj_name] * seq_len
            
            for key in self.data:
                self.data[key].append(torch.tensor(seq_data[key]))
        
        self.data['object_points'] = obj_list                
        self.data = {key:torch.cat(self.data[key], dim=0) for key in self.data}
        
        self.data['object_names'] = obj_name_list
        
        
        print(f"Split {self.mode}: {self.data['qpos'].shape[0]} pieces of data.")         


class RealDexMeshData(Dataset):
    def __init__(self, cfg, mode):
        super(RealDexMeshData, self).__init__()
        self.cfg = cfg
        self.object_name_list = split_data('object')[mode]
        