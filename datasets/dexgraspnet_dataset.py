
import sys
sys.path.append(".")
sys.path.append("..")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from models.hand_model import ShadowHandModel
import transforms3d

class DexGraspNetDataset(Dataset):
    def __init__(self, data_dir, split_type='train'):
        self.data_path = os.path.join(data_dir, "dataset")
        self.name_list = os.listdir(self.data_path)
        self.obj_mesh_path = os.path.join(data_dir, "meshdata")
        
        # For single object
        # self.grasp_code = "core-bottle-1a7ba1f4c892e2da30711cdbdbc73924"
        # self.grasp_data = np.load(os.path.join(self.data_path, self.grasp_code + ".npy"), allow_pickle=True)
        # self.object_mesh_origin = os.path.join(self.obj_mesh_path, self.grasp_code, "coacd/decomposed.obj")

        # For multiple object
        self.grasp_data = []
        with open(os.path.join(data_dir, "splits", f"{split_type}.txt")) as f:
            names = [line.split(".")[0] for line in f]
            for grasp_code in names:
                seq = np.load(os.path.join(self.data_path, grasp_code + ".npy"), allow_pickle=True)
                for i in range(len(seq)):
                    self.grasp_data.append((grasp_code, i))



    def __len__(self):
        return len(self.grasp_data)
    
    def __getitem__(self, index):
        grasp_code, frame_id = self.grasp_data[index]
        obj_path = os.path.join(self.obj_mesh_path, self.grasp_code, "coacd/decomposed.obj")

        frame_data = np.load(os.path.join(self.data_path, grasp_code + ".npy"), allow_pickle=True)[frame_id]
        qpos = frame_data['qpos']
        obj_scale = frame_data['scale']

        rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in ShadowHandModel.rot_names]))
        rot = rot[:, :2].T.ravel().tolist()

        transl = torch.tensor([qpos[name] for name in ShadowHandModel.translation_names], dtype=torch.float).unsqueeze(0)
        
        hand_pose = torch.tensor([qpos[name] for name in ShadowHandModel.joint_names], dtype=torch.float).unsqueeze(0)
        
        
        return hand_pose, transl, rot, obj_path, obj_scale
    

def split_train_val_test(dataset, split_dir, train_percent=0.6, val_percent=0.2, test_percent=0.2, random_seed=42):
    # Creating data indices for training and validation splits:   
    size = len(dataset.name_list)
    print(size)
    indices = list(range(size))
    val_num = int(val_percent * size)
    test_num = int(test_percent * size)
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split_indices = {'val': indices[:val_num], 'test': indices[val_num:val_num+test_num], 'train': indices[val_num+test_num:]}

    os.makedirs(split_dir, exist_ok=True)

    for data_type in ['train', 'val', 'test']:
        with open(os.path.join(split_dir, f"{data_type}.txt"), "w") as f:
            for i in split_indices[data_type]:
                f.write(dataset.name_list[i])  
                f.write("\n")      


if __name__ == '__main__':
    data_dir = "/Users/yumeng/Working/Project2023/data/dexgraspnet"

    # sys.path.append("./models/")
    hand_file = "./mjcf/shadow_hand_wrist_free.xml"
    hand_model = ShadowHandModel(hand_file, "./mjcf/meshes", device="cpu")

    ds = DexGraspNetDataset(data_dir=data_dir)
    # split_train_val_test(ds, os.path.join(data_dir, "splits"))
    print(len(ds))