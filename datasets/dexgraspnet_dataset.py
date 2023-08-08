
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
    def __init__(self, data_dir):
        self.data_path = os.path.join(data_dir, "dataset")
        self.grasp_list = os.listdir(self.data_path)
        self.obj_mesh_path = os.path.join(data_dir, "meshdata")
        
        # For single object
        self.grasp_code = "core-bottle-1a7ba1f4c892e2da30711cdbdbc73924"
        self.grasp_data = np.load(os.path.join(self.data_path, self.grasp_code + ".npy"), allow_pickle=True)
        self.object_mesh_origin = os.path.join(self.obj_mesh_path, self.grasp_code, "coacd/decomposed.obj")


    def __len__(self):
        return len(self.grasp_data)
    
    def __getitem__(self, index):
        # grasp_code = self.grasp_list[index].split(".")[0]
        qpos = self.grasp_data[index]['qpos']

        rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in ShadowHandModel.rot_names]))
        rot = rot[:, :2].T.ravel().tolist()
        hand_pose = torch.tensor([qpos[name] for name in ShadowHandModel.translation_names] + rot + [qpos[name]
                            for name in ShadowHandModel.joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
        return hand_pose


if __name__ == '__main__':
    data_dir = "/Users/yumeng/Working/Project2023/data/dexgraspnet"

    # sys.path.append("./models/")
    hand_file = "./mjcf/shadow_hand_wrist_free.xml"
    hand_model = ShadowHandModel(hand_file, "./mjcf/meshes", device="cpu")

    ds = DexGraspNetDataset(data_dir=data_dir)
    for i in range(3):
        print(ds[i])
        print(ds[i].shape)