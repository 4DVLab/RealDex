import torch
from torch.utils.data import Dataset

import numpy as np

import transforms3d
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import matrix_to_axis_angle

import glob
import json

import sys
import os
from os.path import join as pjoin

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from network.models.loss import contact_map_of_m_to_n
from datasets.shadow_hand_builder import ShadowHandBuilder


def plane2pose(plane_parameters):
    r3 = np.array(plane_parameters[:3])
    r2 = np.zeros_like(r3)
    r2[0], r2[1], r2[2] = (-r3[1], r3[0], 0) if r3[2] * r3[2] <= 0.5 else (-r3[2], 0, r3[0])
    r1 = np.cross(r2, r3)
    pose = np.zeros([4, 4], dtype=np.float32)
    pose[0, :3] = r1
    pose[1, :3] = r2
    pose[2, :3] = r3
    pose[2, 3] = np.array(plane_parameters[3])
    pose[3, 3] = 1
    return pose


def load_splits(root_folder, categories=None):
    """
    :return: split_dict[category]["train"/"test"]: ["1aj2u9a10s43dia8qwe11oqwe", ...]
    """
    split_dict = {}
    if categories is None:
        # load the splits for all categories
        split_paths = glob.glob(os.path.join(root_folder, "DFCData", "splits/*.json"))
        for split_p in split_paths:
            category = os.path.basename(split_p).split('.json')[0]
            splits = json.load(open(split_p, 'r'))
            split_dict[category] = {}
            split_dict[category]['train'] = [obj_p for obj_p in splits['train']]
            split_dict[category]['test'] = [obj_p for obj_p in splits['test']]
    else:
        for category in categories:
            splits_path = pjoin(root_folder, "DFCData", "splits", category + ".json")
            splits_data = json.load(open(splits_path, "r"))
            split_dict[category] = {}
            split_dict[category]["train"] = [obj_p for obj_p in splits_data['train']]
            split_dict[category]["test"] = [obj_p for obj_p in splits_data['test']]

    return split_dict


class DFCDataset(Dataset):
    def __init__(self, cfg, mode):
        super(DFCDataset, self).__init__()

        self.cfg = cfg
        self.mode = mode

        dataset_cfg = cfg["dataset"]

        self.root_path = dataset_cfg["root_path"]
        self.dataset_cfg = dataset_cfg
        self.num_obj_points = dataset_cfg["num_obj_points"]
        self.num_hand_points = dataset_cfg["num_hand_points"]
        # if originally self.categories is None, it means we want all categories
        self.categories = dataset_cfg.get("categories", None)

        # For constructing ShadowHandBuilder
        if cfg["use_Shadow"]:
            self.hand_mesh_dir = pjoin(self.root_path, dataset_cfg["shadow_hand_mesh_dir"])
            self.hand_urdf_path = pjoin(self.root_path, dataset_cfg["shadow_urdf_path"])
            self.hand_builder = ShadowHandBuilder(self.hand_mesh_dir,
                                                  self.hand_urdf_path)
        else:
            # use Adroit
            # self.hand_mesh_dir = pjoin(self.root_path, dataset_cfg["adroit_hand_mesh_dir"])
            # self.hand_urdf_path = pjoin(self.root_path, dataset_cfg["adroit_urdf_path"])
            # self.hand_builder = AdroitHandBuilder(self.hand_mesh_dir,
            #                                       self.hand_urdf_path)
            raise NotImplementedError("Adroit is not supported yet")

        self.splits = [mode]

        # if originally self.categories is None, it means we want all categories
        self.splits_data = load_splits(self.root_path, self.categories)
        self.categories = list(self.splits_data.keys())
        self.file_list = self.get_file_list(self.root_path,
                                            self.splits_data,
                                            self.splits,
                                            self.categories)
        

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        """
        :param item: int
        :return:
        """
        file_path = self.file_list[item]  # e.g., "./data/DFCData/datasetv3.1/core/bottle-asdasfja12jaios9012/00000.npz"
        instance_no: str = file_path.split("/")[-2]
        category = file_path.split("/")[-3]  # e.g., core
        recorded_data = np.load(file_path, allow_pickle=True)

        qpos_dict = recorded_data["qpos"].item()
        global_translation = np.array([qpos_dict['WRJTx'], qpos_dict['WRJTy'], qpos_dict['WRJTz']])  # [3]
        global_rotation_mat = np.array(
            transforms3d.euler.euler2mat(qpos_dict['WRJRx'], qpos_dict['WRJRy'], qpos_dict['WRJRz']))  # [3, 3]
        object_scale = recorded_data["scale"]

        qpos = self.hand_builder.qpos_dict_to_qpos(qpos_dict)

        plane = recorded_data["plane"]
        obj_pc_path = pjoin(self.root_path, "DFCData", "meshes",
                            category, instance_no, "pcs_table.npy")
        pose_path = pjoin(self.root_path, "DFCData", "meshes",
                            category, instance_no, "poses.npy")
        pcs_table = torch.tensor(np.load(obj_pc_path, allow_pickle=True), dtype=torch.float)
        
        pose_matrices = torch.tensor(np.load(pose_path, allow_pickle=True), dtype=torch.float)
        index = (torch.tensor(plane[:3], dtype=torch.float) - pose_matrices[:, 2, :3]).norm(dim=1).argmin()
        pose_matrix = pose_matrices[index]
        pose_matrix[:2, 3] = 0
        pc = (pcs_table[index] @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]) / recorded_data['scale'].item()

        object_pc = pc[:3000]
        obj_pc = (object_pc - pose_matrix[:3, 3] / recorded_data['scale'].item()) @ pose_matrix[:3, :3]  # [N, 3]
                
        if self.cfg["network_type"] == "affordance_cvae":
            plane_pose = plane2pose(plane)
            # place the table horizontally
            obj_pc = obj_pc @ plane_pose[:3, :3].T + plane_pose[:3, 3]
            global_rotation_mat = plane_pose[:3, :3] @ global_rotation_mat
            # rotation_6d = matrix_to_rotation_6d(global_rotation_mat)
            hand_translation = global_translation @ plane_pose[:3, :3].T
            rotation = matrix_to_axis_angle(torch.tensor(global_rotation_mat).unsqueeze(0))[0]
            
            ret_dict = {
                "obj_pc": obj_pc,
                "hand_qpos": qpos,
                "rotation": rotation,
                "translation": hand_translation
            }
            
        else:
            print("WARNING: entered undefined dataset type!")
            ret_dict = {
                "obj_pc": obj_pc,
                "hand_qpos": qpos,
                "world_frame_hand_rotation_mat": global_rotation_mat,
                "world_frame_hand_translation": global_translation
            }
        ret_dict["obj_scale"] = object_scale
        return ret_dict


    def get_file_list(self, root_dir, splits_data, splits, categories):
        """
        :param root_dir: e.g. "./data"
        :return: e.g. ["./data/DFCData/bottle/poses/1a7ba1f4c892e2da30711cdbdbc73924/00000.npz", ...]
                 or ["./data/DFCData/poses/bottle/1a7ba1f4c892e2da30711cdbdbc73924/00000.npz"]
        """
        file_list = []
        for category in categories:
            for split in splits:
                for instance in splits_data[category][split]:
                    file_dir_path = pjoin(root_dir, self.dataset_cfg["dataset_dir"], "poses", category, instance)
                    files = list(filter(lambda file: file[0] != "." and (file.endswith(".npz")),
                                        os.listdir(file_dir_path)))
                    files = list(map(lambda file: pjoin(file_dir_path, file),
                                     files))
                    file_list += files

        print(f"Split {splits}: {len(file_list)} pieces of data.")
        return file_list
