import open3d as o3d
import numpy as np
from pathlib import Path
import os
import json


def load_one_cam_pc(folder,cam_index,pc_index):
    post_path = f"cam{cam_index}/points2/{pc_index}.ply"
    pc_path = Path(folder)/post_path

    pc = o3d.io.read_point_cloud(str(pc_path))
    return pc

def load_four_cam_pc(folder,pc_index):

    cam_num = 4
    all_pc = []
    for cam_index in np.arange(cam_num):
        pc = load_one_cam_pc(folder,cam_index,pc_index)
        all_pc.append(pc)
    return all_pc

def load_one_cam_pc_global_transform(folder,cam_index,time_index = 0):
    post_path = f"global_name_position/{time_index}.txt"
    global_transform_path = Path(folder)/post_path
    with open(global_transform_path) as json_reader:
        all_global_transform = json.load(json_reader)
    cam_link_name = f"cam{cam_index}_rgb_camera_link"
    cam_global_transform = all_global_transform[cam_link_name]
    cam_global_transform = np.array(cam_global_transform)
    return cam_global_transform

def load_four_pc_global_transforms(folder,time_index = 0):
    cam_num = 4
    all_cam_global_transforms = []
    for cam_index in np.arange(cam_num):
        cam_global_transform = load_one_cam_pc_global_transform(folder,cam_index,time_index)
        all_cam_global_transforms.append(cam_global_transform)
    return all_cam_global_transforms

def transform_four_pc_to_world(folder,pc_index,result_post_folder_path):
    four_cam_pc_transforms = load_four_pc_global_transforms(folder)
    all_pcs = load_four_cam_pc(folder,pc_index)
    all_pcs = [pc.transform(four_cam_pc_transforms[pc_index]) for pc_index,pc in enumerate(all_pcs)]
    result_folder = Path(folder) / f"{result_post_folder_path}"
    os.makedirs(result_folder,exist_ok=True)
    for cam_index,pc in enumerate(all_pcs):
        result_path = result_folder / f"{cam_index}.ply"
        o3d.io.write_point_cloud(str(result_path), pc)


if __name__ == "__main__":
    folder_path = "/media/tony/T7/camera_data/banana"
    pc_index = 1015
    result_post_folder_path = "transforme_to_world_pc"
    transform_four_pc_to_world(folder_path,pc_index,result_post_folder_path)
