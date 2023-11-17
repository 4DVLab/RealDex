import open3d as o3d
from pathlib import Path
import numpy as np

def load_pcd_legth(pcd_path):
    pcd_time_stamp_path = pcd_path / Path("info.txt")
    pcd_time_stamp = np.loadtxt(pcd_time_stamp_path)
    return pcd_time_stamp.shape[0]

def show_obj_env_pcd(bag_folder_path):
    env_pcd_folder = Path(bag_folder_path) / Path("merged_pcd_filter")
    pcd_length = load_pcd_legth(env_pcd_folder)
    obj_folder = Path(bag_folder_path) / Path("object_pose_in_every_frame")

    for index in np.arange(pcd_length):



if __name__ == "__main__":
    bag_folder_path = "/media/tony/新加卷/yyx_tmp"
    viz_
    show_obj_env_pcd(bag_folder_path)
