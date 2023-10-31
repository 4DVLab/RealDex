import numpy as np
import open3d as o3d
from pathlib import Path
import os
import json

def find_time_closet(slot,time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index

def load_transform(folder_path):
    transform_path = Path(folder_path) / Path("global_name_position/0.txt")
    with open(transform_path,"r") as json_file:
        data = json.load(json_file)
    four_pcd_transform = [data[f"cam{index}_rgb_camera_link"]  for index in np.arange(4)]
    return four_pcd_transform

def load_pcd(folder,cam_num,pcd_index):

    pcd_path = Path(folder) / Path(f"cam{cam_num}/points2/{pcd_index}.ply")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    return pcd 

def four_cam_pc_align(bag_path = "/media/tony/新加卷/camera_data/banana/"):
    

    bag_path = Path(bag_path)
    cam_num = 4

    four_cam_time_stamps = [np.loadtxt(str(bag_path/Path(f"cam{index}/points2/info.txt"))) for index in np.arange(cam_num)]

    os.makedirs(bag_path / Path("merge_env_pcd"),exist_ok=True)
    print(bag_path / Path("merge_env_pcd"))
    for time_index in np.arange(four_cam_time_stamps[0].shape[0]):

        slot = four_cam_time_stamps[0][time_index]
        
        indexs = [find_time_closet(slot,time_stamps) for time_stamps in four_cam_time_stamps]

        transform_data = load_transform(bag_path)
        pcds = [load_pcd(bag_path,cam_inex,index) for cam_inex,index in enumerate(indexs)]
        pcds = [pcds[pcd_index].transform(transform_data[pcd_index]) for pcd_index in np.arange(cam_num)]
        merge_pcd = o3d.geometry.PointCloud()
        for pcd in pcds:
            merge_pcd += pcd
        
        merge_pcd_path = bag_path / Path(f"merge_env_pcd/{time_index}.ply")
        o3d.io.write_point_cloud(str(merge_pcd_path),merge_pcd)






if __name__ == "__main__":
    four_cam_pc_align()