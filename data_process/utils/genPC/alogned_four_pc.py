import open3d as o3d
import numpy as np
import json
from pathlib import Path 
import os
import shutil






def find_time_closet(slot,time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index




def gen_four_pc_merge(bag_folder:str):
    cam_num = 4

    pc_merge_folder = Path(bag_folder) / 
    for cam_index in np.arange(cam_num):
        