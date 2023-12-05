import rosbag
import numpy as np
import os
from pathlib import Path 
import shutil
from tqdm import tqdm
from collections import defaultdict
import open3d as o3d

from utils.generate_multiple_view_ply import gen_merge_pcd_ply




if __name__ == "__main__":
    floder_path = Path("/home/lab4dv/data/bags/test/test_3")
    four_cam_intrisics_extrisics_save_folder = Path(
        "/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data")
    merge_class = gen_merge_pcd_ply(floder_path, four_cam_intrisics_extrisics_save_folder)

    pcd = merge_class.gen_pcd_with_depth_and_rgb_paths(47, 0)

    o3d.visualization.draw_geometries([pcd])
    