import shutil
from pathlib import Path
import os

root_path = "/home/lab4dv/data/bags/apple/apple_1_20231105"


for i in range(4):
    rgb_cam_path = Path(root_path) / Path("cam"+str(i)+"/rgb/camera_info/info.txt")
    depth_cam_path = Path(root_path) / Path("cam"+str(i)+"/depth_to_rgb/camera_info/info.txt")
    shutil.copyfile(rgb_cam_path, Path("/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data/cam"+str(i)+"_rgb.txt"))
    shutil.copyfile(depth_cam_path, Path("/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data/cam"+str(i)+"_depth.txt"))



