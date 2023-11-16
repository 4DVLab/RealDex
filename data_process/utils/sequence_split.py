import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import struct
import ctypes
from cv_bridge import CvBridge
import cv2
import json
from pathlib import Path
import json
import copy
import threading
import shutil
import filecmp


def sequence_split(root_path):
    see_time_list = False
    rename_split_file = False
    
    print(root_path)
    good_sequence_num = 0
    root_path = Path(root_path)

    # to avoid chaos, mv original directory to sub-directory
    original_path = root_path / Path("original")
    if not original_path.exists():
        os.mkdir(original_path)
    
        for folder in os.listdir(root_path):
            # read the segment text file
            if not folder.endswith(".md") and not folder.endswith("original"):
                shutil.move(root_path/Path(folder), original_path/Path(folder))

    for folder in os.listdir(original_path):
        # read the segment text file
        if folder.endswith(".md"):
            continue

        if not folder.endswith("yogurt_1_20231105"):
            continue
        print("\n", original_path/Path(folder))
 
        
        split_file_path = original_path/Path(folder)/Path(folder.split(folder.split("_")[-1])[0]+"segment.txt")
        tracking_file_path = original_path/Path(folder)/Path(folder.split(folder.split("_")[-1])[0]+"tracking.txt")
        if rename_split_file:
            old_split_file_path = Path()
            shutil.move(old_split_file_path, split_file_path)
           
            continue

        if not split_file_path.exists():
            print(split_file_path, "do not exist")
            continue

        if not tracking_file_path.exists():
            print(tracking_file_path, "do not exist")
            continue
        
        rgb_timestamp_list = []
        depth_timestamp_list = []


        for cam_num in range(4):
            rgb_timestamp_list.append(np.loadtxt(original_path/Path(folder+"/cam"+str(cam_num)+"/rgb/image_raw/info.txt"), dtype=np.float128))
            depth_timestamp_list.append(np.loadtxt(original_path/Path(folder+"/cam"+str(cam_num)+"/depth_to_rgb/image_raw/info.txt"),  dtype=np.float128))

        tracking_list = np.loadtxt(tracking_file_path , dtype=np.float128)

        print(len(tracking_list))

        with open(split_file_path ) as f:
            lines = f.readlines()
        for line in lines:
            if line.endswith("end\n") or line in "\n" or line.endswith("touch\n"):
                continue
            line = line.replace(" ", "")
            line = line.replace("\n", "")
            if line.find("\t\t")!=-1:
                time_list=line.split("\t\t")
            else:
                time_list = line.split("\t")
            if see_time_list:
                print(split_file_path)
                print(time_list)
                continue
            start_time = int(time_list[0])
            end_time = int(time_list[1])
            good_sequence_num = good_sequence_num + 1
            # print(good_sequence_num)
            # mv sequences to single directory and rename
            sequence_dir = root_path / Path(str(root_path).split("/")[-1]+"_"+str(good_sequence_num))
            if not sequence_dir.exists():
                os.mkdir(sequence_dir)
                for cam_num in range(4):
                    os.mkdir(sequence_dir/Path("cam"+str(cam_num)))
                    os.mkdir(sequence_dir/Path("cam"+str(cam_num)+"/rgb"))
                    os.mkdir(sequence_dir/Path("cam"+str(cam_num)+"/depth_to_rgb"))
            print(sequence_dir)
            for cam_num in range(4):
                print("cam"+str(cam_num))
                line_rgb_timestamp = rgb_timestamp_list[cam_num][start_time:end_time+1]
                line_depth_timestamp = depth_timestamp_list[cam_num][start_time: end_time+1]
                line_tracking = tracking_list[start_time:end_time+1]
                tmp_num = 0
                for index in range(end_time - start_time +1):
                    tmp_num = tmp_num + 1
                    #shutil.move(original_path/Path(folder+"/cam"+str(cam_num)+"/rgb/image_raw/"+str(start_time+index)+".png"), sequence_dir / Path("cam"+str(cam_num)+"/rgb/"+str(index)+".png"))
                    #shutil.move(original_path/Path(folder+"/cam"+str(cam_num)+"/depth_to_rgb/image_raw/"+str(start_time+index)+".png"), sequence_dir / Path("cam"+str(cam_num)+"/depth_to_rgb/"+str(index)+".png"))
                    shutil.copy2(original_path/Path(folder+"/cam"+str(cam_num)+"/rgb/image_raw/"+str(start_time+index)+".png"), sequence_dir / Path("cam"+str(cam_num)+"/rgb/"+str(index)+".png"))
                    shutil.copy2(original_path/Path(folder+"/cam"+str(cam_num)+"/depth_to_rgb/image_raw/"+str(start_time+index)+".png"), sequence_dir / Path("cam"+str(cam_num)+"/depth_to_rgb/"+str(index)+".png"))
                print(len(line_depth_timestamp), tmp_num)
                return
                np.savetxt(sequence_dir/Path("cam"+str(cam_num)+"/rgb")/Path(str(root_path).split("/")[-1]+"_"+str(good_sequence_num)+"_rgb.txt"), line_rgb_timestamp )
                np.savetxt(sequence_dir/Path("cam"+str(cam_num)+"/depth_to_rgb")/Path(str(root_path).split("/")[-1]+"_"+str(good_sequence_num)+"_depth.txt"), line_depth_timestamp )
                np.savetxt(sequence_dir/Path(str(root_path).split("/")[-1]+"_"+f"{good_sequence_num}_tracking.txt"), line_tracking)
root = "/home/lab4dv/data/sda/"
for folder in os.listdir(root):
    print(folder)
    # if folder in ["urdf", "lost+found" , ".Trash-1000", "flower_bread"]:
    if folder not in "yogurt":
        continue
    sequence_split(Path(root)/Path(folder))

    



# need
# i can select the sequence i want to split
# at the same time, i can select a root path to process the split sequence