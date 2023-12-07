import rosbag
import xml.etree.ElementTree as ET
from pathlib import Path
from cv_bridge import CvBridge
import cv2
import re
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pylab as plt
import open3d as o3d
from tqdm import tqdm
import math
import copy
import os
from numba import jit
import scipy
from scipy.spatial.transform import Rotation
import pywavefront
import sensor_msgs.point_cloud2 as pc2
import ctypes
import time
import struct
from collections import defaultdict
import copy
import json
from pprint import pprint


def camera_param_init():
    camera_param = {
        "intrisics": None,
        "extrisics": None,
        "width": None,
        "height": None,
        "distortion": None
    }
    return camera_param


def ros_camera_info_to_camera_param(camera_info):
    camera_param = camera_param_init()
    camera_param["intrinsic"] = camera_info["K"]
    camera_param["width"] = camera_info["width"]
    camera_param["height"] = camera_info["height"]
    camera_param["distortion"] = camera_info["D"]
    return camera_param


def load_camera_intrics(folder_path, cam_index=0):
    camera_param = camera_param_init()
    camera_info_path = Path(folder_path) / \
        Path(f"cam{cam_index}/rgb/camera_info/info.txt")
    with open(camera_info_path, "r") as json_file:
        camera_info = json.load(json_file)
    camera_param = ros_camera_info_to_camera_param(camera_info)
    return camera_param


def load_camera_extrinsic(folder_path):
    global_name_postion_path = Path(
        folder_path) / Path("global_name_position/0.txt")
    global_name_postion = None
    with open(global_name_postion_path, "r") as json_file:
        global_name_postion = json.load(json_file)
    return global_name_postion["cam0_rgb_camera_link"]


def load_camera_param(folder_path, cam_index=0):
    # load_intrisics
    camera_param = load_camera_intrics(folder_path, cam_index)
    camera_param["extrinsic"] = np.linalg.inv(
        load_camera_extrinsic(folder_path))
    return camera_param



def chang_trinsic(trinsic):
    length = np.sqrt(len(trinsic))
    length = round(length)
    trinsic = np.array(trinsic).reshape((length, length))
    trinsic = trinsic.T
    trinsic = trinsic.flatten()
    trinsic = trinsic.tolist()
    return trinsic


def change_o3d_camera_param(o3d_camera_param_path, mine_camera_param):

    o3d_camera_param = json.load(open(o3d_camera_param_path, "r"))
    extrisic = [item for list_item in mine_camera_param["extrinsic"]
                for item in list_item]
    o3d_camera_param["extrinsic"] = chang_trinsic(extrisic)
    # mine_camera_param["intrinsic"][2] = mine_camera_param["width"] / 2 -0.5
    # mine_camera_param["intrinsic"][5] = mine_camera_param["height"] / 2 -0.5
    o3d_camera_param["intrinsic"]["intrinsic_matrix"] = chang_trinsic(
        mine_camera_param["intrinsic"])
    o3d_camera_param["intrinsic"]["width"] = mine_camera_param["width"]
    o3d_camera_param["intrinsic"]["height"] = mine_camera_param["height"]
    return o3d_camera_param


def viz_project_object_to_2d(object_folder,my_cam=True):

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    camera_param_path = object_folder / \
        Path("project_arm_hand_to_2d/camera_params.json")
    if my_cam:
        camera_param = load_camera_param(bag_folder)
        o3d_params = change_o3d_camera_param(camera_param_path, camera_param)
        json.dump(o3d_params, open(camera_param_path, "w"), indent=4)
    camera_params = o3d.io.read_pinhole_camera_parameters(
        str(camera_param_path))
    
    file_num = len(os.listdir(object_folder))

    for object_index in np.arange(file_num):
        object_path = object_folder / Path(f"{object_index}.obj")
        mesh = o3d.io.read_triangle_mesh(str(object_path))
        vis.add_geometry(mesh)
        if object_index != 0:
            vis.clear_geometries()
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True)

        vis.update_renderer()

    vis.destroy_window()





def simplify_mesh(mesh, proportion=0.5):
    mesh = mesh.simplify_quadric_decimation(
        int(proportion * len(mesh.triangles)))
    mesh = mesh.remove_unreferenced_vertices()
    return mesh



# 也需要只有一个arm_hand_mesh的
if __name__ == "__main__":

    ros_prefix_path = "/media/tony/T7/camera_data/configuration/hand_arm_mesh"
    configuration_path = "/media/tony/T7/camera_data/configuration"
    bag_folder = "/media/tony/T7/yyx_tmp/for_dust_cleanning_sprayer_tracking/dust_cleanning_spreyer/dust_cleanning_spreyer_1_20231105"

