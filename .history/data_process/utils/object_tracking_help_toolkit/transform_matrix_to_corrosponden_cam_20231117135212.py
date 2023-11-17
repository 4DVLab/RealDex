from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import json


def load_four_cam_to_world_tranform_matrix(bag_folder):
    file_path = Path(bag_folder) / "global_name_position" / '0.txt'
    with open(file_path, 'r') as json_file:
        transform_data = json.load(json_file)
    four_cam_tranform_matrix = [for index in np.arange(4) transform_data['cam2_rgb_camera_link']]



bag_folder = '/media/tony/新加卷/yyx_tmp'

tranform_matrix_in_world_frame = np.loadtxt(bag_folder + '/init_pose.txt')

four_cam_tranform_matrix = np.loadtxt(bag_folder + '/four_cam_pose.txt')
