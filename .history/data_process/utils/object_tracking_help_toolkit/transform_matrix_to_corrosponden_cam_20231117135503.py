from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import json


def load_four_cam_to_world_tranform_matrix(bag_folder):
    file_path = Path(bag_folder) / "global_name_position" / '0.txt'
    with open(file_path, 'r') as json_file:
        transform_data = json.load(json_file)
    four_cam_tranform_matrix = {
        f"cam{index}" : transform_data[f'cam{index}_rgb_camera_link']
        for index in np.arange(4)}
    return four_cam_tranform_matrix



if __name__ == "__main__":
    cam = 0
    bag_folder = '/media/tony/新加卷/yyx_tmp'

    tranform_matrix_in_world_frame = np.loadtxt(bag_folder + '/init_pose.txt')

    four_cam_tranform_matrix = load_four_cam_to_world_tranform_matrix(
        bag_folder)
    print("the matrix you want", @tranform_matrix_in_world_frame)
