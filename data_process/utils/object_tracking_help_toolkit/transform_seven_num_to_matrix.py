import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import json

def seven_num2matrix(seven_num):#translation x,y,z rotation x,y,z,w
    translation = seven_num[:4]
    roatation = seven_num[4:]
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix

def matrix_2_seven_num(matrix):
    quat = Rotation.from_matrix(matrix[:3,:3]).as_quat()
    seven_num = np.concatenate((quat, matrix[:3, 3]))
    return seven_num

def load_global_transforms(folder_path):
    file_path = Path(folder_path) / f"global_name_position/0.txt"
    with open(file_path,"r") as json_reader:
        transforms = json.load(json_reader)
    return transforms

def get_new_cam_transform(original_camera_index,target_cam_index,tracking_index,bag_folder_path,transform_file_name):
    tracking_result = np.loadtxt(Path(bag_folder_path) / f"tracking_result/tracking_result.txt")
    transform_matrix = seven_num2matrix(tracking_result[tracking_index])
    global_transforms = load_global_transforms(bag_folder_path)
    new_cam_transform_matrix = global_transforms[f"cam{target_cam_index}_rgb_camera_link"] @ np.linalg.inv(global_transforms[f"cam{original_camera_index}_rgb_camera_link"]) @ transform_matrix
    np.savetxt(bag_folder_path / f"poses/{transform_file_name}",new_cam_transform_matrix)













if __name__ == "__main__":
    camera_index = 0
    tracking_index = 100
    bag_folder_path =""
    transform_file_name = ""
    original_camera_index = 0
    target_cam_index = 1
    get_new_cam_transform(original_camera_index,target_cam_index,tracking_index,bag_folder_path,transform_file_name)