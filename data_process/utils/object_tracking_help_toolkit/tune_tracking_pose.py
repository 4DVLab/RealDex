from scipy.spatial.transform import Rotation as R
import numpy as np
from pathlib import Path
import json
from scipy.spatial.transform import Rotation

def load_four_cam_to_world_tranform_matrix(bag_folder):
    file_path = Path(bag_folder) / "global_name_position" / '0.txt'
    with open(file_path, 'r') as json_file:
        transform_data = json.load(json_file)
    four_cam_tranform_matrix = {
        f"cam{index}" : transform_data[f'cam{index}_rgb_camera_link']
        for index in np.arange(4)}
    return four_cam_tranform_matrix

def transform_matrix_to_seven_num(transform_matrix):
    quat_rotation = Rotation.from_matrix(transform_matrix[:3,:3]).as_quat()
    translation = transform_matrix[:3,3]
    result_concat = np.concatenate((translation,quat_rotation))
    return result_concat

def seven_num2matrix(seven_num):
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix

def load_tracking_result_transform(folder_path,frame_index,cam_index):
    bag_name = str(folder_path).split('/')[-1][:-9]
    tracking_result_file_path = Path(folder_path) / f"tracking_result/{bag_name}_cam_index_{cam_index}_tracking_result.txt"

    tracking_result = None
    with open(tracking_result_file_path,"r"):
        tracking_result = np.loadtxt(tracking_result_file_path,dtype=np.float32)
    
    seven_num_transform = np.array(tracking_result[frame_index])
    transform_matrix = seven_num2matrix(seven_num_transform)
    return transform_matrix

def from_test_file(bag_folder,frame_index,is_init = True):
    cam_index = 0
    bag_name = bag_folder.split('/')[-1][:-9]
    four_cam_tranform_matrix = load_four_cam_to_world_tranform_matrix(bag_folder)
    target_transform = None
    if is_init:
        tranform_matrix_in_world_frame = np.loadtxt(bag_folder + f'/two_merged_pcd_filter/{bag_name}_init_pose.txt')
        target_transform = tranform_matrix_in_world_frame
        print("you are using the init pose")
    else:
        position_fine_tune_transform = np.loadtxt(bag_folder + f'/two_merged_pcd_filter/test.txt')
        target_transform = position_fine_tune_transform
        # tracking_result_transform = load_tracking_result_transform(bag_folder,frame_index,cam_index)
        # print(tracking_result_transform)
        # target_transform = position_fine_tune_transform@tracking_result_transform
        print("you are using the test.txt")
    target_matrix = np.linalg.inv(four_cam_tranform_matrix[f"cam{cam_index}"]) @ target_transform

    seven_num = transform_matrix_to_seven_num(target_matrix).reshape((1,7))

    file_save_path = Path(bag_folder) / f"poses/gt_{frame_index}.txt"
    np.savetxt(file_save_path,seven_num,delimiter=' ')






if __name__ == "__main__":
    folder_path = '/home/lab4dv/data/sda/yogurt/original/yogurt_1_20231105'
    from_test_file(folder_path,347,is_init=False)




