from scipy.spatial.transform import Rotation as R
import numpy as np

def load_four_cam_to_world_tranform_matrix(bag_folder):
    file_path = bag_folder + '/four_cam_pose.txt'




bag_folder = '/media/tony/新加卷/yyx_tmp'

tranform_matrix_in_world_frame = np.loadtxt(bag_folder + '/init_pose.txt')

four_cam_tranform_matrix = np.loadtxt(bag_folder + '/four_cam_pose.txt')
