
from pathlib import Path
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation
import json
import matplotlib.pylab as plt


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


def load_camera_extrinsic(global_position_path, cam_index):
    with open(global_position_path, "r") as json_file:
        global_name_position = json.load(json_file)
    link_name = f"cam{cam_index}_rgb_camera_link"
    return global_name_position[link_name]


def load_camera_param(data_dir, cam_index=0, frame_id=0):
    # load_intrisics
    camera_param = camera_param_init()
    camera_info_path = os.path.join(data_dir, f"cam{cam_index}/rgb/camera_info/info.txt")
    
    with open(camera_info_path, "r") as json_file:
        camera_info = json.load(json_file)
    camera_param = ros_camera_info_to_camera_param(camera_info)
    
    global_position_path = os.path.join(data_dir, "global_name_position", f"{frame_id}.txt")
    camera_param["extrinsic"] = np.linalg.inv(load_camera_extrinsic(global_position_path, cam_index))
    return camera_param



def change_param(intrinsic):
    length = np.sqrt(len(intrinsic))
    length = round(length)
    intrinsic = np.array(intrinsic).reshape((length, length))
    intrinsic = intrinsic.T
    intrinsic = intrinsic.flatten()
    intrinsic = intrinsic.tolist()
    return intrinsic


def change_o3d_camera_param(o3d_camera_param_path, mine_camera_param):

    o3d_camera_param = json.load(open(o3d_camera_param_path, "r"))
    extrisic = [item for list_item in mine_camera_param["extrinsic"]
                for item in list_item]
    o3d_camera_param["extrinsic"] = change_param(extrisic)
    # mine_camera_param["intrinsic"][2] = mine_camera_param["width"] / 2 -0.5
    # mine_camera_param["intrinsic"][5] = mine_camera_param["height"] / 2 -0.5
    o3d_camera_param["intrinsic"]["intrinsic_matrix"] = change_param(
        mine_camera_param["intrinsic"])
    o3d_camera_param["intrinsic"]["width"] = mine_camera_param["width"]
    o3d_camera_param["intrinsic"]["height"] = mine_camera_param["height"]
    return o3d_camera_param

def tf_to_mat(tf):
    transl = tf[:3]
    rot = Rotation.from_quat(tf[3:])
    mat = np.zeros((4, 4))
    mat[:3, :3] = rot.as_matrix()
    mat[:3, -1] = transl
    mat[-1, -1] = 1
    return mat



def viz_project_object_to_2d(data_dir, object_name, pose_dir, cam_index, frame_id=0):

    camera_param_path = os.path.join(pose_dir, "camera_param.json")
    if not os.path.exists(camera_param_path):
        camera_param = load_camera_param(data_dir, cam_index)
        o3d_params = change_o3d_camera_param(camera_param_path, camera_param)
        json.dump(o3d_params, open(camera_param_path, "w"), indent=4)
        
    camera_params = o3d.io.read_pinhole_camera_parameters(str(camera_param_path))
    
    object_path = os.path.join(data_dir, "models", f"{object_name}.obj")
    mesh = o3d.io.read_triangle_mesh(str(object_path))
    mesh.paint_uniform_color([1, 0, 0])
    
    # lose poses
    pose = np.loadtxt(os.path.join(pose_dir, "pose.txt"))
    pose = tf_to_mat(pose)
    global_position_path = os.path.join(data_dir, "global_name_position", f"{frame_id}.txt")
    cam_mat = load_camera_extrinsic(global_position_path, cam_index=0)
    cam_mat = np.array(cam_mat)
    cam_mat_inv = np.linalg.inv(cam_mat)
    print(cam_mat)
    mesh = mesh.transform(cam_mat_inv @ pose @ cam_mat )
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(False)
    
    # merge two images
    background_img = plt.imread(os.path.join(data_dir, f"cam{cam_index}", "rgb/image_raw", f"{frame_id}.png"))
    print(np.array(image).shape)
    merged_img = np.asarray(background_img) * 0.6 + np.asarray(image) * 0.4
    
    plt.imsave(os.path.join(pose_dir, f"output.png"), merged_img, dpi=1)
    vis.destroy_window()




# 也需要只有一个arm_hand_mesh的
if __name__ == "__main__":

    data_folder = "/Users/yumeng/Working/data/CollectedDataset/test_1_obj_pose"
    pose_dir = os.path.join(data_folder, "pose_labeling")

    viz_project_object_to_2d(data_folder, "yogurt", pose_dir, cam_index=3)