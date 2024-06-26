
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


def load_camera_intrics(folder_path, cam_index=0):
    camera_param = camera_param_init()
    camera_info_path = Path(folder_path) / \
        Path(f"cam{cam_index}/rgb/camera_info/info.txt")
    with open(camera_info_path, "r") as json_file:
        camera_info = json.load(json_file)
    camera_param = ros_camera_info_to_camera_param(camera_info)
    return camera_param


def load_camera_extrinsic(folder_path,cam_index):
    global_name_postion_path = Path(
        folder_path) / Path("global_name_position/0.txt")
    global_name_postion = None
    with open(global_name_postion_path, "r") as json_file:
        global_name_postion = json.load(json_file)
    return global_name_postion[f"cam{cam_index}_rgb_camera_link"]


def load_camera_param(folder_path, cam_index=0):
    # load_intrisics
    camera_param = load_camera_intrics(folder_path, cam_index)
    camera_param["extrinsic"] = np.linalg.inv(
        load_camera_extrinsic(folder_path,cam_index))
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


def viz_project_object_to_2d(object_folder,my_cam=True,cam_index = 0):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    camera_param_path = object_folder / \
        Path("camera_param.json")
    if my_cam:
        camera_param = load_camera_param(object_folder,cam_index)
        o3d_params = change_o3d_camera_param(camera_param_path, camera_param)
        json.dump(o3d_params, open(camera_param_path, "w"), indent=4)
        print("use own camera pram")
    camera_params = o3d.io.read_pinhole_camera_parameters(
        str(camera_param_path))
    

    image_save_folder = object_folder / f"icg_capture_image"

    os.makedirs(image_save_folder, exist_ok=True)
    for object_index in np.arange(223):
        object_path = object_folder / \
            Path(f"object_pose_in_every_frame/{object_index}.ply")
        mesh = o3d.io.read_triangle_mesh(str(object_path))
        mesh.paint_uniform_color([1, 0, 0])
        if object_index != 0:
            vis.clear_geometries()
        vis.add_geometry(mesh)
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(False)

        plt.imsave(image_save_folder /
                   Path(f"{object_index}.png"), np.asarray(image), dpi=1)

    vis.destroy_window()




# 也需要只有一个arm_hand_mesh的
if __name__ == "__main__":

    folder_path = Path("/home/lab4dv/data/bags/yangtao/yangtao_3_20231210")

    viz_project_object_to_2d(folder_path,3)

