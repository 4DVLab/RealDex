import open3d as o3d
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
import os
from time import sleep


def load_pcd_legth(bag_folder_path):
    pcd_time_stamp_path = bag_folder_path / Path("cam0/rgb/image_raw/info.txt")
    pcd_time_stamp = np.loadtxt(pcd_time_stamp_path)
    return pcd_time_stamp.shape[0]


def show_obj_env_pcd(bag_folder_path, viz_camera_info_path,constrain_bound,sleep_time_):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    camera_params = o3d.io.read_pinhole_camera_parameters(viz_camera_info_path)
    env_pcd_folder = Path(bag_folder_path) / Path("merged_pcd_filter")
    pcd_length = load_pcd_legth(bag_folder_path)
    obj_folder = Path(bag_folder_path) / \
        Path("object_pose_in_every_frame")
    image_save_folder = obj_folder / f"capture_image"
    os.makedirs(image_save_folder, exist_ok=True)
    hand_arm_mesh_folder_path = Path(bag_folder_path) / Path("arm_hand_mesh")
    object_mesh_folder  = Path(bag_folder_path) /Path("object_pose_in_every_frame")
    # object_mesh.compute_vertex_normals()
    for index in np.arange(constrain_bound[0],min(constrain_bound[1],pcd_length)):
        sleep(sleep_time_)
        print(index)
        env_pcd = o3d.io.read_point_cloud(
            str(env_pcd_folder / Path(f"merge_pcd_{index}.ply")))
        # obj_pcd = o3d.io.read_point_cloud(str(obj_folder / f"{index}.ply"))
        # hand_arm_mesh = o3d.io.read_triangle_mesh(str(hand_arm_mesh_folder_path / Path(f"{index}.ply")))
        # hand_arm_mesh.compute_vertex_normals()

        object_mesh = o3d.io.read_triangle_mesh(str(object_mesh_folder / Path(f"{index}.ply")))
        object_mesh.compute_vertex_normals()
        object_mesh.paint_uniform_color([1,0,0])
        if index != 0:
            vis.clear_geometries()
        
        vis.add_geometry(env_pcd)
        # vis.add_geometry(hand_arm_mesh)
        # vis.add_geometry(object_mesh)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        vis.poll_events()
        vis.update_renderer()
        # image = vis.capture_screen_float_buffer(False)
        # plt.imsave(image_save_folder /
        #            Path(f"{index}.png"), np.asarray(image), dpi=1)
    vis.destroy_window()

if __name__ == "__main__":

    # os.environ['DISPLAY'] = ':1'  # 指定显示器编号
    bag_folder_path = "/home/lab4dv/data/bags/yogurt/yogurt_1_20231207"

    sleep_time = 0
    
    
    viz_camera_info_path = "/home/lab4dv/data/camera_param.json"
    
    constrain_bound = [0,2000]
    
    
    
    show_obj_env_pcd(bag_folder_path, viz_camera_info_path,constrain_bound,sleep_time_ = sleep_time)
