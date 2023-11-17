import open3d as o3d
from pathlib import Path
import numpy as np

def load_pcd_legth(pcd_path):
    pcd_time_stamp_path = pcd_path / Path("info.txt")
    pcd_time_stamp = np.loadtxt(pcd_time_stamp_path)
    return pcd_time_stamp.shape[0]

def show_obj_env_pcd(bag_folder_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    camera_params = o3d.io.read_pinhole_camera_parameters()
    env_pcd_folder = Path(bag_folder_path) / Path("merged_pcd_filter")
    pcd_length = load_pcd_legth(env_pcd_folder)
    obj_folder = Path(bag_folder_path) / Path("object_pose_in_every_frame")

    for index in np.arange(pcd_length):
        env_pcd = o3d.io.read_point_cloud(str(env_pcd_folder / Path("merge_pcd_{index}.ply")))
        obj_pcd = o3d.io.read_point_cloud(str(obj_folder / f"{index}.ply"))

    if index != 
        vis.clear_geometries()
        for mesh_item in mesh_show.values():
            vis.add_geometry(mesh_item)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        vis.poll_events()
        vis.update_renderer()

if __name__ == "__main__":
    bag_folder_path = "/media/tony/新加卷/yyx_tmp"
    viz_camera_info_path = "/media/tony/新加卷/yyx_tmp/rh_palm_0.obj_camera_params.json"
    show_obj_env_pcd(bag_folder_path)
