import open3d as o3d
from path


def show_obj_env_pcd(bag_folder_path):
    env_pcd_folder = Path(bag_folder_path) / Path("object_pose_in_every_frame")



if __name__ == "__main__":
    bag_folder_path = "/media/tony/新加卷/yyx_tmp"
    
    show_obj_env_pcd(bag_folder_path)
