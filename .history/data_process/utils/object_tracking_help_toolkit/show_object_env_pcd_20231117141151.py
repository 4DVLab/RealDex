import open3d as o3d
from pathlib import Path
im

def load_pcd_legth(pcd_path):
    pcd_time_stamp_path = pcd_path / Path("info.txt")
    pcd_time_stamp = np.loadtxt(pcd_time_stamp_path)
    return len(pcd.points)

def show_obj_env_pcd(bag_folder_path):
    env_pcd_folder = Path(bag_folder_path) / Path("merged_pcd_filter")
    


if __name__ == "__main__":
    bag_folder_path = "/media/tony/新加卷/yyx_tmp"
    
    show_obj_env_pcd(bag_folder_path)
