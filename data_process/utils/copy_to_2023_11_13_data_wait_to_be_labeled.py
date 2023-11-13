import shutil
import os
from pathlib import Path

def save_camera_data_to_traget_folder(src_dir:Path,dst_dir:Path,pose_path = "",cam_index = 0):
    post_path = f"cam{cam_index}/rgb/image_raw"
    data_save_folder = src_dir / post_path
    data_transfer_folder = dst_dir / post_path
    # os.makedirs(data_transfer_folder, exist_ok=True)
    shutil.copytree(data_save_folder, data_transfer_folder)#这个是不允许目标文件夹存在的




def transfer_camera_data_to_traget_folder(src_dir:Path,dst_dir:Path,post_path:Path = Path("")):
    now_path = src_dir / post_path
    data_transfer_path = dst_dir / post_path
    try:
        if 'TF' in os.listdir(now_path):
            save_camera_data_to_traget_folder(now_path, data_transfer_path)
            return 
        for subfolder in os.listdir(now_path):
            if os.path.isdir(now_path / subfolder):
                transfer_camera_data_to_traget_folder(src_dir, dst_dir, post_path / subfolder)
    except PermissionError:
        print(f"PermissionError {now_path}")
        return


if __name__ == "__main__":
    src_dir = Path("/home/lab4dv/data/")
    dst_dir = Path("/media/lab4dv/新加卷/11_13_data_wait_to_be_labeled")
    transfer_camera_data_to_traget_folder(src_dir, dst_dir)




# src_dir = Path("/home/lab4dv/data/sda")
# dst_dir = Path("/media/lab4dv/新加卷/data_wait_to_be_labeled/sda")

# for folder in os.listdir(src_dir):
#     if folder not in ["girl_toy", "goji_jar", "instance_noodles", "thunder_toy", "yogurt"]:
#         continue
#     print(folder)
#     src_path = src_dir / Path(folder)
#     if (src_path / Path("original")).exists():
#         src_path = src_path / Path("original")
#         for sub_folder in os.listdir(src_path):
#             src_files = src_path /Path(sub_folder)/ Path("cam0/rgb/image_raw")
#             if not dst_dir.exists():
#                 os.mkdir(dst_dir)
#             if not (dst_dir/ Path(folder)).exists():
#                 os.mkdir(dst_dir/ Path(folder))
               
            
                
#             shutil.copytree(src_files, dst_dir/ Path(folder)/Path(sub_folder))