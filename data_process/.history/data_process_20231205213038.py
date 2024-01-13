from utils.TFExtract import extract_rosbag_tf
from utils.extract_arm_hand_obj import extract_arm_hand_obj
from utils.extract_exerything_frombag import extract_everything_from_bag
import argparse
import os
from pathlib import Path
import shutil
#以后可能产生变化的因素全部都要用命令行输入
#之后需要把extract_arm中的urdf位置给变
#ros_path_prefix 变
# pinhole_camera_parameters_path

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--folder_path', default=None, type=str,
    #                     help='training stage')
    # parser.add_argument('--bag_name', default=None, type=str,)
    parser.add_argument("--outsize_folder",default=None,type=str,help="the folder outsize the bag folder")
    
    return parser.parse_args()



if __name__ == "__main__":
    # args = get_args()
    # folder_path = args.folder_path
    # bag_name = args.bag_name

    #以后看到机械臂全部offline去得到，不再通过这个程序来直接看到
    # args = get_args()

    # folder_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/"
    # bag_name = "sample" 
    # outsize_folder = args.outsize_folder


    # folder_path = args.folder_path
    # bag_name = args.bag_name
    # camera_param_path = args.camera_param_path
    root_folder = "/media/tony/新加卷/test_data/"
    for folder in os.listdir(root_folder):
        if "urdf" in folder or "config_data" in folder or "hand_arm_mesh" in folder or "lost+found" in folder or "wd40" not in folder:
            print("continue",folder)
            continue
        else:
            print(folder)
        middle_folder = Path(root_folder) / Path(folder)
        for file in os.listdir(middle_folder):
            print(file)
            if file.endswith(".bag") :
                bag_name = file[:-4]
                file_origin_path = middle_folder / Path(file)
                folder_path = middle_folder / Path(bag_name)
                file_transfer_path = folder_path / Path(file)
                os.makedirs(folder_path,exist_ok=True)
                shutil.move(str(file_origin_path),str(file_transfer_path))
                
                middle_folder = str(middle_folder)
                print("folder",middle_folder)
                print("bag_name",bag_name)
                extract_rosbag_tf(middle_folder,bag_name)
                extract_arm_hand_obj(middle_folder,bag_name, root_folder)
                extract_everything_from_bag(middle_folder,bag_name)
                shutil.move(str(file_transfer_path), str(file_origin_path))
                # os.remove(Path(file_transfer_path))
               
        # shutil.move(str(middle_folder), str(Path("/home/lab4dv/data/ssd") / Path(folder)))