import os
from pathlib import Path
import numpya as np



def check_camera_depth(depth_folder_path):   
    pass
def check_camera_rgb(rgb_folder_path):
    pass


def check_camera(camera_folder:Path):#时间戳
    if not check_camera_rgb:
        print(f"{camera_folder} rgb is not correct")
    return True


def check_this_bag_folder(bag_folder,data_check_item):
    cam_num  = 4
    bag_name = str(bag_folder).split("/")[-1]
    camera_jud = True
    for cam_index in np.arange(cam_num):
        if not check_camera(bag_folder,cam_index):
            print(f"{bag_name} camera {cam_index} is not correct")

#all the folder must contain the similar number of data

#how to reconize the folder can be  the bag folder that save the data


# check the folder by the assumption that every bag folder must have a TF folder 
def check_data(root_folder:Path,data_check_item):
    """
    scan all the subfolder in the root_folder
    
        
        """
    if "TF" in os.listdir(root_folder):
        check_this_bag_folder(root_folder,data_check_item)
    
    for file in os.listdir(root_folder):
        concat_folder_path = root_folder / file
        if os.path.isdir(concat_folder_path):
            check_data(concat_folder_path,data_check_item)




if __name__ == "__main__":
    data_check_item = ["depth_to_rgb","rgb"]