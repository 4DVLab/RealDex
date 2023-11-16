import os
from pathlib import Path

def check_camera_depth():   
    pass
def check_camera_rgb():
    pass


def check_camera():
    pass


def check_this_bag_folder(bag_folder,data_check_item):

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
        t
        if os.path.isdir(file):
            check_data(file,data_check_item)

