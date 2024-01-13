import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.pcd_util import PCDGenerator
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm, trange


def wrapper_function(time_index):
    pcd_generator.gen_pcd(time_index, cam_index=cam_index)
    return

def update(*a):
    pbar.update(1)
    pbar.refresh() 
    return

if __name__ == '__main__':
    urdf_path = "../../data_process/bimanual_srhand_ur.urdf"
    prefix ="/public/home/v-liuym/data/ShadowHand/description/"
    struct_file = "./assets/srhand_ur.json"
    
    base_dir = "/public/home/v-liuym/data/IntelligentHand_data/"
    model_name = "crisps"
    exp_code = "crisps_4"
    data_dir = os.path.join(base_dir, model_name, exp_code)
    tf_data_dir = os.path.join(data_dir, "TF")
    time_stamp_file = os.path.join(data_dir, "rgbimage_timestamp.txt")
    time_stamp_list = np.loadtxt(time_stamp_file)
    
    cam_param_dir = "../../calibration_ws/calibration_process/data"
    pcd_generator = PCDGenerator(data_dir, cam_param_dir)
    
    cams_time = pcd_generator.gen_cams_time_stamp()
    seq_length = len(cams_time[0])
    # time_list = list(range(seq_length))
    
    cam_index = 3
    cam_pcd_dir = os.path.join(data_dir, f"cam{cam_index}", "pcd")
    if not os.path.exists(cam_pcd_dir):
        os.makedirs(cam_pcd_dir)
    
    pbar = tqdm(total=seq_length)
    
    # create process pool
    pool = Pool(16)
    # pool.map(wrapper_function, time_list)
    
    for i in trange(seq_length):
        pool.apply_async(wrapper_function, args=(i,)) #, callback=update)
        
    pbar.close()
    
    pool.close()
    pool.join()