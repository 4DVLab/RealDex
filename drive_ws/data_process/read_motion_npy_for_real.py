
import sys
import json
import numpy as np
import os
sys.path.append("/home/lab4dv/IntelligentHand/data_preprocess/")
sys.path.append("/home/lab4dv/IntelligentHand/data_preprocess/IntelligentHand")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    struct_file = '../../data_preprocess/IntelligentHand/assets/sr_arm_hand_ur.json'

    with open(struct_file, 'r') as f:
                # Load the JSON data
                sr_struct = json.load(f)
    npy_file = "../config/0_50.npy"
    if os.path.exists(npy_file): 
        data = np.load(npy_file, allow_pickle=True).item()
    else:
        print("config file error")

    trajectory_path = "../config/hand_points.txt"
    trajectory_list = []
    mapping_dir= {}
   
    for link in sr_struct["hand_info"]:
         mapping_dir[sr_struct["hand_info"][link]["joint"]] = link


    const_added = 1
    num = 0
    total_num = len(data)
    for timestamp in data:
        trajectory = []

        points = data[timestamp]
        trajectory.append(timestamp)
        # print(timestamp, points)
            
        trajectory.append(points[mapping_dir[ "rh_WRJ2"]])
        trajectory.append(points[mapping_dir[ "rh_WRJ1"]])
        
        trajectory.append(points[mapping_dir[ "rh_FFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ1"]]+ const_added*(num/total_num))

        trajectory.append(points[mapping_dir[ "rh_LFJ5"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ1"]]+ const_added*(num/total_num))

        trajectory.append(points[mapping_dir[ "rh_MFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ1"]]+ const_added*(num/total_num))

        trajectory.append(points[mapping_dir[ "rh_RFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ1"]]+ const_added*(num/total_num))
        
        trajectory.append(points[mapping_dir[ "rh_THJ5"]])
        trajectory.append(points[mapping_dir[ "rh_THJ4"]])
        trajectory.append(points[mapping_dir[ "rh_THJ3"]])
        trajectory.append(points[mapping_dir[ "rh_THJ2"]])
        trajectory.append(points[mapping_dir[ "rh_THJ1"]]+ const_added*(num/total_num))
        trajectory_list.append(trajectory)
        # print(len(trajectory))
    print(len(trajectory_list))
    np.savetxt(trajectory_path, trajectory_list)
