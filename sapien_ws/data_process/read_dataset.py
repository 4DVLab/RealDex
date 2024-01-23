# use unidexgrap environment
import sys
import json
import numpy as np
import os
sys.path.append("/home/lab4dv/IntelligentHand/data_preprocess/")
sys.path.append("/home/lab4dv/IntelligentHand/data_preprocess/IntelligentHand")
from  IntelligentHand.utils.kintree import load_sequence


if __name__ == "__main__":
    struct_file = '/home/lab4dv/IntelligentHand/data_preprocess/IntelligentHand/assets/sr_arm_hand_ur.json'

    with open(struct_file, 'r') as f:
                # Load the JSON data
                sr_struct = json.load(f)
                
    tf_data_dir = "/home/lab4dv/data/bags/small_sprayer/small_sprayer_1/TF"
    seq_file = os.path.join(tf_data_dir, "tf_seq.npy")
    if os.path.exists(seq_file): 
        seq_data = np.load(seq_file, allow_pickle=True)
        seq_data = seq_data.item()
    else:
        seq_data = load_sequence(tf_data_dir, struct_file)
        np.save(seq_file, seq_data) # tf stored in 4*4 matrix form

    trajectory_path = "/home/lab4dv/IntelligentHand/sapien_ws/config/trajectory_points.txt"
    trajectory_list = []
    mapping_dir= {}
   
    for link in sr_struct["hand_info"]:
         mapping_dir[sr_struct["hand_info"][link]["joint"]] = link

    for timestamp in seq_data['joint_angle']:
        trajectory = []

        points = seq_data['joint_angle'][timestamp]
        trajectory.append(timestamp)
        # print(timestamp, points)
        if(len(points) == 0):
            continue
        trajectory.append(points[mapping_dir[ "ra_shoulder_pan_joint"]])
        trajectory.append(points[mapping_dir[ "ra_shoulder_lift_joint"]])
        trajectory.append(points[mapping_dir[ "ra_elbow_joint"]])
        trajectory.append(points[mapping_dir[ "ra_wrist_1_joint"]])
        trajectory.append(points[mapping_dir[ "ra_wrist_2_joint"]])
        trajectory.append(points[mapping_dir[ "ra_wrist_3_joint"]])
        trajectory.append(points[mapping_dir[ "rh_WRJ2"]])
        trajectory.append(points[mapping_dir[ "rh_WRJ1"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_FFJ1"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_MFJ1"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_RFJ1"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ5"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ4"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ3"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ2"]])
        trajectory.append(points[mapping_dir[ "rh_LFJ1"]])
        trajectory.append(points[mapping_dir[ "rh_THJ5"]])
        trajectory.append(points[mapping_dir[ "rh_THJ4"]])
        trajectory.append(points[mapping_dir[ "rh_THJ3"]])
        trajectory.append(points[mapping_dir[ "rh_THJ2"]])
        trajectory.append(points[mapping_dir[ "rh_THJ1"]])
        trajectory_list.append(trajectory)
        # print(len(trajectory))
    print(len(trajectory_list))
    np.savetxt(trajectory_path, trajectory_list)
