
import numpy as np
import time
import math
import os
import sys
sys.path.append("/home/lab4dv/IntelligentHand/data_preprocess/")
sys.path.append("/home/lab4dv/IntelligentHand/data_preprocess/IntelligentHand")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":

    load_path = "../config/pose_list.txt"
    joints_tmp = np.loadtxt(load_path,delimiter=",")
    joints = []   
 
    for points in joints_tmp:
        joints.append(list(points[2::-1])+ list(points[3:6]) +  list(points[29:27:-1])+ list(points[9:5:-1]) + list(points[14:9:-1])+ list(points[18:14:-1])
                      + list(points[22:18:-1])  + list(points[27:22:-1])  )
    print(len(joints[0]))

    np.savetxt("../config/real_pose_list.txt", joints)