
from json.tool import main
import rosbag
from tqdm import tqdm
import numpy as np
from pathlib import Path


def read_bag(bag_file:str, topic:str, save_path:str):

    bag_data = rosbag.Bag(bag_file, "r")
    bag_data = bag_data.read_messages(topics=[topic])

    msg_list = []
    for topic, msg, t in tqdm(bag_data):
        
        msg_list_tmp = msg.points[0].positions
        msg_tmp = [t.to_nsec()]
        for x in msg_list_tmp[:]:
            msg_tmp.append(x)
        # print(msg_tmp)
        msg_list.append(msg_tmp)

        # msg_list.append(msg.points[0].positions)

    np.savetxt(save_path, msg_list)


    

if __name__ =="__main__" :

    read_bag("/home/lab4dv/IntelligentHand/drive_ws/bags/test.bag", '/ra_trajectory_controller/command', "/home/lab4dv/IntelligentHand/drive_ws/bags/test_ra_points.txt")
    read_bag("/home/lab4dv/IntelligentHand/drive_ws/bags/test.bag", '/rh_wr_trajectory_controller/command', "/home/lab4dv/IntelligentHand/drive_ws/bags/test_rh_wr_points.txt" )
    read_bag("/home/lab4dv/IntelligentHand/drive_ws/bags/test.bag", '/rh_trajectory_controller/command', "/home/lab4dv/IntelligentHand/drive_ws/bags/test_rh_points.txt" )