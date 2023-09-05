import rosbag
import os
import json
import yaml
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from tqdm import tqdm
from bagpy import bagreader

def msg2json(msg):
    ''' Convert a ROS message to JSON format'''
    y = yaml.safe_load(str(msg))
    return y

def rosbag2json():
    bag_path = "/Users/yumeng/Working/data/CollectedDataset/IntelligentHand/ros_bag_test/"
    bag = rosbag.Bag(os.path.join(bag_path, 'test.bag'))

    out_path = './out_json/'
    os.makedirs(out_path, exist_ok=True)

    for topic, msg, t in bag.read_messages(topics=['/tf']):
        # file_object  = open(os.path.join(out_path, f"frame_{t}.txt"), "w+")
        # file_object.write(str(msg))
        # file_object.close()
        file_object = open(os.path.join(out_path, f"frame_{t}.json"), "w+")
        with open(os.path.join(out_path, f"frame_{t}.json"), "w+") as file_object:
            json.dump(msg2json(msg), file_object, indent=4)
    bag.close()
    



def rosbag2video():
    bag_path = "/remote-home/share/intelligent_hand/test_bag/"
    bag = rosbag.Bag(os.path.join(bag_path, 'yangtao_2grasp_20230901.bag'))

    out_path = '/remote-home/liuym/data/0902/'
    os.makedirs(out_path, exist_ok=True)

    print(bag)
    bridge = CvBridge()

    topics = ["/cam0/depth_to_rgb/image_raw", "/cam0/rgb/image_raw"]

    for topic, msg, t in tqdm(bag.read_messages(topics=topics)):
        # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image_data = msg.data
        im = np.frombuffer(image_data, dtype=np.uint16).reshape(msg.height, msg.width, -1)
        
        path = os.path.join(out_path, topic.split("/")[-2])
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f"frame_{t}.png"), im)
        
    bag.close()

def readbag():
    bag_path = "/remote-home/share/intelligent_hand/test_bag/yangtao_2grasp_20230901.bag"
    b = bagreader(bag_path)
    speed_file = b.message_by_topic('/cam0/depth_to_rgb/image_raw')
    print(speed_file)
    

if __name__ == "__main__":
    # rosbag2json()
    rosbag2video()
    # readbag()

