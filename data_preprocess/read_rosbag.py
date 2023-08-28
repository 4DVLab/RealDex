import rosbag
import os
import json
import yaml

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


if __name__ == "__main__":
    rosbag2json()

