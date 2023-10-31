import rosbag
from tf2_msgs.msg import TFMessage
import rospy
from collections import defaultdict
import sys


def compute_tf_frequencies(bagfile):
    bag = rosbag.Bag(bagfile)

    tf_time_stamps = defaultdict(list)

    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        if isinstance(msg, TFMessage):
            for transform in msg.transforms:
                key = transform.child_frame_id
                tf_time_stamps[key].append(t.to_sec())

    tf_frequencies = {}
    for key, stamps in tf_time_stamps.items():
        if len(stamps) > 1:
            freqs = []
            for i in range(1, len(stamps)):
                freqs.append(1.0 / (stamps[i] - stamps[i-1]))
            avg_freq = sum(freqs) / len(freqs)
            tf_frequencies[key] = avg_freq
        else:
            tf_frequencies[key] = 0.0

    bag.close()
    return tf_frequencies


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python script_name.py BAGFILE_PATH")
    #     exit(0)

    bagfile = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/meal_spoon_0_20230921.bag"
    frequencies = compute_tf_frequencies(bagfile)
    for key, freq in frequencies.items():
        print(f"TF {key} has average frequency: {freq} Hz")
