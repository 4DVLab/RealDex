import open3d as o3d
import numpy as np
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backward_steps", type=int, default=10)
    parser.add_argument("--bag_path",type=str)
    return parser.parse_args()


# the final target is to gen the ICP transform matrix in the world frame
def backward():
    pass


if __name__ == "__main__":
    arg
    bag_path = Path("/home/lab4dv/data/bags/apple/apple_1_20231105")
