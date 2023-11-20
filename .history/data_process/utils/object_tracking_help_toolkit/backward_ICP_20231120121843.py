import open3d as o3d
import numpy as np
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtrack_steps", type=int, default=10)
    parser.add_argument("--bag_path",type=str)
    return parser.parse_args()


# the final target is to gen the ICP transform matrix in the world frame
def backward_nsteps(bag_path,n_steps_backward):
    backtrack_file_path = bag_path / f"trakcing_result/tracking_index.txt"
    with open(backtrack_file_path,"r") as index_reader:
        tracking_index = np.loadtxt(index_reader).item
    


if __name__ == "__main__":
    args = get_args()  

