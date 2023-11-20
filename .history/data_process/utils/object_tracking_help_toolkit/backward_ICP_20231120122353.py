import open3d as o3d
import numpy as np
from pathlib import Path
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtrack_steps", type=int, default=10)
    parser.add_argument("--bag_path",type=str)
    return parser.parse_args()

def ICP_between_two_pcd(source_pcd,target_pcd):
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance=0.1,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    return icp_result/


# the final target is to gen the ICP transform matrix in the world frame
def backward_nsteps(bag_path,n_steps_backward):
    backtrack_file_path = bag_path / f"trakcing_result/tracking_index.txt"
    tracking_index = 0
    with open(backtrack_file_path,"r") as index_reader:
        tracking_index = np.loadtxt(index_reader).item()
    new_begin_tracking_index = tracking_index - n_steps_backward

    


if __name__ == "__main__":
    args = get_args()  
    backward_nsteps(args.bag_path, args.backtrack_steps)

