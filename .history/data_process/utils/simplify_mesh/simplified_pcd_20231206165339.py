import open3d as o3d
from pathlib import Path
import numpy as np
import os
def simplify_pcd(pcd):
    simplified_cloud = pcd.simplify_voxel_size(0.1)


if __name__ == "__main__":
    folder_path = Path(
        "/media/tony/新加卷/test_data/test/test_1/merged_pcd_filter/")
    pcd_num = len(os.listdir(folder_path))
    for pcd_index in np.arange(pcd):

    pcd = o3d.io.read_point_cloud("merge_pcd_0.ply")
    simplified_cloud = pcd.voxel_down_sample(0.01)
    o3d.io.write_point_cloud("/media/tony/新加卷/test_data/test/test_1/merged_pcd_filter/merge_pcd_0_simplified.ply",simplified_cloud)