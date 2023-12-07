import open3d as o3d



def simplify_pcd(pcd):
    simplified_cloud = pcd.simplify_voxel_size(0.1)


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("/media/tony/新加卷/test_data/test/test_1/merged_pcd_filter/merge_pcd_0.ply")
