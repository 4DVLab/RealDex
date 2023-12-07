import open3d as o3d



def simplify_pcd():
    simplified_cloud = point_cloud.simplify_voxel_size(0.1)


if __name__ == "__main__":
    pcd = o3d.io.