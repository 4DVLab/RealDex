from scipy.spatial.transform import Rotation 
import numpy as np
from pathlib import Path
import json
import open3d as o3d


def ico_mesh_pcd(mesh_model,pcd):
    model_point_cloud = o3d.geometry.PointCloud()
    model_point_cloud.points = mesh_model.vertices
    mesh_model.compute_vertex_normals()
    model_point_cloud.normals = mesh_model.vertex_normals
    icp_result = o3d.pipelines.registration.registration_icp(
        # 最大的crospondence距离，不能设置太小了，不然找不到对应点
        pcd, model_point_cloud, max_correspondence_distance=0.004,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
    )    
    transform = np.linalg.inv(icp_result.transformation)# cuz this is the env to icp the model
    return transform

def load_mesh_model(folder_path,model_name):

    model
    mesh_path = Path("/media/tony/新加卷/test_data/test/test_1/models/obj_0.obj")

    return mesh_model


if __name__ == "__main__":
    bag_folder = Path("/media/tony/新加卷/test_data/test/test_1")




