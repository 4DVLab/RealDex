from scipy.spatial.transform import Rotation 
import numpy as np
from pathlib import Path
import json
import open3d as o3d


def seven_num2matrix(seven_num):
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix


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

    model_path = Path(folder_path) / Path("models") / Path(model_name + ".obj")
    mesh_model = o3d.io.read_triangle_mesh(str(model_path))
    return mesh_model


def load_tracking_result(folder_path,model_name,cam_index):
    pose_path = Path(folder_path) / Path(f"tracking_result/{bag_name}_cam_index_{cam_index}_tracking_result.txt")
    with open(pose_path,"r") as pose_reader:
        pose_list = np.loadtxt(pose_reader,dtype=np.float32)
    pose_list = [np.array(seven_num2matrix(pose)) for pose in pose_list]
    pose_list = np.array(pose_list)
    return pose_list






if __name__ == "__main__":
    bag_folder = Path("/media/tony/新加卷/test_data/test/test_1")




