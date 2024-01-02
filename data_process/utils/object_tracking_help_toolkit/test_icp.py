import open3d as o3d
from pathlib import Path
import numpy as np
from copy import deepcopy

folder_path = "/home/lab4dv/data/ssd/shower_cleaner/shower_cleaner_1"

folder_path = Path(folder_path)

index = 627

env_pcd_path = folder_path / f"merged_pcd_filter/merge_pcd_{627}.ply"

mesh_model_path = folder_path / f"icp_object_pose_in_every_frame/{626}.ply"

env_pcd = o3d.io.read_point_cloud(str(env_pcd_path))

mesh_model = o3d.io.read_triangle_mesh(str(mesh_model_path))
mesh_model.compute_vertex_normals()

def p2pl(mesh_model,env_pcd):
    loss = o3d.pipelines.registration.TukeyLoss(k=2)
    model_point_cloud = o3d.geometry.PointCloud()
    model_point_cloud.points = mesh_model.vertices
    mesh_model.compute_vertex_normals()
    model_point_cloud.normals = mesh_model.vertex_normals
    print(np.array(mesh_model.vertex_normals).shape)
    icp_result = o3d.pipelines.registration.registration_icp(
        # 最大的crospondence距离，不能设置太小了，不然找不到对应点
        env_pcd, model_point_cloud, max_correspondence_distance=0.03,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness= 1e-06,relative_rmse= 1e-06,max_iteration=2000)
    )
    # cuz this is the env to icp the model
    temp_model = deepcopy(mesh_model)
    transform = np.linalg.inv(icp_result.transformation)
    o3d.visualization.draw_geometries([env_pcd,temp_model.paint_uniform_color([0,1,0])])


def env_pcd_to_mesh(mesh_model,env_pcd):
    source_pcd = mesh_model.sample_points_uniformly(number_of_points=50000)
    icp_result = o3d.pipelines.registration.registration_icp(
        env_pcd, source_pcd, max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
    transform = np.linalg.inv(icp_result.transformation)
    temp_model = deepcopy(mesh_model)
    o3d.visualization.draw_geometries([env_pcd,mesh_model.paint_uniform_color([1,0,0]),temp_model.transform(transform).paint_uniform_color([0,1,0])])



def mesh_to_pcd(mesh_model,env_pcd):
    source_pcd = mesh_model.sample_points_uniformly(number_of_points=50000)
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, env_pcd, max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness= 1e-06,relative_rmse= 1e-05,max_iteration=10000))
    transform = icp_result.transformation
    temp_model = deepcopy(mesh_model)
    # path = "/home/lab4dv/data/bags/cosmetics/cosmetics_1/icp_object_pose_in_every_frame/93.ply"
    # icp_model = o3d.io.read_triangle_mesh(path)

    o3d.visualization.draw_geometries([env_pcd,temp_model.transform(transform).paint_uniform_color([0,1,0])])#icp_model.paint_uniform_color([0,0,1]),mesh_model.paint_uniform_color([1,0,0]),temp_model.transform(transform).paint_uniform_color([0,1,0])])


p2pl(mesh_model,env_pcd)


