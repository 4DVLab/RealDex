from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import json
import open3d as o3d
import os
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
import multiprocessing

from generate_multiple_view_ply import gen_mvp
from transform_mesh_with_tracking_result import gen_tracking_result_model

def seven_num2matrix(seven_num):
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix


def icp_mesh_pcd(mesh_model, pcd):
    icp_mode = 3
    if icp_mode == 1:
        model_point_cloud = o3d.geometry.PointCloud()
        model_point_cloud.points = mesh_model.vertices
        mesh_model.compute_vertex_normals()
        model_point_cloud.normals = mesh_model.vertex_normals
        icp_result = o3d.pipelines.registration.registration_icp(
            # 最大的crospondence距离，不能设置太小了，不然找不到对应点
            pcd, model_point_cloud, max_correspondence_distance=0.01,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=20)
        )
        # cuz this is the env to icp the model
        transform = np.linalg.inv(icp_result.transformation)
    elif icp_mode == 2:
        source_pcd = mesh_model.sample_points_uniformly(number_of_points=100000)
        icp_result = o3d.pipelines.registration.registration_icp(
            pcd, source_pcd, max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
        transform = np.linalg.inv(icp_result.transformation)
    else:
        source_pcd = mesh_model.sample_points_uniformly(number_of_points=10000)
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, pcd, max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness= 1e-06,relative_rmse= 1e-06,max_iteration=2000000))
        transform = icp_result.transformation
        # print(icp_result.inlier_rmse)
    return transform


def load_mesh_model(folder_path, model_name):

    model_path = Path(folder_path).parent / Path("models") / Path(model_name)
    mesh_model = o3d.io.read_triangle_mesh(str(model_path))
    mesh_model.compute_vertex_normals()
    return mesh_model

def load_pose_result(folder_path,file_name, cam_index=0):

    pose_path = Path(folder_path) / \
        Path(f"tracking_result/{file_name}")
    with open(pose_path, "r") as pose_reader:
        pose_list = np.loadtxt(pose_reader, dtype=np.float32)
    pose_list = [np.array(seven_num2matrix(pose)) for pose in pose_list]
    pose_list = np.array(pose_list)
    return pose_list


def transform_matrix_to_seven_num(transform_matrix):
    quat = Rotation.from_matrix(transform_matrix[:3, :3]).as_quat()
    translation = transform_matrix[:3, 3]
    seven_num = np.concatenate((translation, quat))
    return seven_num

def write_pose_result(pose_result,save_path):
    seven_num_transform_list = [transform_matrix_to_seven_num(pose_result[transform_index]) for transform_index in pose_result.shape[0]]
    seven_num_transform_list = np.concatenate(seven_num_transform_list,axis=0)
    with open(save_path, "w+") as data_saver:
            np.savetxt(data_saver, np.array(seven_num_transform_list).reshape((-1, 7)))

    
def icp_tune_global_icp_result(folder_path, model_name, constrain_bound,use_origin_icp_result):
    pcd_folder = folder_path / f"merged_pcd_filter"
    if use_origin_icp_result:
        tracking_result_list = load_pose_result(str(folder_path),"tracking_and_icp_mode_3.txt")
    else:
        tracking_result_list = load_pose_result(str(folder_path),"local_icp.txt")
    mesh_model = load_mesh_model(folder_path, f"{model_name}.obj")
    mesh_model.compute_vertex_normals()

    temp_pose = tracking_result_list[constrain_bound[0]]
    
    for index in tqdm(np.arange(constrain_bound[0], constrain_bound[1] + 1)):
        pcd_path = pcd_folder / Path(f"merge_pcd_{index}.ply")
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        temp_mesh_model = deepcopy(mesh_model)
        temp_mesh_model.transform(temp_pose)
        temp_mesh_model.paint_uniform_color([0, 1, 0])
        # return the mesh_model transform matrix
        icp_transform_matrix = icp_mesh_pcd(temp_mesh_model, pcd)
        # temp_mesh_model.transform(icp_transform_matrix)
        final_transform_matrix = icp_transform_matrix @ temp_pose
        tracking_result_list[index] = final_transform_matrix.reshape((4,4))

        # icp_result_mesh = temp_mesh_model.transform(icp_transform_matrix)
        # icp_result_mesh.paint_uniform_color([1, 0, 0])

        write_pose_result(tracking_result_list,folder_path / f"tracking_result/local_icp.txt")

    write_pose_result(tracking_result_list,folder_path / f"tracking_result/local_icp.txt")


def get_model_name(bag_folder_path):
    bag_folder_path = Path(bag_folder_path)
    model_folder_path = bag_folder_path.parent / Path("models")
    print(model_folder_path)
    model_name = None
    for file in os.listdir(model_folder_path):
        if file.endswith(".obj"):
            model_name = file[:-4]
            return model_name

def icp_pre_todo(bag_folder_path):
    gen_mvp(bag_folder_path)
    gen_tracking_result_model(bag_folder_path)

def icp_tune(folder_path,use_origin_icp_result):
    # icp_pre_todo(folder_path)
    bag_folder = Path(folder_path)
    model_name = get_model_name(bag_folder)
    # cam_parameter_path = "/media/tony/T7/bags/demo/yogurt_1_20231207/camera_param.json"
    constrain_bound = [0, 200]
    icp_tune_global_icp_result(bag_folder, model_name,
                                          constrain_bound,use_origin_icp_result)
    gen_tracking_result_model(folder_path,True)
    # os.system(f"rm -r {folder_path}/merged_pcd_filter")


if __name__ == "__main__":
    # search_all_folder_do_icp("/home/lab4dv/data/bags")

    use_origin_icp_result = True
    
    folder_path = "/media/lab4dv/HighSpeed/experiment/shower_cleaner_1"

    icp_tune(folder_path)
    