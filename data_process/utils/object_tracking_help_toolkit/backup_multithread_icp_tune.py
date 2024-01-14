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
                max_iteration=1000)
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
        source_pcd = mesh_model.sample_points_uniformly(number_of_points=100000)
        icp_result = o3d.pipelines.registration.registration_icp(
            source_pcd, pcd, max_correspondence_distance=0.05,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
        transform = icp_result.transformation
    return transform


def load_mesh_model(folder_path, model_name):

    model_path = Path(folder_path).parent / Path("models") / Path(model_name)
    mesh_model = o3d.io.read_triangle_mesh(str(model_path))
    mesh_model.compute_vertex_normals()
    return mesh_model


def load_cam0_to_world_transform(folder_path):
    file_path = Path(folder_path) / f"global_name_position/0.txt"
    with open(file_path, "r") as json_reader:
        transform_file = json.load(json_reader)
    cam0_to_world_transform = transform_file["cam0_rgb_camera_link"]
    cam0_to_world_transform = np.array(cam0_to_world_transform).reshape((4, 4))
    return cam0_to_world_transform


def load_tracking_result(folder_path, cam_index=0):
    # cam0_to_world_transform = load_cam0_to_world_transform(folder_path)
    # bag_name = folder_path.split('/')[-1] #[:-9]
    pose_path = Path(folder_path) / \
        Path(f"tracking_result/tracking_result.txt")
    with open(pose_path, "r") as pose_reader:
        pose_list = np.loadtxt(pose_reader, dtype=np.float32)
    pose_list = [np.array(seven_num2matrix(pose)) for pose in pose_list]
    pose_list = np.array(pose_list)
    return pose_list


class pcd_iter():
    def __init__(self, folder_path, file_name_prefix="", file_name_postfix="") -> None:
        self.folder_path = folder_path
        self.pcd_num = len(os.listdir(Path(folder_path)))
        self.pcd_index = 0
        self.file_name_prefix = file_name_prefix
        self.file_name_postfix = file_name_postfix

    def __iter__(self):
        return self

    def __next__(self):
        if self.pcd_index < self.pcd_num:
            pcd_path = Path(self.folder_path) / Path(
                f"{self.file_name_prefix}{self.pcd_index}{self.file_name_postfix}")
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            self.pcd_index += 1
            return pcd
        else:
            raise StopIteration


def load_pcd_iter(folder_path, file_name_prefix="", file_name_postfix=""):
    folder_path = Path(folder_path) / f"merged_pcd_filter"
    return pcd_iter(folder_path, file_name_prefix, file_name_postfix)


def transform_matrix_to_seven_num(transform_matrix):
    quat = Rotation.from_matrix(transform_matrix[:3, :3]).as_quat()
    translation = transform_matrix[:3, 3]
    seven_num = np.concatenate((translation, quat))
    return seven_num


def process_pcd(index, pcd_folder, tracking_result_list, model_name):
    print("process index", index)
    mesh_model = load_mesh_model(pcd_folder.parent, f"{model_name}.obj")

    pcd_path = pcd_folder / Path(f"merge_pcd_{index}.ply")
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    mesh_model.transform(tracking_result_list)

    # return the mesh_model transform matrix
    icp_transform_matrix = icp_mesh_pcd(mesh_model, pcd)
    # model_point_cloud = o3d.geometry.PointCloud()
    # model_point_cloud.points = mesh_model.vertices
    # mesh_model.compute_vertex_normals()
    # model_point_cloud.normals = mesh_model.vertex_normals
    # icp_result = o3d.pipelines.registration.registration_icp(
    #     # 最大的crospondence距离，不能设置太小了，不然找不到对应点
    #     pcd, model_point_cloud, max_correspondence_distance=0.03,
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    # )
    # # cuz this is the env to icp the model
    # icp_transform_matrix = np.linalg.inv(icp_result.transformation)

    final_transform_matrix = icp_transform_matrix @ tracking_result_list
    seven_num = transform_matrix_to_seven_num(final_transform_matrix)

    return seven_num


def multi_thread_icp_tune_tracking_result(folder_path, model_name, constrain_bound, cam_parameter_path=None, max_workers=None):
    pcd_folder = folder_path / f"merged_pcd_filter"
    tracking_result_list = load_tracking_result(str(folder_path))

    pcd_num = len(os.listdir(pcd_folder))
    tracking_result_num = tracking_result_list.shape[0]
    constrain_bound[1] = min(constrain_bound[1], pcd_num, tracking_result_num)
    index_range = list(np.arange(constrain_bound[0], constrain_bound[1]))
    transform_list = {}

    ctx = multiprocessing.get_context('forkserver')
    # Setup the ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as executor:
        # 提交任务到executor，保留index和future的映射
        future_to_index = {executor.submit(
            process_pcd, index, pcd_folder, tracking_result_list[index], model_name): index for index in index_range}
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]  # 获取对应的index
            try:
                result = future.result()  # 尝试获取结果
            except Exception as exc:
                print(f'Index {index} generated an exception: {exc}')
            else:
                transform_list[index] = result  # 将结果放在正确的位置
                temp_list = np.concatenate(
                    [value.reshape(1, -1) for value in transform_list.values()], axis=0)
                # Save results
                with open(folder_path / f"tracking_result/tracking_and_icp.txt", "w+") as data_saver:
                    np.savetxt(data_saver, np.array(
                        temp_list).reshape((-1, 7)))

    transform_list = np.concatenate(
        [value.reshape(1, -1) for value in transform_list.values()], axis=0)
    # Save results
    with open(folder_path / f"tracking_result/tracking_and_icp.txt", "w+") as data_saver:
        np.savetxt(data_saver, np.array(transform_list).reshape((-1, 7)))


def icp_tune_tracking_result(folder_path, model_name, constrain_bound, cam_parameter_path):
    pcd_folder = folder_path / f"merged_pcd_filter"
    tracking_result_list = load_tracking_result(str(folder_path))
    mesh_model = load_mesh_model(folder_path, f"{model_name}.obj")
    mesh_model.compute_vertex_normals()
    transform_list = []
    for index in tqdm(np.arange(constrain_bound[0], constrain_bound[1] + 1)):
        pcd_path = pcd_folder / Path(f"merge_pcd_{index}.ply")
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        temp_mesh_model = deepcopy(mesh_model)
        temp_mesh_model.transform(tracking_result_list[index])
        temp_mesh_model.paint_uniform_color([0, 1, 0])
        # return the mesh_model transform matrix
        icp_transform_matrix = icp_mesh_pcd(temp_mesh_model, pcd)
        # temp_mesh_model.transform(icp_transform_matrix)
        final_transform_matrix = icp_transform_matrix @ tracking_result_list[index]

        seven_num = transform_matrix_to_seven_num(final_transform_matrix)
        icp_result_mesh = temp_mesh_model.transform(icp_transform_matrix)
        icp_result_mesh.paint_uniform_color([1, 0, 0])
        transform_list.append(seven_num)
        with open(folder_path / f"tracking_result/tracking_and_icp.txt", "w+") as data_saver:
            np.savetxt(data_saver, np.array(transform_list).reshape((-1, 7)))

    with open(folder_path / f"tracking_result/tracking_and_icp.txt", "w+") as data_saver:
        np.savetxt(data_saver, np.array(transform_list).reshape((-1, 7)))


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

def icp_tune(folder_path):
    # icp_pre_todo(folder_path)
    bag_folder = Path(folder_path)
    model_name = get_model_name(bag_folder)
    # cam_parameter_path = "/media/tony/T7/bags/demo/yogurt_1_20231207/camera_param.json"
    constrain_bound = [0, 15]
    multi_thread_icp_tune_tracking_result(bag_folder, model_name,
                                          constrain_bound)
    gen_tracking_result_model(folder_path,True)
    # os.system(f"rm -r {folder_path}/merged_pcd_filter")


def search_all_folder_do_icp(root_folder_path):
    if "TF" in os.listdir(root_folder_path):
        print(root_folder_path)
        icp_tune(root_folder_path)
        return
    for folder in os.listdir(root_folder_path):
        
        try:
            folder_path = Path(root_folder_path) / Path(folder)
            if os.path.isdir(folder_path):
                search_all_folder_do_icp(folder_path)
        except PermissionError:
            print("PermissionError")
            continue


if __name__ == "__main__":
    # search_all_folder_do_icp("/home/lab4dv/data/bags")


    
    folder_path = "/home/lab4dv/data/bags/cosmetics/cosmetics_1"

    icp_tune(folder_path)
    