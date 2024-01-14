from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import open3d as o3d
import os
from copy import deepcopy
from tqdm import tqdm

# from generate_multiple_view_ply import gen_mvp
# from transform_mesh_with_tracking_result import gen_tracking_result_model

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
            source_pcd, pcd, max_correspondence_distance=0.02,
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

def load_tracking_result(folder_path, cam_index=0):

    pose_path = Path(folder_path) / \
        Path(f"tracking_result/tracking_result.txt")
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

def load_gt_pose(bag_folder_path ,frame_index = 0):
    pose_file_path = Path(bag_folder_path) / f"poses/0.txt"  
    with open(pose_file_path,"r") as file_reader:
        transform = np.loadtxt(file_reader).reshape((4,4))
    return transform

def icp_tune_tracking_result(folder_path, model_name, constrain_bound):
    pcd_folder = folder_path / f"merged_pcd_filter"
    # tracking_result_list = load_tracking_result(str(folder_path))
    mesh_model = load_mesh_model(folder_path, f"{model_name}.obj")
    mesh_model.compute_vertex_normals()
    mesh_model.paint_uniform_color([1, 0, 0])
    transform_list = []
    temp_pose = load_gt_pose(folder_path)
    mesh_model.transform(temp_pose)
    os.makedirs(Path(folder_path) / "tracking_result",exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    viz_camera_info_path = "/home/lab4dv/data/camera_param.json"
    camera_params = o3d.io.read_pinhole_camera_parameters(viz_camera_info_path)

    for index in tqdm(np.arange(constrain_bound[0], constrain_bound[1] + 1)):
        pcd_path = pcd_folder / Path(f"merge_pcd_{index}.ply")
        pcd = o3d.io.read_point_cloud(str(pcd_path))


        # return the mesh_model transform matrix
        icp_transform_matrix = icp_mesh_pcd(mesh_model, pcd)
        mesh_model.transform(icp_transform_matrix)

        temp_pose = icp_transform_matrix @ temp_pose

        seven_num = transform_matrix_to_seven_num(temp_pose)
        
        if index != 0:
            vis.clear_geometries()
        
        vis.add_geometry(pcd)
        # vis.add_geometry(hand_arm_mesh)
        vis.add_geometry(mesh_model)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        vis.poll_events()
        vis.update_renderer()


        transform_list.append(seven_num)
        with open(folder_path / f"tracking_result/icp_use_pre_frame.txt", "w+") as data_saver:
            np.savetxt(data_saver, np.array(transform_list).reshape((-1, 7)))

    with open(folder_path / f"tracking_result/icp_use_pre_frame.txt", "w+") as data_saver:
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

# def icp_pre_todo(bag_folder_path):
#     gen_mvp(bag_folder_path)
#     gen_tracking_result_model(bag_folder_path)

def icp_tune(folder_path):
    # icp_pre_todo(folder_path)
    bag_folder = Path(folder_path)
    model_name = get_model_name(bag_folder)
    # cam_parameter_path = "/media/tony/T7/bags/demo/yogurt_1_20231207/camera_param.json"
    constrain_bound = [0, 2000]
    icp_tune_tracking_result(bag_folder, model_name,
                                          constrain_bound)
    # gen_tracking_result_model(folder_path,True,file_name="icp_use_pre_frame.txt")
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


    
    folder_path = "/media/lab4dv/film/bags/charmander/charmander_1_20240107"

    icp_tune(folder_path)
    