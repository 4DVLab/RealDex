import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation
from copy import deepcopy
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import multiprocessing
import concurrent

def simplify_mesh(mesh, fraction=0.9):
    """
    Simplify the mesh by decimating it.
    :param mesh: The input mesh
    :param fraction: Fraction of triangles to remove. e.g., 0.1 means remove 10% of triangles.
    :return: Simplified mesh
    """
    return mesh.simplify_quadric_decimation(int((1-fraction) * len(mesh.triangles)))


def load_seven_num_pose(pose_path):
    with open(pose_path,"r") as pose_reader:
        pose_list = np.loadtxt(pose_reader,dtype=np.float32)
    pose_list = [np.array(seven_num2matrix(pose)) for pose in pose_list]
    pose_list = np.array(pose_list)
    return pose_list

def seven_num2matrix(seven_num):
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix


def transform_mesh_with_matrix(transform_matrix,mesh):
    vertices = np.asarray(mesh.vertices)
    vertices = np.hstack((vertices,np.ones((vertices.shape[0],1))),dtype=float).T
    transformed_vertices = np.dot(transform_matrix[:3,:],vertices).T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh

# def save_transformed_mesh(index, pose, cam2world_transform, mesh, mesh_to_save_folder_path):
#     obj_under_world_pose = cam2world_transform @ pose
#     mesh_to_save = transform_mesh_with_matrix(obj_under_world_pose,deepcopy(mesh))
#     mesh_save_path = mesh_to_save_folder_path / Path(str(index) + ".obj")
#     o3d.io.write_triangle_mesh(str(mesh_save_path),mesh_to_save)

def sub_mesh_transform_and_save(transform_matrix,mesh_path,path_to_save):
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.transform(transform_matrix)
        # o3d.visualization.draw_geometries([temp_mesh])
        o3d.io.write_triangle_mesh(str(path_to_save), mesh)


def transform_obj2result_pose(bag_folder_path, model_name, transform_mesh_interval = [0,99999], cam2world_transform=None,cam_index=0,icp_result = False,icp_mode_ = 1,file_name = "tracking_result.txt"):

    mesh_path = Path(bag_folder_path).parent / Path("models") /  Path(model_name + ".obj")
    # bag_name = bag_folder_path.split('/')[-1] #[:-9]
    # mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    # mesh = simplify_mesh(mesh)
    # pcd = mesh.sample_points_poisson_disk(9000)#don't use mesh simplify, but the sample points
    print("simple mesh done")
    # pose_path = Path(bag_folder_path) / Path(f"tracking_result/{bag_name}_cam_index_{cam_index}_tracking_result.txt")
    pose_path = Path(bag_folder_path) / \
        Path(
            f"tracking_result/{file_name}")
    if icp_result :
        pose_path = Path(bag_folder_path) / \
        Path(
            f"tracking_result/tracking_and_icp_mode_{icp_mode_}.txt")
    tranform_matrixs = load_seven_num_pose(pose_path)
    mesh_to_save_folder_path = Path(bag_folder_path) / Path("object_pose_in_every_frame")
    if icp_result:
        mesh_to_save_folder_path = Path(bag_folder_path) / Path("icp_object_pose_in_every_frame")
    
    if not os.path.exists(mesh_to_save_folder_path):
        os.makedirs(mesh_to_save_folder_path)
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(save_transformed_mesh, index, pose, cam2world_transform, mesh, mesh_to_save_folder_path) for index, pose in enumerate(tranform_matrixs)]
    #     for future in futures:
    #         future.result()
    transform_mesh_interval[1] += 1
    transform_mesh_interval[1] = min(transform_mesh_interval[1],tranform_matrixs.shape[0])

    ctx = multiprocessing.get_context('forkserver')
    with ProcessPoolExecutor(max_workers=32, mp_context=ctx) as executor:
        # 提交任务到executor，保留index和future的映射
        future_to_index = {executor.submit(
            sub_mesh_transform_and_save, tranform_matrixs[index], mesh_path, mesh_to_save_folder_path / Path(str(index) + ".ply")): \
            index for index in np.arange(transform_mesh_interval[0],transform_mesh_interval[1])}
        # for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index)):
        #     index = future_to_index[future]  # 获取对应的index
        #     print(index)

    # for index in np.arange(transform_mesh_interval[0],transform_mesh_interval[1]):
    #     pose = tranform_matrixs[index]
    #     # obj_under_world_pose = pose
    #     temp_mesh = deepcopy(mesh)

    #     temp_mesh.transform(pose)
    #     if cam2world_transform is not None:
    #         temp_mesh.transform(cam2world_transform)
    #     # mesh_to_save = transform_mesh_with_matrix(obj_under_world_pose,deepcopy(pcd))
    #     mesh_save_path = mesh_to_save_folder_path / Path(str(index) + ".ply")
    #     # o3d.visualization.draw_geometries([temp_mesh])
    #     o3d.io.write_triangle_mesh(str(mesh_save_path), temp_mesh)






def get_model_name(bag_folder_path):
    bag_folder_path = Path(bag_folder_path)
    model_folder_path = bag_folder_path.parent / Path("models")
    model_name = None
    for file in os.listdir(model_folder_path):
        if file.endswith(".obj"):
            model_name = file[:-4]
            return model_name




def gen_tracking_result_model(bag_folder_path,icp_mode = 1,icp = False,file_name = "tracking_result.txt"):
    bag_folder_path = bag_folder_path
    model_name = get_model_name(bag_folder_path)
    
    cam_index = 0

    # transforms = json.load(open(str(Path(bag_folder_path) / Path("global_name_position/0.txt")),"r"))
    # cam0_rgb_camera_link2world = np.array(transforms["cam3_rgb_camera_link"])
    # #simplify_persentage = 0.9
    
    
    
    transform_mesh_interval = [0,2000]

    transform_obj2result_pose(bag_folder_path, model_name,
                              transform_mesh_interval,
                              None,
                              icp_result=icp,
                              icp_mode_ = icp_mode,
                              file_name = file_name)

def get_model_name(bag_folder_path):
    bag_folder_path = Path(bag_folder_path)
    model_folder_path = bag_folder_path.parent / Path("models")
    print(model_folder_path)
    model_name = None
    for file in os.listdir(model_folder_path):
        if file.endswith(".obj"):
            model_name = file[:-4]
            return model_name


if __name__ == "__main__":
    bag_folder_path = "/home/lab4dv/data/bags/sprayer/sprayer_1_20231209"

    model_name = get_model_name(bag_folder_path)
    icp_result = False




    cam_index = 0

    # transforms = json.load(open(str(Path(bag_folder_path) / Path("global_name_position/0.txt")),"r"))
    # cam0_rgb_camera_link2world = np.array(transforms["cam3_rgb_camera_link"])
    # #simplify_persentage = 0.9
    
    transform_mesh_interval = [0,2000]

    transform_obj2result_pose(bag_folder_path, model_name,
                              transform_mesh_interval,
                              None,
                            icp_result=icp_result,
                            icp_mode_=3,
                            file_name="tracking_and_icp_mode_3.txt"
)



