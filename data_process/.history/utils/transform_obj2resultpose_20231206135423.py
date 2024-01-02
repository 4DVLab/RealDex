import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation
from copy import deepcopy
import os
from numba import njit
from concurrent.futures import ProcessPoolExecutor
import json

def simplify_mesh(mesh, fraction=0.8):
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

def seven_num2matrix(seven_num):#translation x,y,z rotation x,y,z,w
    translation = seven_num[4:]
    roatation = seven_num[[1,2,3,0]]
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


def transform_obj2result_pose(folder_path, bag_name,cam2world_transform,model_name):

    mesh_path = Path(folder_path) / Path(bag_name) / Path("models") /  Path(model_name + ".obj")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh = simplify_mesh(mesh)
    print("simple mesh done")
    pose_path = Path(folder_path) / Path(bag_name) / Path("pose_result") /Path("result.txt")
    tranform_matrixs = load_seven_num_pose(pose_path)
    mesh_to_save_folder_path = Path(folder_path) / Path(bag_name) / Path("object_pose")
    if not os.path.exists(mesh_to_save_folder_path):
        os.makedirs(mesh_to_save_folder_path)
    # with ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(save_transformed_mesh, index, pose, cam2world_transform, mesh, mesh_to_save_folder_path) for index, pose in enumerate(tranform_matrixs)]
    #     for future in futures:
    #         future.result()
    for index,pose in enumerate(tranform_matrixs):
        obj_under_world_pose = cam2world_transform @ pose
        mesh_to_save = transform_mesh_with_matrix(obj_under_world_pose,deepcopy(mesh))
        mesh_save_path = mesh_to_save_folder_path / Path(str(index) + ".obj")
        o3d.io.write_triangle_mesh(str(mesh_save_path),mesh_to_save)

def load_transform(folder_path):
    global_transform_path = Path(
        folder_path) / Path("global_name_position") / Path("0.txt")
    
    with open(global_transform_path,"r") as global_transform_reader:
        global_transform = json.load(global_transform_reader,dtype=np.float32)
    return global_transform
if __name__ == "__main__":
    # folder_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/"
    # bag_name = "test_tracking_duck"
    # cam0_rgb_camera_link2world = np.array([
    #     [
    #         0.31842582813078124,
    #         0.5476966118699068,
    #         -0.7737140384699324,
    #         1.9855345041922787
    #     ],
    #     [
    #         0.9331753409732655,
    #         -0.32464135256597637,
    #         0.15424582718363705,
    #         -0.11211594046023274
    #     ],
    #     [
    #         -0.16669965500461342,
    #         -0.7711267169216478,
    #         -0.6144711640679202,
    #         1.2681138982911153
    #     ],
    #     [
    #         0.0,
    #         0.0,
    #         0.0,
    #         1.0
    #     ]
    # ])

    # transform_obj2result_pose(folder_path,bag_name,cam0_rgb_camera_link2world,"duck")
    seven_num = np.array([-0.23955, -0.381539, 0.500804,0.449507, -0.0513386 ,0.546432,0.704783])
    transform_matrix = seven_num2matrix(seven_num)
    folder_path = "/media/tony/新加卷/test_data/test/test_1"
    load_transform(folder_path,"")
    mesh = o3d.io.read_triangle_mesh(
        "/media/tony/新加卷/test_data/test/test_1/models/simplified_yogurt.obj")
    mesh = mesh.transform(transform_matrix)
    o3d.io.write_triangle_mesh(
        "/media/tony/新加卷/test_data/test/test_1/models/transformed_yogurt.obj", mesh)


