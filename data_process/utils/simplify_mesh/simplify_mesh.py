import open3d as o3d
from pathlib import Path
import trimesh
import numpy as np

import torch


def change_mesh_from_trimesh_to_open3d(trimesh_mesh):#if the mesh is watertight originally,so simplify the mesh is watertight

    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)

    # 可以对Open3D网格执行其他操作
    # 例如，计算法线
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def convert_open3d_mesh_to_trimesh(open3d_mesh):
    # 从Open3D的mesh中获取顶点和面
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)
    
    # 创建一个trimesh的mesh对象
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return trimesh_mesh

def simplify_mesh(mesh,proportion = 0.02):
    mesh = mesh.simplify_quadric_decimation(int(proportion * len(mesh.triangles)))
    mesh = mesh.remove_unreferenced_vertices()
    return mesh

def make_mesh_2_watertight(trimesh_mesh):#this func can't gurantee make mesh to watertight
    trimesh_mesh.fill_holes()

    return trimesh_mesh

def down_sample_mesh_points(pcd,voxel_size = 0.01):
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def downsample_mesh_triangle():
    folder = Path("/media/tony/T7/camera_data/test_object_position_optimize/")
    obj_name = Path("only_hand_1015.ply")
    path = folder / obj_name
    mesh = trimesh.load_mesh(str(path))
    mesh = make_mesh_2_watertight(mesh)
    mesh = change_mesh_from_trimesh_to_open3d(mesh)
    print("nothing")
    # mesh = o3d.io.read_triangle_mesh(str(path))
    print(mesh.is_watertight)
    mesh = simplify_mesh(mesh)
    output_path = folder / f"simplify{obj_name}"
    o3d.io.write_triangle_mesh(str(output_path),mesh)


def resample_and_reconstruct_pcd():
    folder = Path("/media/tony/T7/camera_data/test_object_position_optimize/")
    obj_name = Path("banana.obj")
    path = folder / obj_name
    mesh = o3d.io.read_triangle_mesh(str(path))
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=3000)
    voxel_size = 0.001
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    output_path = folder / f"simplify{obj_name}"
    o3d.io.write_point_cloud(str(output_path), downsampled_point_cloud)
        # # 从网格中降采样点云
    # point_cloud = mesh.sample_points_uniformly(number_of_points=1000)

    # output_path = folder / f"simplify{obj_name}"
    # point_cloud.export(output_path)


def downsample_mesh_pcd_output_pcd():
    folder = Path("/media/tony/T7/camera_data/test_object_position_optimize/")
    obj_name = Path("only_hand_1015.ply")
    path = folder / obj_name

    mesh = o3d.io.read_triangle_mesh(str(path))
    pcd = mesh.sample_points_poisson_disk(number_of_points=6000)
    output_path = folder / f"simplify{obj_name}"
    o3d.io.write_point_cloud(str(output_path), pcd)

if __name__ == "__main__":
    downsample_mesh_pcd_output_pcd()