import numpy as np
import os
from pathlib import Path

def move_mesh_to_origin(mesh_vertices):
    # 计算网格的几何中心
    mesh_vertices = mesh_vertices * 0.001
    center = np.mean(mesh_vertices, axis=0)

    # 将网格几何中心移动到坐标中心
    centered_vertices = mesh_vertices - center

    return centered_vertices

def read_obj_file(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = line.strip().split()[1:]
                vertex = [float(coord) for coord in vertex]
                vertices.append(vertex)
            elif line.startswith('f '):
                face = line.strip().split()[1:]
                face = [int(index.split('/')[0]) for index in face]
                faces.append(face)

    return np.array(vertices), np.array(faces)

def write_obj_file(file_path, vertices, faces):
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f"v {' '.join(str(coord) for coord in vertex)}\n")
        for face in faces:
            f.write(f"f {' '.join(str(index) for index in face)}\n")

folder_path = "/media/lab4dv/Elements/charmander/charmander"

for file in os.listdir(folder_path):
    if file.endswith(".obj"):
        read_path = Path(folder_path) / file 
        # 读取OBJ文件
        mesh_vertices, mesh_faces = read_obj_file(read_path)

        # 将网格的几何中心移动到坐标中心
        centered_vertices = move_mesh_to_origin(mesh_vertices)

        write_path = Path(folder_path) / file
        # 写入OBJ文件
        write_obj_file(write_path, centered_vertices, mesh_faces)
        