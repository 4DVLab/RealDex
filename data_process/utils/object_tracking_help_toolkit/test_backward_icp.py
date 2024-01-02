import numpy as np
from pathlib import Path
import os
import open3d as o3d
import json


def load_cam0_to_world_trasnform(bag_folder_path):
    file_path = Path(bag_folder_path) / f"global_name_position/0.txt"
    with open(file_path, "r") as file_reader:
        all_transforms = json.load(file_reader)
    return all_transforms["cam0_rgb_camera_link"]


def test_back_ward_transform(bag_folder_path,test_index,model_name):
    backward_matrix_file_path = Path(bag_folder_path) / f"poses/gt_{test_index}.txt"
    backward_matrix = np.loadtxt(backward_matrix_file_path).reshape((4,4))

    model_folder_path = Path(bag_folder_path) / f"models"
    print(model_folder_path)
    model_mesh = o3d.io.read_triangle_mesh(str(model_folder_path / f"{model_name}.obj"))

    cam0rgb_transform_to_world = load_cam0_to_world_trasnform(bag_folder_path)


    model_mesh.transform(backward_matrix)

    write_path = model_folder_path / f"transform_{model_name}.obj"
    o3d.io.write_triangle_mesh(str(write_path),model_mesh)


if __name__ == "__main__":
    test_back_ward_transform("/home/lab4dv/data/sda/yogurt/original/yogurt_1_20231105",
                            8,"simplication_yogurt")