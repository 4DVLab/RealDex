import numpy as np
from pathlib import Path
import os
import open3d as o3d
import json


def load_cam0rgb_to_world_transform(bag_path):
    all_transform_file_path = Path(bag_path) / f"global_name_position/0.txt"
    with open(all_transform_file_path, "r") as transform_json_reader:
        all_transforms = json.load(transform_json_reader)
    cam0rgb_to_world_tranform = all_transforms["cam0_rgb_camera_link"]
    cam0rgb_to_world_tranform = np.array(
        cam0rgb_to_world_tranform).reshape((4, 4))
    return cam0rgb_to_world_tranform

def transform_with_one_matrix():
    index = 1130
    bag_folder_path = "/home/lab4dv/data/bags/croissant/croissant_1_20231027"
    cam0_rgb_to_world = load_cam0rgb_to_world_transform(bag_folder_path)
    model_folder_path = Path(bag_folder_path) / f"models/"

    model_name = "simplication_croissant.obj"
    model_path = model_folder_path / model_name
    # for file_name in os.listdir(model_folder_path):
    #     if file_name.endswith(".obj"):
    #         model_name = file_name
    #         model_path = model_folder_path / file_name
    #         break

    model_mesh = o3d.io.read_triangle_mesh(str(model_path))

    back_file_path = Path(bag_folder_path) / f"poses/gt_{index}.txt" 
    matrix = np.loadtxt(back_file_path).reshape((4,4))
    # matrix = np.linalg.inv(cam0_rgb_to_world) @ matrix
    # mid_transform = np.array([
    #         [0.914520, 0.332331 ,0.230671, -0.146380],
    #         [-0.308515 ,0.941778 ,-0.133691, 0.116627],
    #         [-0.261671, 0.051098, 0.963804, 0.060597],
    #         [0.000000, 0.000000, 0.000000, 1.000000]

    # ])
    # matrix = cam0_rgb_to_world @ mid_transform @ matrix
    # np.savetxt(Path(bag_folder_path) / f"poses/gt{index}.txt",matrix )
    # matrix = np.array([
    # [6.905995014183935465e-02, 9.955201037226367733e-01 ,-6.457899326001942386e-02, 1.000000248875259290e+00],
    # [-6.458022483198286312e-02 ,6.905896942971585795e-02, 9.955200918625184414e-01 ,9.999999345670946838e-01],
    # [9.955200238302885918e-01 ,-6.458004200506956005e-02, 6.906012110882978061e-02 ,1.000000001596028421e+00],
    # [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    #  ]
    # )

    model_mesh.transform(matrix)

    write_path = model_folder_path / f"transform_{model_name}"
    o3d.io.write_triangle_mesh(str(write_path),model_mesh)


if __name__ == "__main__":
    transform_with_one_matrix()