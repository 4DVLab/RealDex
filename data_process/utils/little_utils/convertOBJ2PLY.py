import open3d as o3d
import sys
import numpy as np
def convert_obj_to_ply(obj_path, ply_path):
    # 读取OBJ文件
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 写入PLY文件
    o3d.io.write_triangle_mesh(ply_path, mesh)

if __name__ == "__main__":
    folder = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/sample/models/"
    convert_obj_to_ply(folder + "sample.obj", folder + "sample.ply")
    # for index in np.arange(246,603):
    #     obj_path = folder +"world_"+ str(index) + ".obj"
    #     ply_path = folder + str(index) + ".ply"
    #     convert_obj_to_ply(obj_path, ply_path)
    # print(f"{obj_path} converted to {ply_path} successfully!")


