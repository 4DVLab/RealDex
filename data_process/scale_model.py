import open3d as o3d
import numpy as np

def scale_mesh(mesh_path, scale_factor=0.001):
    """
    读入一个 mesh，并对其 xyz 坐标乘以指定的放缩因子。

    参数:
        mesh_path (str): mesh 文件路径。
        scale_factor (float): 放缩因子，默认为 0.001。

    返回:
        o3d.geometry.TriangleMesh: 放缩后的 mesh。
    """
    # 读取 mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # 对所有顶点坐标进行放缩
    vertices = np.asarray(mesh.vertices).copy()  * scale_factor
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    return mesh
path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/test_tracking_duck/models/duck.obj"
# 使用示例
scaled_mesh = scale_mesh(path)
o3d.io.write_triangle_mesh(path, scaled_mesh)



