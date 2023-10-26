import os
from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d



def seven_num2matrix(seven_num):#translation x,y,z rotation x,y,z,w
    translation = seven_num[:3]
    roatation = seven_num[3:]
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix



def quat2Rotation(quat):
    rotation = Rotation.from_quat(quat)
    return rotation.as_matrix()

def transform_pc(transform,vertices):

    vertices = np.hstack((vertices,np.ones((vertices.shape[0],1),dtype=np.float32))).T
    vertices = seven_num2matrix(transform)[:3,:] @ vertices
    return vertices

hand_arm_transform = np.array([
    [0.998383, 0.010974 ,0.055773 ,-0.070593],
    [-0.008112 ,0.998651, -0.051283, 0.105587],
    [-0.056260, 0.050747, 0.997126, 0.154701],
    [0.000000 ,0.000000 ,0.000000 ,1.000000]
    ])

# for pc_index in np.arange(3):

# file_path = cam_folder + str(pc_index) + ".ply"
mesh = o3d.io.read_triangle_mesh(str("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/hand_ICP_backup/3/environment_pc.ply"))
vertices = np.asarray(mesh.vertices).copy()
vertices = np.hstack((vertices,np.ones((vertices.shape[0],1),dtype=np.float32))).T
vertices = hand_arm_transform[:3,:] @ vertices
# vertices = transform_pc(debug_dict["cam0_depth_camera_link"]["cam0_rgb_camera_link"],vertices)
# vertices = transform_pc(debug_dict["cam0_camera_base"]["cam0_depth_camera_link"],vertices.T)
# vertices = transform_pc(debug_dict["ra_base_link"]["cam0_camera_base"],vertices.T)
# vertices = transform_pc(debug_dict["world"]["ra_base_link"],vertices.T)
vertices = (vertices.T)
mesh.vertices = o3d.utility.Vector3dVector(vertices)
o3d.io.write_triangle_mesh("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/hand_ICP_backup/3/transform_environment_pc.ply", mesh)