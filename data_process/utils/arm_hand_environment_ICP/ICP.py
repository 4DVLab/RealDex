import open3d as o3d
import numpy as np


# 加载第一个PLY文件
target_pcd = o3d.io.read_point_cloud("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/pc_environment_only_hand.ply")



source_pcd = o3d.io.read_point_cloud("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/only_hand.ply")

transform_matrix = np.array([
   [ 1.000000, 0.000000 ,0.000000 ,0.004169 ],
   [ 0.000000, 1.000000, 0.000000 ,-0.018018 ],
   [ 0.000000, 0.000000 ,1.000000, -0.030953 ],
    [0.000000 ,0.000000 ,0.000000, 1.000000 ]
])
source_pcd.transform(transform_matrix)
o3d.visualization.draw_geometries([source_pcd])
# source_pcd = source_pcd.sample_points_poisson_disk(100000)
# point_cloud.paint_uniform_color([0.7, 0.7, 0.7])  
# transform = np.array([
#     [-0.152401 ,0.957698, -0.244109, 1.386118 ],
#     [-0.882691 ,-0.242996 ,-0.402256, 0.105873 ],
#     [-0.444557 ,0.154168 ,0.882384, 0.570952 ],
#     [0.000000, 0.000000, 0.000000, 1.000000 ]
# ])

# source_pcd.transform(transform)

# 执行ICP
icp_result = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, max_correspondence_distance=0.000000000000001,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))

# 打印ICP结果
print(icp_result)

final_pcd = source_pcd.transform(icp_result.transformation)
print(icp_result.transformation)
o3d.visualization.draw_geometries([final_pcd,target_pcd])
o3d.io.write_point_cloud(str("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/tranform_arm_hand_mesh.ply"), final_pcd)
# o3d.io.write_triangle_mesh(str("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/fusion_align/world_0.ply"), target_pcd)
# final_pcd += target_pcd
# o3d.visualization.draw_geometries([final_pcd])