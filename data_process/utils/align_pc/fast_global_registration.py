import open3d as o3d
import numpy as np


# 加载第一个PLY文件
target_pcd = o3d.io.read_point_cloud(
    "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/only_hand_arm.ply")


source_pcd = o3d.io.read_triangle_mesh(
    "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/hand_arm_mesh.obj")
# o3d.visualization.draw_geometries([source_pcd])

source_pcd = source_pcd.sample_points_poisson_disk(10000)
# # point_cloud.paint_uniform_color([0.7, 0.7, 0.7])
# transform = np.array([
# [0.946047 ,0.007798 ,0.323937, -0.251488],
# [0.032185 ,0.992505, -0.117887, 0.076864],
# [-0.322428 ,0.121952, 0.938705, 0.530148],
# [0.000000, 0.000000, 0.000000, 1.000000]
# ])

# source_pcd.transform(transform)
radius_normal = 0.05  # 选择一个适当的值进行法线估计
radius_feature = 0.1  # 选择一个适当的值进行特征估计

# 为两个点云估计法线
source_pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
target_pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

# 计算FPFH特征
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))



# Use Fast Global Registration
fg_result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
    source_pcd, target_pcd, source_fpfh, target_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=0.000005)
)

# Optional: refine with ICP
icp_result = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, max_correspondence_distance=0.00000001,
    init=fg_result.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
)

# 打印ICP结果
print(icp_result)

final_pcd = source_pcd.transform(icp_result.transformation)
print(icp_result.transformation)
o3d.visualization.draw_geometries([final_pcd, target_pcd])
o3d.io.write_point_cloud(str(
    "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/hand_ICP/tranform_arm_hand_mesh.ply"), final_pcd)
# o3d.io.write_triangle_mesh(str("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/fusion_align/world_0.ply"), target_pcd)
# final_pcd += target_pcd
# o3d.visualization.draw_geometries([final_pcd])
