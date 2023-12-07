import open3d as o3d


# 保存open3d参数

def save_camera_parameters(vis):
    parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(
        "/media/tony/新加卷/test_data/test/test_1/object_pose_in_every_frame_with_icp/ca", parameters)


bag_path = "/media/tony/新加卷/test_data/test/test_1/object_pose_in_every_frame_with_icp/17.ply"
mesh = o3d.io.read_triangle_mesh(bag_path)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(mesh)
# 81 is the key code for "Q"
vis.register_key_callback(81, save_camera_parameters)
vis.run()
vis.destroy_window()


# # pcd点云显示
# pcd = o3d.io.read_point_cloud('xxx.pcd')
# print(pcd)
# o3d.visualization.draw_geometries([pcd], window_name="Open3D0")

# obj面片显示


# def display_like_meshlab(obj_path):
#     # 读取OBJ文件
#     mesh = o3d.io.read_triangle_mesh(obj_path)

#     # 确保mesh具有法线信息，如果没有，则为其计算法线
#     if not mesh.has_vertex_normals():
#         print("not normal")
#         mesh.compute_vertex_normals()

#     # 创建可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()

#     # 添加mesh到可视化窗口
#     vis.add_geometry(mesh)

#     # 配置渲染选项以模仿MeshLab效果
#     render_option = vis.get_render_option()
#     render_option.light_on = True
#     render_option.mesh_show_back_face = True

#     # 其他可能的渲染选项调整（根据需要调整）
#     render_option.point_size = 1.0
#     render_option.line_width = 1.0
#     render_option.mesh_show_wireframe = False
#     render_option.background_color = [1, 1, 1]  # 白色背景

#     # 设置视角和光源以模仿MeshLab效果
#     view_ctrl = vis.get_view_control()
#     cam_params = view_ctrl.convert_to_pinhole_camera_parameters()

#     # 根据需要调整摄像机和光源位置
#     cam_params.intrinsic.set_intrinsics(800, 800, 400, 400, 400, 400)
#     cam_params.extrinsic = [[1, 0, 0, 0],
#                             [0, 1, 0, 0],
#                             [0, 0, 1, 2],  # 修改Z值以缩放视图
#                             [0, 0, 0, 1]]
#     view_ctrl.convert_from_pinhole_camera_parameters(cam_params)

#     # 启动可视化
#     vis.run()
#     vis.destroy_window()

# if __name__ == '__main__':
#     obj_path = '/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/ur_description/meshes/ur10e/visual/base.obj'
#     display_like_meshlab(obj_path)


# obj_path = '/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/ur_description/meshes/ur10e/visual/base.obj'
# textured_mesh= o3d.io.read_triangle_mesh(obj_path)
# # print(textured_mesh)
# # textured_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([textured_mesh], window_name="Open3D1")

# # obj顶点显示
# pcobj = o3d.geometry.PointCloud()
# pcobj.points = o3d.utility.Vector3dVector(textured_mesh.vertices)
# o3d.visualization.draw_geometries([pcobj], window_name="Open3D2")

# # obj顶点转array
# textured_pc =np.asarray(textured_mesh.vertices)
# print(textured_pc)


# import pywavefront
# import open3d as o3d

# # 读取 OBJ 文件
# mesh = pywavefront.Wavefront('/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/ur_description/meshes/ur10e/visual/base.obj')

# # 获取顶点和面数据
# vertices = mesh.vertices
# faces = mesh.mesh_list[0].faces

# print(faces)

# # 创建 Open3D 点云对象
# point_cloud = o3d.geometry.PointCloud()

# # 设置点云的顶点坐标
# point_cloud.points = o3d.utility.Vector3dVector(vertices)

# # 创建 Open3D 三角面片对象
# triangle_mesh = o3d.geometry.TriangleMesh()

# # 设置三角面片的顶点坐标和面数据
# triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices)
# triangle_mesh.triangles = o3d.utility.Vector3iVector(faces)

# # 创建可视化窗口
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # 将点云和三角面片添加到可视化窗口
# vis.add_geometry(point_cloud)
# vis.add_geometry(triangle_mesh)

# # 设置观察视角
# vis.get_view_control().set_front([0, 0, -1])
# vis.get_view_control().set_up([0, -1, 0])
# vis.get_view_control().set_lookat([0, 0, 0])
# vis.get_view_control().set_zoom(0.8)

# # 显示点云和三角面片
# vis.run()
# vis.destroy_window()
