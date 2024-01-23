import open3d as o3d
import numpy as np

hand_arm_mesh_path = "/home/lab4dv/data/bags/blue_cup/blue_cup_1_20240106/arm_hand_mesh/64.ply"
env_path = "/home/lab4dv/data/bags/blue_cup/blue_cup_1_20240106/sample_63.ply"
viz_camera_info_path = "/home/lab4dv/data/camera_param.json"

env_pcd = o3d.io.read_point_cloud(env_path)
hand_arm_mesh = o3d.io.read_triangle_mesh(hand_arm_mesh_path)
hand_arm_mesh.compute_vertex_normals()

print(np.array(env_pcd.colors))

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(env_pcd)
# vis.add_geometry(hand_arm_mesh)
# camera_params = o3d.io.read_pinhole_camera_parameters(viz_camera_info_path)
# vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
# vis.poll_events()
# vis.update_renderer()
vis.run()
# image = vis.capture_screen_float_buffer(False)
# plt.imsave(image_save_folder /
#            Path(f"{index}.png"), np.asarray(image), dpi=1)
