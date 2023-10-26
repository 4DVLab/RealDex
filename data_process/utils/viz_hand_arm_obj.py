import open3d as o3d
import os
import numpy as np

def display_meshes_in_sequence( file_pattern="{}.ply", start_index=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    
    camera_params = o3d.io.read_pinhole_camera_parameters("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/test_tracking_duck/rh_palm_0.obj_camera_params.json")
    index = start_index

    folder_arm_hand_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/test_tracking_duck/merge_mesh/"
    folder_banana_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/yyx_tmp/test_tracking_duck/object_pose/"


    for index in np.arange(40):
        banana_path = folder_banana_path + str(index) +".obj"
        arm_hand_path = folder_arm_hand_path + "world_"+str(index) + ".obj"

        banana_mesh = o3d.io.read_triangle_mesh(banana_path)
        arm_hand_mesh = o3d.io.read_triangle_mesh(arm_hand_path)
        
        arm_hand_mesh += banana_mesh
        # Recompute the normals
        arm_hand_mesh.compute_vertex_normals()
        
        if index == 0:
            vis.add_geometry(arm_hand_mesh)
        else:
            vis.clear_geometries()
            vis.add_geometry(arm_hand_mesh)
        
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        
        vis.run()  # q
        

    vis.destroy_window()

if __name__ == "__main__":
    display_meshes_in_sequence()


# import open3d as o3d
# import os

# def display_meshes_in_sequence(folder_path, file_pattern="{}.ply", start_index=0):
#     vis = o3d.visualization.Visualizer()
#     vis.create_window()
#     camera_params = o3d.io.read_pinhole_camera_parameters("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/camera_params.json")
#     index = start_index

#     while True:
#         file_path = os.path.join(folder_path, file_pattern.format(index))
#         if not os.path.exists(file_path):
#             break
        
#         mesh = o3d.io.read_triangle_mesh(file_path)
        
#         if index == start_index:
#             vis.add_geometry(mesh)
#         else:
#             vis.clear_geometries()
#             vis.add_geometry(mesh)
#             vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
#             vis.poll_events()
#             vis.update_renderer()
        
#         print("Showing mesh:", file_path)
#         vis.run()  # q
        
#         index += 1

#     vis.destroy_window()

# if __name__ == "__main__":

#     folder_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/merge_mesh/"
#     display_meshes_in_sequence(folder_path)

    
