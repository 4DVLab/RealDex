import open3d as o3d
from pathlib import Path
import numpy as np




def add_lines(vis,direction,pcd):
    vertices = np.array(pcd.points)
    for vert in vertices:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.array([vert,vert+direction * 0.1]))
        line.lines = o3d.utility.Vector2iVector(np.array([[0,1]]))
        line.colors = o3d.utility.Vector3dVector(np.array([[1,0,0]]))
        vis.add_geometry(line)


def show_hand_arm_with_the_direction_ray():


    vis = o3d.visualization.Visualizer()
    vis.create_window()


    folder = "/home/tony/mine/Projects/test_object_position_optimize"
    folder = Path(folder)

    camera_param_path = folder / Path("camera_params.json")
    camera_params = o3d.io.read_pinhole_camera_parameters(str(camera_param_path))



    hand_pcd_path = folder / Path("viz_model/1000times_only_hand_1015.ply")
    hand_pcd = o3d.io.read_point_cloud(str(hand_pcd_path))
    obj_mesh_path = folder / Path("viz_model/1000times_banana.obj")
    obj_mesh = o3d.io.read_point_cloud(str(obj_mesh_path))
    # obj_pcd = obj_mesh.sample_points_uniformly(500)


    # original_obj_mesh.compute_vertex_normals()

    



    direction = np.array([0.4395064455, 0.617598629942, 0.652231566745])
    # add_lines(vis,direction,hand_pcd)



    vis.add_geometry(hand_pcd)
    vis.add_geometry(obj_mesh)

    vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params,allow_arbitrary = True)


    vis.poll_events()
    vis.update_renderer()
    
    vis.run()

    vis.destroy_window()

if __name__ == "__main__":
    show_hand_arm_with_the_direction_ray()