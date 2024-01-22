import numpy as np
import os
import re
from tqdm import tqdm, trange
import json
# os.environ['EGL_PLATFORM'] = 'surfaceless'   # Ubunu 20.04+
# os.environ['OPEN3D_CPU_RENDERING'] = 'true'  # Ubuntu 18.04
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation




def segment_scene_point_cloud(scene_pcd, sr_mesh):
    scene_pcd_tree = o3d.geometry.KDTreeFlann(scene_pcd)
    
    mesh_pcd = sr_mesh.sample_points_poisson_disk(number_of_points=4000)
    mesh_pcd.estimate_normals()
    

    idx_list = []
    distance = 0
    counter = 0
    for point in mesh_pcd.points:
        [_, idx, _] = scene_pcd_tree.search_radius_vector_3d(point, radius=0.01)
        if len(idx) > 0:
            closet_pt = scene_pcd.points[idx[0]]
            distance += np.linalg.norm(point - closet_pt)
            counter += 1      
        idx_list += idx
        
    distance /= counter
    idx_list = np.unique(idx_list)
    # seg_points_ratio = len(idx_list)/len(scene_pcd.points)
    seg_points_ratio = counter/len(mesh_pcd.points)

    return seg_points_ratio, distance, idx_list

# Function to extract the id number from the filename
def extract_id(filename):
    match = re.search(r'index(\d+)\.ply', filename)
    if match is not None:
        return int(match.group(1))
    return None  # return a default value if no id is found

def find_closest(lst, key):
    return min(lst, key=lambda x: abs(x - key))

def extract_hand_mesh(data_dir, start_time_sr, start_scene_id=0):
    time_stamp_file = os.path.join(data_dir, "rgbimage_timestamp.txt")
    scene_time_list = np.loadtxt(time_stamp_file)
    start_time_scene = scene_time_list[start_scene_id]
    sr_mesh_dir = os.path.join(data_dir, "srhand_ur_meshes")
    out_path = os.path.join(data_dir, "scene_to_mesh.json")
    ret_dict = {}
    
    sr_mesh_file_list = os.listdir(sr_mesh_dir)
    sr_time_list = [int(x.split('.')[0]) for x in sr_mesh_file_list]
    sr_time_list = sorted(sr_time_list)
    
    for i, time_scene in enumerate(scene_time_list):
        delta_t = int(time_scene - start_time_scene)
        approx_sr_time = start_time_sr + delta_t
        sr_time = find_closest(sr_time_list, approx_sr_time)
        ret_dict[i] = f"{sr_time}.ply"
    
    with open(out_path, 'w') as f:
        f.write(json.dumps(ret_dict, indent=4))
    

def time_synchronization(sr_mesh_dir, scene_dir, scene_start=0, scene_end=0, save_seg=False, ratio_threshold = 0.06):
    scene_file_list = []
    '''Sort the file list based on the id number'''
    pattern = re.compile(r'^\d+\.ply$')
    for filename in os.listdir(scene_dir):
        if pattern.match(filename):
            scene_file_list.append(filename)
    # scene_file_list = [filename for filename in scene_file_list if extract_id(filename) is not None]
    # scene_file_list = sorted(scene_file_list, key=extract_id)
    scene_file_num = len(scene_file_list)
    scene_end = min(scene_file_num, scene_end)
    
    
    '''if the output file already exists, load it'''
    out_path = os.path.join(scene_dir, "../scene_to_mesh.json")
    print(out_path)
    if os.path.exists(out_path) and scene_start>0:
        with open(out_path, 'r') as file:
            scene_to_mesh = json.load(file)
        latest_scene_file = f"{scene_start-1}.ply"
        latest_sr_file = scene_to_mesh[latest_scene_file]
        print(latest_scene_file, latest_sr_file)
    else:
        scene_to_mesh = {}
        latest_sr_file = None
    sr_mesh_file_list = sorted(os.listdir(sr_mesh_dir), key=lambda x: int(x.split('.')[0]))
    
    sr_start = 0 if latest_sr_file is None else sr_mesh_file_list.index(latest_sr_file)
    print("start from: ", sr_start, latest_sr_file)
    num_sr_mesh = len(sr_mesh_file_list)
    
    seg_dir = os.path.join(scene_dir, "segmented")
    os.makedirs(seg_dir, exist_ok=True)
    # for scene_file in tqdm(scene_file_list):
    for id in trange(scene_start, scene_end):
        scene_file = f"{id}.ply"
        scene_pcd = o3d.io.read_point_cloud(os.path.join(scene_dir,scene_file))
        gotit = False
        progress_bar = trange(sr_start, num_sr_mesh, desc="Processing items", unit="item")
        potential_list = []
        for i in progress_bar:
            sr_mesh_file = sr_mesh_file_list[i]
            sr_mesh = o3d.io.read_triangle_mesh(os.path.join(sr_mesh_dir, sr_mesh_file))
            seg_points_ratio, distance, scene_idx_list = segment_scene_point_cloud(scene_pcd, sr_mesh)
            progress_bar.set_postfix({  "potential_len": len(potential_list),
                                        "ratio": seg_points_ratio, 
                                        "distance":distance}, refresh=True)

            if seg_points_ratio > ratio_threshold and len(potential_list) < 3:
                metric = seg_points_ratio - 50 * distance
                potential_list.append((i, metric, scene_idx_list))
            elif len(potential_list) > 0:
                potential_list = sorted(potential_list, key=lambda x: x[1], reverse=True) # the higher the better
                selected_id, metric, scene_idx_list = potential_list[0]
                scene_to_mesh[id] = int(sr_mesh_file_list[selected_id].split('.')[0])
                sr_start = selected_id + 1
                gotit = True
                break
        if save_seg:
            segmented_scene_pcd = o3d.geometry.PointCloud() 
            segmented_scene_pcd.points.extend(np.asarray(scene_pcd.points)[scene_idx_list])
            segmented_scene_pcd.colors.extend(np.asarray(scene_pcd.colors)[scene_idx_list])
            o3d.io.write_point_cloud(os.path.join(seg_dir, scene_file), segmented_scene_pcd)
        else:
            np.save(os.path.join(seg_dir,f"{id}.npy"), scene_idx_list)
        if gotit == False:
            print(scene_file)
        with open(out_path, 'w') as f:
            f.write(json.dumps(scene_to_mesh, indent=4))
            
    return scene_to_mesh
        
def offline_render():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render = rendering.OffscreenRenderer(640, 480)
    image = vis.capture_screen_float_buffer(False)
    
            
def vis_result(sr_mesh_dir, scene_dir, scene_to_mesh_path=None, export_img = True, out_path=None):
    if scene_to_mesh_path is None:
        scene_to_mesh_path = os.path.join(scene_dir, "scene_to_mesh.json")
    with open(scene_to_mesh_path, 'r') as f:
        scene_to_mesh = json.load(f)
        
    current_pcd = o3d.geometry.PointCloud()
    current_mesh = o3d.geometry.TriangleMesh()
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the point cloud to the visualization window
    vis.add_geometry(current_pcd)
    vis.add_geometry(current_mesh)
    # Callback function for animation
    iterator = iter(scene_to_mesh.items())
    counter = 0
    camera_parameters = o3d.io.read_pinhole_camera_parameters("./utils/camera_param.json")
    
    def load_next(vis):
        nonlocal iterator, current_pcd, current_mesh, counter, camera_parameters
        try:
            scene_file, mesh_file = next(iterator)
        except StopIteration:
            return True
        scene_file = os.path.join(scene_dir, scene_file)
        mesh_file = os.path.join(sr_mesh_dir, mesh_file)
        

        # Load the next point cloud and mesh
        current_pcd = o3d.io.read_point_cloud(scene_file)
        current_mesh = o3d.io.read_triangle_mesh(mesh_file)
        current_mesh.compute_vertex_normals()

        # Clear the old geometries and add the new ones
        vis.clear_geometries()
        vis.add_geometry(current_pcd)
        vis.add_geometry(current_mesh)
        
        # Set the camera parameters for the current frame
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_parameters)
        
        if export_img:
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(out_path, '{:05d}.png'.format(counter)), np.asarray(image), dpi=1)
        counter += 1

        return False
    
    vis.register_animation_callback(load_next)
    
    
    vis.run()
    vis.destroy_window()
    
            
            
def single_test():
    '''
    single frame test fot time sync
    '''
    mesh_file = os.path.join(sr_mesh_dir, "1702129107982117888.ply")
    sr_mesh = o3d.io.read_triangle_mesh(mesh_file)

    pcd_file = os.path.join(scene_dir, "264.ply")
    scene_pcd = o3d.io.read_point_cloud(pcd_file)

    seg_points_ratio, distance, scene_idx_list = segment_scene_point_cloud(scene_pcd, sr_mesh)
    print(seg_points_ratio, distance)
    segmented_scene_pcd = o3d.geometry.PointCloud() 
    segmented_scene_pcd.points.extend(np.asarray(scene_pcd.points)[scene_idx_list])
    segmented_scene_pcd.colors.extend(np.asarray(scene_pcd.colors)[scene_idx_list])

    o3d.io.write_point_cloud(os.path.join(scene_dir, "264_seg.ply"), segmented_scene_pcd)
    o3d.visualization.draw_geometries([segmented_scene_pcd])
    
def pairwise_icp(source, target, threshold=0.02):
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return icp_result.transformation

def merge_scene_pcd(data_dir, scene_id, scene_to_mesh):
    os.makedirs(os.path.join(data_dir, "merged"), exist_ok=True)
    '''Load scence pcd files'''
    pcd_list = []
    for i in range(4):
        scene_file = os.path.join(data_dir, f"cam{i}", "pcd", f"{scene_id}.ply")
        scene_pcd = o3d.io.read_point_cloud(scene_file)
        pcd_list.append(scene_pcd)
    
    '''Load hand meshes'''
    mesh_file = os.path.join(data_dir, "srhand_ur_meshes", scene_to_mesh[f"{scene_id}.ply"])
    sr_mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh_pcd = sr_mesh.sample_points_poisson_disk(number_of_points=4000)
    
    '''Segment pcd'''
    seg_scene_pcd_list = []
    tf_icp_list = []
    for i, scene_pcd in enumerate(pcd_list):
        pcd_tree = o3d.geometry.KDTreeFlann(scene_pcd)
        segmented_scene_pcd = o3d.geometry.PointCloud() 
        for point in mesh_pcd.points:
            [_, idx, _] = pcd_tree.search_radius_vector_3d(point, radius=0.01)
            if len(idx) > 0:
                segmented_scene_pcd.points.extend(np.asarray(scene_pcd.points)[idx])
                segmented_scene_pcd.colors.extend(np.asarray(scene_pcd.colors)[idx])
        seg_scene_pcd_list.append(segmented_scene_pcd)
        
        transformation_icp = pairwise_icp(segmented_scene_pcd, mesh_pcd)
        tf_icp_list.append(transformation_icp)
        
        o3d.io.write_point_cloud(os.path.join(data_dir, "merged", f"seg_cam{i}_{scene_id}.ply"), segmented_scene_pcd)
        segmented_scene_pcd.transform(transformation_icp)
        o3d.io.write_point_cloud(os.path.join(data_dir, "merged", f"seg_trans_cam{i}_{scene_id}.ply"), segmented_scene_pcd)
        
    
    merged = o3d.geometry.PointCloud() 
    for i, tf in enumerate(tf_icp_list):
        scene_pcd = pcd_list[i]
        scene_pcd.transform(tf)
        merged.points.extend(np.asarray(scene_pcd.points))
        merged.colors.extend(np.asarray(scene_pcd.colors))
    o3d.io.write_point_cloud(os.path.join(data_dir, "merged", f"merged_{scene_id}.ply"), merged)

def tf_to_mat(tf):
    transl = tf[:3]
    rot = Rotation.from_quat(tf[3:])
    mat = np.zeros((4, 4))
    mat[:3, :3] = rot.as_matrix()
    mat[:3, -1] = transl
    mat[-1, -1] = 1
    return mat

def rotmat_to_angleaxis(mat):
    rot = Rotation.from_matrix(mat)
    angle_xis = rot.as_rotvec()
    return angle_xis

def rpy_to_mat(rpy):
    rot = Rotation.from_euler('XYZ', rpy, degrees=False)
    return rot.as_matrix()

def xyzrpy_to_mat(xyz, rpy):
    rot = Rotation.from_euler('XYZ', rpy, degrees=False)
    tran = np.array(xyz)
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = tran
    return tf

def rotate_axis(axis, rpy):
    rot = Rotation.from_euler('XYZ', rpy, degrees=False)
    axis = rot.apply(axis)
    return axis

def check_rotation_axis(mat, axis):
    '''Normalize the axis to make sure it's a unit vector'''
    axis = axis / np.linalg.norm(axis)
    # Apply the rotation matrix to the rotation axis
    axis_rotated = mat @ axis

    # Check if v_rotated is the same as v (within a tolerance)
    if np.allclose(axis_rotated, axis, atol=1e-2):

        # rot = Rotation.from_matrix(mat)
        # angle_axis = rot.as_rotvec() # angle * axis
        # angle = np.linalg.norm(angle_axis)
        # gt_axis = angle_axis / angle
        # if np.dot(angle_axis, gt_axis)<0:
        #     gt_axis *= -1
        #     angle *= -1

        # print(axis, gt_axis, angle) 
        # The axis is an eigenvector of the rotation matrix with eigenvalue 1.
        return True
    else:
        rot = Rotation.from_matrix(mat)
        angle_axis = rot.as_rotvec() # angle * axis
        angle = np.linalg.norm(angle_axis)
        gt_axis = angle_axis / angle
        if np.dot(angle_axis, gt_axis)<0:
            gt_axis *= -1
            angle *= -1
        
        print("The rotation matrix R does not correspond to a rotation around the given axis.")
        print(axis, gt_axis, angle_axis , angle) 
        return False

def compute_joint_angle(rot_mat, axis):
    '''Normalize the axis to make sure it's a unit vector'''
    axis = axis / np.linalg.norm(axis)
    rot = Rotation.from_matrix(rot_mat)
    angle_axis = rot.as_rotvec() # angle * axis
    angle = np.dot(axis, angle_axis)
    return angle

def batched_rotmat_to_vec(batched_rotation_matrices):
    # Calculate the angle of rotation for each matrix
    angles = np.arccos(((np.trace(batched_rotation_matrices, axis1=1, axis2=2) - 1) / 2).clip(-1, 1))
    
    # Allocate space for the batched axis of rotation
    batch_size = batched_rotation_matrices.shape[0]
    axis_of_rotation = np.zeros((batch_size, 3))
    
    # Epsilon to prevent division by zero
    epsilon = 1e-6
    
    # Calculate each axis of rotation
    for i in range(batch_size):
        # The axis is the normalized vector of the matrix's off-diagonal elements
        # [R32 - R23, R13 - R31, R21 - R12] / (2*sin(θ))
        R = batched_rotation_matrices[i]
        axis_unnormalized = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        sin_theta = np.sin(angles[i])
        if sin_theta > epsilon:
            axis_of_rotation[i] = axis_unnormalized / (2 * sin_theta)
        else:
            # The axis does not matter for zero angle, but we set to [1, 0, 0] for convention
            axis_of_rotation[i] = np.array([1, 0, 0])
    
    # Multiply the angle with the axis to get the angle-axis representation
    angle_axis_batch = angles[:, np.newaxis] * axis_of_rotation
    return angle_axis_batch

def vis_hand_object(data_dir, tracking_file, obj_mesh_file, scene_to_mesh, out_path):    
    '''Load object mesh'''
    obj_tf = np.loadtxt(tracking_file)
    
    obj_mesh = o3d.io.read_triangle_mesh(obj_mesh_file)
    obj_center = np.asarray(obj_mesh.get_center())
    obj_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(obj_mesh.vertices)-obj_center)
    
    
    '''Load Object Mesh'''
    current_obj_mesh = o3d.geometry.TriangleMesh()
    current_hand_mesh = o3d.geometry.TriangleMesh()
    current_obj_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(obj_mesh.triangles))
    obj_meshes_list = []
    for scene_file in scene_to_mesh:
        # Load the object pose
        scene_id = int(scene_file.split('.')[0])
        obj_poses_mat = tf_to_mat(obj_tf[scene_id])
        current_obj_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(obj_mesh.vertices))
        current_obj_mesh.transform(obj_poses_mat)
        current_obj_mesh.compute_vertex_normals()
        obj_meshes_list.append(current_obj_mesh)
        
        o3d.io.write_triangle_mesh(os.path.join(data_dir, "obj_mesh", f"{scene_id}.ply"), current_obj_mesh)
    
    '''Init the animation setting'''
    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add the point cloud to the visualization window
    vis.add_geometry(current_obj_mesh)
    vis.add_geometry(current_hand_mesh)
    # Callback function for animation
    iterator = iter(scene_to_mesh.items())
    counter = 0
    camera_parameters = o3d.io.read_pinhole_camera_parameters("./utils/camera_param.json")
    
    def load_next(vis):
        nonlocal iterator, current_obj_mesh, current_hand_mesh, counter, camera_parameters, obj_meshes_list
        try:
            scene_file, mesh_file = next(iterator)
            scene_id = int(scene_file.split('.')[0])  
        except StopIteration:
            return True
        scene_file = os.path.join(scene_dir, scene_file)
        mesh_file = os.path.join(sr_mesh_dir, mesh_file)
        
        # obj_poses_mat = cam_pose @ obj_poses_mat
        current_obj_mesh = obj_meshes_list[scene_id]
        
        current_hand_mesh = o3d.io.read_triangle_mesh(mesh_file)
        current_hand_mesh.compute_vertex_normals()

        # Clear the old geometries and add the new ones
        vis.clear_geometries()
        vis.add_geometry(current_obj_mesh)
        vis.add_geometry(current_hand_mesh)
        
        # Set the camera parameters for the current frame
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_parameters)
        
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(os.path.join(out_path, '{:05d}.png'.format(counter)), np.asarray(image), dpi=1)
        counter += 1

        return False
    
    vis.register_animation_callback(load_next)
    
    
    vis.run()
    vis.destroy_window()
    


if __name__ == '__main__':
    # sr_mesh_dir = "/home/lab4dv/yumeng/results/srhand_ur_meshes/test_1"
    # sr_mesh_dir = "/Users/yumeng/Working/data/CollectedDataset/sprayer_1_20231209/srhand_ur_meshes"
    sr_mesh_dir = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207/srhand_ur_meshes/"
    
    # scene_dir = "/home/lab4dv/data/bags/test/backup/test_1/merged_pcd_filter/cam3"
    scene_dir = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207/cam3/pcd"
    
    data_dir = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207"

    #<------------------------->#
    # single_test()
    
    #<------------------------->#
    # time_synchronization(sr_mesh_dir, scene_dir, scene_start=115, ratio_threshold=0.09)
    
    #<------------------------->#
    # out_path=os.path.join(data_dir, "time_sync_vis-cam3")
    # os.makedirs(out_path, exist_ok=True)
    # vis_result(sr_mesh_dir, scene_dir, out_path=out_path)
    
    #<------------------------->#
    # scene_to_mesh_path = os.path.join(scene_dir, "scene_to_mesh.json")
    # with open(scene_to_mesh_path, 'r') as f:
    #     scene_to_mesh = json.load(f)
    
    # merge_scene_pcd(data_dir, scene_id=0, scene_to_mesh=scene_to_mesh)
    
    #<---------------------------#
    tracking_file = os.path.join(os.path.dirname(data_dir), "tracking_result/yogurt_1.txt")
    scene_to_mesh_path = os.path.join(scene_dir, "scene_to_mesh.json")
    with open(scene_to_mesh_path, 'r') as f:
        scene_to_mesh = json.load(f)
    obj_mesh_file = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt.obj"
    
    global_pose_file = os.path.join(data_dir, "global_name_position/0.txt")
    global_pose = json.load(open(global_pose_file, 'r'))
    cam_pose = np.asarray(global_pose["cam0_rgb_camera_link"])
    # print(cam_pose)
    out_path = os.path.join(data_dir, "hand_obj_vis")
    os.makedirs(out_path, exist_ok=True)
    vis_hand_object(data_dir,tracking_file, obj_mesh_file, scene_to_mesh, out_path)
        
    
    