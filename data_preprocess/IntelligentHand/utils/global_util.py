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


def segment_scene_point_cloud(scene_pcd, object_mesh):
    distance_threshold = 0.005
    scene_pcd_tree = o3d.geometry.KDTreeFlann(scene_pcd)
    
    mesh_pcd = object_mesh.sample_points_poisson_disk(number_of_points=4000)
    mesh_pcd.estimate_normals()
    

    idx_list = []
    distance = 0
    counter = 0
    for point in mesh_pcd.points:
        [_, idx, _] = scene_pcd_tree.search_radius_vector_3d(point, distance_threshold)
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

def time_synchronization(sr_mesh_dir, scene_dir, scene_start=0, save_seg=False):
    scene_file_list = []
    '''Sort the file list based on the id number'''
    pattern = re.compile(r'^\d+\.ply$')
    for filename in os.listdir(scene_dir):
        if pattern.match(filename):
            scene_file_list.append(filename)
    # scene_file_list = [filename for filename in scene_file_list if extract_id(filename) is not None]
    # scene_file_list = sorted(scene_file_list, key=extract_id)
    scene_file_num = len(scene_file_list)
    
    
    '''if the output file already exists, load it'''
    out_path = os.path.join(scene_dir, "scene_to_mesh.json")
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
    for id in trange(scene_start, scene_file_num):
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

            if seg_points_ratio > 0.05:
                metric = seg_points_ratio - 50 * distance
                potential_list.append((i, metric, scene_idx_list))
            elif len(potential_list) > 0:
                potential_list = sorted(potential_list, key=lambda x: x[1], reverse=True) # the higher the better
                selected_id, metric, scene_idx_list = potential_list[0]
                scene_to_mesh[scene_file] = sr_mesh_file_list[selected_id] 
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
        
def offline_render():
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render = rendering.OffscreenRenderer(640, 480)
    image = vis.capture_screen_float_buffer(False)
            
def vis_result(sr_mesh_dir, scene_dir, export_img = True, out_path=None):
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


if __name__ == '__main__':
    # sr_mesh_dir = "/home/lab4dv/yumeng/results/srhand_ur_meshes/test_1"
    sr_mesh_dir = "/Users/yumeng/Working/data/CollectedDataset/sprayer_1_20231209/srhand_ur_meshes"
    # scene_dir = "/home/lab4dv/data/bags/test/backup/test_1/merged_pcd_filter/cam3"
    scene_dir = "/Users/yumeng/Working/data/CollectedDataset/sprayer_1_20231209/cam3/pcd"
    
    # single_test()
    out_path = "/Users/yumeng/Working/data/CollectedDataset/sprayer_1_20231209/time_sync_vis/cam3"
    time_synchronization(sr_mesh_dir, scene_dir, scene_start=265)
    # vis_result(sr_mesh_dir, scene_dir, out_path=out_path)
    