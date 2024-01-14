import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
from scipy.spatial.transform import Rotation as R
import json
from utils.kintree import load_sequence
from utils.urdf_util import load_mesh_from_urdf
from utils.global_util import segment_scene_point_cloud, extract_hand_mesh, find_closest
from utils.pcd_util import PCDGenerator
import shutil
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm, trange
import shutil
import re
import open3d as o3d
from pathlib import Path


class DataProcesser():
    def __init__(self, data_dir, cam_param_dir):
        
        self.data_dir = data_dir
        self.tf_data_dir = os.path.join(data_dir, "TF")
        self.hand_mesh_path = None
        
        '''Load Hand Mesh'''
        with open(struct_file, 'r') as f:
            # Load the JSON data
            sr_struct = json.load(f)
            
        link_list = sr_struct['node_names']
        self.mesh_dict = load_mesh_from_urdf(urdf_path,link_list, prefix)
        print(self.mesh_dict.keys())
        
        '''Load TF Data'''
        tf_data_file = os.path.join(self.tf_data_dir, "global_tf_all_in_one.npy")
        if os.path.exists(tf_data_file): 
            tf_data_all_in_one = np.load(tf_data_file, allow_pickle=True)
            tf_data_all_in_one = tf_data_all_in_one.item()
        else:
            tf_data_all_in_one = load_sequence(self.tf_data_dir, struct_file)
            np.save(tf_data_file, tf_data_all_in_one)
        self.tf_data_all_in_one = tf_data_all_in_one
        self.num_tf = len(self.tf_data_all_in_one)
        
        '''PCD generator'''
        self.pcd_generator = PCDGenerator(data_dir, cam_param_dir)
        time_stamp_file = os.path.join(data_dir, "rgbimage_timestamp.txt")
        self.scene_time_list = np.loadtxt(time_stamp_file)
        
        

    def gen_single_hand_mesh(self, time):
        updated_meshes = {}
        tf_data = self.tf_data_all_in_one[time]
        for key in list(self.mesh_dict.keys()):
            tf = tf_data[key]
            # print(tf)
            mesh = self.mesh_dict[key]
            verts = mesh.vertices
            verts = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
            # print(verts.shape)
            verts = verts @ tf.T
            
            new_mesh = trimesh.Trimesh(vertices=verts[:, :3], faces=mesh.faces)
            updated_meshes[key] = new_mesh
            
        combined_mesh = list(updated_meshes.values())
        combined_mesh = trimesh.util.concatenate(combined_mesh)
        
        # Create an Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        # Assign vertices and faces to Open3D mesh
        o3d_mesh.vertices = o3d.utility.Vector3dVector(combined_mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(combined_mesh.faces)
        
        return o3d_mesh

    def gen_hand_mesh(self):
        '''Set out path'''
        self.hand_mesh_path = os.path.join(data_dir, "srhand_ur_meshes")
        os.makedirs(self.hand_mesh_path, exist_ok=True)
    
        for time in self.tf_data_all_in_one:
            combined_mesh = self.gen_single_hand_mesh(time)
            combined_mesh.export(os.path.join(self.hand_mesh_path, f"{time}.ply"))
        
    def remove_hand_mesh(self):
        if self.hand_mesh_path is None:
            return
        try:
            shutil.rmtree(self.hand_mesh_path)
            print(f"The directory {self.hand_mesh_path} has been removed with all its contents")
        except OSError as error:
            print(f"Error: {error.strerror}")
            
            
    def align_scene_handmesh(self, scene_id, max_search, cam_index=3, ratio_threshold = 0.07):
        scene_pcd = self.pcd_generator.gen_pcd(scene_id, cam_index)
        
        sr_time_list = self.tf_data_all_in_one.keys()
        progress_bar = tqdm(sorted(sr_time_list), desc="Processing items", unit="item")
        potential_list = []
        count = 0
        for sr_time in progress_bar:
            sr_mesh = self.gen_single_hand_mesh(sr_time)
            seg_points_ratio, distance, scene_idx_list = segment_scene_point_cloud(scene_pcd, sr_mesh)
            progress_bar.set_postfix({  "potential_len": len(potential_list),
                                        "ratio": seg_points_ratio, 
                                        "distance":distance}, refresh=True)

            if seg_points_ratio > ratio_threshold and len(potential_list) < 3:
                metric = seg_points_ratio - 50 * distance
                potential_list.append((sr_time, metric, scene_idx_list))
                selected_id = sr_time
            elif len(potential_list) > 0:
                potential_list = sorted(potential_list, key=lambda x: x[1], reverse=True) # the higher the better
                selected_id, metric, scene_idx_list = potential_list[0]
                break 
            
            if count > max_search:
                return None  
            
            count += 1
            
        return selected_id
    
    def time_sync(self):
        '''Check if this data is processed'''
        out_path = os.path.join(data_dir, "scene_to_mesh.json")
        if os.path.exists(out_path):
            print("Already Done!")
            return
        
        '''Find the first pair that can be aligned'''
        num_scene = len(self.scene_time_list)
        for start_scene_id in trange(num_scene):
            start_time_scene = self.scene_time_list[start_scene_id]
            start_time_sr = self.align_scene_handmesh(scene_id=start_scene_id, max_search=20)
            if start_time_sr is not None:
                break
        ret_dict = {}
        
        '''Align the remain pairs'''
        sr_time_list = self.tf_data_all_in_one.keys()
        sr_time_list = sorted(sr_time_list)
        
        for i, time_scene in enumerate(self.scene_time_list):
            delta_t = int(time_scene - start_time_scene)
            approx_sr_time = start_time_sr + delta_t
            sr_time = find_closest(sr_time_list, approx_sr_time)
            ret_dict[i] = f"{sr_time}.ply"
        
        with open(out_path, 'w') as f:
            f.write(json.dumps(ret_dict, indent=4))
        
       
def export_meshes(sr_mesh_dir, scene_dir, out_dir, scene_to_mesh_path=None):
    if scene_to_mesh_path is None:
        scene_to_mesh_path = os.path.join(scene_dir, "scene_to_mesh.json")
    with open(scene_to_mesh_path, 'r') as f:
        scene_to_mesh = json.load(f)
        
    for scene_t in scene_to_mesh:
        scene_pcd = os.path.join(scene_dir, f"{scene_t}.ply")
        hand_mesh = os.path.join(sr_mesh_dir, scene_to_mesh[scene_t])
        out_path = os.path.join(out_dir, f"scene_{scene_t}.ply")
        shutil.copy(scene_pcd, out_path)
        out_path = os.path.join(out_dir, f"hand_{scene_t}.ply")
        shutil.copy(hand_mesh, out_path)



def time_sync_script():
    urdf_path = "../../data_process/bimanual_srhand_ur.urdf"
    prefix ="/public/home/v-liuym/data/ShadowHand/description/"
    struct_file = "./assets/srhand_ur.json"
    
    base_dir = "/public/home/v-liuym/data/IntelligentHand_data/"
    cam_param_dir = "../../calibration_ws/calibration_process/data"
    
    model_name_list = os.listdir(base_dir)
    
    for model_name in model_name_list:
        path = os.path.join(base_dir, model_name)
        for exp_code in os.listdir(path):
            subpath = os.path.join(path, exp_code)
            if os.path.isdir(subpath) and re.match(rf"{model_name}_\d+", exp_code):
                data_dir = os.path.join(base_dir, model_name, exp_code)
                print(data_dir)
                data_processer = DataProcesser(data_dir, cam_param_dir)
                data_processer.time_sync()
    
def load_scene_to_mesh_by_step(data_dir):
    file_path = f"{data_dir}/scene_to_mesh_by_step.json"
    with open(file_path,"r") as json_reader:
        file = json.load(json_reader)
    return file

if __name__ == '__main__': 
    urdf_path = "../../data_process/bimanual_srhand_ur.urdf"
    prefix ="/home/lab4dv/data/hand_arm_mesh"
    struct_file = "./assets/srhand_ur.json"
    
    cam_param_dir = "../../calibration_ws/calibration_process/data"
    

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    bag_folder_path = Path("/media/lab4dv/HighSpeed/charger/charger_1")
    env_pcd_folder = Path(bag_folder_path) / Path("merged_pcd_filter")
    viz_camera_info_path = "/home/lab4dv/data/bags/camera_param.json"
    camera_params = o3d.io.read_pinhole_camera_parameters(viz_camera_info_path)
    

    data_dir = str(bag_folder_path)
    time_sync_json = load_scene_to_mesh_by_step(data_dir)
    # print(time_sync_json)
    data_processer = DataProcesser(data_dir, cam_param_dir)
    for key, value in time_sync_json.items():
        time = value.split('.')[0]
        
        env_pcd = o3d.io.read_point_cloud(str(env_pcd_folder / Path(f"cam0/{key}.ply")))

        hand_mesh = data_processer.gen_single_hand_mesh(int(time))
        hand_mesh.compute_vertex_normals()
        if key != 0:
            vis.clear_geometries()
        vis.add_geometry(env_pcd)
        # vis.add_geometry(hand_arm_mesh)
        vis.add_geometry(hand_mesh)
        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()

    