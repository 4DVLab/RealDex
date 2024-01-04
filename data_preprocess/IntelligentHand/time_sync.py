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
        tf_data_file = os.path.join(tf_data_dir, "global_tf_all_in_one.npy")
        if os.path.exists(tf_data_file): 
            tf_data_all_in_one = np.load(tf_data_file, allow_pickle=True)
            tf_data_all_in_one = tf_data_all_in_one.item()
        else:
            tf_data_all_in_one = load_sequence(tf_data_dir, struct_file)
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
        
        return combined_mesh

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
            
            
    def align_scene_handmesh(self, scene_id = 0, cam_index=3, ratio_threshold = 0.06):
        scene_pcd = pcd_generator.gen_pcd(scene_id, cam_index)
        
        sr_time_list = self.tf_data_all_in_one.keys()
        progress_bar = tqdm(sorted(sr_time_list), desc="Processing items", unit="item")
        potential_list = []
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
            
        return selected_id
    
    def time_sync(self, start_scene_id=0):
        start_time_scene = self.scene_time_list[start_scene_id]
        sr_mesh_dir = os.path.join(data_dir, "srhand_ur_meshes")
        out_path = os.path.join(data_dir, "scene_to_mesh.json")
        ret_dict = {}
        
        sr_mesh_file_list = os.listdir(sr_mesh_dir)
        sr_time_list = [int(x.split('.')[0]) for x in sr_mesh_file_list]
        sr_time_list = sorted(sr_time_list)
        
        for i, time_scene in enumerate(self.scene_time_list):
            delta_t = int(time_scene - start_time_scene)
            approx_sr_time = self.start_time_sr + delta_t
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
        

if __name__ == '__main__':
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
            if os.path.isdir(subpath) and re.match(r'model_\d+', exp_code):
                data_dir = os.path.join(base_dir, model_name, exp_code)
                tf_data_dir = os.path.join(data_dir, "TF")
                
                pcd_generator = PCDGenerator(data_dir, cam_param_dir)
                
                # gen_hand_mesh()
                
                sr_mesh_dir = os.path.join(data_dir, "srhand_ur_meshes")
                scene_dir = os.path.join(data_dir, "cam3/pcd")
                scene_to_mesh = time_synchronization(sr_mesh_dir, scene_dir, scene_start=0, scene_end=1, ratio_threshold=0.04)
                print(scene_to_mesh)
                
                extract_hand_mesh(data_dir, start_time_sr=scene_to_mesh[0], start_scene_id=0)
                out_dir = os.path.join(data_dir, "temp_result")
                os.makedirs(out_dir, exist_ok=True)
                # export_meshes(sr_mesh_dir, scene_dir, out_dir, os.path.join(data_dir, "scene_to_mesh_by_step.json"))

                
                # scene_to_mesh_path = os.path.join(data_dir, "scene_to_mesh_by_step.json")
                # vis_result(sr_mesh_dir, scene_dir, scene_to_mesh_path)
                
                remove_hand_mesh()
    
    
    model_name = "crisps"
    exp_code = "crisps_2"
    
    
    
    
    