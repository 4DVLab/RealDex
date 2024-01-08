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
from utils.global_util import segment_scene_point_cloud, tf_to_mat, find_closest
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
        self.scene_to_mesh = None
        
        '''Load Hand Mesh'''
        with open(struct_file, 'r') as f:
            # Load the JSON data
            sr_struct = json.load(f)
            
        link_list = sr_struct['node_names']
        self.mesh_dict = load_mesh_from_urdf(urdf_path,link_list, prefix)
        print(self.mesh_dict.keys())
        
        '''Load TF Sequence Data'''
        # tf_data_file = os.path.join(self.tf_data_dir, "global_tf_all_in_one.npy")
        seq_file = os.path.join(self.tf_data_dir, "tf_seq.npy")
        if os.path.exists(seq_file): 
            seq_data = np.load(seq_file, allow_pickle=True)
            seq_data = seq_data.item()
        else:
            seq_data = load_sequence(self.tf_data_dir, struct_file)
            np.save(seq_file, seq_data) # tf stored in 4*4 matrix form
        self.global_tf_seq = seq_data['global_tf']
        self.qpos_seq = seq_data['joint_angle']
        self.num_tf = len(self.global_tf_seq)
        
        '''PCD generator'''
        self.pcd_generator = PCDGenerator(data_dir, cam_param_dir)
        time_stamp_file = os.path.join(data_dir, "rgbimage_timestamp.txt")
        self.scene_time_list = np.loadtxt(time_stamp_file)
        
        '''Load Object Poses'''
        obj_tracking_path = os.path.join(self.data_dir, "tracking_result/tracking_and_icp.txt")
        object_poses = np.loadtxt(obj_tracking_path)
        self.object_poses = np.array([tf_to_mat(tf) for tf in object_poses])
        print(self.object_poses.shape)
        
        '''URDF info'''
        urdf_info_path = "./assets/srhand_ur.json"
        self.urdf_info = json.load(open(urdf_info_path))
        
        

    def gen_single_hand_mesh(self, time):
        updated_meshes = {}
        tf_data = self.global_tf_seq[time]
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
    
        for time in self.global_tf_seq:
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
            
    def gen_pcd(self, scene_id, cam_index):
        return self.pcd_generator.gen_pcd(scene_id, cam_index)
            
    def align_scene_handmesh(self, scene_id, approx_start_time_sr, max_search, cam_index=3, ratio_threshold = 0.07):
        scene_pcd = self.gen_pcd(scene_id, cam_index)
        
        sr_time_list = self.global_tf_seq.keys()
        progress_bar = tqdm(sorted(sr_time_list), desc="Processing items", unit="item")
        potential_list = []
        count = 0
        for sr_time in progress_bar:
            if sr_time < approx_start_time_sr:
                continue
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
        sr_time_list = self.global_tf_seq.keys()
        sr_time_list = sorted(sr_time_list)
        num_scene = len(self.scene_time_list)
        for start_scene_id in trange(num_scene):
            start_time_scene = self.scene_time_list[start_scene_id]
            approx_start_time_sr = find_closest(sr_time_list, start_time_scene) - 0.5*1e8 #0.5s
            start_time_sr = self.align_scene_handmesh(scene_id=start_scene_id, 
                                                      approx_start_time_sr=approx_start_time_sr, 
                                                      max_search=20,
                                                      ratio_threshold = 0.05)
            if start_time_sr is not None:
                break
        ret_dict = {}
        
        '''Align the remain pairs'''
        for i, time_scene in enumerate(self.scene_time_list):
            delta_t = int(time_scene - start_time_scene)
            approx_sr_time = start_time_sr + delta_t
            sr_time = find_closest(sr_time_list, approx_sr_time)
            ret_dict[i] = f"{sr_time}.ply"
        
        with open(out_path, 'w') as f:
            f.write(json.dumps(ret_dict, indent=4))
            
        self.scene_to_mesh = ret_dict
            
    def export_final_data(self):
        if self.scene_to_mesh is None:
            path = os.path.join(data_dir, "scene_to_mesh.json")
            if os.path.exists(path):
                self.scene_to_mesh = json.load(open(path))
            else:
                print("Do time sync first!")
                return
        
        qpos_key_list = list(self.urdf_info['hand_info'].keys())
        qpos_seq = []
        global_transl_seq = []
        global_orient_seq = []
        for time_scene in tqdm(self.scene_to_mesh):
            time_sr = self.scene_to_mesh[time_scene]
            time_sr = int(time_sr.split('.')[0])
            qpos_data = [self.qpos_seq[time_sr][k] for k in qpos_key_list]
            qpos_seq.append(qpos_data)
            
            hand_root_frame = self.global_tf_seq[time_sr]['rh_forearm']
            global_transl_seq.append(hand_root_frame[:3, -1])
            global_orient_seq.append(hand_root_frame[:3, :3])
        qpos_seq = np.array(qpos_seq)
        global_transl_seq = np.array(global_transl_seq)
        global_orient_seq = np.array(global_orient_seq)
        
        out_dict = {'qpos': qpos_seq,
                    'global_transl': global_transl_seq,
                    'global_orient': global_orient_seq,
                    'object_transl': self.object_poses[:, :3, -1],
                    'object_orient': self.object_poses[:, :3, :3]}
        for key in out_dict:
            print(key, out_dict[key].shape)
        path = os.path.join(data_dir, "hand_pose.npy")
        np.save(path, out_dict)
        
        # return ret_dict
               
       
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
            if os.path.isdir(subpath) and re.match(rf"{model_name}_\d+", exp_code):
                data_dir = os.path.join(base_dir, model_name, exp_code)
                print(data_dir)
                data_processer = DataProcesser(data_dir, cam_param_dir)
                data_processer.time_sync()
                data_processer.export_final_data()
    
    
    
    