import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
import json
from utils.kintree import load_sequence
from utils.urdf_util import load_mesh_from_urdf
from utils.global_util import segment_scene_point_cloud, tf_to_mat, find_closest, batched_rotmat_to_vec
from utils.pcd_util import PCDGenerator
from models.shadow_hand_builder import ShadowHandBuilder
import shutil
from tqdm import tqdm, trange
import shutil
import re
import open3d as o3d
from copy import deepcopy
import random

class DataProcesser():
    def __init__(self, data_dir, cam_param_dir, obj_mesh):
        
        self.data_dir = data_dir
        self.obj_mesh = obj_mesh
        self.tf_data_dir = os.path.join(data_dir, "TF")
        self.arm_hand_path = None
        self.scene_to_mesh = None
        
        
        '''Load Hand Mesh'''
        with open(struct_file, 'r') as f:
            # Load the JSON data
            sr_struct = json.load(f)
            
        link_list = sr_struct['node_names']
        self.mesh_dict = load_mesh_from_urdf(urdf_path,link_list, prefix)
        # print(self.mesh_dict.keys())
        
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
        
        '''URDF info'''
        urdf_info_path = "./assets/srhand_ur.json"
        self.urdf_info = json.load(open(urdf_info_path))
        # self.qpos_key_list = list(self.urdf_info['hand_info'].keys())
        self.qpos_key_list = []
        self.joints_name_list = ShadowHandBuilder.joint_names
        for query_jn in self.joints_name_list:
            for key, value in self.urdf_info['hand_info'].items():
                if value['joint'] == query_jn:
                    self.qpos_key_list.append(key) 
        print(self.joints_name_list)
        print(self.qpos_key_list)
        
        '''Load Time Snyc Results'''
        path = os.path.join(self.data_dir, "scene_to_mesh.json")
        if os.path.exists(path):
            self.scene_to_mesh = json.load(open(path))
        
        

    def gen_single_arm_hand(self, time, only_hand):
        updated_meshes = {}
        tf_data = self.global_tf_seq[time]
        if only_hand:
            key_list = set(self.qpos_key_list) & set(self.mesh_dict.keys())
            key_list = list(key_list)
        else:
            # key_list = list(self.mesh_dict.keys())
            key_list = set(self.mesh_dict.keys()) - set(self.qpos_key_list)
            key_list = list(key_list)
            
        for key in key_list:
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

    def gen_all_arm_hand(self, out_type='both', num_samples = None, seq_length=20):
        '''Set out path'''
        self.arm_hand_path = os.path.join(self.data_dir, "srhand_ur_meshes")
        os.makedirs(self.arm_hand_path, exist_ok=True)
        
        if num_samples is not None:
            global_tf_seq = list(self.global_tf_seq.keys())
            start_points = random.sample(range(len(global_tf_seq) - seq_length), num_samples)
            sequences = []
            for start in start_points:
                sequences += global_tf_seq[start:start + seq_length]
            global_tf_seq = sequences
        else:
            global_tf_seq = self.global_tf_seq
    
        for time in tqdm(global_tf_seq):
            
            filename = os.path.join(self.arm_hand_path, f"{time}.ply")
            if out_type == 'hand':
                mesh = self.gen_single_arm_hand(time, only_hand=True)
                filename = os.path.join(self.arm_hand_path, f"{time}_hand.ply")
                o3d.io.write_triangle_mesh(filename, mesh)
            elif out_type == 'arm_hand':
                mesh = self.gen_single_arm_hand(time, only_hand=False)
                filename = os.path.join(self.arm_hand_path, f"{time}.ply")
                o3d.io.write_triangle_mesh(filename, mesh)
            elif out_type == 'both':
                o3d.io.write_triangle_mesh(os.path.join(self.arm_hand_path, f"{time}_hand.ply"), 
                                           self.gen_single_arm_hand(time, only_hand=True))
                o3d.io.write_triangle_mesh(os.path.join(self.arm_hand_path, f"{time}.ply"), 
                                           self.gen_single_arm_hand(time, only_hand=False))
                
        
    def remove_all_arm_hand(self):
        if self.arm_hand_path is None:
            return
        try:
            shutil.rmtree(self.arm_hand_path)
            print(f"The directory {self.arm_hand_path} has been removed with all its contents")
        except OSError as error:
            print(f"Error: {error.strerror}")
            
    def gen_pcd(self, scene_id, cam_index):
        return self.pcd_generator.gen_pcd(scene_id, cam_index)
    
    def gen_object_pcd(self, scene_id):
        out_dir = os.path.join(self.data_dir, "merged_pcd")
        out_path = os.path.join(out_dir, f"{scene_id}.ply")
        if os.path.exists(out_path):
            pcd = o3d.io.read_point_cloud(out_path)
            return pcd
        os.makedirs(out_dir, exist_ok=True)
        pcd = self.pcd_generator.gen_object_pcd(scene_id, out_dir, export=False)
        return pcd
        
    def check_sr_valid(self, sr_time):
        '''Check if at this time, the qpos for sr is all updated'''
        is_valid = True
        for k in self.qpos_key_list:
            if k not in self.qpos_seq[sr_time]:
                is_valid = False
                break
        return is_valid
            
    def align_scene_handmesh(self, scene_id, approx_start_time_sr, max_search, cam_index=3, ratio_threshold = 0.07):
        scene_pcd = self.gen_pcd(scene_id, cam_index)
        
        sr_time_list = self.global_tf_seq.keys()
        progress_bar = tqdm(sorted(sr_time_list), desc="Processing items", unit="item")
        potential_list = []
        count = 0
        
        for sr_time in progress_bar:
            
            '''Check if at this time, the qpos for sr is all updated'''
            if not self.check_sr_valid(sr_time):
                continue
            
            if sr_time < approx_start_time_sr:
                continue
                    
            '''Start'''
            sr_mesh = self.gen_single_arm_hand(sr_time)
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
        out_path = os.path.join(self.data_dir, "scene_to_mesh.json")
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
                                                      ratio_threshold = 0.02)
            if start_time_sr is not None:
                break
        ret_dict = {}
        
        '''Align the remain pairs'''
        for i, time_scene in enumerate(self.scene_time_list):
            delta_t = int(time_scene - start_time_scene)
            approx_sr_time = start_time_sr + delta_t
            sr_time = find_closest(sr_time_list, approx_sr_time)
            if self.check_sr_valid(sr_time):
                ret_dict[i] = f"{sr_time}.ply"
            else:
                sr_time = self.align_scene_handmesh(scene_id=i,
                                                    approx_start_time_sr=sr_time,
                                                    max_search=10, ratio_threshold=0.03)
                ret_dict[i] = f"{sr_time}.ply"
        
        with open(out_path, 'w') as f:
            f.write(json.dumps(ret_dict, indent=4))
            
        self.scene_to_mesh = ret_dict
            
    def export_final_data(self):
        if self.scene_to_mesh is None:
            path = os.path.join(self.data_dir, "scene_to_mesh.json")
            if os.path.exists(path):
                self.scene_to_mesh = json.load(open(path))
            else:
                print("Do time sync first!")
                return
        
        qpos_seq = []
        global_transl_seq = []
        global_orient_seq = []
        for time_scene in tqdm(self.scene_to_mesh):
            time_sr = self.scene_to_mesh[time_scene]
            time_sr = int(time_sr.split('.')[0])
            
            for k in self.qpos_key_list:
                if k not in self.qpos_seq[time_sr]:
                    print(k)
                    print(self.qpos_seq[time_sr].keys())
            
            qpos_data = [self.qpos_seq[time_sr][k] for k in self.qpos_key_list[2:]]
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
        path = os.path.join(self.data_dir, "final_data.npy")
        np.save(path, out_dict)
        
        return out_dict
    
    def seg_sequence(self, out_dir):
        seg_file = os.path.join(self.data_dir, "segment.txt")
        seg_list = list(np.loadtxt(seg_file))
        data_file = os.path.join(self.data_dir, "final_data.npy")
        full_data = np.load(data_file, allow_pickle=True).item()
        obj_tf_mat = np.eye(4)
        hand_orient_full = batched_rotmat_to_vec(full_data['global_orient'])
        obj_orient_full = batched_rotmat_to_vec(full_data['object_orient'])
        
        global counter
        
        
        for seg in tqdm(seg_list):
            start, end = seg
            start, end = int(start), int(end)
            
            qpos = full_data['qpos'][start:end, :]
            
            hand_transl = full_data['global_transl'][start:end, :]
            obj_transl = full_data['object_transl'][start:end, :]
            
            obj_tf_mat[:3, :3] = full_data['object_orient'][start]
            obj_tf_mat[:3, -1] = obj_transl[0]
            obj_mesh = deepcopy(self.obj_mesh)
            obj_mesh.apply_transform(obj_tf_mat)
            bb_min, bb_max = obj_mesh.bounding_box.bounds
            
            crop_pcd_path = os.path.join(self.data_dir, "merged_pcd", f"{start}_crop.ply")
            if os.path.exists(crop_pcd_path):
                obj_pcd = o3d.io.read_point_cloud(crop_pcd_path)
            else:
                obj_pcd = self.gen_object_pcd(start)
                PCDGenerator.crop_pcd(obj_pcd, bb_min, bb_max)          
                o3d.io.write_point_cloud(crop_pcd_path,obj_pcd)
            
            np.savez_compressed(os.path.join(out_dir, f"{counter}.npz"), 
                                qpos = qpos,
                                hand_transl = hand_transl,
                                hand_orient = hand_orient_full[start:end, :],
                                object_transl = obj_transl,
                                object_orient = obj_orient_full[start:end, :],
                                object_points = np.array(obj_pcd.points),
                                object_colors = np.array(obj_pcd.colors)
                                )
            counter += 1
    
    def compute_contact(self, qpos):
        pass
        
    def split_data(self, out_dir, split_type='object'):
        if split_type == 'object':
            train = ['blue_magnet_toy', 'body_lotion', 'crisps', 'dust_cleaning_sprayer', 
                     'laundry_detergent', 'toilet_cleaning_sprayer']
            val = ['goji_jar', 'small_sprayer', 'yogurt']
            test = ['duck_toy', 'cosmetics', 'sprayer']
        
    def export_meshes(self):
        out_dir = os.path.join(self.data_dir, "vis_meshes")
        os.makedirs(out_dir, exist_ok=True)
        
        seg_file = os.path.join(self.data_dir, "segment.txt")
        seg_list = list(np.loadtxt(seg_file))
        data_file = os.path.join(self.data_dir, "final_data.npy")
        
        full_data = np.load(data_file, allow_pickle=True).item()
            
        for seg in tqdm(seg_list):
            start, end = seg
            start, end = int(start), int(end)
            obj_tf_mat = np.eye(4)
            # for t in trange(start, end):
            for t in trange(0, start):
                time_sr = int(self.scene_to_mesh[str(t)].split('.')[0])
                
                obj_tf_mat[:3, :3] = full_data['object_orient'][t]
                obj_tf_mat[:3, -1] = full_data['object_transl'][t]
                obj_mesh = deepcopy(self.obj_mesh)
                obj_mesh.apply_transform(obj_tf_mat)
                
                obj_mesh.export(os.path.join(out_dir, f"{t}_object.ply"))
                o3d.io.write_triangle_mesh(os.path.join(out_dir, f"{t}_hand.ply"), 
                                           self.gen_single_arm_hand(time_sr, only_hand=True))
                o3d.io.write_triangle_mesh(os.path.join(out_dir, f"{t}.ply"), 
                                           self.gen_single_arm_hand(time_sr, only_hand=False))
            break
                

def run(data_processer):
    data_processer.gen_object_pcd(0)
    data_processer.time_sync()
    data_processer.export_final_data()

def segment_sequence(data_processer, model_name):
    out_dir = "/storage/group/4dvlab/yumeng/IntelligentHand/collected_data/"
    out_dir = os.path.join(out_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    data_processer.seg_sequence(out_dir)
    
def run_single(model_name, exp_code):
    data_dir = os.path.join(base_dir, model_name, exp_code)
    obj_path = os.path.join(obj_model_dir, f"{model_name}.obj")
    obj_mesh = trimesh.load(obj_path)
    print(data_dir)
    data_processer = DataProcesser(data_dir, cam_param_dir, obj_mesh)
    # data_processer.gen_all_arm_hand(out_type='both', num_samples=1)
    data_processer.export_meshes()
    
if __name__ == '__main__':
    urdf_path = "../../data_process/bimanual_srhand_ur.urdf"
    prefix ="/public/home/v-liuym/data/ShadowHand/description/"
    struct_file = "./assets/srhand_ur.json"
    
    base_dir = "/public/home/v-liuym/data/IntelligentHand_data/"
    cam_param_dir = "../../calibration_ws/calibration_process/data"
    obj_model_dir = "/storage/group/4dvlab/youzhuo/models"
    
    
    model_name_list = os.listdir(base_dir)
    
    # run_single(model_name="blue_magnet_toy", exp_code="blue_magnet_toy_1_20231209")
    
    for model_name in model_name_list:
        path = os.path.join(base_dir, model_name)
        global counter 
        counter = 0
        for exp_code in os.listdir(path):
            subpath = os.path.join(path, exp_code)
            if os.path.isdir(subpath) and re.match(rf"{model_name}_\d+", exp_code):
                data_dir = os.path.join(base_dir, model_name, exp_code)
                obj_path = os.path.join(obj_model_dir, f"{model_name}.obj")
                obj_mesh = trimesh.load(obj_path)
                print(data_dir)
                data_processer = DataProcesser(data_dir, cam_param_dir, obj_mesh)
                run(data_processer)
                segment_sequence(data_processer, model_name)
                
    
    
    
    