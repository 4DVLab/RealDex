import sys
sys.path.append(".")
sys.path.append("..")
import trimesh
import os
import torch
import pytorch_kinematics as pk
# os.chdir("/remote-home/liuym/Project/IntelligentHand")

from utils.rot6d import robust_ortho6d_to_rot_mat
import json
import numpy as np
import random
import transforms3d
from transforms3d.quaternions import quat2mat
# from urdfpy import URDF


class LinkNode():
    def __init__(self):
        """_summary_

        Args:
            time_stamp_list (list): This is a list of the exact times 
            when the ROS message was recorded. 
        """
        self.link_name = None
        # self.time_stamp_list = []
        self.tf_list = []
        
    def add(self, new_time, new_tf):
        # self.time_stamp_list.append(new_time)
        self.tf_list.append({'time': new_time, 'transform': new_tf})
        self.tf_list = sorted(self.tf_list, key=lambda x: x['time'])
        
        
    def query_tf_by_time(self, query):
        if len(self.tf_list) == 0:
            return -1
        left = 0
        right = len(self.tf_list)
        while left < right:
            mid = int((left + right) / 2)
            if self.tf_list[mid]['time'] < query:
                left = mid + 1
            elif self.tf_list[mid]['time'] >= query:
                right = mid
                
        # select the nearest time that is earlier than the query in the time stamp list.
        tf = self.tf_list[left]['transform']
        quat = torch.tensor([tf['rotation'][k] for k in ['w', 'x', 'y', 'z']])
        quat_w = torch.tensor(tf['rotation']['w'])
        joint_angle = torch.arccos(quat_w) * 2
        
        return joint_angle
    
class KinematicTree():
    def __init__(self, mjcf_path, mesh_path, device):
        self.mesh_path = mesh_path
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.device = device
        self.mesh = {}
        self.build_mesh_recurse(self.chain._root)
        # self.build_mesh(self.chain._root)
        self.link_nodes = {}
        self.joints_name = self.chain.get_joint_parameter_names()
        
    def build_mesh_recurse(self, body):
        if(len(body.link.visuals) > 0):
            mesh = self.load_link_visuals(body.link)
            self.mesh.update(mesh)
            
        for child in body.children:
            self.build_mesh_recurse(child)
            
            
    def load_link_visuals(self, link):
        mesh = {}
        link_name = link.name
        link_vertices = []
        link_faces = []
        n_link_vertices = 0
        for visual in link.visuals:
            scale = torch.tensor([1, 1, 1]).float()
            if visual.geom_type == "box":
                # link_mesh = trimesh.primitives.Box(extents=2 * visual.geom_param)
                link_mesh = trimesh.load_mesh(os.path.join(self.mesh_path, 'box.obj'), process=False)
                link_mesh.vertices *= visual.geom_param.numpy()
            elif visual.geom_type == "capsule":
                continue
                link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0],
                                                       height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
            elif visual.geom_type == "mesh":
                mesh_name = visual.geom_param[0]
                if len(mesh_name.split(":")) > 1:
                    link_mesh = trimesh.load_mesh(os.path.join(self.mesh_path, mesh_name.split(":")[1]+".obj"), process=False)
                else:
                    link_mesh = trimesh.load_mesh(os.path.join(self.mesh_path, mesh_name + ".obj"), process=False)
                    
                if visual.geom_param[1] is not None:
                    scale = torch.tensor(visual.geom_param[1])
            vertices = torch.tensor(link_mesh.vertices).float()
            faces = torch.tensor(link_mesh.faces).long()
            pos = visual.offset
            vertices = vertices * scale
            vertices = pos.transform_points(vertices)
            link_vertices.append(vertices)
            link_faces.append(faces + n_link_vertices)
            n_link_vertices += len(vertices)

        link_vertices = torch.cat(link_vertices, dim=0)
        link_faces = torch.cat(link_faces, dim=0)
        mesh[link_name] = {
            'vertices': link_vertices,
            'faces': link_faces,
        }
        return mesh
        
                    
class URModel(KinematicTree):
    def __init__(self, mjcf_path, mesh_path, device):
        super().__init__(mjcf_path, mesh_path, device)
        print(self.joints_name)
        self.name_mapping = {'ra_shoulder_link':"shoulder_pan_joint", 
                            'ra_upper_arm_link':"shoulder_lift_joint",
                            'ra_forearm_link':"elbow_joint",
                            'ra_wrist_1_link':"wrist_1_joint",
                            'ra_wrist_2_link':"wrist_2_joint",
                            'ra_wrist_3_link':"wrist_3_joint"}
        
    def load_world_frame(self, file_path):
        with open(file_path, 'r') as f:
            static_tf = json.load(f)['transforms']
        for link in static_tf:
            if link['header']['frame_id'] == "world":
                world = link['transform']
                transl = world['translation']
                quat = world['rotation']
                quat = np.array([quat[k] for k in ['x', 'y', 'z', 'w']])
                rot_mat = quat2mat(quat)
                self.world_translation = np.array([transl[k] for k in ['x', 'y', 'z']])
                self.world_rotation = np.array(rot_mat)
                break
                
    def load_base_link(self, file_path):
        with open(file_path, 'r') as f:
            base_tf = json.load(f)['transforms']
        for link in base_tf:
            if link['header']['frame_id']=="ra_base":
                base = link['transform']
                transl = base['translation']
                quat = base['rotation']
                quat = np.array([quat[k] for k in ['x', 'y', 'z', 'w']])
                rot_mat = quat2mat(quat)
                
                base_translation = torch.tensor([transl[k] for k in ['x', 'y', 'z']]).float()
                base_rot = rot_mat
                
                world_trans = transforms3d.affines.compose(T=self.world_translation, R=self.world_rotation, Z=np.ones(3))
                base_trans = transforms3d.affines.compose(T=base_translation, R=base_rot, Z=np.ones(3))
                
                base_to_world = base_trans @ world_trans
                
                self.global_translation = torch.tensor(base_to_world[:3, -1]).float()
                self.global_rotation = torch.tensor(base_to_world[:3, :3]).unsqueeze(0).float()
                break
                
    
    def load_tf_file(self, file_path):
        with open(file_path, 'r') as f:
            tf_data = json.load(f)['transforms']
        self.update_node(tf_data)
        return tf_data
            
        
    def update_node(self, tf_data):
        for link in tf_data:
            child_frame_id = link['child_frame_id']
            if child_frame_id not in self.name_mapping:
                continue
            child_frame_id = self.name_mapping[child_frame_id]
            time_stamp = link['header']['stamp']['secs']
            transform = link['transform']
            if child_frame_id not in self.link_nodes:
                self.link_nodes[child_frame_id] = LinkNode()
            self.link_nodes[child_frame_id].add(time_stamp, transform)
            
                    
    def build_mesh(self, body):
        curr_body = body
        body_stack = [curr_body]
        while len(body_stack) > 0:
            curr_body = body_stack.pop()
            if len(curr_body.link.visuals) > 0:
                mesh = self.load_link_visuals(curr_body.link)
                self.mesh.update(mesh)
            body_stack += curr_body.children
            
    
    
    def set_arm_parameters(self, time=0):
        arm_pose = {}
        for i, joint in enumerate(self.joints_name):
            arm_pose[joint]=self.link_nodes[joint].query_tf_by_time(time)
            
        self.arm_pose = arm_pose
        self.global_translation = torch.zeros([1,3])
        self.global_rotation = torch.eye(3).unsqueeze(0)
        self.current_status = self.chain.forward_kinematics(
            arm_pose)
        
        
    
    def set_parameters(self, hand_pose):
        """
        Set translation, rotation, and joint angles of grasps
        
        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        """
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_ortho6d_to_rot_mat(
            self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(
            self.hand_pose[:, 9:])
    
    def get_trimesh_data(self, i):
        """
        Get full mesh
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        data = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += trimesh.Trimesh(vertices=v, faces=f)
        return data
    
if __name__ == '__main__':
    arm_file = "./mjcf/ur10e_mujoco/ur10e.xml"
    mesh_file = "./mjcf/ur10e_mujoco/assets"
    # arm_file = "./mjcf/shadow_hand/shadow_hand_wrist_free.xml" 
    # mesh_file = "./mjcf/shadow_hand/meshes"
    
    ur_model = URModel(arm_file, mesh_file,device="cpu")
    
    tf_file_path = "/remote-home/liuym/data/0721/out_tf_json/frame_1687317998456300066.json"
    world_file_path = "/remote-home/liuym/data/0721/tf_static/frame_1687317997982037854.json"
    base_file_path = "/remote-home/liuym/data/0721/out_tf_json/frame_1687317998454985545.json"
    ur_model.load_tf_file(tf_file_path)
    # ur_model.load_world_frame(world_file_path)
    # ur_model.load_base_link(base_file_path)
    print(ur_model.chain.get_joint_parameter_names())
    ur_model.set_arm_parameters(time=0)
    
    ur_mesh = ur_model.get_trimesh_data(0)
    
    ur_mesh.export("/remote-home/liuym/data/0721/ur_mesh.obj")
    
    