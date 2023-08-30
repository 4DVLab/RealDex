import sys
sys.path.append(".")
sys.path.append("..")
import trimesh
import os
import torch
import pytorch_kinematics as pk
# os.chdir("/remote-home/liuym/Project/IntelligentHand")

from utils.rot6d import robust_ortho6d_to_rot_mat

import numpy as np
import random
import transforms3d
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
        
        
    def query_time(self, query):
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
        return left
                    
class KinematicTree():
    def __init__(self, mjcf_path, mesh_path, device):
        self.mesh_path = mesh_path
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.device = device
        self.mesh = {}
        print(os.getcwd())
        # self.build_mesh_recurse(self.chain._root)
        self.build_mesh(self.chain._root)
        self.link_nodes = {}
        
        
    def update_node(self, tf_data):
        for link in tf_data:
            child_frame_id = link['child_frame_id']
            time_stamp = link['header']['stamp']['secs']
            transform = link['transform']
            if child_frame_id not in self.link_nodes:
                self.link_nodes[child_frame_id] = LinkNode()
            else:
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
                link_mesh = trimesh.primitives.Capsule(radius=visual.geom_param[0],
                                                       height=visual.geom_param[1] * 2).apply_translation((0, 0, -visual.geom_param[1]))
            elif visual.geom_type == "mesh":
                link_mesh = trimesh.load_mesh(os.path.join(self.mesh_path, visual.geom_param[0].split(":")[1]+".obj"), process=False)
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
    
if __name__ == '__main__':
    arm_file = "./mjcf/ur10e_mujoco/ur10e.xml"
    mesh_file = "./mjcf/ur10e_mujoco/assets"
    # arm_file = "./mjcf/shadow_hand/shadow_hand_wrist_free.xml"
    # mesh_file = "./mjcf/shadow_hand/meshes"
    
    robot_model = KinematicTree(arm_file, mesh_file,device="cpu")