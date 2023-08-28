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

class KinematicTree():
    def __init__(self, mjcf_path, mesh_path, device):
        self.mesh_path = mesh_path
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.device = device
        self.mesh = {}
        print(os.getcwd())
        # self.build_mesh_recurse(self.chain._root)
        self.build_mesh(self.chain._root)
        
        
    class TimeNode():
        def __init__(self, l=None, r=None):
            self.l_time = l
            self.r_time = r
            self.parent = None
            self.is_leaf = False
            self.value = None
            
        
    class KLink():
        def __init__(self, time_stamp_list):
            self.time_stamp_list = time_stamp_list
            self.leaf_num = len(time_stamp_list)
            # allocate memories
            self.bi_tree = [KinematicTree.TimeNode()] * 4 * self.leaf_num
            self.build_tree_for_query()
            
        def build_tree_for_query(self):
            # insert leaf node in tree
            for i, time in enumerate(self.time_stamp_list):
                self.bi_tree[self.leaf_num + i] = KinematicTree.TimeNode(l=time, r=time)
                self.bi_tree[self.leaf_num + i].is_leaf = True
                
            # build the tree by calculate parents
            for i in range(self.leaf_num - 1, 0, -1):
                lchild = i << 1
                rchild = i << 1 | 1
                
                # get the interval from its children
                self.bi_tree[i] = KinematicTree.TimeNode(l=self.bi_tree[lchild].l_time,r=self.bi_tree[rchild].r_time)
                
                self.bi_tree[lchild].parent = self.bi_tree[i]
                self.bi_tree[rchild].parent = self.bi_tree[i]
                
        def query(self, node_id, q_time):
            node = self.bi_tree[node_id]
            if node.is_leaf:
                return node.value
            if q_time <= node.l_time:
                lchild = node_id << 1
                self.query(lchild, q_time)
            if q_time >= node.r_time:
                rchild = node_id << 1 | 1
                self.query(rchild, q_time)
            
            
            
            
                
        # # function to query on interval [l, r)
        # def query(self, l, r): 
        #     res = 0
        #     # loop to find the sum in the range
        #     while l < r and l >>= 1 and r >>= 1:
        #         l += n
        #         r += n
        #         if (l&1): 
        #             res += tree[l++]
            
        #         if (r&1): 
        #             res += tree[--r]
            
            # return res
            
            
        def insert(self, time):
            pass
        
        
        
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
    
    def update(self, time, qpos):
        