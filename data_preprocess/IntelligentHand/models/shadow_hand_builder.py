import pytorch_kinematics as pk
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.ops import sample_points_from_meshes
from utils.urdf_util import load_mesh_from_urdf

import xml.etree.ElementTree as ET
import torch
import os
import json

import numpy as np
import trimesh


class ShadowHandBuilder():
    joint_names = [
                'WRJ2', 
                'WRJ1',
                'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1',
                'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1',
                'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1',
                'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1',
                'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1',
                ]
    joint_names = ["rh_" + name for name in joint_names]


    def __init__(self,
                 device,
                 assets_dir="./assets",
                 mesh_prefix ="/public/home/v-liuym/data/ShadowHand/description/",
                 num_sample=500
                 ):
        urdf_path=os.path.join(assets_dir, "bimanual_srhand_ur.urdf")
        self.urdf_info = json.load(open(os.path.join(assets_dir, "srhand_ur.json")))
        self.mesh_dict, self.points_dict = self._load_mesh_from_urdf(urdf_path, mesh_prefix)
        self.chain = self.get_hand_chain(urdf_path)
        joints_name = self.chain.get_joint_parameter_names()
        print(len(joints_name), joints_name)
        self.mesh = {}
        self.device = device
        # self.mesh_dir = mesh_dir
        self.num_sample = num_sample
        
        # self.build_mesh(self.chain._root)
        
    def get_hand_chain(self, urdf_path):
        urdf_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=device, dtype=torch.float)
        hand_chains = {}
        root_frame_name = 'rh_palm_frame'
        root_frame = urdf_chain.find_frame(root_frame_name)
        chain = pk.chain.Chain(root_frame)
        # for key in qpos_key_list:
        #     chain = pk.chain.SerialChain(urdf_chain, 
        #                                  end_frame_name=key,
        #                                  root_frame_name=root_frame_name)
        #     hand_chains[key] = chain
        return chain
    
    def _load_mesh_from_urdf(self, urdf_path, prefix):
        link_name_list = list(self.urdf_info['hand_info'].keys())
        trimesh_dict = load_mesh_from_urdf(urdf_path, link_name_list, prefix)  
        verts = [mesh.vertices for mesh in trimesh_dict.values()]
        faces = [mesh.faces for mesh in trimesh_dict.values()]
        meshes = Meshes(verts=verts, faces=faces)
          
        sampled_pts, pts_normal = sample_points_from_meshes(meshes=meshes, 
                                                            num_samples=self.num_sample, 
                                                            return_normals=True)
        
        ret_dict = {}
        for i, link in enumerate(link_name_list):
            ret_dict[link] = {
                'vertices': verts[i],
                'faces': faces[i],
                'points': sampled_pts[i],
                'points_normal': pts_normal[i]
            }
            
        return ret_dict
        
    def build_mesh(self, body):
        
        link_vertices = []
        link_faces = []
        n_link_vertices = 0
        for visual in body.link.visuals:
            scale = torch.tensor(
                [1, 1, 1], dtype=torch.float, device=self.device)
            if visual.geom_type == "box":
                link_mesh = trimesh.primitives.Box(
                    extents=2*visual.geom_param)
            elif visual.geom_type == "capsule":
                link_mesh = trimesh.primitives.Capsule(
                    radius=visual.geom_param[0], height=visual.geom_param[1]*2).apply_translation((0, 0, -visual.geom_param[1]))
            else:
                link_mesh = trimesh.load_mesh(
                    os.path.join(self.mesh_dir, visual.geom_param[0].split(":")[1]+".obj"), process=False)
                if visual.geom_param[1] is not None:
                    scale = (visual.geom_param[1]).to(dtype=torch.float, device=self.device)
            vertices = torch.tensor(
                link_mesh.vertices, dtype=torch.float, device=self.device)
            faces = torch.tensor(
                link_mesh.faces, dtype=torch.float, device=self.device)
            # print(body.link.name, link_mesh.is_watertight)
            
            pos = visual.offset.to(dtype=torch.float, device=self.device)
            vertices = vertices * scale
            vertices = pos.transform_points(vertices)
            link_vertices.append(vertices)
            link_faces.append(faces + n_link_vertices)
            n_link_vertices += len(vertices)
            
        if (len(link_vertices) > 0):
            # link_vertices = torch.cat(link_vertices, dim=0)
            # link_faces = torch.cat(link_faces, dim=0)
            mesh = Meshes(verts=link_vertices, faces=link_faces)
            
            bbox = mesh.get_bounding_boxes()
            bbox_vol = 1
            for i in range(3):
                bbox_vol *= bbox[:, i, 1] - bbox[:, i, 0]
            # print(body.link.name, bbox_vol)
            
            num_sample = bbox_vol / 1e-5 * self.num_sample
            num_sample = self.num_sample / 10 if num_sample < self.num_sample / 10 else num_sample
            
            sampled_pts, pts_normal = sample_points_from_meshes(meshes=mesh, num_samples=int(num_sample), return_normals=True)
            self.mesh[body.link.name] = {   'vertices': mesh.verts_packed(),
                                            'faces': mesh.faces_packed(),
                                            'face_normals': mesh.faces_normals_packed(),
                                            'sampled_pts':sampled_pts.squeeze(),
                                            'sampled_pts_normal':pts_normal.squeeze()
                                        }
        
        for children in body.children:
            self.build_mesh(children)
        

    def qpos_to_qpos_dict(self, qpos,hand_qpos_names):
        """
        :param qpos: [22]
        WARNING: The order must correspond with the joint_names
        """
        assert len(qpos) == len(hand_qpos_names)
        return dict(zip(hand_qpos_names, qpos))

    def qpos_dict_to_qpos(self, qpos_dict, hand_qpos_names):
        """
        :return: qpos: [22]
        WARNING: The order must correspond with the joint_names
        """
        return np.array([qpos_dict[name] for name in hand_qpos_names])
    
    def compute_status(self, qpos):
        '''
        qpos: tensor [batch_size, 22]
        '''
        batch_size = qpos.shape[0]
        new_qpos = torch.cat([torch.zeros(batch_size, 2), qpos], dim=1) # 22->24
        current_status = self.chain.forward_kinematics(new_qpos[None, :])
        
        return current_status
        

    def get_hand_model(self, world_rotation, world_translation, qpos):
        """
        Either qpos or qpos_dict should be provided.
        :param qpos: [22] numpy array
        :world_rotation: [3, 3]
        :world_translation: [3]
        :return:
        """
        current_status = self.compute_status(qpos)

        verts = []
        faces = []
        face_normals = []
        points = []
        pts_normals_list = []
        

        for link_name in self.mesh_dict:
            mesh = self.mesh_dict[link_name]
            verts = torch.tensor(mesh.vertices)
            verts = current_status[link_name].transform_points(verts)
            fn = torch.tensor(mesh.face_normals())
            fn = current_status[link_name].transform_normals(fn)
            pts = torch.tensor(self.points_dict[link_name])
            pts = current_status[link_name].transform_points(pts)
            
            # new_mesh = trimesh.Trimesh(vertices=verts[:, :3], faces=mesh.faces)
            # updated_meshes[key] = new_mesh
            
            pts = current_status[link_name].transform_points(self.mesh[link_name]['sampled_pts'])
            normals = current_status[link_name].transform_normals(self.mesh[link_name]['face_normals'])
            pts_normals = current_status[link_name].transform_normals(self.mesh[link_name]['sampled_pts_normal'])
            
            
            v = v @ world_rotation.T + world_translation
            pts = pts @ world_rotation.T + world_translation
            normals = normals @ world_rotation.T
            f = self.mesh[link_name]['faces']
            
            verts.append(v)
            faces.append(f)
            points.append(pts)
            face_normals.append(normals)
            pts_normals_list.append(pts_normals)

        meshes = Meshes(verts=verts, faces=faces)
        ret_dict = {'meshes': meshes, 
                    'sampled_pts': points, 
                    'face_normals': face_normals, 
                    'sampled_pts_normal':pts_normals_list}
            
        return ret_dict

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sr_builder = ShadowHandBuilder(device=device)
    # qpos= torch.zeros(22).cuda()
    # rotation_mat = torch.eye(3, 3).cuda()
    # transl = torch.zeros(3).cuda()
    # ret_dict = sr_builder.get_hand_model(rotation_mat, transl, qpos, without_arm=False)
    
    # points = torch.concat(ret_dict['sampled_pts']).reshape(-1, 3)
    # print(points.shape)
    
    # meshes = trimesh.Trimesh(vertices=ret_dict['meshes'].verts_packed().cpu(), faces=ret_dict['meshes'].faces_packed().cpu())
    # points = trimesh.PointCloud(vertices=points.cpu())
    # meshes.export("./test.ply")
    # points.export("./test_pts.ply")
    # print(meshes.is_watertight)