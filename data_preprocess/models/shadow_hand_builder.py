from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from scipy.spatial.transform import Rotation

import xml.etree.ElementTree as ET
import torch
import os
import json
import pytorch_kinematics as pk

import numpy as np
import trimesh

def relative_to_absolute_path(rel_path, abs_prefix):
    parts = rel_path.split('/')
    parts[-1] = parts[-1].replace(".dae", ".obj")
    parts = parts[2:]
    abs_path = os.path.join(abs_prefix, "/".join(parts))
    return abs_path

def init_component(attrib,mesh_origin,abs_prefix):
    #scale:np.array((1x3))
    file_path = relative_to_absolute_path(attrib['filename'],abs_prefix)
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    if 'scale' in attrib.keys():
        scale = np.array([float(item) for item in attrib['scale'].split(' ')]).reshape(1,3).squeeze(0)
        vertices = vertices * scale
        
    if mesh_origin is not None:
        origin_rpy = mesh_origin.attrib['rpy'].split(' ')
        origin_xyz = mesh_origin.attrib['xyz'].split(' ') #translation
        origin_rpy = np.array([float(item) for item in origin_rpy])
        origin_xyz = np.array([float(item) for item in origin_xyz])
        rot = Rotation.from_euler('xyz', origin_rpy, degrees=False)
        rot_mat = rot.as_matrix()
        vertices = vertices @ rot_mat.T + origin_xyz
        
    mesh.vertices = vertices
    return mesh

def load_mesh_from_urdf(urdf_path, name_list:set,abs_path_prefix):
    urdf_tree = ET.parse(urdf_path)
    #names:mesh + position
    root = urdf_tree.getroot()
    links = root.findall('link')
    # link_names = set(link.attrib['name'] for link in links)
    mesh_dict = {}
    for link in links:
        name = link.attrib['name']    
        link_visuals = link.findall('visual')
        for visual in link_visuals:
            if name in name_list and visual is not None:
                geometry = visual.find('geometry')
                mesh_origin = visual.find('origin')
                if geometry is not None:
                    # name = geometry.attrib['name']
                    mesh = geometry.find('mesh')
                    if mesh is None:
                        continue
                    vis_component = init_component(mesh.attrib, mesh_origin,abs_path_prefix)
                    if name in mesh_dict:
                        mesh_dict[name] += vis_component
                    else:
                        mesh_dict[name] = vis_component
                        
    return  mesh_dict #,link_names

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
                 num_sample=5000
                 ):
        
        self.device = device
        # self.mesh_dir = mesh_dir
        self.num_sample = num_sample
        
        urdf_path=os.path.join(assets_dir, "bimanual_srhand_ur.urdf")
        print(urdf_path)
        self.urdf_info = json.load(open(os.path.join(assets_dir, "srhand_ur.json")))
        self.chain = self.get_hand_chain(urdf_path).to(device=self.device)
        self.hand_joints_name = self.chain.get_joint_parameter_names()
        self.hand_frame_name = self.chain.get_frame_names()
        print(len(self.hand_joints_name), self.hand_joints_name)
        print(len(self.hand_frame_name), self.hand_frame_name)
        
        self.mesh_dict = self._load_mesh_from_urdf(urdf_path, mesh_prefix)
        
        
    def get_hand_chain(self, urdf_path):
        urdf_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=device, dtype=torch.float)
        root_frame_name = 'rh_wrist'
        root_frame = urdf_chain.find_frame(root_frame_name)
        chain = pk.chain.Chain(root_frame)
        
        return chain
    
    def _load_mesh_from_urdf(self, urdf_path, prefix):
        # link_name_list = list(self.urdf_info['hand_info'].keys())
        trimesh_dict = load_mesh_from_urdf(urdf_path, self.hand_frame_name, prefix)  
       
        verts = [torch.tensor(mesh.vertices).float() for mesh in trimesh_dict.values()]
        faces = [torch.tensor(mesh.faces).float() for mesh in trimesh_dict.values()]
        meshes = Meshes(verts=verts, faces=faces)
        faces_normal = meshes.faces_normals_list()
          
        ret_dict = {}
        for i, link in enumerate(trimesh_dict.keys()):
            ret_dict[link] = {
                'vertices': verts[i].unsqueeze(0).to(self.device),
                'faces': faces[i].unsqueeze(0).to(self.device),
                'face_normals': faces_normal[i].unsqueeze(0).to(self.device),
            }
            
        return ret_dict
        

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
        zeros = torch.zeros(batch_size, 2).to(self.device)
        new_qpos = torch.cat([zeros, qpos], dim=1) # 22->24
        current_status = self.chain.forward_kinematics(new_qpos)
        
        return current_status
        

    def get_hand_model(self, world_rotation, world_translation, qpos):
        """
        Either qpos or qpos_dict should be provided.
        :param qpos: [B, 24] numpy array
        :world_rotation: [B, 3, 3]
        :world_translation: [B, 3]
        :return:
        """
        batch_size = qpos.shape[0]
        current_status = self.compute_status(qpos)

        ret_dict = {
            'verts': [],
            'faces': [],
            'face_normals': [],
        }
        world_translation = world_translation.unsqueeze(1)

        counter = 0
        for link_name in self.mesh_dict:
            
            '''Get Mesh Data and Apply FK'''
            mesh = self.mesh_dict[link_name]
            
            verts = current_status[link_name].transform_points(mesh['vertices'])
            face_normals = current_status[link_name].transform_normals(mesh['face_normals'])
            
            f = mesh['faces'] + counter
            f = f.expand(batch_size, -1, -1)
            counter += verts.shape[1]
            
            '''Apply Global Transform'''
            verts = torch.einsum('bik,bjk->bij', verts, world_rotation)
            verts = verts + world_translation
            
            face_normals = torch.einsum('bik,bjk->bij', face_normals, world_rotation)
            
            ret_dict['verts'].append(verts)
            ret_dict['faces'].append(f)
            ret_dict['face_normals'].append(face_normals)
    
        for key in ret_dict:
            ret_dict[key] = torch.cat(ret_dict[key], dim=1)
        meshes = Meshes(verts=ret_dict['verts'], 
                        faces=ret_dict['faces'])
        print(len(meshes))
        print(meshes.num_verts_per_mesh(), meshes.num_faces_per_mesh())
        print(meshes.valid)
        # num_sample = [self.num_sample] * batch_size
        sampled_pts, pts_normal = sample_points_from_meshes(meshes=meshes, 
                                                            num_samples=2 * self.num_sample, 
                                                            return_normals=True)
        print(sampled_pts.shape)
        
        sampled_pts, _ = sample_farthest_points(sampled_pts, K=self.num_sample)
        print(sampled_pts.shape)

        
        ret_dict['points'] = sampled_pts
        ret_dict['point_normals'] = pts_normal
        ret_dict.pop('verts')
        ret_dict.pop('faces')
        ret_dict['meshes'] = meshes
                    
            
        return ret_dict

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sr_builder = ShadowHandBuilder(device=device)
    qpos= torch.zeros(2, 22).cuda()
    rotation_mat = torch.eye(3, 3).cuda()
    rotation_mat = rotation_mat.unsqueeze(0).expand(2, -1, -1)
    transl = torch.zeros(2, 3).cuda()
    ret_dict = sr_builder.get_hand_model(rotation_mat, transl, qpos)
    
    points = ret_dict['points']
    print(points.shape)
    meshes = ret_dict['meshes']
    
    meshes = trimesh.Trimesh(vertices=meshes.verts_list()[0].cpu(), 
                             faces=meshes.faces_list()[0].cpu())
    points = trimesh.PointCloud(vertices=points[0].cpu())
    
    out_dir = "/storage/group/4dvlab/yumeng/results/sr_hand_builder"
    os.makedirs(out_dir, exist_ok=True)
    meshes.export(os.path.join(out_dir, "test_hand.ply"))
    points.export(os.path.join(out_dir, "test_points.ply"))
    print(meshes.is_watertight)