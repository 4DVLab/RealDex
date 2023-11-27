import pytorch_kinematics as pk
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.vis.plotly_vis import plot_scene
from pytorch3d.ops import sample_points_from_meshes
import xml.etree.ElementTree as ET
import torch
import os

import numpy as np
import trimesh


class ShadowHandBuilder():
    joint_names = [
                # 'WRJ1', 'WRJ0',
                'FFJ3', 'FFJ2', 'FFJ1', 'FFJ0',
                'MFJ3', 'MFJ2', 'MFJ1', 'MFJ0',
                'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0',
                'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0',
                'THJ4', 'THJ3', 'THJ2', 'THJ1', 'THJ0',
                ]
    joint_names = ["robot0:" + name for name in joint_names]

    mesh_filenames = [  "forearm_electric.obj",
                        "forearm_electric_cvx.obj",
                        "wrist.obj",
                        "palm.obj",
                        "knuckle.obj",
                        "F3.obj",
                        "F2.obj",
                        "F1.obj",
                        "lfmetacarpal.obj",
                        "TH3_z.obj",
                        "TH2_z.obj",
                        "TH1_z.obj"]

    def __init__(self,
                 device,
                 mesh_dir="assets/mjcf/meshes",
                 mjcf_path="assets/mjcf/shadow_hand.xml",
                 num_sample=500
                 ):
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(device=device, dtype=torch.float)
        self.sr_xml_tree = ET.parse(mjcf_path)
        self.mesh = {}
        self.device = device
        self.mesh_dir = mesh_dir
        self.num_sample = num_sample
        
        self.build_mesh(self.chain._root)
        
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
        

    def qpos_to_qpos_dict(self, qpos,
                          hand_qpos_names=None):
        """
        :param qpos: [24]
        WARNING: The order must correspond with the joint_names
        """
        if hand_qpos_names is None:
            hand_qpos_names = ShadowHandBuilder.joint_names
        assert len(qpos) == len(hand_qpos_names)
        return dict(zip(hand_qpos_names, qpos))

    def qpos_dict_to_qpos(self, qpos_dict,
                          hand_qpos_names=None):
        """
        :return: qpos: [24]
        WARNING: The order must correspond with the joint_names
        """
        if hand_qpos_names is None:
            hand_qpos_names = ShadowHandBuilder.joint_names
        return np.array([qpos_dict[name] for name in hand_qpos_names])

    def get_hand_model(self,
                      rotation_mat,
                      world_translation,
                      qpos=None,
                      hand_qpos_dict=None,
                      hand_qpos_names=None,
                      without_arm=False):
        """
        Either qpos or qpos_dict should be provided.
        :param qpos: [24] numpy array
        :rotation_mat: [3, 3]
        :world_translation: [3]
        :return:
        """
        if qpos is None:
            if hand_qpos_names is None:
                hand_qpos_names = ShadowHandBuilder.joint_names
            assert hand_qpos_dict is not None, "Both qpos and qpos_dict are None!"
            qpos = np.array([hand_qpos_dict[name] for name in hand_qpos_names], dtype=np.float32)
        jn = self.chain.get_joint_parameter_names()
        # print(len(jn), jn)
        current_status = self.chain.forward_kinematics(qpos[None, :])

        verts = []
        faces = []
        face_normals = []
        points = []
        pts_normals_list = []
        

        for link_name in self.mesh:
            v = current_status[link_name].transform_points(self.mesh[link_name]['vertices'])
            pts = current_status[link_name].transform_points(self.mesh[link_name]['sampled_pts'])
            normals = current_status[link_name].transform_normals(self.mesh[link_name]['face_normals'])
            pts_normals = current_status[link_name].transform_normals(self.mesh[link_name]['sampled_pts_normal'])
            
            
            v = v @ rotation_mat.T + world_translation
            pts = pts @ rotation_mat.T + world_translation
            normals = normals @ rotation_mat.T
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
    sr_builder = ShadowHandBuilder(device=device,
                                   mjcf_path="assets/mjcf/shadow_hand_vis.xml"
                                )
    qpos= torch.zeros(22).cuda()
    rotation_mat = torch.eye(3, 3).cuda()
    transl = torch.zeros(3).cuda()
    ret_dict = sr_builder.get_hand_model(rotation_mat, transl, qpos, without_arm=False)
    
    points = torch.concat(ret_dict['sampled_pts']).reshape(-1, 3)
    print(points.shape)
    
    meshes = trimesh.Trimesh(vertices=ret_dict['meshes'].verts_packed().cpu(), faces=ret_dict['meshes'].faces_packed().cpu())
    points = trimesh.PointCloud(vertices=points.cpu())
    meshes.export("./test.ply")
    points.export("./test_pts.ply")
    print(meshes.is_watertight)