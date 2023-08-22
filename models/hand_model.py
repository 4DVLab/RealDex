import sys
sys.path.append(".")
sys.path.append("..")
import trimesh
import os
import torch
import pytorch_kinematics as pk
from utils.rot6d import robust_ortho6d_to_rot_mat
import numpy as np
import random
import transforms3d

class ShadowHandModel():
    joint_names = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]
    translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
    rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
    
    def __init__(self, mjcf_path, mesh_path, device):
        self.mesh_path = mesh_path
        self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        self.device = device
        self.mesh = {}
        print(os.getcwd())
        self.build_mesh_recurse(self.chain._root)

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

    def build_mesh_recurse(self, body):
        if(len(body.link.visuals) > 0):
            mesh = self.load_link_visuals(body.link)
            self.mesh.update(mesh)
            
        for child in body.children:
            self.build_mesh_recurse(child)

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
    grasp_code = "core-camera-51176ec8f251800165a1ced01089a2d6"
    data_path = "/Users/yumeng/Working/Project2023/data/dexgraspnet/dataset/"
    mesh_path = "/Users/yumeng/Working/Project2023/data/dexgraspnet/meshdata/"
    grasp_data = np.load(os.path.join(data_path, grasp_code + ".npy"), allow_pickle=True)
    object_mesh_origin = trimesh.load(os.path.join(mesh_path, grasp_code, "coacd/decomposed.obj"))

    # index = random.randint(0, len(grasp_data) - 1)
    index = 150
    qpos = grasp_data[index]['qpos']
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in ShadowHandModel.rot_names]))
    rot = rot[:, :2].T.ravel().tolist()
    hand_pose = torch.tensor([qpos[name] for name in ShadowHandModel.translation_names] + rot + [qpos[name]+np.random.random()*0
                            for name in ShadowHandModel.joint_names], dtype=torch.float, device="cpu").unsqueeze(0)
    
    use_visual_mesh = True
    # os.chdir('./models')
    # print(os.getcwd())
    hand_file = "./mjcf/shadow_hand_vis.xml" if use_visual_mesh else "./mjcf/shadow_hand_wrist_free.xml"
    hand_model = ShadowHandModel(hand_file,"./mjcf/meshes",device="cpu")

    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0)
    object_mesh = object_mesh_origin.copy().apply_scale(grasp_data[index]["scale"])

    # (hand_mesh+object_mesh).show()

    (hand_mesh+object_mesh).export("/Users/yumeng/Working/Project2023/result/SynthesizedGraspPose/150_" + grasp_code + ".obj")

