import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.shadow_hand_builder import ShadowHandBuilder
from utils.tta_util import get_NN, inter_penetr_loss, contact_map_of_m_to_n
from utils.tta_util import check_ray_triangle_intersection, batched_index_select

import numpy as np
import open3d as o3d

class SR_Loss(nn.Module):
    """Loss related to shadow hand"""
    def __init__(self, hand_builder):
        super(SR_Loss, self).__init__()
        self.hand_builder = hand_builder
        self.mse_loss = torch.nn.MSELoss()
        
    def setup_tgt(self, tgt_pc):
        self.tgt_pc = tgt_pc
        self.tgt_feat = self.compute_feature(tgt_pc.cpu())
        self.tgt_kdtree = o3d.geometry.KDTreeFlann()
        # print(self.tgt_feat.dimension(), self.tgt_feat.num())
        # print(self.tgt_feat.data)
        self.tgt_kdtree.set_feature(self.tgt_feat)
        
    
    def compute_feature(self, pc, pc_normal=None):
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc)
        if pc_normal is None:
            pc_o3d.estimate_normals()
        else:
            pc_o3d.normals = o3d.utility.Vector3dVector(pc_normal)
        pc_feature = o3d.pipelines.registration.compute_fpfh_feature(pc_o3d, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        return pc_feature
    
    def find_correspondences(self, source_feat, target_kdtree, max_distance=np.inf):
        """
        :param source: 
        :param target: 
        :param max_distance: correspndence will be eliminated if the distance is larger than max_distance
        :return: correspondence index
        """
        source_indices = []
        target_indices = []
        for i in range(len(source_feat.data)):
            feat = np.array(source_feat.data[:, i],dtype=np.float64)
            ret = target_kdtree.search_knn_vector_xd(feat.reshape(-1,1), 1)
            
            _, idx, dist = ret
            if dist[0] < max_distance:
                source_indices.append(i)
                target_indices.append(idx[0])
        
        return source_indices, target_indices

    def pc_diff_loss(self, sr_points, sr_points_normal):
        with torch.no_grad():
            sr_points_feat = self.compute_feature(sr_points.detach().cpu(), sr_points_normal.detach().cpu())
            src_index, tgt_index = self.find_correspondences(sr_points_feat, self.tgt_kdtree)
        # nn_dists, _ = get_NN(sr_points[src_index], self.tgt_pc[tgt_index])
        # loss = torch.mean(nn_dists)
        sr_points = sr_points[src_index]
        tgt_pc =  self.tgt_pc[tgt_index]
        dists = torch.norm(sr_points - tgt_pc, dim=-1)
        loss = torch.mean(dists[dists<0.05])
        # print(max(dists))
        # print(src_index, tgt_index)
        # loss = self.mse_loss(sr_points[src_index], self.tgt_pc[tgt_index])
        return loss
        
    def forward(self, global_rotation, transl, hand_pose, obj_points):
        
        rh_m = self.hand_builder.get_hand_model(global_rotation, transl, hand_pose)
        sr_meshes = rh_m['meshes']
        sr_points_list = rh_m['sampled_pts']
        face_normals_list = rh_m['face_normals']
        hand_verts = sr_meshes.verts_list()
        hand_faces = sr_meshes.faces_list()
        
        pen_dist = 0
        for i in range(len(sr_points_list)):
            v = hand_verts[i]
            f = hand_faces[i]
            fn = face_normals_list[i]
            sr_points = sr_points_list[i]
            hand_face_norm = face_normals_list[i]
            
            nn_dists, nn_idx = get_NN(src_xyz=obj_points, trg_xyz=sr_points)
            trg_pts = batched_index_select(sr_points, nn_idx)
            intersection = check_ray_triangle_intersection(ray_origins=obj_points, 
                                                                               ray_direction=trg_pts - obj_points,
                                                                               faces_verts=v[f],
                                                                               face_normals=fn)
            hit_number = torch.sum(intersection, dim=0) #intersection is boolean tensor
            interior_mask = (hit_number % 2 != 0)
            pen_dist += torch.norm(trg_pts[interior_mask] - obj_points[interior_mask])
            
        loss_dict = {}
        loss_dict['penetration'] = pen_dist
        loss_dict['cmap'] = contact_map_of_m_to_n(obj_points, sr_points)
            
        return loss_dict