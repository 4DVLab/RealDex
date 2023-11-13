import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.shadow_hand_builder import sr_builder
from utils.tta_util import get_NN, inter_penetr_loss, contact_map_of_m_to_n
from utils.tta_util import check_ray_triangle_intersection, batched_index_select

class SR_Loss(nn.Module):
    """Loss related to shadow hand"""
    def __init__(self, hand_builder:sr_builder):
        super(SR_Loss, self).__init__()
        self.hand_builder = hand_builder

    def pc_diff_loss(self, rh_m, pc):
        sr_meshes = rh_m['meshes']
        sr_points_list = rh_m['sampled_pts']
        
        
    
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