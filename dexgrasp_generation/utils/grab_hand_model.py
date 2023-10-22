"""
Last modified date: 2022.08.07
Author: mzhmxzh
Description: class HandModel
"""

import os
import json
import torch
from torch import nn
import pytorch_kinematics as pk
import trimesh
import pytorch3d.structures
import pytorch3d.ops
import pytorch3d.transforms
from csdf import index_vertices_by_faces, compute_sdf
import pickle
from manotorch.manolayer import ManoLayer, MANOOutput

class HandModel(nn.Module):
    def __init__(self, mano_path):
        super().__init__()
        
        self.mano_path = os.path.join(mano_path, 'MANO_RIGHT.pkl')
            
        self.mano_layer = ManoLayer(use_pca=False, flat_hand_mean=False)
        self.mano_faces = self.mano_layer.get_mano_closed_faces()
        
    

class AdditionalLoss(nn.Module):
    def __init__(self, tta_cfg, device, num_obj_points, num_hand_points, cmap_net):
        super().__init__()
        self.num_obj_points = num_obj_points
        self.hand_model = HandModel(
            mjcf_path='data/mjcf/shadow_hand.xml',
            mesh_path='data/mjcf/meshes',
            n_surface_points=num_hand_points,
            contact_points_path='data/mjcf/contact_points.json',
            penetration_points_path='data/mjcf/penetration_points.json',
            device=device,
        )
        self.cmap_func = cmap_net.forward
        self.normalize_factor=tta_cfg['normalize_factor']
        self.weights = dict(
            weight_cmap=tta_cfg['weight_cmap'],
            weight_pen=tta_cfg['weight_pen'],
            weight_dis=tta_cfg['weight_dis'],
            weight_spen=tta_cfg['weight_spen'],
            weight_tpen=tta_cfg['weight_tpen'],
            thres_dis=tta_cfg['thres_dis'],
        )
    
    def forward(self, points, translation, hand_qpos):
        hand_pose = torch.cat([translation,torch.zeros_like(translation), hand_qpos],dim=-1)
        hand = self.hand_model(hand_pose, points, with_penetration=True, with_surface_points=True, with_contact_candidates=True, with_penetration_keypoints=True)
        discretized_cmap_pred = self.cmap_func(dict(canon_obj_pc=points, observed_hand_pc=hand['surface_points']))['contact_map'].detach().exp()# [B, N, 10]
        arange = (torch.arange(0, discretized_cmap_pred.shape[-1], dtype=discretized_cmap_pred.dtype, device=discretized_cmap_pred.device)+0.5)
        cmap_pred = torch.mean(discretized_cmap_pred * arange, dim=-1).detach()
        cmap = 2 - 2 * torch.sigmoid(self.normalize_factor * (hand['distances'].abs() + 1e-8).sqrt())  # calculate pseudo contactmap: 0~3cm mapped into value 1~0
        loss, losses = cal_loss(hand, cmap, cmap_pred, points, self.num_obj_points, **self.weights, verbose=True)
        return loss, losses
    
    def tta_loss(self, hand_pose, points, cmap_pred, plane_parameters):
        hand = self.hand_model(hand_pose, points, with_penetration=True, with_surface_points=True, with_contact_candidates=True, with_penetration_keypoints=True)
        cmap = 2 - 2 * torch.sigmoid(self.normalize_factor * (hand['distances'].abs() + 1e-8).sqrt())
        loss = cal_loss(hand, cmap, cmap_pred, points, plane_parameters, self.num_obj_points, **self.weights)
        return loss

def cal_loss(hand, cmap_labels, cmap_pred, object_pc, num_obj_points, verbose=False, weight_cmap=1., weight_pen=1., weight_dis=1., weight_spen=1., weight_tpen=1., thres_dis=0.02):

    batch_size = len(cmap_labels)
    
    distances = hand['penetration']  # signed squared distances from object_pc to hand, inside positive, outside negative
    contact_candidates = hand['contact_candidates']
    penetration_keypoints = hand['penetration_keypoints']
    # plane_distances = hand['plane_distances']

    # loss_cmap
    loss_cmap = torch.nn.functional.mse_loss(cmap_pred[:, :num_obj_points], cmap_labels[:, :num_obj_points], reduction='sum') / batch_size

    # loss_pen
    loss_pen = distances[distances > 0].sum() / batch_size

    # loss_dis
    dis_pred = pytorch3d.ops.knn_points(object_pc, contact_candidates).dists[:, :, 0]  # squared chamfer distance from object_pc to contact_candidates_pred
    small_dis_pred = dis_pred < thres_dis ** 2
    loss_dis = dis_pred[small_dis_pred].sqrt().sum()# / (small_dis_pred.sum() + 1e-4)

    # loss_spen
    dis_spen = (penetration_keypoints.unsqueeze(1) - penetration_keypoints.unsqueeze(2) + 1e-13).square().sum(3).sqrt()
    dis_spen = torch.where(dis_spen < 1e-6, 1e6 * torch.ones_like(dis_spen), dis_spen)
    dis_spen = 0.02 - dis_spen
    dis_spen[dis_spen < 0] = 0
    loss_spen = dis_spen.sum() / batch_size

    # loss_tpen
    # plane_distances[plane_distances > 0] = 0
    # loss_tpen = - weight_tpen * plane_distances.sum() / batch_size
    # dis_tpen = (penetration_keypoints * plane_parameters[:, :3].unsqueeze(1)).sum(2) + plane_parameters[:, 3].unsqueeze(1) - 0.01
    # dis_tpen[dis_tpen > 0] = 0
    # loss_tpen = - dis_tpen.sum() / batch_size


    # loss
    loss = weight_cmap * loss_cmap + weight_pen * loss_pen + weight_dis * loss_dis + weight_spen * loss_spen #+ weight_tpen * loss_tpen
    if verbose:
        return loss, dict(loss=loss, loss_cmap=loss_cmap, loss_pen=loss_pen, loss_dis=loss_dis, loss_spen=loss_spen) #, loss_tpen=loss_tpen)
    else:
        return loss

def add_rotation_to_hand_pose(hand_pose, rotation):
    translation = hand_pose[..., :3]
    added_rot_aa = hand_pose[..., 3:6]
    added_rot_mat = pytorch3d.transforms.axis_angle_to_matrix(added_rot_aa)
    hand_qpos = hand_pose[..., 6:]

    new_translation = torch.einsum('na,nba->nb', translation, rotation)
    new_rotation_mat = rotation @ added_rot_mat
    new_rotation_aa = pytorch3d.transforms.matrix_to_axis_angle(new_rotation_mat)

    return torch.cat([new_translation, new_rotation_aa, hand_qpos], dim=-1)