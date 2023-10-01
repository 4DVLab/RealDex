import torch
import numpy as np
from pytorch3d.loss import chamfer_distance as CD
from emd import earth_mover_distance
from typing import Union
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.structures import Meshes
from utils import utils_loss, utils
import chamfer_distance as chd

def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y
    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.contiguous().view(-1, 1, 3), y2x.contiguous().view(-1, 3, 1)).contiguous().view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return y2x_signed, x2y_signed, yidx_near

def loss_cnet(verts_rhand, verts_rhand_gt, faces, verts_obj, v_weights):
    device = verts_rhand.device
    dtype = verts_rhand.dtype

    B, NPoints_obj, _ = verts_obj.size()
    kl_coef = 0.005 # 5e-3
    v_weights2 = torch.pow(v_weights, 1.0 / 2.5)

    rh_mesh = Meshes(verts=verts_rhand, faces=faces).to(device).verts_normals_packed().view(-1, 778, 3)
    rh_mesh_gt = Meshes(verts=verts_rhand_gt, faces=faces).to(device).verts_normals_packed().view(-1, 778, 3)

    o2h_signed, h2o, _ = point2point_signed(verts_rhand, verts_obj, rh_mesh)
    o2h_signed_gt, h2o_gt, o2h_idx = point2point_signed(verts_rhand_gt, verts_obj, rh_mesh_gt)

    # addaptive weight for penetration and contact verts
    w_dist = (o2h_signed_gt < 0.01) * (o2h_signed_gt > -0.005)
    w_dist_neg = o2h_signed < 0.
    w = torch.ones([B, NPoints_obj]).to(device)
    w[~w_dist] = .1 # less weight for far away vertices
    w[w_dist_neg] = 1.5 # more weight for penetration
    ######### dist loss
    loss_dist_h = 35 * (1. - kl_coef) * torch.mean(torch.einsum('ij,j->ij', torch.abs(h2o.abs() - h2o_gt.abs()), v_weights2))
    loss_dist_o = 30 * (1. - kl_coef) * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h_signed - o2h_signed_gt), w))
    return loss_dist_h.sum() + loss_dist_o.sum()

def Coarsenet_cvae_loss(recon_x, x, mean, log_var):
    # L2 mesh reconstruction loss
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='none').sum() / x.size(0)
    # KLD loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    return recon_loss + KLD, recon_loss.item(), KLD.item()

################################################################################################
def CVAE_loss(recon_x, x, mean, log_var, loss_type, l2_weight=1):
    '''
    :param recon_x: reconstructed hand xyz [B,3,778]
    :param x: ground truth hand xyz [B,3,778]
    :param mean:
    :param log_var:
    :return:
    '''
    # L2 reconstruction loss
    recon_loss = torch.sqrt(torch.sum((recon_x-x)**2)) / x.size(0)
    # KLD loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
    if loss_type == 'CD':
        aux_loss, _ = CD(recon_x.permute(0,2,1), x.permute(0,2,1), point_reduction='sum')
    elif loss_type == 'EMD':
        aux_loss = earth_mover_distance(recon_x, x, transpose=True).sum() / x.size(0)
    return l2_weight * recon_loss + KLD + aux_loss, recon_loss.item(), KLD.item(), aux_loss.item()

def CVAE_loss_mano(recon_x, x, mean, log_var, loss_tpye, mode='train'):
    '''
    :param recon_x: reconstructed hand xyz [B,778,3]
    :param x: ground truth hand xyz [B,778,6]
    :param mean: [B,z]
    :param log_var: [B,z]
    :return:
    '''
    if loss_tpye == 'L2':
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='none').sum() / x.size(0)
    elif loss_tpye == 'CD':
        recon_loss, _ = CD(recon_x, x, point_reduction='sum', batch_reduction='mean')
    elif loss_tpye == 'EMD':
        recon_loss = earth_mover_distance(recon_x, x, transpose=False).sum() / x.size(0)
    if mode != 'train':
        return recon_loss
    # KLD loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0) * 10.0
    if mode == 'train':
        return recon_loss + KLD, recon_loss.item(), KLD.item()


def CMap_loss(obj_xyz, hand_xyz, cmap):
    '''
    # prior cmap loss on contactdb cmap
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1, 10] for 10 types of contact map
    :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
    :return:
    '''

    # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
    #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
    #                    350, 351, 352, 353, 354, 355,  # 2nd finger
    #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
    #                    467,  # 3rd
    #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
    #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
    #                    692, 693, 694, 695,  # 5th
    #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
    #                    767, 768,  # 1st
    #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
          355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[:, prior_idx, :]  # only using prior points for contact map

    B = obj_xyz.size(0)
    T = cmap.size(2)

    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz_prior)  # [B, N1] NN distance from obj pc to hand pc

    # compute contact map loss
    cmap_loss_list = []
    for i in range(B):
        tmp_list = []
        for j in range(T):
            mask = cmap[i, :, j] # [N1]
            n_points = torch.sum(mask)
            if n_points == 0:  # skip nan
                continue
            tmp_list.append(obj_CD[i][mask].sum() / n_points)  # point reduction
        cmap_loss_list.append(
            torch.min(torch.stack(tmp_list)))
    cmap_loss = torch.stack(cmap_loss_list).sum() / B  # batch reduction

    return 3000.0 * cmap_loss

def CMap_loss1(obj_xyz, hand_xyz, cmap):
    '''
    # prior cmap loss on contactdb cmap
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1, 10] for 10 types of contact map
    :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
    :return:
    '''

    # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
    #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
    #                    350, 351, 352, 353, 354, 355,  # 2nd finger
    #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
    #                    467,  # 3rd
    #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
    #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
    #                    692, 693, 694, 695,  # 5th
    #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
    #                    767, 768,  # 1st
    #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
          355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[:, prior_idx, :]  # only using prior points for contact map

    B = obj_xyz.size(0)
    T = cmap.size(2)

    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz_prior)  # [B, N1] NN distance from obj pc to hand pc
    hand_CD, _ = utils_loss.get_NN(hand_xyz_prior, obj_xyz)
    # compute contact map loss
    cmap_loss_list = []
    for i in range(B):
        tmp_list = []
        for j in range(T):
            mask = cmap[i, :, j] # [N1]
            n_points = torch.sum(mask)
            if n_points == 0:  # skip nan
                continue
            tmp_list.append(obj_CD[i][mask].sum() / n_points)  # point reduction
        cmap_loss_list.append(
            torch.min(torch.stack(tmp_list)))
    cmap_loss = torch.stack(cmap_loss_list).sum() / B  # batch reduction

    return 3000.0 * cmap_loss + 10.0 * hand_CD.sum() / B

def CMap_loss2(obj_xyz, hand_xyz):
    '''
    # self cmap loss with prior
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :return:
    '''
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz = hand_xyz[:, prior_idx, :]

    B = obj_xyz.size(0)
    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz)  # [B, N1] NN dists
    obj_cmap = obj_CD < 0.01**2  # NN dist smaller than 1 cm as True
    cpoint_num = torch.sum(obj_cmap) + 0.001  # in case of 0 denom
    return 20.0 * obj_CD[obj_cmap].sum() / cpoint_num  # point and batch reduction

def CMap_loss3(obj_xyz, hand_xyz, cmap):
    '''
    # prior cmap loss on gt cmap
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1] for contact map from NN dist thresholding
    :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
    :return:
    '''

    # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
    #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
    #                    350, 351, 352, 353, 354, 355,  # 2nd finger
    #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
    #                    467,  # 3rd
    #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
    #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
    #                    692, 693, 694, 695,  # 5th
    #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
    #                    767, 768,  # 1st
    #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
          355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[:, prior_idx, :]  # only using prior points for contact map

    B = obj_xyz.size(0)

    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz_prior)  # [B, N1] NN distance from obj pc to hand pc

    # compute contact map loss
    n_points = torch.sum(cmap)
    cmap_loss = obj_CD[cmap].sum() / (B * n_points)
    #print(cmap_loss.requires_grad)
    return 3000.0 * cmap_loss

def CMap_loss_chamfer(obj_xyz, hand_xyz):
    B = obj_xyz.size(0)
    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz)
    hand_CD, _ = utils_loss.get_NN(hand_xyz, obj_xyz)
    cmap_loss = 1000.0 * obj_CD.sum() / (B * 3000.0) + 778.0 * hand_CD.sum() / (B * 778)
    return cmap_loss


def CMap_loss4(obj_xyz, hand_xyz, cmap):
    '''
    # prior cmap loss on gt cmap, also minimize hand NN
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N1] for contact map from NN dist thresholding
    :param hand_faces_index: [B, 1538, 3] hand index in [0,N2] for 3 vertices in a face
    :return:
    '''

    # finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
    #                    326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
    #                    350, 351, 352, 353, 354, 355,  # 2nd finger
    #                    429, 433, 434, 435, 436, 437, 438, 439, 442, 443, 444, 455, 461, 462, 463, 465, 466,
    #                    467,  # 3rd
    #                    547, 548, 549, 550, 553, 566, 573, 578,  # 4th
    #                    657, 661, 662, 664, 665, 666, 667, 670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691,
    #                    692, 693, 694, 695,  # 5th
    #                    736, 737, 738, 739, 740, 741, 743, 753, 754, 755, 756, 757, 759, 760, 761, 762, 763, 764, 766,
    #                    767, 768,  # 1st
    #                    73, 96, 98, 99, 772, 774, 775, 777]  # hand
    f1 = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
          750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]
    f2 = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323, 324, 325, 326,
          327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354,
          355]
    f3 = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433, 434, 435, 436, 437, 438,
          439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459, 460, 461, 462, 463, 464, 465, 466, 467]
    f4 = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
          550, 551, 552, 553, 555, 563, 564, 565, 566, 567, 570, 572, 573, 574, 575, 576, 577, 578]
    f5 = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
          668, 670, 672, 680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]
    f0 = [73, 96, 98, 99, 772, 774, 775, 777]
    prior_idx = f1 + f2 + f3 + f4 + f5 + f0
    hand_xyz_prior = hand_xyz[:, prior_idx, :]  # only using prior points for contact map

    B = obj_xyz.size(0)

    obj_CD, _ = utils_loss.get_NN(obj_xyz, hand_xyz_prior)  # [B, N1] NN distance from obj to hand
    hand_CD, _ = utils_loss.get_NN(hand_xyz_prior, obj_xyz)  # [B, N2] NN distance from hand to obj
    # compute contact map loss
    n_points = torch.sum(cmap)
    cmap_loss_obj = obj_CD[cmap].sum() / (B * n_points)
    cmap_loss_hand = hand_CD.sum() / B
    return 3000.0 * cmap_loss_obj + 4.0 * cmap_loss_hand

def CMap_consistency_loss(obj_xyz, recon_hand_xyz, gt_hand_xyz, recon_dists, gt_dists):
    '''
    :param recon_hand_xyz: [B, N2, 3]
    :param gt_hand_xyz: [B, N2, 3]
    :param obj_xyz: [B, N1, 3]
    :return:
    '''
    # if not recon_dists or not gt_dists:
    #     recon_dists, _ = utils_loss.get_NN(obj_xyz, recon_hand_xyz)  # [B, N1]
    #     gt_dists, _ = utils_loss.get_NN(obj_xyz, gt_hand_xyz)  # [B, N1]
    recon_dists = torch.sqrt(recon_dists)
    gt_dists = torch.sqrt(gt_dists)
    # hard cmap
    recon_cmap = recon_dists < 0.005
    gt_cmap = gt_dists < 0.005
    gt_cpoint_num = gt_cmap.sum() + 0.0001
    consistency = (recon_cmap * gt_cmap).sum() / gt_cpoint_num
    # soft cmap
    #consistency2 = torch.nn.functional.mse_loss(recon_dists, gt_dists, reduction='none').sum() / recon_dists.size(0)
    return -5.0 * consistency #+ consistency2

def CMap_consistency_loss_soft(gt_dists, recon_dists):
    gt_dists = torch.sqrt(gt_dists)
    recon_dists = torch.sqrt(recon_dists)
    #gt_cmap = utils.get_pseudo_cmap(gt_dists)
    #recon_cmap = utils.get_pseudo_cmap(recon_dists)
    consistency = torch.nn.functional.mse_loss(recon_dists, gt_dists, reduction='none').sum() / recon_dists.size(0)
    return consistency

def CMap_consistency_loss_soft2(gt_dists, recon_dists):
    #gt_dists = torch.sqrt(gt_dists)
    #recon_dists = torch.sqrt(recon_dists)
    gt_cmap = utils.get_pseudo_cmap(gt_dists)
    recon_cmap = utils.get_pseudo_cmap(recon_dists)
    consistency = torch.nn.functional.mse_loss(recon_dists, gt_dists, reduction='none').sum() / recon_dists.size(0)
    return consistency

def inter_penetr_loss(hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
    '''
    get penetrate object xyz and the distance to its NN
    :param hand_xyz: [B, 778, 3]
    :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
    :param obj_xyz: [B, 3000, 3]
    :return: inter penetration loss
    '''
    B = hand_xyz.size(0)
    mesh = Meshes(verts=hand_xyz, faces=hand_face)
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

    # if not nn_dist:
    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
    interior = utils_loss.get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)  # True for interior
    penetr_dist = nn_dist[interior].sum() / B  # batch reduction
    return 100.0 * penetr_dist

