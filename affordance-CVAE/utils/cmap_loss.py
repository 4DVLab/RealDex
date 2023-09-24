from typing import Union
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds


def cmap_loss(obj_xyz, hand_xyz, cmap):
    '''
    :param obj_xyz: [B, N1, 3]
    :param hand_xyz: [B, N2, 3]
    :param cmap: [B, N, 10] for 10 types of contact map
    :return:
    '''
    since = time.time()
    B = obj_xyz.size(0)
    T = cmap.size(2)
    obj_lengths = torch.full(
        (obj_xyz.shape[0],), obj_xyz.shape[1], dtype=torch.int64, device=obj_xyz.device
    ) # [B], N for each num
    hand_lengths = torch.full(
        (hand_xyz.shape[0],), hand_xyz.shape[1], dtype=torch.int64, device=hand_xyz.device
    )
    obj_nn = knn_points(obj_xyz, hand_xyz, lengths1=obj_lengths, lengths2=hand_lengths, K=1) #[dists, idx]
    obj_CD = obj_nn.dists[..., 0]  # [B, N1] NN distance from obj pc to hand pc

    cmap_loss_list = []
    print('time for compute CD {}'.format(time.time() - since))
    for i in range(B):
        tmp_list = []
        for j in range(T):
            mask = cmap[i, :, j] # [N1]
            n_points = torch.sum(mask)
            tmp_list.append(obj_CD[i][mask].sum() / n_points) # point reduction
        cmap_loss_list.append(
            torch.min(torch.stack(tmp_list))
        )
    loss = torch.stack(cmap_loss_list).sum() / B # batch reduction
    print('time for index {}'.format(time.time() - since))
    return loss


if __name__ == '__main__':
    a = torch.randn(96,3000,3, requires_grad=True).cuda()
    b = torch.randn(96,778,3, requires_grad=True).cuda()
    cmap = torch.randn(96,3000,10, requires_grad=True)
    #cmap = torch.zeros(96, 3000, 10, requires_grad=True)
    cmap = cmap >= 0.5
    loss = cmap_loss(a, b, cmap)
    print(loss)
    loss.backward()