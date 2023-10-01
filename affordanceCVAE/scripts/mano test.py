import torch
import mano
import json
import numpy as np
import pickle
import open3d
from utils import utils_vis

rot_file = json.load(open('C:\\Users\\think\\server\\global_rot.json', 'r'))
trans_file = json.load(open('C:\\Users\\think\\server\\mano_trans.json', 'r'))
gt_file = pickle.load(open('C:\\Users\\think\\server\\00004478.pkl', 'rb'))

model_path = 'PATH_TO_MANO_MODELS'
n_comps = 15
batch_size = 10
rhm_path = '../models/mano/MANO_RIGHT.pkl'
rh_model = mano.load(model_path='../models/mano/MANO_RIGHT.pkl',
                     model_type='mano',
                     use_pca=True,
                     num_pca_comps=n_comps,
                     batch_size=1,
                     flat_hand_mean=True)

betas = torch.tensor(gt_file['shape']).view(1,-1)
pose = torch.tensor(gt_file['hand_pose']).view(1,-1)[:,:n_comps]
global_orient = torch.tensor(rot_file[2795]).view(1,-1)
transl        = torch.tensor(trans_file[2795]).view(1,-1)

print(betas.size(), pose.size(), global_orient.size(), transl.size())

print(global_orient)
print(transl)

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl).vertices

gt_xyz = torch.tensor(gt_file['verts_3d']).permute(1,0) # [3, 778]
print(gt_xyz.size())
hand_xyz = output[0].permute(1,0)
print(hand_xyz.size())

finger_vertices = [309, 317, 318, 319, 320, 322, 323, 324, 325,
      326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
      350, 351, 352, 353, 354, 355,
                   429, 433, 434, 435, 436, 437, 438,
      439, 442, 443, 444, 455, 461, 462, 463, 465, 466, 467,
                   547, 548,
      549, 550, 553, 566, 573, 578,
                   657, 661, 662, 664, 665, 666, 667,
      670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691, 692,
      693, 694, 695,
                   736, 737, 738, 739, 740, 741, 743, 753, 754, 755,
      756, 757, 759, 760, 761, 762, 763, 764, 766, 767, 768,
                  73,  96,  98,  99, 772, 774, 775, 777]
#
bigfinger_vertices = [697, 698, 699, 700, 712, 713, 714, 715, 737, 738, 739, 740, 741, 743, 744, 745, 746, 748, 749,
                      750, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768]

indexfinger_vertices = [46, 47, 48, 49, 164, 165, 166, 167, 194, 195, 223, 237, 238, 280, 281, 298, 301, 317, 320, 323,
                        324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 340, 341, 342, 343, 344, 345, 346, 347, 348,
                        349, 350, 351, 352, 353, 354, 355]

middlefinger_vertices = [356, 357, 358, 359, 375, 376, 386, 387, 396, 397, 402, 403, 413, 429, 433,
                         434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 452, 453, 454, 455, 456, 459,
                         460, 461, 462, 463, 464, 465, 466, 467]

fourthfinger_vertices = [468, 469, 470, 471, 484, 485, 486, 496, 497, 506, 507, 513, 514, 524, 545, 546, 547, 548, 549,
                         550, 551, 552, 553, 555, 563, 564, 565, 566, 567,  570, 572, 573, 574, 575, 576, 577, 578]

smallfinger_vertices = [580, 581, 582, 583, 600, 601, 602, 614, 615, 624, 625, 630, 631, 641, 663, 664, 665, 666, 667,
                        668, 670, 672,680, 681, 682, 683, 684, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695]

hand_vertices = [73, 96, 98, 99, 772, 774, 775, 777]

finger_vertices = bigfinger_vertices + indexfinger_vertices + middlefinger_vertices + fourthfinger_vertices + smallfinger_vertices + hand_vertices

#pc = np.hstack((gt_xyz.detach().numpy(), hand_xyz.detach().numpy())).T
pc = np.hstack((gt_xyz.detach().numpy(), 0.0001 + gt_xyz[:, finger_vertices].detach().numpy())).T
#utils_vis.show_pointcloud_hand(pc)
utils_vis.show_pointcloud_fingertips(pc)

print(torch.nn.functional.mse_loss(hand_xyz, gt_xyz, reduction='none').sum())