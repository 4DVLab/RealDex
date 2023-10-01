import numpy as np
import open3d
from utils import utils_vis

id = '3'

#pc_path_baseline = 'C:\\Users\\think\\server\\baseline_mano_v2p2\\{}.npy'.format(id)
pc_path_soft = 'C:\\Users\\think\\server\\baseline_soft\\{}.npy'.format(id) # cmap using finger inside vertex
pc_path_hard = 'C:\\Users\\think\\server\\baseline_hard\\{}.npy'.format(id)  # self cmap
#pc_path_sh_cmaploss0 = 'C:\\Users\\think\\server\\baseline_softhard_cmaploss0\\{}.npy'.format(id)
#pc_path_sh_cmaploss1 = 'C:\\Users\\think\\server\\baseline_softhard_cmaploss4\\{}.npy'.format(id)
#pc_path_sh_cmaploss3 = 'C:\\Users\\think\\server\\baseline_softhard_cmaploss3\\{}.npy'.format(id)
#pc_path_sh_cmaploss3 = 'C:\\Users\\think\\server\\baseline_hard_cmap3thre1\\{}.npy'.format(id)
pc_path_sh_cmaploss4 = 'C:\\Users\\think\\server\\baseline_hard_cmap3thre1_2\\{}.npy'.format(id)


# pc_baseline = np.load(pc_path_baseline).T
# pc_soft = np.load(pc_path_soft).T
# pc_hard = np.load(pc_path_hard).T
# pc_sh_l0 = np.load(pc_path_sh_cmaploss0).T
# pc_sh_l1 = np.load(pc_path_sh_cmaploss1).T
# pc_sh_l3 = np.load(pc_path_sh_cmaploss3).T
pc_sh_l4 = np.load(pc_path_sh_cmaploss4).T

obj = np.load(pc_path_soft.replace('.npy', '_obj.npy')).T
utils_vis.show_pointcloud_objhand(pc_sh_l4[:778,:], obj)  # gt
# utils_vis.show_pointcloud_objhand(pc_soft[778:,:], obj)
# utils_vis.show_pointcloud_objhand(pc_hard[778:,:], obj)
# utils_vis.show_pointcloud_objhand(pc_sh_l0[778:,:], obj)
#utils_vis.show_pointcloud_objhand(pc_sh_l3[778:,:], obj)
utils_vis.show_pointcloud_objhand(pc_sh_l4[778:,:], obj)
# utils_vis.show_pointcloud_objhand(pc_sh_l4[778:,:], obj)
