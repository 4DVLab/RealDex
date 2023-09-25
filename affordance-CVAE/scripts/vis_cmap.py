import numpy as np
import open3d
from utils import utils_vis

dist_path = 'C:\\Users\\think\\server\\NN_dist.npy'
dist = np.load(dist_path)  # [N,]
print(dist.shape)
print(dist.max(), dist.min())

handobj_path = 'C:\\Users\\think\\server\\handobj.npy'
handobj = np.load(handobj_path)  # [N+778, 3]
print(handobj)
print(handobj.shape)

utils_vis.show_dist_objhand(handobj, dist)  # gt

