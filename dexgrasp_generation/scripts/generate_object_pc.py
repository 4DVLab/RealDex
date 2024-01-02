"""
Last modified date: 2023.06.06
Author: Jialiang Zhang
Description: Generate object point clouds
"""

import os

# os.chdir(os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import transforms3d
import torch
import pytorch3d.io
import pytorch3d.ops
import pytorch3d.structures
from multiprocessing import Pool, current_process
from tqdm import tqdm
import trimesh


def sample_projected(_):
    args, object_code, idx = _

    worker = current_process()._identity[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_list[(worker - 1) % len(args.gpu_list)]
    print(idx)

    object_path = os.path.join(args.data_root_path, object_code, 'coacd', 'decomposed.obj')
    # sample pc
    obj_mesh = trimesh.load(object_path)
    sampled_pc = trimesh.sample.sample_surface(obj_mesh, 4000)

    np.save(os.path.join(args.data_root_path, object_code, 'obj_sampled_pcs.npy'), sampled_pc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # experiments settings
    parser.add_argument('--data_root_path', type=str, default='../data/DFCData/meshes')
    parser.add_argument('--n_poses', type=int, default=100)
    parser.add_argument('--max_n_points', type=int, default=9000)
    parser.add_argument('--num_samples', type=int, default=3000)
    parser.add_argument('--n_cpu', type=int, default=8)
    # parser.add_argument('--n_cameras', type=int, default=6)
    # parser.add_argument('--theta', type=float, default=np.pi / 4)
    parser.add_argument('--scale', type=float, default=0.1)
    parser.add_argument('--gpu_list', type=str, nargs='*', default=['0', '1', '2', '3'])
    # camera settings
    parser.add_argument('--camera_distance', type=float, default=0.5)
    parser.add_argument('--camera_height', type=float, default=0.05)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--near', type=float, default=0.1)
    parser.add_argument('--far', type=float, default=100)
    args = parser.parse_args()

    object_category_list = os.listdir(args.data_root_path)
    object_code_list = []
    for object_category in object_category_list:
        object_code_list += [os.path.join(object_category, object_code) for object_code in sorted(os.listdir(os.path.join(args.data_root_path, object_category)))]
    # object_code_list = [object_code for object_code in object_code_list if not os.path.exists(os.path.join(args.data_root_path, object_code, 'pcs.npy'))]

    # object_code_list = object_code_list[:1]

    parameters = []
    for idx, object_code in enumerate(object_code_list):
        parameters.append((args, object_code, idx))
    
    with Pool(args.n_cpu) as p:
        it = tqdm(p.imap(sample_projected, parameters), desc='sampling', total=len(parameters))
        list(it)
