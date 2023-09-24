from torch.utils.data import Dataset
import torch
import os
import pickle
from torchvision import transforms
import numpy as np
from utils import utils
import time
from PIL import Image
import json
from utils import utils_loss
from utils import utils
import trimesh

class obman(Dataset):
    def __init__(self, img_root='/hand-object-3/download/dataset/ObMan/obman',
                 obj_root='/hand-object-3/download/dataset',
                 mode="train", vis=False, batch_size=160):

        self.mode = mode
        self.file_path = '/hand-object-3/download/dataset/ObMan/obman/{}.txt'.format(self.mode)
        self.mano_trans_path = '/hand-object-3/download/dataset/ObMan/obman/mano-fit/{}/mano_trans.json'.format(self.mode)
        self.mano_rot_path = '/hand-object-3/download/dataset/ObMan/obman/mano-fit/{}/global_rot.json'.format(self.mode)
        self.img_list = utils.readTxt_obman(self.file_path)
        self.mano_trans = json.load(open(self.mano_trans_path, 'r'))
        self.mano_rot = json.load(open(self.mano_rot_path, 'r'))

        self.img_root = img_root
        self.obj_root = obj_root
        self.dataset_size = len(self.img_list)

        self.transform = transforms.ToTensor()
        self.sample_nPoint = 3000
        self.batch_size = batch_size
        self.vis = vis


    def __len__(self):
        return self.dataset_size #- (self.dataset_size % self.batch_size)  # in case of unmatched mano batch size

    def __getitem__(self, idx):
        line = self.img_list[idx].strip()
        img_path = os.path.join(self.img_root, self.mode, 'rgb_obj', line)
        meta_path = img_path.replace('rgb_obj', 'meta').replace('jpg', 'pkl')
        if self.vis:
            img = Image.open(img_path)
        meta = pickle.load(open(meta_path, 'rb'))

        # hand information
        # hand_side = meta['side']
        hand_pose = torch.tensor(meta['hand_pose']) # [45]
        hand_shape = torch.tensor(meta['shape']) # [10]
        #hand_trans = torch.tensor(meta['trans']) # [3]
        hand_trans = torch.tensor(self.mano_trans[idx])
        hand_rot = torch.tensor(self.mano_rot[idx]) # [3]
        hand_param = torch.cat((hand_shape, hand_rot, hand_pose, hand_trans), dim=0) # [61]
        hand_xyz = torch.tensor(meta['verts_3d']).permute(1,0) # [778, 3] -> [3, 778]

        # object informationvim
        obj_id = meta['sample_id']
        obj_path = meta['obj_path']
        obj_path_seg = obj_path.split('/')[4:]
        obj_path_seg = [it + '/' for it in obj_path_seg]
        obj_mesh_path = ''.join(obj_path_seg)[:-1]
        obj_mesh_path = os.path.join(self.obj_root, obj_mesh_path)

        # obj pointcloud
        # obj_xyz_normalized = np.array(utils.fast_load_obj(open(obj_mesh_path))[0]['vertices']) # [N, 3]
        # nPoint = obj_xyz_normalized.shape[0]
        # obj_scale = meta['obj_scale']
        # obj_xyz = obj_xyz_normalized
        #choice = np.random.choice(nPoint, self.sample_nPoint, replace=True)
        #obj_xyz_normalized = obj_xyz_normalized[choice] # [N', 3]
        obj_mesh = utils.fast_load_obj(open(obj_mesh_path))[0]
        obj_mesh = trimesh.Trimesh(vertices=obj_mesh['vertices'],
                                   faces=obj_mesh['faces'])
        obj_xyz_resampled = trimesh.sample.sample_surface(obj_mesh, 10000)[0]
        obj_xyz_resampled_path = '/hand-object-3/download/dataset/ObMan/obman/{}/models_resampled/{}.npy'.format(self.mode, str(idx))
        np.save(obj_xyz_resampled_path, obj_xyz_resampled)
        #obj_xyz_normalized = np.array(utils.fast_load_obj(open(obj_mesh_path))[0]['vertices']) # [N, 3]
        obj_xyz_normalized = np.array(obj_xyz_resampled)
        nPoint = obj_xyz_normalized.shape[0]
        obj_scale = meta['obj_scale']
        obj_xyz = obj_xyz_normalized

        obj_pose = np.array((meta['affine_transform']))
        obj_xyz_transformed = utils.vertices_transformation(obj_xyz, obj_pose)
        obj_xyz_transformed = torch.tensor(obj_xyz_transformed, dtype=torch.float32)

        obj_scale_tensor = torch.tensor(obj_scale).type_as(obj_xyz_transformed).repeat(nPoint, 1) # [N', 1]
        obj_pc = torch.cat((obj_xyz_transformed, obj_scale_tensor), dim=-1)# [N', 4]
        obj_pc = obj_pc.permute(1, 0)  # [4, N']

        # NN distance
        NN_dists, _ = utils_loss.get_NN(obj_xyz_transformed.unsqueeze(0), hand_xyz.permute(1, 0).unsqueeze(0))
        #NN_dists_path = obj_mesh_path.replace('model_normalized', 'NN_dist').replace('.obj', '.npy')
        NN_dists = NN_dists.squeeze(0).detach().cpu().numpy()
        NN_dists_path = '/hand-object-3/download/dataset/ObMan/obman/{}/nn_dist/{}.npy'.format(self.mode, str(idx))
        np.save(NN_dists_path, NN_dists)


        return hand_xyz

