import sys
sys.path.append(".")
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from network.models.backbones.pointnet_encoder import PointNetEncoder
from utils.hand_model import HandModel
from pytorch3d.transforms import matrix_to_axis_angle
import os
from pytorch3d.ops import sample_farthest_points



class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, condition_size=1024):
        super().__init__()

        if conditional:
            assert condition_size > 0

        # assert type(encoder_layer_sizes) == list
        # assert type(latent_size) == int
        # assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_size)

    def forward(self, x, c=None):
        # x: [B, 58]
        # if x.dim() > 2:
        #     x = x.view(-1, 58)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means
        # z = means
        

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):
        batch_size = n
        z = torch.randn([batch_size, self.latent_size], device=c.device) * 0
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        #print('encoder', self.MLP)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1) # [B, 1024+61]
        #print('x size before MLP {}'.format(x.size()))
        x = self.MLP(x)
        #print('x size after MLP {}'.format(x.size()))
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        #print('mean size {}, log_var size {}'.format(means.size(), log_vars.size()))
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #print('decoder', self.MLP)

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            #print('z size {}'.format(z.size()))

        x = self.MLP(z)

        return x

class AffordanceCVAE(nn.Module):
    def __init__(self, cfg, contact_net=None):
        super(AffordanceCVAE, self).__init__()

        self.obj_inchannel = cfg["model"]["obj_inchannel"]
        
        self.num_obj_feature = cfg["model"]["obj_feature_dim"]
        self.num_hand_points = cfg["dataset"]["num_obj_points"]
        self.num_obj_points = cfg["dataset"]["num_obj_points"]
        
        self.cvae_encoder_sizes = cfg["model"]["cvae_encoder_sizes"]
        self.cvae_latent_size = cfg["model"]["cvae_latent_size"]
        self.cvae_decoder_sizes = cfg["model"]["cvae_decoder_sizes"]
        
        if cfg['model']['network']['type'] == 'pointnet':
            self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3, use_stn=True)
            self.hand_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3, use_stn=True)
        else:
            raise NotImplementedError(f"backbone {cfg['model']['network']['type']} not implemented")

        self.cvae = VAE(encoder_layer_sizes=self.cvae_encoder_sizes,
                        latent_size=self.cvae_latent_size,
                        decoder_layer_sizes=self.cvae_decoder_sizes,
                        condition_size=self.num_obj_feature)
        
        # self.num_obj_points = cfg['dataset']['num_obj_points']
        self.device = cfg['device']
        self.hand_model = HandModel(
            mjcf_path='data/mjcf/shadow_hand.xml',
            mesh_path='data/mjcf/meshes',
            n_surface_points=self.num_hand_points,
            contact_points_path='data/mjcf/contact_points.json',
            penetration_points_path='data/mjcf/penetration_points.json',
            device=self.device,
        )
        self.cmap_func = contact_net.forward
        
        # self.data_info = torch.load("./assets/DFCData/pose_mean_std.pt")
        
        data_info_path = os.path.join(cfg["dataset"]["data_info_path"], "pose_mean_std.pt")
        self.data_info = torch.load(data_info_path)
        print(self.data_info)
        
        
        
    def inference(self, obj_pc):
        B = obj_pc.shape[0]
        obj_glb_feature, _, _ = self.obj_encoder(obj_pc) # [B, 1024]
        #hand_glb_feature, _, _ = self.hand_encoder(hand_xyz) # [B, 1024]
        
        pose_mean = self.data_info['pose_mean'].to(self.device)
        pose_std = self.data_info['pose_std'].to(self.device)
        
        recon = self.cvae.inference(B, obj_glb_feature)
        # recon = recon * pose_std + pose_mean
        
        recon = recon.contiguous().view(B, -1)
        
        ret_dict = {
                "translation": recon[:, :3],
                "rotation": recon[:, 3:6],
                "hand_qpos": recon[:, 6:],
                "hand_pose": recon
            }
        
        return ret_dict

    def forward(self, dic):
        '''
        :param x: [B, N, 3]
        :return: ret_dict:
            {
                "rotation": [B, R],
                "translation": [B, 3],
                "hand_qpos": [B, H]
            }
        '''
        # fetch data
        obj_pc = dic['obj_pc']
        obj_pc, _ = sample_farthest_points(obj_pc, K=self.num_obj_points)
        
        
        transl = dic['translation']
        rotation = dic['rotation']
        qpos = dic['hand_qpos']
        B = qpos.shape[0]
        qpos_dim = qpos.shape[-1]
        # print(transl.shape, rotation.shape, qpos.shape)
        gt_hand_pose = torch.cat([transl, rotation, qpos],dim=-1).detach()
        gt_hand = self.hand_model(gt_hand_pose, obj_pc, with_penetration=True, with_surface_points=True)
        # get the object center
        obj_center = torch.mean(obj_pc, dim=1).detach() #[B, 3]
        # canon_obj_pc = obj_pc - obj_center[:, None, :]
        obj_glb_feature, _, _ = self.obj_encoder(obj_pc.transpose(1, 2)) # [B, 1024]
        hand_glb_feature, _, _ = self.hand_encoder(gt_hand['surface_points'].transpose(1,2)) # [B, 1024]
        
        # get the new recon hand
        recon, mean, log_var, z = self.cvae(hand_glb_feature, obj_glb_feature) # recon: [B, 28]
        
        pose_mean = self.data_info['pose_mean'].to(self.device)
        pose_std = self.data_info['pose_std'].to(self.device)
        recon = recon.contiguous().view(B, 6 + qpos_dim)
        # recon = recon * pose_std + pose_mean
        
        recon_hand = self.hand_model(recon, obj_pc, with_penetration=True, with_surface_points=True, with_contact_candidates=True)
        
        # cmap predicted by the contactnet
        # canon_hand_points = gt_hand['surface_points']-obj_center[:, None, :]
        # discretized_cmap_pred = self.cmap_func(dict(canon_obj_pc=canon_obj_pc, 
        #                                             observed_hand_pc=canon_hand_points))
        discretized_cmap_pred = self.cmap_func(dict(canon_obj_pc=obj_pc, 
                                                    observed_hand_pc=gt_hand['surface_points']))
        discretized_cmap_pred = discretized_cmap_pred['contact_map'].detach().exp()# [B, N, 10]
        
        arange = (torch.arange(0, discretized_cmap_pred.shape[-1], dtype=discretized_cmap_pred.dtype, device=discretized_cmap_pred.device)+0.5)
        cmap_pred = torch.mean(discretized_cmap_pred * arange, dim=-1)
        
        ret_dict = {
                "translation": recon[:, :3],
                "rotation": recon[:, 3:6],
                "hand_qpos": recon[:, 6:],
                "recon_hand_points": recon_hand['surface_points'],
                "gt_hand_points": gt_hand['surface_points'],
                "log_var": log_var,
                "mean": mean,
                "z": z,
                "pen_dist": recon_hand['penetration'],
                "cmap_pred": cmap_pred,
                "o2h_dist": recon_hand['distances']
                
            }
        return ret_dict
        


if __name__ == '__main__':
    model = AffordanceCVAE(obj_inchannel=4,
                          cvae_encoder_sizes=[22, 512, 256],
                          cvae_decoder_sizes=[1024, 256, 22])
    obj_xyz = torch.randn(13, 4, 3000)
    hand_param = torch.randn(13, 22)
    print('params {}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.eval()
    #model.train()
    #recon, _, _, _ = model(obj_xyz, hand_param)
    recon = model(obj_xyz, hand_param)
    print(recon.size())
