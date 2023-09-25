import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from network.pointnet_encoder import PointNetEncoder
from network.CVAE import VAE

class affordanceNet(nn.Module):
    def __init__(self, obj_inchannel=4,
                 cvae_encoder_sizes=[1024,512,256], cvae_latent_size=64,
                 cvae_decoder_sizes=[1024, 2048, 778*3], cvae_condition_size=1024):
        super(affordanceNet, self).__init__()

        self.obj_inchannel = obj_inchannel
        self.cvae_encoder_sizes = cvae_encoder_sizes
        self.cvae_latent_size = cvae_latent_size
        self.cvae_decoder_sizes = cvae_decoder_sizes
        self.cvae_condition_size = cvae_condition_size

        self.obj_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=self.obj_inchannel)
        self.hand_encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
        self.cvae = VAE(encoder_layer_sizes=self.cvae_encoder_sizes,
                        latent_size=self.cvae_latent_size,
                        decoder_layer_sizes=self.cvae_decoder_sizes,
                        condition_size=self.cvae_condition_size)

    def forward(self, obj_pc, hand_xyz):
        '''
        :param obj_pc: [B, 3+n, N1]
        :param hand_xyz: [B, 3, N2]
        :return: reconstructed hand vertex
        '''
        B, D, N_2 = hand_xyz.size()
        obj_glb_feature, _, _ = self.obj_encoder(obj_pc) # [B, 1024]
        hand_glb_feature, _, _ = self.hand_encoder(hand_xyz) # [B, 1024]

        if self.training:
            recon, means, log_var, z = self.cvae(hand_glb_feature, obj_glb_feature) # recon: [B, 778*3]
            recon = recon.contiguous().view(B, 3, 778)
            return recon, means, log_var, z
        else:
            # inference
            recon = self.cvae.inference(B, obj_glb_feature)
            recon = recon.contiguous().view(B, 3, 778)
            return recon


if __name__ == '__main__':
    model = affordanceNet(obj_inchannel=4,
                          cvae_encoder_sizes=[1024, 512, 256],
                          cvae_decoder_sizes=[1024, 256, 778*3])
    obj_xyz = torch.randn(3, 4, 3000)
    hand_xyz = torch.randn(3, 3, 778)
    print('params {}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.eval()
    #network.train()
    #recon, _, _, _ = network(obj_xyz, hand_xyz)
    recon = model(obj_xyz, hand_xyz)
    print(recon.size())
