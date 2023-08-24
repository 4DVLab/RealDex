import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from networks.point_cloud_transformer.model import Pct

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x
    

class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class GeoEncoder(nn.Module):
    def __init__(self, model_path, device):
        """
        Pretrained Geometric Encoder
        """
        super(GeoEncoder).__init__()
        self.device = device
        pct_model = Pct().to(self.device)
        if device=='cuda':
            self.pct_model = nn.DataParallel(pct_model) 
            self.pct_model.load_state_dict(torch.load(model_path))
        else:
            pct_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.pct_model = pct_model
        
        self.pct_model = self.pct_model.eval()

    def forward(self, pc_data):
        feature = self.pct_model.get_feature(pc_data)
        return feature

class GraspPoseNet(nn.Module):
    def __init__(self, n_neurons = 64, latentD = 16, geo_dim = 256, global_pose_dim = 3+6, hand_pose_dim = 22):
        # 一个自编码器,输入机械手的pose（不包括global）和condition，输出机械手的pose。condition是物体的点云，机械手global pose

        super(GraspPoseNet).__init__()
        self.latentD = latentD

        self.condition_dim = geo_dim + global_pose_dim
        self.enc_bn1 = nn.BatchNorm1d(self.condition_dim + hand_pose_dim)
        self.enc_rb1 = ResBlock(self.condition_dim + hand_pose_dim, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + self.condition_dim + hand_pose_dim, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, latentD)
        self.enc_var = nn.Linear(n_neurons, latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(self.condition_dim)
        self.dec_rb1 = ResBlock(latentD + self.condition_dim, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + latentD + self.condition_dim, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, hand_pose_dim)
        self.dec_transl_offset = nn.Linear(n_neurons, 3)
        self.dec_global_ori_offset = nn.Linear(n_neurons, 6)

    def forward(self, geo_condition, global_pose_condition, hand_pose):
        condition = torch.cat([geo_condition, global_pose_condition], dim=1)

        Z = self.encoder(condition, hand_pose)
        hand_pose, transl_offset, global_ori_offset = self.decoder(Z, condition)

        return hand_pose, transl_offset, global_ori_offset

    def encoder(self, condition, hand_pose):
        x = torch.cat([condition, hand_pose], dim=1)
        x0 = self.enc_bn1(x)
        x = self.enc_rb1(x0, True)
        x = self.enc_rb2(torch.cat([x0, x], dim=1), True)
        latent_code = self.enc_mu(x)
        return latent_code
    
    def decoder(self, Zin, condition):
        condition = self.dec_bn1(condition)
        x0 = torch.cat([Zin, condition], dim=1)
        x = self.dec_rb1(x0, True)
        x = self.dec_rb2(torch.cat([x0, x], dim=1), True)

        hand_pose = self.dec_pose(x)
        transl_offset = self.dec_transl_offset(x)
        global_ori_offset = self.dec_global_ori_offset(x)

        return hand_pose, transl_offset, global_ori_offset





