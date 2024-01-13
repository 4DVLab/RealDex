import os
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize
from datasets.realdex_dataset import RealDexDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def to_cpu(tensor):
    return tensor.cpu().float()
def compute_mean_std(dataloader):
    key_list = ["translation","rotation","hand_qpos"]
    
    pose_data_list = []
    for data in tqdm(dataloader):
        data_list = [data[key] for key in key_list]
            
        pose_data = torch.cat(data_list, dim=-1).to(device) # [B, 6 + 22]
        # print(pose_data.shape)
        pose_data_list.append(pose_data)
    
    pose_data = torch.cat(pose_data_list, dim=0)
    pose_mean = torch.mean(pose_data, dim=0)
    pose_std = torch.std(pose_data, dim=0)
    
    out_path = "./assets/RealDexData"
    out_path = os.path.join(out_path, "pose_mean_std.pt")
    
    torch.save({'pose_mean': to_cpu(pose_mean), 'pose_std': to_cpu(pose_std)}, out_path)

        

if __name__ == '__main__':
    initialize(version_base=None, config_path="../configs_realdex", job_name="train")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = compose(config_name='cvae_config')
    
    dataset = RealDexDataset(cfg, mode='train')
    dataset = RealDexDataset(cfg, mode='val')
    dataset = RealDexDataset(cfg, mode='test')
    
    # print(dataset[0])
    # dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    # compute_mean_std(dataloader)