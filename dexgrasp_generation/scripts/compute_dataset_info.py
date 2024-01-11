import os
import numpy as np
import sys
import os
sys.path.insert(0, os.getcwd())
from hydra import compose, initialize

from datasets.realdex_dataset import RealDexDataset
import torch

def compute(dataloader):
    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data in dataloader:
        batch_samples = data[0].size(0)
        data = data[0].view(batch_samples, data[0].size(1), -1)
        
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

if __name__ == '__main__':
    initialize(version_base=None, config_path="../configs_realdex", job_name="train")
    
    cfg = compose(config_name='cvae_config')
    
    dataset = RealDexDataset(cfg, mode='train')
    print(dataset[0])