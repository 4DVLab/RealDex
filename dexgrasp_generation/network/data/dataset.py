import torch
from torch.utils.data import DataLoader

import os
from os.path import join as pjoin
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..'))  # data -> model -> root, to import data_proc
sys.path.append(pjoin(base_dir, '..', '..'))  # data -> model -> root, to import data_proc

from datasets.dex_dataset import DFCDataset
from datasets.realdex_dataset import RealDexDataset
from datasets.object_dataset import Meshdata
from datasets.grab_dataset import GRABDataset, GRABMeshData
from utils.global_utils import my_collate_fn
import matplotlib.pyplot as plt
import numpy as np



def get_grab_dataloader(cfg, mode="train", shuffle=None):
    if shuffle is None:
        shuffle = (mode == "train")
    
    dataset = GRABDataset(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"])

def get_dex_dataloader(cfg, mode="train", shuffle=None):
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = DFCDataset(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"])

def get_realdex_dataloader(cfg, mode="train", shuffle=None):
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = RealDexDataset(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"])

def get_mesh_dataloader(cfg, mode="train"):
    dataset = Meshdata(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

def get_grab_mesh_dataloader(cfg, mode="train"):
    dataset = GRABMeshData(cfg, mode)
    return DataLoader(dataset, batch_size=cfg["batch_size"], 
                      shuffle=False, num_workers=cfg["num_workers"], collate_fn=my_collate_fn)
    
def feature_to_color(feature_values):
    colormap = plt.get_cmap('rainbow')  # Choose a colormap (e.g., 'viridis')
    colors = colormap(feature_values)[:, :3] # Apply the colormap and extract RGB values
    # print(np.max(colors), np.min(colors))
    
    colors *= 255
    return colors
