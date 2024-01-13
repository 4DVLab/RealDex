from hydra import compose, initialize
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_mesh_dataloader, get_dex_dataloader
from data.dataset import get_grab_mesh_dataloader, feature_to_color
from trainer import Trainer
# from trainer_grab import Trainer
from utils.global_utils import result_to_loader, flatten_result
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import os
from os.path import join as pjoin
import json
from tqdm import tqdm, trange

import argparse

from train import process_config
from utils.interrupt_handler import InterruptHandler
from network.models.contactnet.contact_network import ContactMapNet
from utils.hand_model import AdditionalLoss, add_rotation_to_hand_pose
from utils.eval_utils import KaolinModel, eval_result
from utils.visualize import visualize
from utils.hand_model import HandModel
import trimesh
from pytorch3d import transforms as pttf
from network.models.model import get_model
from collections import OrderedDict
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

import json

def vis_result(hand, data, result_path):
    num_hand = hand['vertices'].shape[0]
    hand_verts = hand['vertices'].cpu()
    hand_faces = hand['faces'].cpu()
    # print(hand_verts.shape, hand_faces.shape)
    for i in range(num_hand):
        # colors = feature_to_color(result['cmap_pred'][i].cpu())
        obj_pc = data['obj_pc'][i].cpu()
        obj_pc = trimesh.PointCloud(vertices=obj_pc) #, colors=colors)
        hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_faces)
        hand_mesh.export(os.path.join(result_path, f"test_hand_{i}.ply"))
        obj_pc.export(os.path.join(result_path, f"test_obj_{i}.ply"))
        
        if 'obj_mesh' in data.keys():
            obj_mesh = data['obj_mesh'][i]
            (hand_mesh + obj_mesh).export(os.path.join(result_path, f"combined_{i}.ply"))
        if 'mesh_path' in data.keys():
            mesh_path = data['mesh_path'][i]
            obj_mesh = trimesh.load(mesh_path)
            scale = data['scale'][i].cpu()
            pose_matrix = data['pose_matrix'][i].cpu()
            verts = torch.from_numpy(obj_mesh.vertices).float()
            new_verts = scale * ( verts @ pose_matrix[:3, :3].T + pose_matrix[:3, 3])
            obj_mesh.vertices = new_verts
            
            (hand_mesh + obj_mesh).export(os.path.join(result_path, f"combined_{i}.ply"))

def vis_dex_data(cfg):
    cfg = process_config(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Logging """
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("EvalModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    result_path = "/home/liuym/results/unidexgrasp_datavis"
    if os.path.exists(result_path) is not True:
        os.makedirs(result_path)

    """ DataLoaders """
    train_loader = get_dex_dataloader(cfg, "train")
    
    for _, data in enumerate(tqdm(train_loader)):
        print(data.keys())
        hand_model = HandModel(
            mjcf_path='data/mjcf/shadow_hand.xml',
            mesh_path='data/mjcf/meshes',
            contact_points_path='data/mjcf/contact_points.json',
            penetration_points_path='data/mjcf/penetration_points.json',
            device=device,
        )
        
        hand_pose = torch.cat([data['translation'], data['rotation'], data['hand_qpos']],dim=-1).to(device)
        hand_pose = hand_pose.float()
        hand = hand_model(hand_pose=hand_pose, object_pc=data['obj_pc'].float().to(device), with_meshes=True)
        vis_result(hand, data, result_path)
        break
    
def compute_data_mean_std(cfg):
    cfg = process_config(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ Logging """
    log_dir = cfg["exp_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("EvalModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    result_path = "/home/liuym/results/unidexgrasp_datavis"
    if os.path.exists(result_path) is not True:
        os.makedirs(result_path)

    """ DataLoaders """
    train_loader = get_dex_dataloader(cfg, "train")
    pose_list = []
    
    running_mean = 0
    running_var = 0
    batch_count = 0
    count = 0
    for _, data in enumerate(tqdm(train_loader)):
        rotation = data['rotation'].to(device)
        # rotation = matrix_to_axis_angle(rotation)
        transl = data['translation'].to(device)
        qpos = data['hand_qpos'].to(device)
        hand_pose = torch.cat([transl, rotation, qpos],dim=-1)
        hand_pose = hand_pose.float()
        # running_mean, running_var, batch_count = update_stats(hand_pose, running_mean, running_var, batch_count)
        # batch_count += hand_pose.shape[0]
        pose_list.append(hand_pose)
        count += 1
        if count > 10:
            break
        
    
    all_pose = torch.cat(pose_list, dim=0)
    pose_mean = torch.mean(all_pose, dim=0)
    pose_std = torch.std(all_pose, dim=0)
    data_info = {'pose_mean': pose_mean.cpu(), 'pose_std': pose_std.cpu()}
    print(data_info)
    
    out_path = "./assets/DFCData/pose_mean_std.pt"

    # with open(out_path, 'w') as outfile:
    #     json.dump(data_info, outfile)
        
    torch.save(data_info, out_path)
    

def get_file_list(root_dir, splits_data, splits, categories):
    """
    :param root_dir: e.g. "./data"
    :return: e.g. ["./data/DFCData/bottle/poses/1a7ba1f4c892e2da30711cdbdbc73924/00000.npz", ...]
                or ["./data/DFCData/poses/bottle/1a7ba1f4c892e2da30711cdbdbc73924/00000.npz"]
    """
    file_list = []
    for category in categories:
        for split in splits:
            for instance in splits_data[category][split]:
                file_dir_path = pjoin(root_dir, "DFCData", "poses", category, instance)
                files = list(filter(lambda file: file[0] != "." and (file.endswith(".npz")),
                                    os.listdir(file_dir_path)))
                files = list(map(lambda file: pjoin(file_dir_path, file),
                                    files))
                file_list += files

    print(f"Split {splits}: {len(file_list)} pieces of data.")
    return file_list

def update_stats(batch, running_mean, running_var, batch_count):
    batch_mean = torch.mean(batch, dim=0)
    batch_var = torch.var(batch, dim=0)

    delta = batch_mean - running_mean
    tot_count = batch_count + batch.size(0)

    new_mean = running_mean + delta * batch.size(0) / tot_count
    m_a = running_var * (batch_count - 1)
    m_b = batch_var * (batch.size(0) - 1)
    M2 = m_a + m_b + delta ** 2 * batch_count * batch.size(0) / tot_count
    new_var = M2 / (tot_count - 1)

    return new_mean, new_var, tot_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="eval_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './eval_result'.")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    config_name= "configs_cvae"
    # config_name= "configs_grab_mesh"
    
    initialize(version_base=None, config_path="../" + config_name, job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
        
    # vis_dex_data(cfg)
    compute_data_mean_std(cfg)