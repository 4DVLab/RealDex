from hydra import compose, initialize
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_mesh_dataloader, get_dex_dataloader
from data.dataset import get_grab_mesh_dataloader, feature_to_color, get_realdex_dataloader
from trainer import Trainer
# from trainer_grab import Trainer
from utils.global_utils import result_to_loader, flatten_result
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import os
from os.path import join as pjoin
import json
import re
from tqdm import tqdm, trange

import argparse

from train import process_config
from utils.interrupt_handler import InterruptHandler
from network.models.contactnet.contact_network import ContactMapNet
from utils.hand_model import AdditionalLoss, add_rotation_to_hand_pose
from utils.eval_utils import KaolinModel, eval_result
from utils.visualize import visualize
# from utils.hand_model import HandModel
from utils.shadow_hand_builder import ShadowHandBuilder
import trimesh
from pytorch3d import transforms as pttf
from network.models.model import get_model
from collections import OrderedDict
from pytorch3d.transforms import axis_angle_to_matrix
import numpy as np


def main(cfg, result_path):
    cfg = process_config(cfg)

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



    """ DataLoaders """
    test_loader = get_realdex_dataloader(cfg, "test")
    # test_loader = get_grab_mesh_dataloader(cfg, "train")

    """ Trainer """
    trainers = {}
    for key in cfg['models'].keys():
        print(key)
        if key == 'affordance_cvae':
            continue
        net_cfg = compose(f"{cfg['models'][key]['type']}_config")
        print(net_cfg['exp_dir'])
        with open_dict(net_cfg):
            net_cfg['device'] = cfg['device']
        trainer = Trainer(net_cfg, logger)
        trainer.resume()
        trainers[key] = (trainer)
    
    # contact_net = trainers['affordance_cvae'].model.contact_net
    # contact_net = contact_net.to(cfg['device'])
    # contact_net.eval()

    """ Test """
    # sample
    loader = test_loader
    for i, data_tuple in enumerate(tqdm(loader)):
        data, obj_name = data_tuple
        data['object_name'] = obj_name
        for key, trainer in trainers.items():
            # loader = result_to_loader(result, cfg) if result else test_loader
            # result = []
            pred_dict, _ = trainer.test(data)
            data.update(pred_dict)
        result = {k: v.cpu() if type(v) == torch.Tensor else v for k, v in data.items()}
        torch.save(result, os.path.join(result_path, f"data_{i}.pt"))
        

def test_time_opt(contact_net, data):
    pass
                   
def divide(data):
    seen_data = []
    unseen_data = []

    object_code_list_train = []
    for splits_file_name in os.listdir('data/DFCData/splits'):
        with open(os.path.join('data/DFCData/splits', splits_file_name), 'r') as f:
            splits_map = json.load(f)
        object_code_list_train += [os.path.join(splits_file_name[:-5], object_code) for object_code in splits_map['train']]

    category_set_train = set()
    for object_code in object_code_list_train:
        category_set_train.add(object_code.split('-')[0])
    
    for i in range(len(data['object_code'])):
        object_code = data['object_code'][i]
        if 'ddg/' in object_code or 'mujoco/' in object_code or object_code.split('-')[0] in category_set_train:
            seen_data.append({k: v[i] for k, v in data.items()})
        else:
        #if 'ddg/' not in object_code and 'mujoco/' not in object_code and object_code.split('-')[0] not in category_set_train:
            unseen_data.append({k: v[i] for k, v in data.items()})
    
    return seen_data, unseen_data

def output_result(data, name):
    if len(data) == 0:
        return
    for has_tta in ['', 'tta_']:
        for label in ['q1', 'pen', 'tpen', 'valid_q1']:
            result = [dic[has_tta+label] for dic in data]
            print(f'category {name}, {has_tta+label}: {sum(result)/len(result)}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="eval_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './eval_result'.")
    return parser.parse_args()

def vis_test_result(device, result_path, vis_path):
    for filename in tqdm(os.listdir(result_path)):
        if not re.match(r'data_\d+\.pt$', filename):
            continue
        result = torch.load(os.path.join(result_path, filename))
        
        # hand_pose = result['hand_pose'].to(device)
        # hand_pose = hand_pose.float()
        
        global_translation = result['translation'].float().to(device)
        global_rotation = axis_angle_to_matrix(result['rotation'].float().to(device))
        qpos = result['hand_qpos'].float().to(device)
        hand_model = ShadowHandBuilder(device=device, 
                                       assets_dir="/public/home/v-liuym/projects/IntelligentHand/dexgrasp_generation/assets")
        hand_dict = hand_model.get_hand_model(global_rotation, global_translation, qpos)
        
        
        vis_result(hand_dict['meshes'], result, vis_path)
    
def vis_result(hand_meshes, data, result_path, mesh_dir=None):
    if mesh_dir is None:
        mesh_dir = "/storage/group/4dvlab/youzhuo/models/"
    num_hand = len(hand_meshes)
    hand_verts = hand_meshes.verts_padded().cpu()
    hand_faces = hand_meshes.faces_padded().cpu()
    
    if 'object_orient' in data:
        object_rotation = axis_angle_to_matrix(data['object_orient'].float().to(device))
        object_rotation = object_rotation.cpu().numpy()
    obj_tf_mat = np.eye(4)
    
    for i in range(num_hand):
        # colors = feature_to_color(result['cmap_pred'][i].cpu())
        obj_pc = data['obj_pc'][i].cpu()
        obj_pc = trimesh.PointCloud(vertices=obj_pc) #, colors=colors)
        hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_faces[i])
        if 'object_name' in data.keys():
            obj_name = data['object_name'][i]
            out_dir = os.path.join(result_path, obj_name)
            os.makedirs(out_dir, exist_ok=True)
            counter = len(os.listdir(out_dir))
            obj_mesh = trimesh.load(pjoin(mesh_dir, f"{obj_name}.obj"))
            obj_tf_mat[:3, :3] = object_rotation[i]
            obj_tf_mat[:3, -1] = data['object_transl'][i]
            obj_mesh.apply_transform(obj_tf_mat)
            # obj_mesh.export(pjoin(out_dir,f"obj_{counter}.ply"))
            (hand_mesh + obj_mesh).export(os.path.join(out_dir, f"combined_{counter}.ply"))
            obj_mesh.export(os.path.join(out_dir, f"obj_{counter}.ply"))
            hand_mesh.export(os.path.join(out_dir, f"hand_{counter}.ply"))
            
            
        elif 'obj_mesh' in data.keys():
            obj_mesh = data['obj_mesh'][i]
            (hand_mesh + obj_mesh).export(os.path.join(result_path, f"combined_{i}.ply"))
        elif 'mesh_path' in data.keys():
            mesh_path = data['mesh_path'][i]
            # print(mesh_path)
            obj_name = mesh_path.split('/')[-3]
            out_dir = os.path.join(result_path, obj_name)
            os.makedirs(out_dir, exist_ok=True)
            counter = len(os.listdir(out_dir))
            
            
            obj_mesh = trimesh.load(mesh_path)
            scale = data['scale'][i].cpu()
            print("scale: ", scale)
            pose_matrix = data['pose_matrix'][i].cpu()
            # obj_mesh.apply_transform(pose_matrix)
            verts = torch.from_numpy(obj_mesh.vertices).float()
            # new_verts =  verts @ pose_matrix[:3, :3].T + pose_matrix[:3, 3]
            # new_verts = scale * ( verts @ pose_matrix[:3, :3].T + pose_matrix[:3, 3])
            new_verts = (verts /scale - pose_matrix[:3, 3])@pose_matrix[:3,:3]
            
            obj_mesh.vertices = new_verts
            
            (hand_mesh + obj_mesh).export(os.path.join(out_dir, f"{counter}_combined.ply"))
            obj_mesh.export(os.path.join(out_dir, f"{counter}_obj.ply"))
            hand_mesh.export(os.path.join(out_dir, f"{counter}_hand.ply"))
            obj_pc.export(os.path.join(out_dir, f"{counter}_obj_pc.ply"))
            
            
            
            
        else:
            hand_mesh.export(os.path.join(result_path, f"test_hand_{i}.ply"))
            obj_pc.export(os.path.join(result_path, f"test_obj_pc_{i}.ply"))
            


if __name__ == "__main__":
    args = parse_args()
    config_name= "configs_realdex"
    # config_name= "configs_grab_mesh"
    
    initialize(version_base=None, config_path="../" + config_name, job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
        
    
    result_path = "/storage/group/4dvlab/yumeng/results/"
    result_path = os.path.join(result_path, "test_set_0118_"+config_name)
    vis_path = os.path.join(result_path, "vis_0118_"+config_name)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # results_file = os.path.join(result_path, "result_test_set_orig_ckpt.pt")
    main(cfg, result_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_test_result(device, result_path, vis_path)
