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


def main(cfg, result_file):
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
        net_cfg = compose(f"{cfg['models'][key]['type']}_config")
        print(net_cfg['exp_dir'])
        with open_dict(net_cfg):
            net_cfg['device'] = cfg['device']
        trainer = Trainer(net_cfg, logger)
        trainer.resume()
        trainers[key] = (trainer)

    # contact_cfg = compose(f"{cfg['tta']['contact_net']['type']}_config")
    # with open_dict(contact_cfg):
    #     contact_cfg['device'] = cfg['device']
    # contact_net = ContactMapNet(contact_cfg)
    # ckpt_dir = pjoin(contact_cfg['exp_dir'], 'ckpt')
    # model_name = get_model(ckpt_dir, contact_cfg.get('resume_epoch', None))
    # ckpt = torch.load(model_name)['model']
    # new_ckpt = OrderedDict()
    # for name in ckpt.keys():
    #     new_name = name.replace('net.', '')
    #     if new_name.startswith('backbone.'):
    #         new_name = new_name.replace('backbone.', '')
    #     new_ckpt[new_name] = ckpt[name]
    
    # contact_net.load_state_dict(new_ckpt)
    
    contact_net = trainers['affordance_cvae'].model.contact_net
    contact_net = contact_net.to(cfg['device'])
    contact_net.eval()
    
    tta_loss = AdditionalLoss(cfg['tta'], 
                              cfg['device'], 
                              cfg['dataset']['num_obj_points'], 
                              cfg['dataset']['num_hand_points'], contact_net)

    """ Test """
    result = None
    # sample
    for key, trainer in trainers.items():
        loader = result_to_loader(result, cfg) if result else test_loader
        result = []
        for _, data in enumerate(tqdm(loader)):
            pred_dict, _ = trainer.test(data)
            data.update(pred_dict)
            result.append({k: v.cpu() if type(v) == torch.Tensor else v for k, v in data.items()})
            
    print(data.keys())
       
    torch.save(data, result_file)

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

def vis_test_result(filename, device, result_path):
    result = torch.load(filename)
    print(result.keys())
    print(len(result))
    
    hand_pose = result['hand_pose'].to(device)
    hand_model = HandModel(
            mjcf_path='data/mjcf/shadow_hand.xml',
            mesh_path='data/mjcf/meshes',
            contact_points_path='data/mjcf/contact_points.json',
            penetration_points_path='data/mjcf/penetration_points.json',
            device=device,
        )
    hand_pose = hand_pose.float()
    hand = hand_model(hand_pose=hand_pose, object_pc=result['obj_pc'].to(device), with_meshes=True)
    vis_result(hand, result, result_path)
    
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


if __name__ == "__main__":
    args = parse_args()
    config_name= "configs_realdex"
    # config_name= "configs_grab_mesh"
    
    initialize(version_base=None, config_path="../" + config_name, job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
        
    
    result_path = "/public/home/v-liuym/results/unidexgrasp_test_set_"+config_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    results_file = os.path.join(result_path, "result_test_set_orig_ckpt.pt")
    main(cfg, results_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_test_result(results_file, device, result_path)
