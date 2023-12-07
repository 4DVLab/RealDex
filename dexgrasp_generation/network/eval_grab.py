from hydra import compose, initialize
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from data.dataset import get_grab_mesh_dataloader, get_grab_dataloader

from trainer_grab import Trainer
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
from utils.grab_hand_model import AdditionalLoss, add_rotation_to_hand_pose
from utils.eval_utils import KaolinModel, eval_result
from utils.visualize import visualize

from pytorch3d import transforms as pttf
from utils.grab_hand_model import HandModel
import trimesh



def main(cfg):
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
    mesh_data = get_grab_mesh_dataloader(cfg, 'train')

    """ Trainer """
    trainers = []
    for key in cfg['models'].keys():
        net_cfg = compose(f"{cfg['models'][key]['type']}_config")
        print(net_cfg['exp_dir'])
        with open_dict(net_cfg):
            net_cfg['device'] = cfg['device']
        trainer = Trainer(net_cfg, logger)
        trainer.resume()
        trainers.append(trainer)

    contact_cfg = compose(f"{cfg['tta']['contact_net']['type']}_config")
    with open_dict(contact_cfg):
        contact_cfg['device'] = cfg['device']
    contact_net = ContactMapNet(contact_cfg).to(cfg['device'])
    contact_net.eval()
    tta_loss = AdditionalLoss(cfg['tta'], 
                              cfg['device'], 
                              cfg['dataset']['num_obj_points'], 
                              cfg['dataset']['num_hand_points'], contact_net)

    """ Test """
    result = None
    # sample
    for key, trainer in zip(cfg['models'].keys(), trainers):
        loader = result_to_loader(result, cfg) if result else mesh_data
        result = []
        for _, data in enumerate(tqdm(loader)):
            for i in range(cfg['models'][key]['sample_num']):
                pred_dict, _ = trainer.test(data)
                data.update(pred_dict)
                result.append({k: v.cpu() if type(v) == torch.Tensor else v for k, v in data.items()})
    
    # tta
    loader = result_to_loader(result, cfg, cfg['tta']['batch_size'])
    result = []
    for i, data in enumerate(tqdm(loader)):
        points = data['canon_obj_pc'].cuda()
        hand_pose = torch.cat([data['canon_translation'], torch.zeros_like(data['canon_translation']), data['hand_pos']], dim=-1)
        old_hand_pose = hand_pose.clone()
        data['hand_pose'] = add_rotation_to_hand_pose(old_hand_pose, data['sampled_rotation'])

        hand_pose = hand_pose.cuda()
        hand = tta_loss.hand_model(hand_pose, points, with_surface_points=True)
        discretized_cmap_pred = tta_loss.cmap_func(dict(canon_obj_pc=points, observed_hand_pc=hand['surface_points']))['contact_map'].exp()
        cmap_pred = (torch.argmax(discretized_cmap_pred, dim=-1) + 0.5) / discretized_cmap_pred.shape[-1]
        
        hand_pose.requires_grad_()
        optimizer = torch.optim.Adam([hand_pose], lr=cfg['tta']['lr'])
        for t in range(cfg['tta']['iterations']):
            optimizer.zero_grad()
            loss = tta_loss.tta_loss(hand_pose, points, cmap_pred)
            loss.backward()
            optimizer.step()

        data['tta_hand_pose'] = add_rotation_to_hand_pose(hand_pose.detach().cpu(), data['sampled_rotation'])
        result.append(data)
        
    torch.save(data, "/home/liuym/results/unidexgrasp_test_on_grab/result_train_set.pt")

    
def vis_result(filename, device, result_path):
    result = torch.load(filename)
    print(result.keys())
    print(len(result))
    
    hand_pose = result['hand_pose'].to(device)
    
    
    hand_model = HandModel(device=device)
    
    hand = hand_model(hand_pose=hand_pose, obj_points=result['obj_pc'].to(device), with_meshes=True)
    num_hand = hand['vertices'].shape[0]
    hand_verts = hand['vertices'].cpu()
    hand_faces = hand['faces'].cpu()
    for i in range(num_hand):
        obj_pc = trimesh.PointCloud(vertices=result['canon_obj_pc'][i])
        hand_mesh = trimesh.Trimesh(vertices=hand_verts[i], faces=hand_faces[i])
        hand_mesh.export(os.path.join(result_path, f"test_hand_{i}.ply"))
        obj_pc.export(os.path.join(result_path, f"test_obj_{i}.ply"))
        
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="eval_config")
    parser.add_argument("--exp-dir", type=str, help="E.g., './eval_result'.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    initialize(version_base=None, config_path="../configs", job_name="train")
    if args.exp_dir is None:
        cfg = compose(config_name=args.config_name)
    else:
        cfg = compose(config_name=args.config_name, overrides=[f"exp_dir={args.exp_dir}"])
    # main(cfg)
    
    result_path = "/home/liuym/results/unidexgrasp_test_on_grab/"
    results_file = os.path.join(result_path, "result_test_set_orig_ckpt.pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_result(results_file, device, result_path)
