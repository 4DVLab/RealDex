import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.realpath('.'))

from hydra import compose, initialize
import pytorch3d.transforms
import argparse
import random
import numpy as np
import logging
import torch
import plotly.graph_objects as go
from datasets.grab_dataset import GRABDataset
from utils.hand_model import HandModel
from network.train_grab_baseline import process_config
from network.trainer_grab import Trainer
from omegaconf.omegaconf import open_dict
from network.data.dataset import get_mesh_dataloader
from network.models.contactnet.contact_network import ContactMapNet
from utils.hand_model import AdditionalLoss, add_rotation_to_hand_pose
from utils.global_utils import result_to_loader, flatten_result
import tqdm


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
    test_loader = get_mesh_dataloader(cfg, "test")

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
        loader = result_to_loader(result, cfg) if result else test_loader
        result = []
        for _, data in enumerate(tqdm(loader)):
            for i in range(cfg['models'][key]['sample_num']):
                pred_dict, _ = trainer.test(data)
                data.update(pred_dict)
                result.append({k: v.cpu() if type(v) == torch.Tensor else v for k, v in data.items()})
    

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--num', type=int, default=100)
args = parser.parse_args()

initialize(version_base=None, config_path="../configs", job_name="train")
cfg = compose(config_name='glow_config.yaml')

# seed
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
random.seed(args.seed)

# dataset
dataset = GRABDataset(cfg, mode=args.mode)
print(len(dataset))
data_dict = dataset[args.num]
hand_pose = torch.cat([
    torch.tensor(data_dict['canon_translation'], dtype=torch.float), 
    pytorch3d.transforms.matrix_to_axis_angle(torch.tensor(data_dict['canon_rotation'], dtype=torch.float)), 
    torch.tensor(data_dict['hand_qpos'], dtype=torch.float)
])

# hand model
hand_model = HandModel(
    mjcf_path='data/mjcf/shadow_hand.xml',
    mesh_path='data/mjcf/meshes',
)
hand = hand_model(hand_pose.unsqueeze(0), with_meshes=True)

# visualize
object_pc = data_dict['canon_obj_pc'][:3000]
table_pc = data_dict['canon_obj_pc'][3000:]
object_plotly = go.Scatter3d(x=object_pc[:, 0], y=object_pc[:, 1], z=object_pc[:, 2], mode='markers', marker=dict(size=2, color='lightgreen'))
table_pc = go.Scatter3d(x=table_pc[:, 0], y=table_pc[:, 1], z=table_pc[:, 2], mode='markers', marker=dict(size=2, color='lightgrey'))
hand_plotly = go.Mesh3d(x=hand['vertices'][0, :, 0], y=hand['vertices'][0, :, 1], z=hand['vertices'][0, :, 2], i=hand['faces'][:, 0], j=hand['faces'][:, 1], k=hand['faces'][:, 2], color='lightblue', opacity=1)
fig = go.Figure([object_plotly, table_pc, hand_plotly])
fig.show()
