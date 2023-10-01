import os
import time
import torch
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.dataset_obman_mano import obman
from network.affordanceNet_obman_mano_param import affordanceNet
import numpy as np
import random
from utils.loss import CVAE_loss_mano, CMap_loss
from utils import utils_vis
import mano


def eval(args, model, eval_loader, device, rh_mano):
    # validation
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (obj_pc, hand_xyz, hand_param, obj_cmap, img) in enumerate(eval_loader):
            obj_pc, hand_param = obj_pc.to(device), hand_param.to(device)
            recon_param = model(obj_pc, hand_param)
            recon_xyz = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:]).vertices.to(device)  # [B,778,3]
            hand_xyz = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                               hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:]).vertices.to(device)  # [B,778,3]

            loss = torch.sqrt(torch.sum((recon_xyz - hand_xyz) ** 2)) / hand_xyz.size(0)
            total_loss += loss.item()
            if args.vis:
                utils_vis.vis_reconstructed_hand_vertex(batch_idx, recon_xyz[0], img, args.fig_root, hand_xyz[0], obj_pc[0], recon_param[0], p=False)
                print('finish visualization of sample idx {}'.format(batch_idx))


    # val_mean_loss = total_loss / len(eval_loader)
    # out_str = "Mean Recon Loss: {:9.5f}".format(val_mean_loss)
    # with open(log_root, 'a') as f:
    #     f.write(out_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--fig_root", type=str, default='figs/baseline')
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=16)
    parser.add_argument("--use_contactmap", default='False', action='store_true')
    parser.add_argument("--vis", default='False', action='store_true')
    '''network information'''
    parser.add_argument("--model_path", type=str, default='./checkpoints/v1baseline_model_best_test.pth')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[61, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device", device)

    # network
    model = affordanceNet(
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_latent_size=args.latent_size,
        cvae_decoder_sizes=args.decoder_layer_sizes,
        cvae_condition_size=args.condition_size)

    # checkpoint
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))['network']
    model.load_state_dict(checkpoint)
    model = model.to(device)

    # multi-gpu
    if device == torch.device("cuda"):
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)

    # dataset
    dataset = obman(mode="test", vis=args.vis)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                            model_type='mano',
                            use_pca=True,
                            num_pca_comps=45,
                            batch_size=args.batch_size,
                            flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3)  # [1, 1538, 3], face triangle indexes
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device)  # [N, 1538, 3]

    # eval
    eval(args, model, dataloader, device, rh_mano)

