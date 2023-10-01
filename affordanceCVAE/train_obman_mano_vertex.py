import os
import time
import torch
import argparse
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from dataset.dataset_obman_mano2 import obman
from network.affordanceNet_obman_mano_vertex import affordanceNet
import numpy as np
import random
from utils import utils_loss
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss
from pytorch3d.loss import chamfer_distance
import mano


def train(args, epoch, model, train_loader, device, optimizer, log_root, rh_mano, rh_faces):
    since = time.time()
    logs = defaultdict(list)
    a, b, c, d, e = args.loss_weight
    model.train()
    for batch_idx, (obj_pc, hand_param, obj_cmap) in enumerate(train_loader):
        obj_pc, hand_param, obj_cmap = obj_pc.to(device), hand_param.to(device), obj_cmap.to(device)
        gt_mano = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                          hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:])
        hand_xyz = gt_mano.vertices.to(device)  # [B,778,3]
        optimizer.zero_grad()
        recon_param, mean, log_var, z = model(obj_pc, hand_xyz.permute(0,2,1))  # recon [B,61] mano params
        recon_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                             hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
        recon_xyz = recon_mano.vertices.to(device)  # [B,778,3]
        # obj xyz NN dist and idx
        obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(obj_pc.permute(0,2,1)[:,:,:3], hand_xyz)
        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc.permute(0, 2, 1)[:, :, :3], recon_xyz)

        # mano param loss
        param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)
        # mano recon xyz loss, KLD loss
        cvae_loss, recon_loss_num, KLD_loss_num = CVAE_loss_mano(recon_xyz, hand_xyz, mean, log_var, args.loss_type, 'train')
        # cmap loss
        #cmap_loss = CMap_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_cmap)
        cmap_loss = CMap_loss3(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)
        # cmap consistency loss
        consistency_loss = CMap_consistency_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, hand_xyz,
                                                 obj_nn_dist_recon, obj_nn_dist_gt)
        # inter penetration loss
        penetr_loss = inter_penetr_loss(recon_xyz, rh_faces, obj_pc.permute(0,2,1)[:,:,:3],
                                        obj_nn_dist_recon, obj_nn_idx_recon)
        if epoch >= 5:
            loss = a * cvae_loss + b * param_loss + c * cmap_loss + d * penetr_loss + e * consistency_loss
        else:
            loss = a * cvae_loss + b * param_loss + d * penetr_loss + e * consistency_loss
        loss.backward()
        optimizer.step()
        logs['loss'].append(loss.item())
        logs['param_loss'].append(param_loss.item())
        logs['recon_loss'].append(recon_loss_num)
        logs['KLD_loss'].append(KLD_loss_num)
        logs['cmap_loss'].append(cmap_loss.item())
        logs['penetr_loss'].append(penetr_loss.item())
        logs['cmap_consistency'].append(consistency_loss.item())
        if batch_idx % args.print_every == 0 or batch_idx == len(train_loader) - 1:
            print("Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, Total Loss {:9.5f}, Mesh {:9.5f}, KLD {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f}".format(
                    epoch, args.epochs, batch_idx, len(train_loader) - 1, loss.item(),
                    recon_loss_num, KLD_loss_num, param_loss.item(), cmap_loss.item(), consistency_loss.item(), penetr_loss.item()))

    time_elapsed = time.time() - since
    out_str = "Epoch: {:02d}/{:02d}, train, time {:.0f}m, Mean Toal Loss {:9.5f}, Mesh {:9.5f}, KLD {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f}".format(
        epoch, args.epochs, time_elapsed // 60,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['KLD_loss']) / len(logs['KLD_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
    )
    with open(log_root, 'a') as f:
        f.write(out_str+'\n')


def val(args, epoch, model, val_loader, device, log_root, checkpoint_root, best_val_loss, rh_mano, rh_faces, mode='val'):
    # validation
    total_recon_loss, total_param_loss, total_cmap_loss = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (obj_pc, hand_param, obj_cmap) in enumerate(val_loader):
            obj_pc, hand_param, obj_cmap = obj_pc.to(device), hand_param.to(device), obj_cmap.to(device)
            hand_xyz = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                               hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:]).vertices.to(device)  # [B,778,3]
            recon_param = model(obj_pc, hand_xyz.permute(0,2,1))
            recon_xyz = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:]).vertices.to(device)  # [B,778,3]

            # param loss
            param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)
            # mesh recon loss
            recon_loss = CVAE_loss_mano(recon_xyz, hand_xyz, -1, -1, args.loss_type, mode)
            # cmap loss
            #cmap_loss = CMap_loss(obj_pc.permute(0, 2, 1)[:,:,:3], hand_xyz, obj_cmap)
            total_recon_loss += recon_loss.item()
            total_param_loss += param_loss.item()
            #total_cmap_loss += cmap_loss.item()
    mean_recon_loss, mean_param_loss, mean_cmap_loss = total_recon_loss / len(val_loader), total_param_loss / len(val_loader), total_cmap_loss / len(val_loader)
    out_str = "Epoch: {:02d}/{:02d}, {}, Mean Mesh Recon Loss: {:9.5f}, Param loss {:9.5f}, Best Mesh Recon Loss: {:9.5f}".format(
        epoch, args.epochs, mode, mean_recon_loss, mean_param_loss, min(best_val_loss, mean_recon_loss))
    with open(log_root, 'a') as f:
        f.write(out_str + '\n')
    if mean_recon_loss < best_val_loss and args.train_mode != 'Test':
        save_name = os.path.join(checkpoint_root, 'model_best_{}.pth'.format(mode))
        torch.save({
            'network': model.module.state_dict(),
            'epoch': epoch
        }, save_name)
    return min(best_val_loss, mean_recon_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--figs", type=str, default='figs')
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--dataloader_workers", type=int, default=16)
    parser.add_argument("--train_mode", type=str, default='TrainTest')
    '''network information'''
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1024, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    parser.add_argument("--model_type", type=str, default='baseline_mano')
    parser.add_argument("--loss_type", type=str, default='CD')
    parser.add_argument("--loss_weight", type=list, default=[1.0, 0.1, 1000.0, 10.0, 10.0])
    parser.add_argument("--use_contactmap", default='False', action='store_true')
    args = parser.parse_args()

    # log file
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3])
    model_root = os.path.join('./logs2', args.model_type)
    model_info = 'v2p_W{}_hard_cmap3thre1'.format(str(args.loss_weight))
    save_root = os.path.join(model_root, time_str + '_' + model_info)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    log_root = save_root + '/log.txt'
    log_file = open(log_root, 'w+')
    log_file.write(str(args) + '\n')
    log_file.write('weights for cvae, param, cmap, penetr, consistency loss are {}'.format(str(args.loss_weight)) + '\n')
    log_file.close()

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
    device_num = 1

    # network
    model = affordanceNet(
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_latent_size=args.latent_size,
        cvae_decoder_sizes=args.decoder_layer_sizes,
        cvae_condition_size=args.condition_size).to(device)

    # multi-gpu
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)
            device_num = len(device_ids)

    # dataset
    if 'Train' in args.train_mode:
        train_dataset = obman(mode="train", batch_size=args.batch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers)
    if 'Val' in args.train_mode:
        val_dataset = obman(mode="val", batch_size=args.batch_size)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
    if 'Test' in args.train_mode:
        eval_dataset = obman(mode="test", batch_size=args.batch_size)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.3, 0.6, 0.8, 0.9]], gamma=0.5)

    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                              model_type='mano',
                              use_pca=True,
                              num_pca_comps=45,
                              batch_size=args.batch_size,
                              flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
    rh_faces = rh_faces.repeat(args.batch_size, 1, 1).to(device) # [N, 1538, 3]

    best_val_loss = float('inf')
    best_eval_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        if 'Train' in args.train_mode:
            train(args, epoch, model, train_loader, device, optimizer, log_root, rh_mano, rh_faces)
            scheduler.step()
        if 'Val' in args.train_mode:
            best_val_loss = val(args, epoch, model, val_loader, device, log_root, save_root, best_val_loss, rh_mano, rh_faces, 'val')
        if 'Test' in args.train_mode:
            best_eval_loss = val(args, epoch, model, eval_loader, device, log_root, save_root, best_eval_loss, rh_mano, rh_faces, 'test')

