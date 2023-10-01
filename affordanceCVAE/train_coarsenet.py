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
from utils.loss_coarsenet import Coarsenet_cvae_loss, loss_cnet, CVAE_loss_mano
from pytorch3d.loss import chamfer_distance
import mano


def train(args, epoch, model, train_loader, device, optimizer, log_root, rh_mano, rh_faces):
    since = time.time()
    v_weights = torch.from_numpy(np.load('rhand_weight.npy')).to(torch.float32).to(device)
    logs = defaultdict(list)
    a, b, c = args.loss_weight
    model.train()
    for batch_idx, (obj_pc, hand_xyz, hand_param, obj_cmap) in enumerate(train_loader):
        obj_pc, hand_param = obj_pc.to(device), hand_param.to(device)
        optimizer.zero_grad()
        recon_param, mean, log_var, z = model(obj_pc, hand_param) # recon [B,61] mano params

        # mano param loss
        param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0) #+ \
                     #10.0 * torch.nn.functional.mse_loss(recon_param[:, 10:13], hand_param[:, 10:13], reduction='none').sum() / recon_param.size(0)
        # get hand mesh
        recon_xyz = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                            hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:]).vertices.to(device)  # [B,778,3]
        hand_xyz = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                            hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:]).vertices.to(device)  # [B,778,3]
        # cvae loss
        cvae_loss, recon_loss_num, KLD_loss_num = Coarsenet_cvae_loss(recon_xyz, hand_xyz, mean, log_var)
        # ho loss
        ho_loss = loss_cnet(recon_xyz, hand_xyz, rh_faces, obj_pc[:,:3,:].permute(0,2,1), v_weights)

        loss = a * cvae_loss + b * param_loss + c * ho_loss
        loss.backward()
        optimizer.step()
        logs['loss'].append(loss.item())
        logs['param_loss'].append(param_loss.item())
        logs['recon_loss'].append(recon_loss_num)
        logs['KLD_loss'].append(KLD_loss_num)
        logs['ho_loss'].append(ho_loss.item())
        if batch_idx % args.print_every == 0 or batch_idx == len(train_loader) - 1:
            print("Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, weight {}, Total Loss {:9.5f}, Mesh loss {:9.5f}, KLD loss {:9.5f}, Param loss {:9.5f}, HO loss {:9.5f}".format(
                    epoch, args.epochs, batch_idx, len(train_loader) - 1, str(args.loss_weight), loss.item(),
                    recon_loss_num, KLD_loss_num, param_loss.item(), ho_loss.item()))

    time_elapsed = time.time() - since
    out_str = "Epoch: {:02d}/{:02d}, train, time {:.0f}m, weight {}, Mean Toal Loss {:9.5f}, Mesh loss {:9.5f}, KLD loss {:9.5f}, Param loss {:9.5f}, HO loss {:9.5f}".format(
        epoch, args.epochs, time_elapsed // 60, str(args.loss_weight),
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['KLD_loss']) / len(logs['KLD_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['ho_loss']) / len(logs['ho_loss']))
    with open(log_root, 'a') as f:
        f.write(out_str+'\n')


def val(args, epoch, model, val_loader, device, log_root, checkpoint_root, best_val_loss, rh_mano, rh_faces, mode='val'):
    # validation
    total_recon_loss, total_param_loss, total_cmap_loss = 0.0, 0.0, 0.0
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (obj_pc, hand_xyz, hand_param, obj_cmap) in enumerate(val_loader):
            obj_pc, hand_param = obj_pc.to(device), hand_param.to(device)
            recon_param = model(obj_pc, hand_param)
            preds.append(recon_param)

            # param loss
            param_loss = torch.nn.functional.mse_loss(recon_param, hand_param, reduction='none').sum() / recon_param.size(0)
            # mesh recon loss
            recon_xyz = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                                hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:]).vertices.to(device)  # [B,778,3]
            hand_xyz = rh_mano(betas=hand_param[:, :10], global_orient=hand_param[:, 10:13],
                               hand_pose=hand_param[:, 13:58], transl=hand_param[:, 58:]).vertices.to(device)  # [B,778,3]
            recon_loss = CVAE_loss_mano(recon_xyz, hand_xyz, -1, -1, args.loss_type, mode)
            total_recon_loss += recon_loss.item()
            total_param_loss += param_loss.item()

    mean_recon_loss, mean_param_loss = total_recon_loss / len(val_loader), total_param_loss / len(val_loader)
    out_str = "Epoch: {:02d}/{:02d}, {}, weight {}, Mean Mesh Recon Loss: {:9.5f}, Param loss {:9.5f}, Best Mesh Recon Loss: {:9.5f}".format(
        epoch, args.epochs, mode, str(args.loss_weight), mean_recon_loss, mean_param_loss, min(best_val_loss, mean_recon_loss))
    with open(log_root, 'a') as f:
        f.write(out_str + '\n')
    if mean_recon_loss < best_val_loss and args.train_mode != 'Test':
        save_name = os.path.join(checkpoint_root, 'model_best_{}.pth'.format(mode))
        torch.save({
            'network': model.module.state_dict(),
            'epoch': epoch
        }, save_name)
        np.save('preds_best.npy', torch.stack(preds).detach().cpu().numpy())
        np.save(os.path.join(checkpoint_root, 'preds_best_ep{}.npy'.format(epoch)), torch.stack(preds).detach().cpu().numpy())
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
    parser.add_argument("--encoder_layer_sizes", type=list, default=[61, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    parser.add_argument("--model_type", type=str, default='coarsenet')
    parser.add_argument("--loss_type", type=str, default='L2')
    parser.add_argument("--loss_weight", type=list, default=[1.0, 0.1, 10.0])
    parser.add_argument("--use_contactmap", default='False', action='store_true')
    args = parser.parse_args()

    # log file
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3])
    model_root = os.path.join('./logs', args.model_type)
    #model_info = 'p2p_vertLoss{}_paramW{}'.format(args.loss_type, '{}_{}_{}'.format(args.loss_weight[0], args.loss_weight[1], args.loss_weight[2]))
    save_root = os.path.join(model_root, time_str)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    log_root = save_root + '/log.txt'
    log_file = open(log_root, 'w+')
    log_file.write(str(args) + '\n')
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
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.3, 0.6, 0.8, 0.9]], gamma=0.7)

    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                              model_type='mano',
                              use_pca=True,
                              num_pca_comps=45,
                              batch_size=args.batch_size,
                              flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3) # [1, 1538, 3], face triangle indexes
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

