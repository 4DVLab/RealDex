import os
import sys
sys.path.append(".")
sys.path.append("..")
import trimesh
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
import numpy as np
import random
import argparse
from dataset.dataset_grab import GRABDataset
from network.affordanceNet_obman_mano_vertex import affordanceNet
import mano
from mano.utils import Mesh


def vis_grab(model, dataloader, save_path, rh_mano):
    model.eval()
    vis_counter = 0
    with torch.no_grad():
        for batch_idx, (obj_pc, hand_param, next_frame_hand, obj_cmap) in enumerate(data_loader):
            print("batch: ", batch_idx)
            # obj_pc, hand_param, next_frame_hand, obj_cmap = obj_pc.to(device), hand_param.to(device), next_frame_hand.to(device), obj_cmap.to(device)
            obj_pc, hand_param, next_frame_hand, obj_cmap = obj_pc.cuda(), hand_param.cuda(), next_frame_hand.cuda(), obj_cmap.cuda()
            
            input_rh = rh_mano(betas=hand_param[:, :10], transl=hand_param[:, 10:13], global_orient=hand_param[:, 13:16],
                          hand_pose=hand_param[:, 16:])
            gt_rh = rh_mano(betas=next_frame_hand[:, :10], transl=next_frame_hand[:, 10:13], global_orient=next_frame_hand[:, 13:16],
                            hand_pose=next_frame_hand[:, 16:])
            input_hand_xyz = input_rh.vertices.cuda()  # [B,778,3]
            # gt_hand_xyz = gt_mano.vertices.cuda()
            
            recon_param = model(obj_pc, input_hand_xyz.permute(0,2,1))
            recon_rh = rh_mano(betas=recon_param[:, :10], transl=recon_param[:, 10:13], global_orient=recon_param[:, 13:16],
                          hand_pose=recon_param[:, 16:])
            
            gt_h_meshes = rh_mano.hand_meshes(gt_rh)
            gt_j_meshes = rh_mano.joint_meshes(gt_rh)
            recon_h_meshes = rh_mano.hand_meshes(recon_rh)
            recon_j_meshes = rh_mano.joint_meshes(recon_rh)
            obj_pc = obj_pc.transpose(1,2).detach().cpu().numpy()
            obj_pc = obj_pc[:, :, :3]
            for id in range(obj_pc.shape[0]):
                pc = trimesh.PointCloud(obj_pc[id])
                pc.export(os.path.join(save_path, f"{vis_counter}_object.ply"))
                gt_hj_meshes = Mesh.concatenate_meshes([gt_h_meshes[id], gt_j_meshes[id]])
                gt_hj_meshes.export(os.path.join(save_path, f"{vis_counter}_gt_hj.ply"))
                recon_hj_meshes = Mesh.concatenate_meshes([recon_h_meshes[id], recon_j_meshes[id]])
                recon_hj_meshes.export(os.path.join(save_path, f"{vis_counter}_recon_hj.ply"))
                vis_counter += 1
                
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''network information'''
    parser.add_argument("--batch_size", type=int, default=160)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1024, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    parser.add_argument("--model_type", type=str, default='baseline_mano')
    parser.add_argument("--loss_type", type=str, default='CD')
    parser.add_argument("--loss_weight", type=list, default=[1.0, 0.1, 1000.0, 10.0, 10.0])
    parser.add_argument("--use_contactmap", default='False', action='store_true')
    parser.add_argument("--dataloader_workers", type=int, default=16)
    
    args = parser.parse_args()
    
    # device
    use_cuda = torch.cuda.is_available()
        
    # device = torch.device("cuda" if use_cuda else "cpu")

    num_gpus = torch.cuda.device_count()
    
    # print("using device", device)

    # load model file
    model_path = "/home/liuym/Project/IntelligentHand/affordanceCVAE/checkpoints/grab_seq_best_test.pth"
    best_model = torch.load(model_path, map_location=torch.device('cpu'))['network']
    print(best_model.keys())
    
    # set vis result file path
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + '_' + str(local_time[2]) + '_' + str(local_time[3]) + '_' + str(local_time[4])
    save_root = "/remote-home/share/yumeng/results"
    save_root = os.path.join(save_root, model_path.split('/')[-1].split('.')[0], time_str)
    for mode in ["test", "train"]:
        os.makedirs(os.path.join(save_root, mode))

    # seed
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # network
    model = affordanceNet(
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_latent_size=args.latent_size,
        cvae_decoder_sizes=args.decoder_layer_sizes,
        cvae_condition_size=args.condition_size)
    model.load_state_dict(best_model)
    model = model.cuda()
    
    # multi-gpu
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(num_gpus))
        if num_gpus > 1:
            model = torch.nn.DataParallel(model)
    
    
    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                              model_type='mano',
                              use_pca=True,
                              num_pca_comps=45,
                              batch_size=args.batch_size,
                              flat_hand_mean=True).cuda()
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
    rh_faces = rh_faces.expand(args.batch_size, -1, -1).cuda() # [N, 1538, 3]

    # dataset
    grab_dir = "/home/liuym/results/GRAB_V00/"
    for mode in ["test", "train"]:
        dataset = GRABDataset(baseDir=grab_dir, mode=mode)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers)
        save_path = os.path.join(save_root, mode)
        vis_grab(model, data_loader, save_path, rh_mano)