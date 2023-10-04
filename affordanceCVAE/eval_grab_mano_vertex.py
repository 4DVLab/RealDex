import os
import sys
sys.path.append(".")
sys.path.append("..")
# from dataset.dataset_grab import GRABDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.dataset_grab import GRABDataset
from network.affordanceNet_obman_mano_vertex import affordanceNet
import mano
from mano.utils import Mesh
import time
import trimesh
import argparse



class EvalTool():
    def __init__(self, grab_dir, mode):
        self.frame_names = self.get_grab_framenames(grab_dir, mode)
        self.mode = mode
        # mano hand model
        with torch.no_grad():
            self.rh_mano = mano.load(model_path='./models/mano/MANO_RIGHT.pkl',
                                model_type='mano',
                                use_pca=True,
                                num_pca_comps=45,
                                batch_size=args.batch_size,
                                flat_hand_mean=True).cuda()
        rh_faces = torch.from_numpy(self.rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes
        self.rh_faces = rh_faces.expand(args.batch_size, -1, -1).cuda() # [N, 1538, 3]

    def get_grab_framenames(self, grab_dir, mode):
        ds_path = os.path.join(grab_dir, mode)
        # load frame name
        frame_names = np.load(os.path.join(ds_path, 'frame_names.npz'))['frame_names']
        frame_names =np.asarray([file.split('/')[-1] for file in frame_names])
        return frame_names
    
    def split_sequence(self, seq_name):
        groups = []
        current_group = []
        for id, name in enumerate(self.frame_names):
            name:str
            if name.startswith(seq_name):
                current_group.append(id)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
        
        if current_group:
            groups.append(current_group)
            
        return groups
    
    def init_model(self, args, model_path):
        best_model = torch.load(model_path, map_location=torch.device('cpu'))['network']
        # network
        model = affordanceNet(
                                obj_inchannel=args.obj_inchannel,
                                cvae_encoder_sizes=args.encoder_layer_sizes,
                                cvae_latent_size=args.latent_size,
                                cvae_decoder_sizes=args.decoder_layer_sizes,
                                cvae_condition_size=args.condition_size
                            )
        model.load_state_dict(best_model)
        model = model.cuda()
        
        # multi-gpu
        torch.backends.cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
        
        return model
    
    def set_save_path(self, dir):
        # set vis result file path
        local_time = time.localtime(time.time())
        time_str = str(local_time[1]) + '_' + str(local_time[2])
        path = os.path.join(dir, time_str, self.mode)
        os.makedirs(path)
        return path
        
        
        
    
def save_result(model, data, save_path, rh_mano):
    model.eval()
    vis_counter = 0
    with torch.no_grad():
        obj_pc, hand_param, next_frame_hand, obj_cmap = data
        obj_pc, hand_param, next_frame_hand, obj_cmap = obj_pc.cuda(), hand_param.cuda(), next_frame_hand.cuda(), obj_cmap.cuda()
        B, N, _ = obj_pc.shape
        obj_pc = obj_pc.transpose(1, 2) #[B, 3, 3000]
        obj_pc = torch.cat([obj_pc, torch.ones(1, 1, 1).expand(B,-1,N).cuda()], dim=1) # [B, 4, 3000]
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
        recon_h_meshes = rh_mano.hand_meshes(recon_rh)
        obj_pc = obj_pc.transpose(1,2).detach().cpu().numpy()
        obj_pc = obj_pc[:, ::10, :3]
        for id in range(obj_pc.shape[0]):
            pc = Mesh(vertices=obj_pc[id])
            gt_meshes = Mesh.concatenate_meshes([gt_h_meshes[id], pc])
            gt_meshes.export(os.path.join(save_path, f"{vis_counter}_gt.ply"))
            recon_meshes = Mesh.concatenate_meshes([recon_h_meshes[id], pc])
            recon_meshes.export(os.path.join(save_path, f"{vis_counter}_recon.ply"))
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

    grab_dir = "/home/liuym/results/GRAB_V00/"
    
    mode = 'train'
    
    
    
                
    eval_tool = EvalTool(grab_dir, mode)
    
    # load model file
    model_path = "/home/liuym/Project/IntelligentHand/affordanceCVAE/checkpoints/grab_seq_best_test.pth"
    model = eval_tool.init_model(args, model_path)
    
    # set save path
    seq_name = "airplane_fly"
    save_dir = "/remote-home/share/yumeng/results/grasptta"
    save_dir = eval_tool.set_save_path(save_dir)
    
    
    # get dataset
    dataset = GRABDataset(baseDir=grab_dir, mode=mode)
    
    # split sequence
    seq_groups = eval_tool.split_sequence(seq_name)
    for group in seq_groups:
        start = group[0]
        end = start + args.batch_size
        print(f"{group[0]}-{group[-1]}")
        data = dataset[start:end]
        path = os.path.join(save_dir, seq_name, f"{group[0]}-{group[-1]}")
        os.makedirs(path, exist_ok=True)
        save_result(model, data, path, eval_tool.rh_mano)