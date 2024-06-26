import sys
sys.path.append(".")
sys.path.append("..")
import os
import torch
from torch.utils.data import DataLoader
from networks.pose_synthesis import GeoEncoder, GraspPoseNet
from models.hand_model import ShadowHandModel
from datasets.dexgraspnet_dataset import DexGraspNetDataset
import trimesh
from trimesh.sample import sample_surface_even, sample_surface
from pytorch3d.ops import sample_farthest_points
from pytorch3d.io import load_objs_as_meshes


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(self, data_dir, geo_encoder_path, tensorboard_dir, pc_num=2048):
        # hand_file = "./mjcf/shadow_hand_wrist_free.xml"
        self.train_ds = DexGraspNetDataset(data_dir=data_dir, split_type='train')
        self.obj_mesh_path = self.train_ds.obj_mesh_path
        self.pc_num = pc_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # network models
        self.geo_encoder = GeoEncoder(geo_encoder_path, device=self.device)
        self.grasp_pose_net = GraspPoseNet()

        # data
        self.train_ds = DataLoader(self.train_ds, batch_size=4, shuffle=True).to(self.device)
        print('Training set has {} instances'.format(len(self.train_ds)))

        # loss definition and weights for loss
        self.L2Loss = torch.nn.SmoothL1Loss()
        self.loss_weights = {'rec_loss': 1., 'rot_reg_loss':1., 'transl_reg_loss': 1.}

        # for training
        self.optimizer = torch.optim.Adam(self.grasp_pose_net.parameters(), lr=1e-3)
        self.epoch_index = 0
        self.max_epoch_num = 500


        # report in training
        time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.tb_writer = SummaryWriter(os.path.join(tensorboard_dir, time_stamp))
        self.log_per_epoch = 5
        self.num_batch_for_log = len(self.train_ds) // self.log_per_epoch

    def load_pts(self, obj_path_list):
        path_list = []
        pc = load_objs_as_meshes(obj_path_list, device=self.device)
        return pc
        


    def train(self):
        for epoch in range(self.max_epoch_num):
            print(f'EPOCH {epoch}:')

            self.grasp_pose_net.train()
            train_loss_dict = self.train_one_epoch()
            val_loss_dict = self.eval_one_epoch()

            # Log the running loss averaged per batch
            # for both training and validation
            for key in train_loss_dict:
                self.tb_writer.add_scalars(f'Training vs. Validation {key}',
                                { 'Training' : train_loss_dict[key], 'Validation' : val_loss_dict[key] },
                                epoch + 1)
            self.tb_writer.flush()

    def eval_one_epoch(self):
        self.grasp_pose_net.eval()

        running_loss = 0
        running_loss_dict = {key:0 for key in self.loss_weights}
        running_loss_dict["total_loss"] = 0

        with torch.no_grad():
            for i, item in enumerate(self.validation_set):
                hand_pose, transl, rot, obj_path, obj_scale = item
                obj_pts = self.load_pts(obj_path)
                geo_condition = self.geo_encoder(obj_pts)
                global_pose_condition = torch.cat([transl, rot], dim=1)
                pred_pose, transl_offset, global_ori_offset = self.grasp_pose_net( geo_condition, global_pose_condition, hand_pose)

                # loss
                loss_dict = {
                    "rec_loss": self.L2Loss(pred_pose, hand_pose),
                    "rot_reg_loss": torch.norm(global_ori_offset),
                    "transl_reg_loss": torch.norm(transl_offset)
                } 

                loss = 0
                for key in self.loss_weights:
                    loss += self.loss_weights[key] * loss_dict[key] 

                # Gather data and report
                running_loss += loss.item()
                running_loss_dict = {key: running_loss_dict[key] + loss_dict[key].item() for key in loss_dict}
                running_loss_dict["total_loss"] += loss.item()

            avg_running_loss = running_loss / (i+1)
            avg_loss_dict = {key: running_loss_dict[key]/self.num_batch_for_log for key in running_loss_dict}  # average loss

            print('LOSS valid {}'.format(avg_running_loss))
                
            # tb_x = self.epoch_index * len(self.training_set) + i + 1
            # self.tb_writer.add_scalars('Loss/Valid', last_loss_dict, epoch_number + 1)
            

        return avg_loss_dict

                
        

    def train_one_epoch(self):

        running_loss = 0
        last_loss = 0
        running_loss_dict = {key:0 for key in self.loss_weights}
        running_loss_dict["total_loss"] = 0

        self.geo_encoder.eval()
        self.grasp_pose_net.train()

        
        for i, item in enumerate(self.train_ds):

            # clear the gradients in last step
            self.optimizer.zero_grad()

            hand_pose, transl, rot, obj_path, obj_scale = item
            obj_pts = self.load_pts(obj_path)

            geo_condition = self.geo_encoder(obj_pts)
            global_pose_condition = torch.cat([transl, rot], dim=1)
            pred_pose, transl_offset, global_ori_offset = self.grasp_pose_net( geo_condition, global_pose_condition, hand_pose)

            # loss
            rec_loss = self.L2Loss(pred_pose, hand_pose)
            rot_reg_loss = torch.norm(global_ori_offset)
            transl_reg_loss = torch.norm(transl_offset)

            loss_dict = {
                    "rec_loss": self.L2Loss(pred_pose, hand_pose),
                    "rot_reg_loss": torch.norm(global_ori_offset),
                    "transl_reg_loss": torch.norm(transl_offset)
                } 

            loss = 0
            for key in self.loss_weights:
                loss += self.loss_weights[key] * loss_dict[key] 

            

            # compute gradient
            loss.backward()

            # update network parameters
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            running_loss_dict = {key: running_loss_dict[key] + loss_dict[key].item() for key in loss_dict}
            running_loss_dict["total_loss"] += loss.item()
            if i % self.num_batch_for_log == self.num_batch_for_log - 1:
                last_loss = running_loss / self.num_batch_for_log # average loss
                last_loss_dict = {key: running_loss_dict[key]/self.num_batch_for_log for key in running_loss_dict}  # average loss
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = self.epoch_index * len(self.train_ds) + i + 1
                self.tb_writer.add_scalars('Loss/Train', last_loss_dict, tb_x)
                running_loss_dict = dict.fromkeys(running_loss_dict.keys(), 0)

        return last_loss_dict
            
        


if __name__ == '__main__':
    # data_dir = "/Users/yumeng/Working/Project2023/data/dexgraspnet"
    # hand_file = "./mjcf/shadow_hand_wrist_free.xml"
    # ds = DexGraspNetDataset(data_dir=data_dir)
    # obj_mesh = trimesh.load(ds.object_mesh_origin)
    # object_mesh = obj_mesh.copy().apply_scale(ds.object_scale)
    # object_mesh.export("/Users/yumeng/Working/Project2023/result/SynthesizedGraspPose/test.obj")

    data_dir = "/Users/yumeng/Working/Project2023/data/dexgraspnet"
    pct_best_model = "/Users/yumeng/Working/Project2023/3rdparty/PCT_Pytorch-main/checkpoints/best/models/model.t7"
    trainer = Trainer(data_dir=data_dir, geo_encoder_path=pct_best_model, tensorboard_dir="./output/log")
    trainer.train()

    