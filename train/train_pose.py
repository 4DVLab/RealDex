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

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class SingleObjTrainer:
    def __init__(self, geo_net_path, tensorboard_dir, pc_num=2048):
        data_dir = "/Users/yumeng/Working/Project2023/data/dexgraspnet"
        # hand_file = "./mjcf/shadow_hand_wrist_free.xml"
        self.ds = DexGraspNetDataset(data_dir=data_dir)
        self.object_mesh_origin = trimesh.load(ds.object_mesh_origin)

        pc = sample_surface_even(self.object_mesh_origin, 2 * pc_num)
        pc = sample_surface(pc, pc_num)
        self.object_pts = pc
        
        # network models
        self.geo_encoder = GeoEncoder()
        self.geo_encoder.load_state_dict(geo_net_path)
        self.grasp_pose_net = GraspPoseNet()

        # data
        self.training_set = DataLoader(ds, batch_size=4, shuffle=True)
        print('Training set has {} instances'.format(len(self.training_set)))

        # for training
        self.L2Loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.grasp_pose_net.parameters(), lr=1e-3)

        # report in training
        self.epoch_index = 0
        time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.tb_writer = SummaryWriter(os.path.join(tensorboard_dir, time_stamp))
        self.log_per_batch = 100

        
        

    def train_one_epoch(self):

        running_loss = 0
        last_loss = 0
        train_loss_dict = {"total_loss": 0, "rec_loss": 0, "rot_reg_loss": 0, "transl_reg_loss": 0}
        self.geo_encoder.eval()
        self.grasp_pose_net.train()
        
        for i, item in enumerate(self.training_set):

            # clear the gradients in last step
            self.optimizer.zero_grad()


            hand_pose, transl, rot, obj_scale = item
            object_mesh = self.object_mesh_origin.copy().apply_scale(obj_scale)
            geo_condition = self.geo_encoder(self.object_pts * obj_scale)
            global_pose_condition = torch.cat([transl, rot], dim=1)
            pred_pose, transl_offset, global_ori_offset = self.grasp_pose_net( geo_condition, global_pose_condition, hand_pose)

            # loss
            rec_loss = self.L2Loss(pred_pose, hand_pose)
            rot_reg_loss = torch.norm(global_ori_offset)
            transl_reg_loss = torch.norm(transl_offset)

            
            loss = rec_loss + rot_reg_loss + transl_reg_loss

            train_loss_dict["total_loss"] += loss.item()
            train_loss_dict["rec_loss"] += rec_loss.item()
            train_loss_dict["rot_reg_loss"] += rot_reg_loss.item()
            train_loss_dict["transl_reg_loss"] += transl_reg_loss.item()

            # compute gradient
            loss.backward()

            # update network parameters
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.log_per_batch == 0:
                last_loss = running_loss / self.log_per_batch # loss per batch
                last_loss_dict = {key: train_loss_dict[key]/self.log_per_batch for key in train_loss_dict}  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = self.epoch_index * len(self.training_set) + i + 1
                self.tb_writer.add_scalars('Loss/Train', last_loss_dict, tb_x)
                running_loss = 0.
                train_loss_dict = dict.fromkeys(train_loss_dict.keys(), 0)
            
        


if __name__ == '__main__':
    data_dir = "/Users/yumeng/Working/Project2023/data/dexgraspnet"
    hand_file = "./mjcf/shadow_hand_wrist_free.xml"
    ds = DexGraspNetDataset(data_dir=data_dir)
    obj_mesh = trimesh.load(ds.object_mesh_origin)
    object_mesh = obj_mesh.copy().apply_scale(ds.object_scale)
    object_mesh.export("/Users/yumeng/Working/Project2023/result/SynthesizedGraspPose/test.obj")


    # train()

    