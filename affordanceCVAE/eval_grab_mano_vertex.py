import os
import sys
sys.path.append(".")
sys.path.append("..")
# from dataset.dataset_grab import GRABDataset
import numpy as np


class EvalTool():
    def __init__(self, grab_dir, mode) -> None:
        self.frame_names = self.get_grab_framenames(grab_dir, mode)
        pass

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

result_dir="/remote-home/share/yumeng/results/grab_seq_best_test/9_28_15_28/test/"

grab_dir = "/home/liuym/results/GRAB_V00/"

            
for mode in ["test", "train"]:
    print(mode)
    eval_tool = EvalTool(grab_dir, mode)
    seq_groups = eval_tool.split_sequence("gamecontroller_lift")
    for seq in seq_groups:
        print(len(seq))