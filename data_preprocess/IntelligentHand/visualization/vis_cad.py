import os
import sys
sys.path.append(".")
sys.path.append("..")
# os.chdir("/remote-home/liuym/Project/IntelligentHand")

from models.hand_model import ShadowHandModel
from data_preprocess.ours.utils.kintree import Kintree
from scipy.spatial.transform import Rotation
import numpy as np

import json
import torch
# from robot_descriptions.loaders.mujoco import load_robot_description




JOINTS_MAPPING = [
    ('robot0:FFJ3', 'rh_ffproximal'), ('robot0:FFJ2', 'rh_ffknuckle'), ('robot0:FFJ1', 'rh_ffmiddle'), ('robot0:FFJ0', 'rh_ffdistal'), #食指
    ('robot0:MFJ3', 'rh_mfproximal'), ('robot0:MFJ2', 'rh_mfknuckle'), ('robot0:MFJ1', 'rh_mfmiddle'), ('robot0:MFJ0', 'rh_mfdistal'), #中指
    ('robot0:RFJ3', 'rh_rfproximal'), ('robot0:RFJ2', 'rh_rfknuckle'), ('robot0:RFJ1', 'rh_rfmiddle'), ('robot0:RFJ0', 'rh_rfdistal'), #无名指
    ('robot0:LFJ4', 'rh_lfmetacarpal'), ('robot0:LFJ3', 'rh_lfproximal'), ('robot0:LFJ2', 'rh_lfknuckle'), ('robot0:LFJ1', 'rh_lfmiddle'), ('robot0:LFJ0', 'rh_lfdistal'), #小指
    ('robot0:THJ4', 'rh_thbase'), ('robot0:THJ3', 'rh_thproximal'), ('robot0:THJ2', 'rh_thhub'), ('robot0:THJ1', 'trh_thmiddle'), ('robot0:THJ0', 'rh_thdistal') #大拇指
]

RAW_JOINTS_NAME = [
    'rh_ffproximal',  'rh_ffknuckle', 'rh_ffmiddle', 'rh_ffdistal', #食指
    'rh_mfproximal',  'rh_mfknuckle', 'rh_mfmiddle', 'rh_mfdistal', #中指
    'rh_rfproximal',  'rh_rfknuckle', 'rh_rfmiddle', 'rh_rfdistal', #无名指
    'rh_lfmetacarpal','rh_lfproximal', 'rh_lfknuckle', 'rh_lfmiddle', 'rh_lfdistal', #小指
    'rh_thbase', 'rh_thproximal', 'rh_thhub', 'rh_thmiddle', 'rh_thdistal' #大拇指
]

JOINTS_NAME = [
            'robot0:FFJ3', 'robot0:FFJ2', 'robot0:FFJ1', 'robot0:FFJ0',
            'robot0:MFJ3', 'robot0:MFJ2', 'robot0:MFJ1', 'robot0:MFJ0',
            'robot0:RFJ3', 'robot0:RFJ2', 'robot0:RFJ1', 'robot0:RFJ0',
            'robot0:LFJ4', 'robot0:LFJ3', 'robot0:LFJ2', 'robot0:LFJ1', 'robot0:LFJ0',
            'robot0:THJ4', 'robot0:THJ3', 'robot0:THJ2', 'robot0:THJ1', 'robot0:THJ0'
        ]

def load_pose():
    data_file = "/remote-home/liuym/data/0721/out_tf_json/frame_1687318023320834343.json"
    
    tree = Kintree(data_file)
    
    tree.forward_kinematic(base_pos=np.zeros(3), 
                            base_frame=Rotation.identity(),
                            node_name="ra_base_link_inertia")

    base_link = tree.nodes['ra_wrist_1_link']
    parent = base_link.parent

    tree.forward_kinematic(base_pos=base_link.value['position'], 
                        base_frame=parent.value['orient'],
                        node_name="rh_wrist")
        
    pose_data = {}
    for name, raw_name in zip(JOINTS_NAME, RAW_JOINTS_NAME):
        if raw_name in tree.nodes.keys():
            node = tree.nodes[raw_name]
            pnode = node.parent
            child_id = pnode.children.index(node)
            
            tf = pnode.transform['orient_quat'][child_id]
        else:
            print(name)
        
        w = tf[-1]
        joint_angle = np.arccos(w) * 2
        pose_data[name] = joint_angle
        
    global_ori = tree.nodes['ra_wrist_1_link'].value['orient'].as_matrix()
    global_trans = tree.nodes['ra_wrist_1_link'].value['position']
    
    return global_ori, global_trans, pose_data
        
        
def vis_cad():
    use_visual_mesh = False
    # os.chdir('./models')
    # print(os.getcwd())
    hand_file = "./mjcf/shadow_hand/shadow_hand_vis.xml" if use_visual_mesh else "./mjcf/shadow_hand_wrist_free.xml"
    hand_model = ShadowHandModel(hand_file,"./mjcf/meshes",device="cpu")
    
    global_ori, global_trans, poses = load_pose()
    
    rot = np.array(global_ori)
    rot = rot[:, :2].T.ravel().tolist()
    trans = list(global_trans)
    pose = [poses[name] for name in ShadowHandModel.joint_names]
    hand_pose = trans + rot + pose
    hand_pose = np.array(hand_pose)
    hand_pose = torch.tensor(hand_pose, dtype=torch.float, device="cpu").unsqueeze(0)
    print(global_ori, global_trans)
    # print(poses)
    print(len(poses), len(JOINTS_NAME))
    
    hand_model.set_parameters(hand_pose)
    hand_mesh = hand_model.get_trimesh_data(0)
    hand_mesh.export("/remote-home/liuym/data/0721/results/frame_1687318023320834343.obj")
    


if __name__ == '__main__':
    vis_cad()