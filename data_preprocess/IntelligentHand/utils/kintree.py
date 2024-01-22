import json
import numpy as np
from typing import Callable
from functools import lru_cache, wraps
import os
from utils.global_util import tf_to_mat, check_rotation_axis, compute_joint_angle
from utils.global_util import rotate_axis, rpy_to_mat
import sys

class Node(object):
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = [ ]
        self.transform = np.eye(4, 4) # transform relative to its parent node
        self.value = dict()
    
    def __repr__(self):
        return '<tree node representation>'
    
    def set_tf(self, tf):
        self.transform = tf

    def set_value(self, key, value):
        self.value[key] = value
    
    def get_value(self, key):
        return self.value[key]

    def add_child(self, child):
        self.children.append(child)
        

class Kintree(object):
    def __init__(self, info_file):
        self.nodes = dict()
        self.root = Node('world')
        
        with open(info_file, 'r') as f:
            self.info = json.load(f)
        self._load_data()
        
        # optional, added for test
        self.arm_root = self.root
        self.hand_root = self.nodes["rh_forearm"]
        # self.print_tree(self.root)

    def print_tree(self, root, level=0):
        print("\t"*level+repr(root.name)+"\n")
        if len(root.children)==0:
            return "\n"
        for child in root.children:
            self.print_tree(self.nodes[child], level+1)
        return "\n"

    def _load_data(self):
        for node_name in self.info['node_names']:
            self.nodes[node_name] = Node(node_name)
        for link in self.info['link']:
            parent = link['parent']
            child = link['child']
            if parent == "world":
                self.nodes[child].parent = None
                continue
            self.nodes[parent].add_child(child)
            self.nodes[child].parent = self.nodes[parent]

        for name in self.nodes:
            if self.nodes[name].parent is None:
                self.root.add_child(name)
                self.nodes[name].parent = self.root
                
    def get_hand_info(self, name, key):
        hand_info = self.info['hand_info']
        value = hand_info[name][key].split()
        value = [float(number) for number in value]
        value = np.array(value)
        return value
                
    def compute_angle(self, tf_mat, cname):
        axis = self.get_hand_info(cname, 'axis')
        rpy = self.get_hand_info(cname, 'ori_rpy')
        xyz = self.get_hand_info(cname, "ori_xyz")
        rot_mat = rpy_to_mat(rpy)
        
        local_tf_mat = np.linalg.inv(rot_mat) @ tf_mat[:3, :3]

        if check_rotation_axis(local_tf_mat, axis):
            # print(f"good node, with {cname}\n")
            angle = compute_joint_angle(local_tf_mat, axis)
            return angle  
        else:
            sys.stderr.write(f"The rotation axis of the given tf matrix is inconsistent with the axis in the URDF file, with node {cname}.\n" )
            sys.exit(1)  

    def update_joints(self, tf_data, cname):
        try:
            cnode = self.nodes[cname]
        except (ValueError):
            print(f"Couldn't find a node named {cname}")
        mat = tf_to_mat(tf_data)
        cnode.transform = mat
        self.forward_kinematic(root=self.root)
        
        if cname in self.info['hand_info']:
            # print(cname, cnode.parent.name)
            angle = self.compute_angle(mat, cname)
            cnode.set_value('joint_angle', angle)
            
    def forward_kinematic(self, root, base_tf=None):
        if base_tf is None:
            base_tf = np.eye(4, 4)
        
        for child in root.children:
            node = self.nodes[child]
            local_tf = node.transform
            global_tf = base_tf @ local_tf
            node.set_value('global_tf', global_tf)
            self.forward_kinematic(root=node, base_tf=node.value['global_tf'])

    def output(self, key):
        out_dict = {}
        for node_name in self.nodes:
            node:Node = self.nodes[node_name]
            if key in node.value:
                out_dict[node_name] = node.value[key]
        return out_dict


def read_tf_file(file, cname):
    with open(file, 'r') as f:
        lines = f.readlines()
    tf_data_list = []
    for line in lines:
        line = line.strip().split()
        time_stamp = int(float(line[0]))
        tf_data = np.array(line[1:], dtype=float) # transl, quat(x,y,z,w)
        
        tf_data_list.append((time_stamp, tf_data, cname))
    return tf_data_list
    

def rearrange_hand_tf(tf_data_dir, tf_info):
    link_list = tf_info['link']
    tf_data_list = []
    for link in link_list:
        pname = link['parent']
        cname = link['child']
        tf_data_list += read_tf_file(os.path.join(tf_data_dir, f"{pname}-{cname}.txt"), cname)
    tf_data_list = sorted(tf_data_list, key=lambda x:x[0])
    tf_data_list = np.array(tf_data_list, dtype=object)
    # np.save(os.path.join(tf_data_dir, "tf.npy"),tf_data_list)
    return tf_data_list

def load_sequence(tf_data_dir, tf_info_file):
    
    with open(tf_info_file, 'r') as f:
        tf_info = json.load(f)
    
    data = rearrange_hand_tf(tf_data_dir, tf_info)
    ktree = Kintree(tf_info_file)
    out_dict = {'global_tf': {}, 'joint_angle': {}}
    for link_data in data:
        time_stamp, tf_data, cname = link_data
        ktree.update_joints(tf_data, cname)
        global_tf = ktree.output('global_tf')
        joint_angle = ktree.output('joint_angle')
        out_dict['global_tf'][time_stamp] = global_tf 
        out_dict['joint_angle'][time_stamp] = joint_angle 
    # out_file = os.path.join(tf_data_dir, "global_tf_all_in_one.npy")
    # np.save(out_file, out_dict)
    return out_dict


if __name__ == "__main__":
    # tf_data_dir = "/home/lab4dv/data/bags/test/backup/test_1/TF"
    tf_data_dir = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207/TF"
    
    tf_info_file = "./assets/srhand_ur.json"

    with open(tf_info_file, 'r') as f:
        tf_info = json.load(f)

    rearrange_hand_tf(tf_data_dir, tf_info)
    load_sequence(tf_data_dir, tf_info_file)


        










    