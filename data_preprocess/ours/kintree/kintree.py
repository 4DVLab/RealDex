import json
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Callable
from functools import lru_cache, wraps
import os

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
        self.print_tree(self.root)

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
            self.nodes[child].parent = parent

        for name in self.nodes:
            if self.nodes[name].parent is None:
                self.root.add_child(name)
                self.nodes[name].parent = self.root

    def update_joints(self, tf_data, cname):
        try:
            cnode = self.nodes[cname]
        except (ValueError):
            print(f"Couldn't find a node named {cname}")
        cnode.transform = tf_to_mat(tf_data)
            
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
            out_dict[node_name] = node.value[key] if key in node.value else None
        return out_dict



def tf_to_mat(tf):
    transl = tf[:3]
    rot = Rotation.from_quat(tf[3:])
    mat = np.zeros((4, 4))
    mat[:3, :3] = rot.as_matrix()
    mat[:3, -1] = transl
    mat[-1, -1] = 1
    return mat

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
    

def rearrange_tf(tf_data_dir, tf_info):
    link_list = tf_info['link']
    tf_data_list = []
    for link in link_list:
        pname = link['parent']
        cname = link['child']
        tf_data_list += read_tf_file(os.path.join(tf_data_dir, f"{pname}-{cname}.txt"), cname)
    tf_data_list = sorted(tf_data_list, key=lambda x:x[0])
    tf_data_list = np.array(tf_data_list, dtype=object)
    np.save(os.path.join(tf_data_dir, "tf.npy"),tf_data_list)


def load_sequence(tf_data_dir, tf_info_file):
    data_file = os.path.join(tf_data_dir, "tf.npy")
    data = np.load(data_file, allow_pickle=True)
    ktree = Kintree(tf_info_file)
    out_dict = {}
    for link_data in data:
        time_stamp, tf_data, cname = link_data
        ktree.update_joints(tf_data, cname)
        ktree.forward_kinematic(root=ktree.root)
        global_tf = ktree.output('global_tf')
        out_dict[time_stamp] = global_tf
    out_file = os.path.join(tf_data_dir, "global_tf_all_in_one.npy")
    np.save(out_file, out_dict)
    


if __name__ == "__main__":
    tf_data_dir = "/home/lab4dv/data/bags/test/test_3/TF"
    tf_info_file = "./kintree/srhand_ur.json"

    with open(tf_info_file, 'r') as f:
        tf_info = json.load(f)

    # rearrange_tf(tf_data_dir, tf_info)
    load_sequence(tf_data_dir, tf_info_file)
    # tree = Kintree(data_file)
    # tree.forward_kinematic(base_pos=np.zeros(3), base_frame=Rotation.identity())

    # for name, node in tree.nodes.items():
    #     if len(node.value) > 0:
    #         print(name, ":\t", node.value['position'])


        










    