import json
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Callable
from functools import lru_cache, wraps
import os

def np_cache(function):
    @lru_cache(maxsize=None)
    def cached_wrapper(*args, **kwargs):

        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


class Node(object):
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = [ ]
        self.transform = None # transform relative to its parent node
        self.value = dict()

    def print_tree(self, level=0):
        print("\t"*level+repr(self.name)+"\n")
        if len(self.children)==0:
            return "\n"
        for child in self.children:
            child.print_tree(level+1)
        return "\n"
    
    def __repr__(self):
        return '<tree node representation>'
    
    def set_tf(self, tf):
        self.transform = tf

    def add_link(self, child):
        self.children.append(child)
        

class Kintree(object):
    def __init__(self, data_file):
        self.nodes = dict()
        self.link_table = []
        self._load_data(data_file)


        # optional, added for test
        self.arm_root = self.nodes["ra_base_link_inertia"]
        self.hand_root = self.nodes["rh_forearm"]
        self.arm_root.print_tree()
        self.hand_root.print_tree()

    def set_value(self, name, value):
        assert name in self.nodes
        self.value[name] = value

    def _load_data(self, file):
        with open(file) as f:
            kintree_data = json.load(f)['transforms'] # a list for each node
        for link_data in kintree_data:
            parent=link_data['header']['frame_id']
            child = link_data['child_frame_id']
            self.link_table.append((parent, child))
            if parent not in self.nodes:
                self.nodes[parent] = Node(parent)
            if child not in self.nodes:
                self.nodes[child] = Node(child)
            
            tf = link_data['transform']
            self.nodes[parent].add_link(self.nodes[child], tf)
            self.nodes[child].parent = self.nodes[parent]

    def update_joints(self, file):
        with open(file) as f:
            kintree_data = json.load(f)['transforms'] # a list for each node
        for link_data in kintree_data:
            
            tf = link_data['transform']
            parent=link_data['header']['frame_id']
            child = link_data['child_frame_id']

            pnode = self.nodes[parent]
            child_id = pnode.children.index(self.nodes[child])
            pnode.set_tf(child_id, tf)

    # def dfs_position_update(self, global_name_postion,name,time_slot):
    #     if name in TF_tree.keys():#叶子节点不会在keys里面
    #         for child_name in TF_tree[name].keys():
    #             child_time_and_transform = TF_tree[name][child_name]#有些TF只有一个，是static TF
    #             time_index = find_time_closet(time_slot,child_time_and_transform[:,0])
    #             senven_num_transform = child_time_and_transform[time_index,1:]
    #             child_transform = seven_num2matrix(translation=senven_num_transform[:3],roatation=senven_num_transform[3:])
    #             child_transform_position = np.dot(global_name_postion[name],child_transform)
    #             global_name_postion[child_name] = child_transform_position
    #             dfs_position_update(TF_tree,global_name_postion,child_name,time_slot)
            
    # @lru_cache(maxsize=None)
    # def forward_kinematic(self, node_name='rh_palm'):
    #     node = self.nodes[node_name]
    #     while len(node.children) > 0:
    @np_cache
    def forward_kinematic(self, base_pos, base_frame, node_name='rh_forearm'):
        node:Node = self.nodes[node_name]
        node.value['position'] = base_pos
        node.value['orient'] = base_frame
        # base_frame = Rotation.from_quat(base_ori_quat) # orthogonal frame
        
        for cid, child in enumerate(node.children):
            # LocalFrameA @ coord_A = LocalFrameB @ coord_B
            # LocalFrameB = rel_rot @ LocalFrameA
            # LocalFrameA @ coord_A = LocalFrameA ( LocalFrameA^-1 @ rel_rot @ LocalFrameA ) coord_B
            # coord_A = ( LocalFrameA^-1 @ rel_rot @ LocalFrameA ) coord_B
            rel_rot = Rotation.from_quat(node.transform['orient_quat'][cid])
            child_coord_local = node.transform['transl'][cid]
            child_coord_global = base_pos + (base_frame.inv() * rel_rot * base_frame).as_matrix() @ child_coord_local
            child_ori_global = rel_rot * base_frame
            self.forward_kinematic(base_pos=child_coord_global, 
                                base_frame=child_ori_global, 
                                node_name=child.name)
            del rel_rot, child_coord_local, child_coord_global, child_ori_global

def tf_to_mat(tf):
    transl = tf[:3]
    rot = Rotation.from_quat(tf[3:])
    mat = np.zeros(4, 4)
    mat[:3, :3] = rot
    mat[:, 3] = transl
    mat[3, 3] = 1
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


if __name__ == "__main__":
    tf_data_dir = "/home/lab4dv/data/bags/test/test_3/TF"
    tf_info_file = "./kintree/srhand_ur.json"

    with open(tf_info_file, 'r') as f:
        tf_info = json.load(f)

    rearrange_tf(tf_data_dir, tf_info)
    # tree = Kintree(data_file)
    # tree.forward_kinematic(base_pos=np.zeros(3), base_frame=Rotation.identity())

    # for name, node in tree.nodes.items():
    #     if len(node.value) > 0:
    #         print(name, ":\t", node.value['position'])


        










    