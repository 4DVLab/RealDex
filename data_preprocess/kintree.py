import json
import numpy as np
from scipy.spatial.transform import Rotation


class Node(object):
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.children = [ ]
        self.transform = dict()
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

    def add_link(self, child, tf):
        self.children.append(child)
        transl = [tf['translation'][d] for d in ["x", "y", "z"]]
        orient_quat = [tf['rotation'][d] for d in ["x", "y", "z", "w"]]
        self.transform.setdefault("transl", []).append(np.asarray(transl))
        self.transform.setdefault("orient_quat", []).append(np.asarray(orient_quat))
        

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
        



if __name__ == "__main__":
    data_file = "./out_json/frame_1687318023320834343.json"
    tree = Kintree(data_file)
    tree.forward_kinematic(base_pos=np.zeros(3), base_frame=Rotation.identity())

    for name, node in tree.nodes.items():
        if len(node.value) > 0:
            print(name, ":\t", node.value['position'])


        










    