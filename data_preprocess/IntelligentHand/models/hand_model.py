import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
from scipy.spatial.transform import Rotation as R
import json
from utils.urdf_util import load_mesh_from_urdf


if __name__ == '__main__':
    urdf_path = "../../data_process/bimanual_srhand_ur.urdf"
    # prefix = "/home/lab4dv/yumeng/ShadowHand"
    prefix ="/Users/yumeng/Working/data/ShadowHand/description/"
    struct_file = "./assets/srhand_ur.json"
    # tf_data_dir = "/home/lab4dv/data/bags/test/backup/test_1/TF"
    tf_data_dir = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207/TF"
    # out_path = "/home/lab4dv/yumeng/results/srhand_ur_meshes/test_1"
    out_path = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207/srhand_ur_meshes/"
    
    os.makedirs(out_path, exist_ok=True)
    
    with open(struct_file, 'r') as f:
        # Load the JSON data
        sr_struct = json.load(f)
        
    link_list = sr_struct['node_names']
    mesh_dict = load_mesh_from_urdf(urdf_path,link_list, prefix)
    print(mesh_dict.keys())

    tf_data_file = os.path.join(tf_data_dir, "global_tf_all_in_one.npy")
    tf_data_all_in_one = np.load(tf_data_file, allow_pickle=True)
    tf_data_all_in_one = tf_data_all_in_one.item()
    
    
    for time in tf_data_all_in_one:
        updated_meshes = {}
        tf_data = tf_data_all_in_one[time]
        for key in list(mesh_dict.keys()):
            tf = tf_data[key]
            # print(tf)
            mesh = mesh_dict[key]
            verts = mesh.vertices
            verts = np.concatenate([verts, np.ones((verts.shape[0], 1))], axis=1)
            # print(verts.shape)
            verts = verts @ tf.T
            
            new_mesh = trimesh.Trimesh(vertices=verts[:, :3], faces=mesh.faces)
            updated_meshes[key] = new_mesh
            

        combined_mesh = trimesh.util.concatenate(updated_meshes.values())
        combined_mesh.export(os.path.join(out_path, f"{time}.ply"))