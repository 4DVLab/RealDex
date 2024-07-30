import os
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
from scipy.spatial.transform import Rotation as R
import json

tf_link_names = [
                # 'WRJ1', 'WRJ0',
                'rh_ffknuckle', 'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal',
                'rh_mfknuckle', 'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal',
                'rh_rfknuckle', 'rh_rfproximal', 'rh_rfmiddle', 'rh_rfdistal',
                'rh_lfmetacarpal', 'rh_lfknuckle', 'rh_lfproximal', 'rh_lfmiddle', 'rh_lfdistal',
                'rh_thbase', 'rh_thproximal', 'rh_thhub', 'rh_thmiddle', 'rh_thdistal',
]

joint_names = [
                # 'WRJ1', 'WRJ0',
                'FFJ3', 'FFJ2', 'FFJ1', 'FFJ0',
                'MFJ3', 'MFJ2', 'MFJ1', 'MFJ0',
                'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0',
                'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0',
                'THJ4', 'THJ3', 'THJ2', 'THJ1', 'THJ0',
                ]

def relative_to_absolute_path(rel_path, abs_prefix):
    parts = rel_path.split('/')
    parts[-1] = parts[-1].replace(".dae", ".obj")
    parts = parts[2:]
    abs_path = os.path.join(abs_prefix, "/".join(parts))
    return abs_path

def init_component(attrib,mesh_origin,abs_prefix):
    #scale:np.array((1x3))
    file_path = relative_to_absolute_path(attrib['filename'],abs_prefix)
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    if 'scale' in attrib.keys():
        scale = np.array([float(item) for item in attrib['scale'].split(' ')]).reshape(1,3).squeeze(0)
        vertices = vertices * scale
        
    if mesh_origin is not None:
        origin_rpy = mesh_origin.attrib['rpy'].split(' ')
        origin_xyz = mesh_origin.attrib['xyz'].split(' ') #translation
        origin_rpy = np.array([float(item) for item in origin_rpy])
        origin_xyz = np.array([float(item) for item in origin_xyz])
        rot = R.from_euler('xyz', origin_rpy, degrees=False)
        rot_mat = rot.as_matrix()
        vertices = vertices @ rot_mat.T + origin_xyz
        
    mesh.vertices = vertices
    return mesh

def load_mesh_from_urdf(urdf_path, name_list:set,abs_path_prefix):
    urdf_tree = ET.parse(urdf_path)
    #names:mesh + position
    root = urdf_tree.getroot()
    links = root.findall('link')
    # link_names = set(link.attrib['name'] for link in links)
    mesh_dict = {}
    for link in links:
        name = link.attrib['name']    
        link_visuals = link.findall('visual')
        for visual in link_visuals:
            if name in name_list and visual is not None:
                geometry = visual.find('geometry')
                mesh_origin = visual.find('origin')
                if geometry is not None:
                    # name = geometry.attrib['name']
                    mesh = geometry.find('mesh')
                    if mesh is None:
                        continue
                    vis_component = init_component(mesh.attrib, mesh_origin,abs_path_prefix)
                    if name in mesh_dict:
                        mesh_dict[name] += vis_component
                    else:
                        mesh_dict[name] = vis_component
                        
    return  mesh_dict #,link_names

def load_parent_link_from_urdf(urdf_path, link_list):
    urdf_tree = ET.parse(urdf_path)
    root = urdf_tree.getroot()
    parent_dict = {}
    for joint in root.findall(".//joint"):
        joint_name = joint.get('name')
        joint_type = joint.get('type')

        parent = joint.find('parent')
        if parent is not None:
            parent_link = parent.get('link')
        else:
            parent_link = None

        child = joint.find('child')
        if child is not None:
            child_link = child.get('link')
        else:
            child_link = None
            
        # print(f"Joint Name: {joint_name}, Joint Type: {joint_type}, Parent Link: {parent_link}, Child Link: {child_link}")
        
        if child_link is not None and parent_link is not None:
            parent_dict[child_link] = parent_link
    return parent_dict

def load_visible_link_from_urdf(urdf_path):
    urdf_tree = ET.parse(urdf_path)
    root = urdf_tree.getroot()
    link_list = []
    for link in root.findall("link"):
        link_visuals = link.findall('visual')
        if len(link_visuals)>0:
            link_list.append(link.attrib['name'])
    return link_list

def load_hand_info_from_urdf(urdf_path):
    hand_info = {}
    urdf_tree = ET.parse(urdf_path)
    root = urdf_tree.getroot()
    for joint in root.findall('joint'):
        axis = joint.find('axis')
        child = joint.find('child')
        origin = joint.find('origin')
        if axis is not None:
            joint_name = joint.get('name')
            ori_rpy = origin.get('rpy')
            ori_xyz = origin.get('xyz')
            
            if joint_name.startswith('rh_'):
                print(joint_name, axis.get('xyz'), child.get('link'))
                hand_info[child.get('link')] = {'joint': joint_name, 
                                                'axis':axis.get('xyz'),
                                                'ori_rpy': ori_rpy,
                                                'ori_xyz': ori_xyz}
    return hand_info
                

def load_arm_and_hand_info_from_urdf(urdf_path):
    arm_and_hand_info = {}
    urdf_tree = ET.parse(urdf_path)
    root = urdf_tree.getroot()
    for joint in root.findall('joint'):
        axis = joint.find('axis')
        child = joint.find('child')
        origin = joint.find('origin')
        if axis is not None:
            joint_name = joint.get('name')
            ori_rpy = origin.get('rpy')
            ori_xyz = origin.get('xyz')

            if joint_name.startswith('rh_') or joint_name.startswith('ra_'):
                print(joint_name, axis.get('xyz'), child.get('link'))
                arm_and_hand_info[child.get('link')] = {'joint': joint_name, 
                                                'axis':axis.get('xyz'),
                                                'ori_rpy': ori_rpy,
                                                'ori_xyz': ori_xyz}
    return arm_and_hand_info
    
if __name__ == '__main__':
    urdf_path = "./assets/bimanual_srhand_ur.urdf"
    urdf_tree = ET.parse(urdf_path)
    node_list = load_visible_link_from_urdf(urdf_path)
    parent_dict = load_parent_link_from_urdf(urdf_path, node_list)

    out_dict = {'node_names': [], 'link': []}
    for node in node_list:
        prefix = node.split('_')[0]
        if prefix == "la" or prefix == "lh":
            continue
        if node in parent_dict and parent_dict[node] not in node_list:
            out_dict['node_names'].append(parent_dict[node])
        out_dict['node_names'].append(node)

    for node in out_dict['node_names']:
        if node in parent_dict:
            out_dict['link'].append({'parent': parent_dict[node], 'child': node})

    # for hand only  
    #out_dict['hand_info'] = load_hand_info_from_urdf(urdf_path)       
            
    # for arm and hand together
    out_dict["hand_info"] = load_arm_and_hand_info_from_urdf(urdf_path)
    json_string = json.dumps(out_dict, indent=4)

    hand_out_path = "./assets/srhand_ur.json"
    arm_and_hand_out_path = "./assets/sr_arm_hand_ur.json"
    with open(arm_and_hand_out_path, 'w') as outfile:
        outfile.write(json_string)
    print(json_string)
    