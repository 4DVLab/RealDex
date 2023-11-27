import torch
import torch.nn.functional as F
import json
import trimesh
from models.shadow_hand_builder import ShadowHandBuilder
from pytorch3d.transforms import Transform3d, matrix_to_euler_angles, matrix_to_axis_angle
tf_link_names = [
                # 'WRJ1', 'WRJ0',
                'rh_ffknuckle', 'rh_ffproximal', 'rh_ffmiddle', 'rh_ffdistal',
                'rh_mfknuckle', 'rh_mfproximal', 'rh_mfmiddle', 'rh_mfdistal',
                'rh_rfknuckle', 'rh_rfproximal', 'rh_rfmiddle', 'rh_rfdistal',
                'rh_lfmetacarpal', 'rh_lfknuckle', 'rh_lfproximal', 'rh_lfmiddle', 'rh_lfdistal',
                'rh_thbase', 'rh_thproximal', 'rh_thhub', 'rh_thmiddle', 'rh_thdistal',
]

# tf_link_names = [
#                 # 'WRJ1', 'WRJ0',
#                 'rh_ffknuckle', 'rh_ffmiddle', 'rh_ffproximal', 'rh_ffdistal',
#                 'rh_mfknuckle', 'rh_mfmiddle', 'rh_mfproximal',  'rh_mfdistal',
#                 'rh_rfknuckle', 'rh_rfmiddle', 'rh_rfproximal',  'rh_rfdistal',
#                 'rh_lfmetacarpal', 'rh_lfknuckle', 'rh_lfmiddle', 'rh_lfproximal', 'rh_lfdistal',
#                 'rh_thbase', 'rh_thproximal', 'rh_thhub', 'rh_thmiddle','rh_thdistal',
# ]

joint_names = [
                # 'WRJ1', 'WRJ0',
                'FFJ3', 'FFJ2', 'FFJ1', 'FFJ0',
                'MFJ3', 'MFJ2', 'MFJ1', 'MFJ0',
                'RFJ3', 'RFJ2', 'RFJ1', 'RFJ0',
                'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'LFJ0',
                'THJ4', 'THJ3', 'THJ2', 'THJ1', 'THJ0',
                ]
joint_names = ["robot0:" + name for name in joint_names]

def str_to_tensor(string):
    data = [float(num_str) for num_str in string.split()]
    data = torch.tensor(data)
    return data

def load_wrist_data(global_path, palm_correction_path):
    with open(global_path, "r") as file:
        # Load the JSON data
        json_data = json.load(file)
        
    with open(palm_correction_path, "r") as file:
        # Load the JSON data
        correct_tf = json.load(file)['rh_palm']
        correct_tf = torch.tensor(correct_tf)
        
    # rh_wrist = torch.tensor(json_data['rh_wrist'])
    rh_wrist = torch.tensor(json_data['rh_palm'])
    rh_wrist = Transform3d(matrix=rh_wrist.T)
    correct_tf = Transform3d(matrix=correct_tf.T)
    rh_wrist = rh_wrist.compose(correct_tf)
    rot_mat = rh_wrist.get_matrix()
    rot_mat = rot_mat.squeeze().T
    print(rot_mat)
    return rot_mat

def load_tf_data(tf_path, sr_xml_tree):
    root = sr_xml_tree.getroot()
    elements = root.findall(".//joint")
    joints_dict = {}
    for element in elements:
        element_tag = element.tag
        element_attrib = element.attrib
        if 'name' in element_attrib:
            name = element_attrib['name']
            joints_dict[name] = element_attrib
    print(joints_dict)

    with open(tf_path, "r") as file:
        # Load the JSON data
        json_data = json.load(file)
    
    qpos = []
    for i, key in enumerate(tf_link_names):
        transf_mat = torch.tensor(json_data[key])
        transf_euler = matrix_to_euler_angles(transf_mat[:3,:3], convention=["X", "Y", "Z"])
        transf_axis_angle = matrix_to_axis_angle(transf_mat[:3,:3])
        print(transf_axis_angle.shape)
        transf_angle = transf_axis_angle.norm()
        transf_axis = F.normalize(transf_axis_angle, dim=0)
        
        real_axis = joints_dict[joint_names[i]]['axis']
        real_axis = str_to_tensor(real_axis)
        angle_range = joints_dict[joint_names[i]]['range']
        angle_range = str_to_tensor(angle_range)
        # if transf_axis.dot(real_axis) < 0:
        #     transf_angle *= -1
        # transf_angle = transf_axis_angle.dot(real_axis)
        
        if key == 'rh_thbase':
            transf_angle = transf_euler[-1]
            
        if key in ['rh_lfmiddle', 'rh_rfmiddle', 'rh_mfmiddle', 'rh_ffmiddle']:
            transf_angle = angle_range.max() / 1
        # if key in ['rh_lfdistal','rh_rfdistal', 'rh_mfdistal', 'rh_ffdistal']:
        #     transf_angle *= 10
        if key in ['rh_rfproximal', 'rh_mfproximal', 'rh_ffproximal']:
            transf_angle = angle_range.max() / 3
        if key == 'rh_thdistal':
            transf_angle = -transf_euler[1]
        if key in ['rh_lfknuckle', 'rh_rfknuckle', 'rh_mfknuckle', 'rh_thmiddle', 'rh_lfproximal']:
            transf_angle *= -1
        if key in ['rh_lfmetacarpal']:
            transf_angle = angle_range.max()
        
            
        print(key, transf_euler, transf_angle, transf_axis, real_axis)
        
            
        qpos.append(transf_angle)
    qpos = torch.tensor(qpos)
    return qpos

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sr_builder = ShadowHandBuilder(device=device,
                                   mjcf_path="assets/mjcf/shadow_hand_vis.xml"
                                )
    tf_file_path = '/remote-home/share/yumeng/our_data/single_frame/TF/all_links_tf.json'
    global_tf_path = '/remote-home/share/yumeng/our_data/single_frame/global_postion/1015.json'
    palm_correction_path = '/remote-home/share/yumeng/our_data/single_frame/global_postion/palm_pose.json'
    
    
    # qpos = load_tf_data(tf_file_path, sr_builder.sr_xml_tree).cuda()
    qpos = torch.rand(22)
    qpos = qpos.cuda()
    rh_wrist = load_wrist_data(global_tf_path, palm_correction_path)
    
    rotation_mat = rh_wrist[:3, :3].cuda()
    transl = rh_wrist[:3, -1].cuda()
    ret_dict = sr_builder.get_hand_model(rotation_mat, transl, qpos, without_arm=False)
    
    points = torch.concat(ret_dict['sampled_pts']).reshape(-1, 3)
    print(points.shape)
    
    meshes = trimesh.Trimesh(vertices=ret_dict['meshes'].verts_packed().cpu(), faces=ret_dict['meshes'].faces_packed().cpu())
    points = trimesh.PointCloud(vertices=points.cpu())
    meshes.export("./test.ply")
    points.export("./test_pts.ply")
    
    out_path = '/remote-home/share/yumeng/our_data/single_frame/init_pose.pt'
    init_pose = {'global_rotation': rotation_mat.cpu(), 'global_transl': transl.cpu(), 'qpos': qpos.cpu()}
    torch.save(init_pose, out_path)