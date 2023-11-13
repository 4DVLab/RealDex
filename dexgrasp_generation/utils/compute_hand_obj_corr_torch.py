import numpy as np
import os, glob, pickle, argparse
from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import IO
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.transforms import Transform3d, quaternion_apply, axis_angle_to_matrix,quaternion_invert
import torch.nn.functional as F
import trimesh
from tqdm import tqdm, trange
import random
from copy import deepcopy

seed_value = 0
random.seed(seed_value)

def parse_np_to_torch(npdata, device, sample_verts=1000):
    # load data
    #
    # rhand_verts: 778*3
    # rhand_global_orient: 3
    # rhand_hand_pose: 24
    # object_verts: V*3
    # object_vn: V*3
    # object_global_orient: 3
    # object_transl: 3
    # object_id: 1
    # frame_id: 1
    # is_left: 1
    #
    # np_dtype = '(2334)f4, (3)f4, (24)f4, ({})f4, ({})f4, (3)f4, (3)f4, i4, i4, ?'.format(args.num_points*3, args.num_points*3)

    t_data = {}
    t_data['rhand_verts'] = npdata[:]['f0']
    t_data['rhand_global_orient'] = npdata[:]['f1']
    t_data['rhand_hand_pose'] = npdata[:]['f2']
    t_data['full_object_verts'] = npdata[:]['f3']
    t_data['full_object_vn'] = npdata[:]['f4']
    t_data['object_global_orient'] = npdata[:]['f5']
    t_data['object_transl'] = npdata[:]['f6']
    
    for key in t_data:
        narray = np.array(t_data[key])
        t_data[key] = torch.tensor(narray).to(device)
        
    B = t_data['rhand_verts'].shape[0]
    
    t_data['rhand_verts'] = t_data['rhand_verts'].reshape(B, -1, 3) # [B, 778, 3]
    t_data['full_object_verts'] = t_data['full_object_verts'].reshape(B, -1, 3) # [B, V, 3]
    t_data['full_object_vn'] = t_data['full_object_vn'].reshape(B, -1, 3) # [B, V, 3]
    
    V = t_data['full_object_verts'].shape[1]
    obj_sample_index = list(range(V))
    random.shuffle(obj_sample_index)
    obj_sample_index = obj_sample_index[:sample_verts]
    
    # V = 3
    t_data['object_verts'] = t_data['full_object_verts'][:, obj_sample_index, :]
    t_data['object_vn'] = t_data['full_object_vn'][:, obj_sample_index, :]
    
    t_data['object_id'] = npdata[:]['f7'].tolist()
    t_data['frame_id'] = npdata[:]['f8'].tolist()
    t_data['is_left'] = npdata[:]['f9'].tolist()
    
    return t_data
    
def get_mano_closed_faces(th_faces):
    """
    The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
    which looks much better.
    https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py
    """
    close_faces = torch.Tensor([
        [92, 38, 122],
        [234, 92, 122],
        [239, 234, 122],
        [279, 239, 122],
        [215, 279, 122],
        [215, 122, 118],
        [215, 118, 117],
        [215, 117, 119],
        [215, 119, 120],
        [215, 120, 108],
        [215, 108, 79],
        [215, 79, 78],
        [215, 78, 121],
        [214, 215, 121],
    ])
    th_closed_faces = torch.cat([th_faces.clone(), close_faces.long()])
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]
    return th_closed_faces  # , hand_ignore_faces

def get_object_mesh(d, id2objmesh):
    object_id, object_rot, object_transl = d['f7'], d['f5'], d['f6']
    is_left = d['f9']
    object_mesh = trimesh.load_mesh(id2objmesh[object_id], process=False)
    object_mesh.vertices = np.dot(object_mesh.vertices, R.from_rotvec(object_rot).as_matrix()) + object_transl.reshape(1, 3) 
    if is_left:
        object_mesh.vertices[..., 0] = -object_mesh.vertices[..., 0]
        object_mesh.faces = object_mesh.faces[..., [2, 1, 0]]    
    return object_mesh    

def get_transformed_obj_mesh(data, meshes):
    obj_id = data['object_id']
    obj_rot = data['object_global_orient']
    obj_transl = data['object_transl']
    obj_meshes = meshes[obj_id]
    obj_tf = Transform3d().rotate(axis_angle_to_matrix(obj_rot)).translate(obj_transl).to(obj_meshes.device)
    
    faces_padded = obj_meshes.faces_padded()
    verts_padded = obj_tf.transform_points(obj_meshes.verts_padded())
    verts_normals_padded = obj_tf.transform_normals(obj_meshes.verts_normals_padded())
    
    del obj_meshes
    transformed_meshes = Meshes(verts=verts_padded, faces=faces_padded, verts_normals=verts_normals_padded)
    
    return transformed_meshes

def rotate_2_vectors(vector_a, vector_b):
    """
    vector_a, vector_b: [B, V, 3]
    rotate vector_a to vector_b
    """
    unit_vector_a = F.normalize(vector_a, p=2, dim=-1)
    unit_vector_b = F.normalize(vector_b, p=2, dim=-1)

    rotation_axis = torch.cross(unit_vector_a, unit_vector_b, dim=-1)
    rotation_axis = F.normalize(rotation_axis, p=2, dim=-1)

    cos_theta = torch.einsum('bij,bij->bi',unit_vector_a, unit_vector_b)
    theta = torch.acos(cos_theta).unsqueeze(-1)
    
    w = torch.cos(theta / 2)
    xyz = rotation_axis * torch.sin(theta / 2)
    quaternion = torch.cat((w, xyz),dim=-1)
    
    return quaternion

def view_mesh_from_obj(obj_verts, obj_vn, mesh_verts):
    B, V, _ = obj_verts.shape
    device = obj_verts.device
    
    # use the point on obj as viewport to rasterize mesh
    z_dir = torch.tensor([0,0,1], dtype=torch.float32, device=device).reshape(1,1,3).expand([B, V, -1])
    quat = rotate_2_vectors(obj_vn, z_dir)
    
    rotation_center = obj_verts.view(B, -1, 1, 3) # in each batch, there are V viewports
    
    new_mesh_verts = quaternion_apply(quat.view(B,-1, 1, 4), mesh_verts - rotation_center) #[B, V, 778, 3]
    
    new_mesh_verts = new_mesh_verts.contiguous().reshape(B*V, -1, 3)
    return new_mesh_verts

def batched_ray_tracer(obj_verts, obj_vn, rhand_verts, mano_faces, mano_temp_verts):
    B, V, _ = obj_verts.shape
    device = obj_verts.device
    rhand_verts = rhand_verts.view(B, 1, -1, 3)
    mano_temp_verts = mano_temp_verts.expand(B, -1, -1)
    mano_temp_verts = mano_temp_verts.view(B, 1, -1, 3)
    
    new_rhand_verts = view_mesh_from_obj(obj_verts, obj_vn, rhand_verts)
    rhand_faces = mano_faces.expand([B*V, -1, -1])
    
    rhand_mesh = Meshes(verts=new_rhand_verts, faces=rhand_faces)
    pix_to_face, _, barycentric_coords, _ = rasterize_meshes(rhand_mesh, image_size=1, cull_backfaces=False)
    # check if the obj verts is inside of hand mesh
    hit_num = torch.sum((pix_to_face[:, 0, 0, :] > -1).int(), dim=-1)
    verts_in_hand = (hit_num % 2 == 1).cpu()
    
    # modify norm if obj verts is within hand mesh
    obj_vn_new = obj_vn.contiguous().reshape(B*V, 3)
    if torch.sum(verts_in_hand.int()) > 0:
        obj_vn_new[verts_in_hand] = -obj_vn_new[verts_in_hand]
        obj_vn_new = obj_vn_new.contiguous().reshape(B, V, 3)
        new_rhand_verts = view_mesh_from_obj(obj_verts, obj_vn_new, rhand_verts)
        rhand_mesh = Meshes(verts=new_rhand_verts, faces=rhand_faces)
        pix_to_face, _, barycentric_coords, _ = rasterize_meshes(rhand_mesh, image_size=1, cull_backfaces=False)
        
    
    mask = torch.any(pix_to_face > -1, dim=-1)
    corr_index = mask.nonzero() # obj_id that has a corr on hand
    
    
    #only need the first hit position
    face_idx = [pix_to_face[ind[0]][ind[1]][ind[2]][0] for ind in corr_index.tolist()]
    faces = rhand_mesh.faces_packed()[face_idx, :]
    barycentric_coords = barycentric_coords[mask][..., 0, :]
    
    mano_temp_verts = mano_temp_verts.expand(-1, V, -1, -1).contiguous().reshape(B*V, -1, 3)
    rhand_mesh = rhand_mesh.update_padded(mano_temp_verts)
    temp_f_verts = rhand_mesh.verts_packed()[faces] 
    
    rhand_verts = rhand_verts.expand(-1, V, -1, -1).contiguous().reshape(B*V, -1, 3)
    rhand_mesh = rhand_mesh.update_padded(rhand_verts)
    f_verts = rhand_mesh.verts_packed()[faces]
    
    
    temp_location = torch.einsum('bk, bkv -> bv',barycentric_coords, temp_f_verts)
    location = torch.einsum('bk, bkv -> bv',barycentric_coords, f_verts)
    
    # obj_verts = obj_verts.contiguous().reshape(B*V, 3)[corr_index[:, 0]]
    # obj_vn = obj_vn.contiguous().reshape(B*V, 3)[corr_index[:, 0]]
    # quat = quat.contiguous().reshape(B*V, 4)[corr_index[:, 0]]
    # z_dir_recon = quaternion_apply(quat, obj_vn)
    # print(z_dir_recon)
    # print(F.normalize(location - obj_verts))
    # print(F.normalize(obj_vn))
    
    corr_index = corr_index[:, 0]
    
    return verts_in_hand, corr_index, temp_location, location, obj_vn_new

def ray_distance_point_cloud(ray_origin, ray_dir, points_cloud, threshold=0.9):
    # O - - > A (ray)
    #  \ 
    #   \
    #     P
    # points_cloud: [K, V, 3]
    # ray_origin: [K, 3]
    OP = points_cloud - ray_origin[:, None, :] #[K, V, 3]
    ray_dir =  F.normalize(ray_dir,dim=-1) #[K, 3]
    
    dist = torch.einsum('kj,kij -> ki', ray_dir, OP) #[K,V]
    cos_AOP = dist / OP.norm(p=2, dim=-1)
    mask = cos_AOP > threshold #[K,V]
    dist[~mask] = float("Inf") #[K,V]
    ray_mask = torch.any(mask, dim=-1) #[K]
    
    if torch.all(~ray_mask):
        min_dist = None
    else:
        dist = dist[ray_mask] # [K', V]
        min_dist, _ = torch.min(dist, dim=-1) # K'
        # print("concave check, min dist to self: ", min_dist)
    return min_dist, ray_mask
    
    

def compute_corr(data, mano_faces, mano_temp_verts):        
    B = data['rhand_verts'].shape[0]
    
    rhand_verts = data['rhand_verts']# [B, 778, 3]
    
    obj_verts = data['object_verts'] # [B, V, 3]
    obj_vn = data['object_vn']
    
    verts_in_hand, corr_index, temp_location, location, obj_vn_new = batched_ray_tracer(obj_verts, obj_vn, rhand_verts, mano_faces, mano_temp_verts)
    corr_index = corr_index.cpu()
    # check concave hit
    B, V, _ = obj_verts.shape
    device = obj_verts.device
    obj_verts_cor = obj_verts.contiguous().reshape(B*V, 3)[corr_index]
    obj_vn_cor = obj_vn_new.contiguous().reshape(B*V, 3)[corr_index]
    
    """ get toch info """
    """ corresponding points on hand """
    # corr_pts = torch.sparse_coo_tensor(corr_index, temp_location, (B*V, 3))
    corr_pts = torch.zeros(B*V, 3).float()
    corr_pts[corr_index, :] = temp_location.cpu()
    """ mask to indicate which vertex on object has a correspondance on hand """
    corr_mask = torch.zeros(B*V).bool()
    corr_mask[corr_index] = True
    # mask_value = torch.ones(corr_index.shape).bool().to(device)
    # corr_mask = torch.sparse_coo_tensor(corr_index, mask_value, (B*V,))
    """ distance between the correspondance from object to hand """
    corr_dist = torch.zeros(B*V).float()
    dist_value = (location - obj_verts_cor).norm(p=2, dim=-1)
    corr_dist[corr_index] = dist_value.cpu()
    # corr_dist = torch.sparse_coo_tensor(corr_index, dist_value, (B*V,))
    
    
    
    
    obj_verts_cor = obj_verts_cor + 1e-4*obj_vn_cor #[k, 3]
    
    full_obj_verts = data['full_object_verts']
    obj_mesh_verts = full_obj_verts[corr_index//V, :, :] #[k, V, 3]
    dist_to_self, ray_mask = ray_distance_point_cloud(ray_origin=obj_verts_cor,
                                    ray_dir=obj_vn_cor,
                                    points_cloud=obj_mesh_verts)
    ray_mask = ray_mask.cpu()
    if dist_to_self is not None:
        check_concave_index = corr_index[ray_mask]
        dist_to_hand = corr_dist[check_concave_index]
        
        dist_to_self = dist_to_self.cpu()
        concave_index = check_concave_index[dist_to_self < dist_to_hand]
        print('{}/{} concave hits found!'.format((dist_to_self < dist_to_hand).sum(), corr_index.shape[0]))
        concave_index = concave_index.cpu()
        corr_mask[concave_index] = False
        corr_dist[concave_index] = 0
        corr_pts[concave_index, :] *= 0
        
    corr_dist[verts_in_hand] = -corr_dist[verts_in_hand]
    
    
    # save data
    # rhand_verts: [B, 778, 3]
    # rhand_global_orient: [B, 3]
    # rhand_hand_pose: [B, 24]
    # object_verts: [B, V, 3]
    # object_vn: [B, V, 3]
    # object_global_orient: [B, 3]
    # object_transl: [B, 3]
    # object_id: [B,]
    # frame_id: [B,]
    # obj_corr_mask: [B, V]
    # obj_corr_dist: [B, V]
    # obj_corr_pts: [B, V, 3]
    # is_left: [B,]
    out_data = {}
    out_data['obj_corr_mask'] = corr_mask.reshape(B, V)
    out_data['obj_corr_dist'] = corr_dist.reshape(B, V)
    out_data['obj_corr_pts'] = corr_pts.reshape(B, V, 3)
    
    return out_data

def dict_to_cpu(data):
    for key in data:
        if isinstance(data[key], torch.Tensor) and data[key].device.type != 'cpu':
            data[key] = data[key].cpu()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grab_path', type=str)
    # preprocessed sequences
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--mano_path', type=str)
    parser.add_argument('--object', default='all', type=str)
    args = parser.parse_args()

    device=torch.device("cuda")

    # load MANO model
    mano_path = os.path.join(args.mano_path, 'MANO_RIGHT.pkl')
    with open(mano_path, 'rb') as f:
        mano_model = pickle.load(f, encoding='latin1')

    # normalize MANO template
    mano_template = trimesh.Trimesh(vertices=mano_model['v_template'], faces=mano_model['f'],
        process=False)
    with open('data/grab/scale_center.pkl', 'rb') as f:
        scale, center = pickle.load(f)
    mano_template.vertices = mano_template.vertices * scale + center
    mano_temp_verts = torch.from_numpy(mano_template.vertices).float().to(device)
    
    mano_faces = torch.from_numpy(mano_template.faces)
    mano_faces = get_mano_closed_faces(mano_faces).unsqueeze(0).to(device) #[1, 778, 3]

    obj_meshes = sorted(os.listdir(os.path.join(args.grab_path, 
        'tools/object_meshes/contact_meshes')))
    
    meshes = []
    for fn in obj_meshes:
        mesh_file = os.path.join(args.grab_path, 'tools/object_meshes/contact_meshes', fn)
        mesh = IO().load_mesh(path=mesh_file, device=device)
        meshes.append(mesh)
    meshes = join_meshes_as_batch(meshes)
    

    out_dir = "/home/liuym/results/toch-my-implem/" + args.object
    
    for split in ['train', 'train_pert', 'val', 'val_pert', 'test', 'test_pert']:
        out_path = os.path.join(out_dir, split)
        if os.path.exists(out_path) is False:
            os.makedirs(out_path)
            
        clips = glob.glob(os.path.join(args.data_path, split, '*.npy'))
        clips = sorted(clips)
        num_clips = len(clips)
        print('Start processing {} split: {} clips found!'.format(split, num_clips))
        for id, c in enumerate(tqdm(clips)):
            npdata = np.load(c)
            data = parse_np_to_torch(npdata, device=device)
            frame_len = len(data['frame_id'])
            step = 10
            processed_data = {'obj_corr_mask': [],
                              'obj_corr_dist': [],
                              'obj_corr_pts': []}
            
            for start in trange(0, frame_len, step):
                end = start + step
                end = frame_len if end > frame_len else end
                clipped_data = {}
                for key in data:
                    clipped_data[key] = data[key][start:end]
                outdata = compute_corr(clipped_data, mano_faces, mano_temp_verts)
                for key in outdata:
                    processed_data[key].append(outdata[key])
            
            
            
            data = dict_to_cpu(data)
            for key in processed_data:
                data[key] = torch.cat(processed_data[key],dim=0)
                
            clip_name = c.split('/')[-1].split('.')[0]
            torch.save(data, os.path.join(out_path, f"{clip_name}.pt"))
            
            
                
