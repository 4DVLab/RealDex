# code from GraspTTA

import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_gather, knn_points
import numpy as np
import time


def check_ray_triangle_intersection(ray_origins, ray_direction, faces_verts, face_normals, epsilon=1e-6):
    """
    code borrowed from: https://github.com/sibowi/mesh_utils/blob/main/src/mesh_to_voxels.py
    
    Parameters
    ----------
    ray_origin : torch.Tensor, (n_rays, n_dimensions),
    ray_directions : torch.Tensor, (n_rays, n_dimensions),
    faces : torch.Tensor, (n_batches * n_faces, 3, 3), faces should be packed
    face_normals : torch.Tensor, (n_batches * n_faces, 3), normals should be packed
    

    Return
    ------
    intersection : boolean (n_rays,)
    """
    n_rays, _ = ray_origins.shape
    n_faces,_ = faces_verts.shape
    v0 = faces_verts[:, 0] # [n_faces, 3]
    E1 = faces_verts[:, 1] - v0 #vector of edge 1 on triangle 
    E2 = faces_verts[:, 2] - v0 #vector of edge 2 on triangle
    
    # Expand needed vectors
    batch_direction = ray_direction.view(1, n_rays, 3).expand(n_faces, -1, -1)

    # inverse determinant
    invdet = 1. / -torch.einsum('nji, ni -> nj', batch_direction, face_normals) # [n_faces, n_rays]

    A0 = batch_direction - v0[:, None, :] # [n_faces, n_rays, 3]
    # print('A0.shape: ', A0.shape)
    # print('ray_direction.shape: ', ray_direction.shape)
    DA0 = torch.cross(A0, batch_direction) # [n_faces, n_rays, 3]

    u = torch.einsum('nji, ni -> nj', DA0, E2) * invdet #[n_faces, n_rays]
    v = -torch.einsum('nji, ni -> nj', DA0, E1) * invdet 
    t = torch.einsum('nji, ni -> nj', A0, face_normals) * invdet

    intersection = (t >= 0.0) * (u >= 0.0) * (v >= 0.0) * ((u + v) <= 1.0) #[n_faces, n_rays]
    # barycentric_coords = torch.stack([u, v, 1-u-v], dim=-1) #[n_faces, n_rays, 3]

    return intersection


def get_NN(src_xyz, trg_xyz, k=1):
    '''
    :param src_xyz: [B, N1, 3]
    :param trg_xyz: [B, N2, 3]
    :return: nn_dists, nn_dix: all [B, N1] tensor for NN distance and index in N2
    '''
    if len(src_xyz.shape) == 2:
        src_xyz = src_xyz.unsqueeze(0)
        trg_xyz = trg_xyz.unsqueeze(0)
    B = src_xyz.size(0)
    src_lengths = torch.full(
        (src_xyz.shape[0],), src_xyz.shape[1], dtype=torch.int64, device=src_xyz.device
    )  # [B], N for each num
    trg_lengths = torch.full(
        (trg_xyz.shape[0],), trg_xyz.shape[1], dtype=torch.int64, device=trg_xyz.device
    )
    src_nn = knn_points(src_xyz, trg_xyz, lengths1=src_lengths, lengths2=trg_lengths, K=k)  # [dists, idx]
    nn_dists = src_nn.dists[..., 0]
    nn_idx = src_nn.idx[..., 0]
    return nn_dists, nn_idx


def get_interior(src_face_normal, src_xyz, trg_xyz, trg_NN_idx):
    '''
    :param src_face_normal: [B, 778, 3], surface normal of every vert in the source mesh
    :param src_xyz: [B, 778, 3], source mesh vertices xyz
    :param trg_xyz: [B, 3000, 3], target mesh vertices xyz
    :param trg_NN_idx: [B, 3000], index of NN in source vertices from target vertices
    :return: interior [B, 3000], inter-penetrated trg vertices as 1, instead 0 (bool)
    '''
    N1, N2 = src_xyz.size(1), trg_xyz.size(1)

    # get vector from trg xyz to NN in src, should be a [B, 3000, 3] vector
    NN_src_xyz = batched_index_select(src_xyz, trg_NN_idx)  # [B, 3000, 3]
    NN_vector = NN_src_xyz - trg_xyz  # [B, 3000, 3]

    # get surface normal of NN src xyz for every trg xyz, should be a [B, 3000, 3] vector
    NN_src_normal = batched_index_select(src_face_normal, trg_NN_idx)

    interior = (NN_vector * NN_src_normal).sum(dim=-1) > 0  # interior as true, exterior as false
    return interior

def batched_index_select(input, index, dim=1):
    '''
    :param input: [B, N1, *]
    :param dim: the dim to be selected
    :param index: [B, N2]
    :return: [B, N2, *] selected result
    '''
    views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim=dim, index=index)

def get_faces_xyz(faces_idx, xyz):
    '''
    :param faces_idx: [B, N1, 3]. N1 is number of faces (1538 for MANO), index of face vertices in N2
    :param xyz: [B, N2, 3]. N2 is number of points.
    :return: faces_xyz: [B, N1, 3, 3] faces vertices coordinate
    '''
    B, N1, D = faces_idx.size()
    N2 = xyz.size(1)
    xyz_replicated = xyz.cpu().unsqueeze(1).repeat(1,N1,1,1)  # use cpu to save CUDA memory
    faces_idx_replicated = faces_idx.unsqueeze(-1).repeat(1,1,1,D).type(torch.LongTensor)
    return torch.gather(xyz_replicated, dim=2, index=faces_idx_replicated).to(faces_idx.device)

def batch_mesh_contains_points(
    ray_origins, # point cloud as origin of rays
    obj_triangles,
    direction=torch.Tensor([0.4395064455, 0.617598629942, 0.652231566745]),
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh
    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    tol_thresh: To determine if ray and triangle are //
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    direction = direction.to(ray_origins.device)
    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2)
    dets = torch.bmm(
        v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1)
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    )
    pvec = pvec.repeat(1, point_nb, 1)
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    try:
        not_parallel = 1 - parallel
    except:
        not_parallel = parallel==False
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior

def inter_penetr_loss(hand_xyz, hand_face, obj_xyz, nn_dist, nn_idx):
    '''
    get penetrate object xyz and the distance to its NN
    :param hand_xyz: [B, 778, 3]
    :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
    :param obj_xyz: [B, 3000, 3]
    :return: inter penetration loss
    '''
    B = hand_xyz.size(0)
    mesh = Meshes(verts=hand_xyz, faces=hand_face)
    hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

    # if not nn_dist:
    #     nn_dist, nn_idx = utils_loss.get_NN(obj_xyz, hand_xyz)
    interior = get_interior(hand_normal, hand_xyz, obj_xyz, nn_idx).type(torch.bool)  # True for interior
    penetr_dist = nn_dist[interior].sum() / B  # batch reduction
    return 100.0 * penetr_dist

def min_distance_from_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    m_num = m.shape[-2]
    n_num = n.shape[-2]

    # m_: [..., M, N, 3]
    # n_: [..., M, N, 3]
    m_ = m.unsqueeze(-2)  # [..., M, 1, 3]
    n_ = n.unsqueeze(-3)  # [..., 1, N, 3]

    m_ = torch.repeat_interleave(m_, n_num, dim=-2)  # [..., M, N, 3]
    n_ = torch.repeat_interleave(n_, m_num, dim=-3)  # [..., M, N, 3]

    # [..., M, N]
    pairwise_dis = torch.sqrt(((m_ - n_) ** 2).sum(dim=-1))

    ret_dis = torch.min(pairwise_dis, dim=-1)[0]  # [..., M]

    return ret_dis


def soft_distance(distance):
    """
    :param distance: [..., M]
    :return: [..., M]
    """
    sigmoid = torch.nn.Sigmoid()
    normalize_factor = 60  # decided by visualization
    return 1 - 2 * (sigmoid(normalize_factor * distance) - 0.5)


def contact_map_of_m_to_n(m, n):
    """
    :param m: [..., M, 3]
    :param n: [..., N, 3]
    :return: [..., M]
    """
    distances = min_distance_from_m_to_n(m, n)  # [..., M]
    distances = soft_distance(distances)
    return distances