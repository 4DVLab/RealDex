
import pytorch3d.io
from pytorch3d.transforms import Rotate, Translate, Transform3d
from pytorch3d.structures import Meshes
import kornia
import torch
import pickle
from functools import lru_cache
from pathlib import Path
import numpy as np
import open3d as o3d
import os
import numpy as np
from copy import deepcopy
import trimesh


# Full batch mode
# for every point, there is crosspondent ray,and for every ray, it cals for every triangle in the obj,
# very time consuming
def batch_mesh_contains_points(
    ray_origins,  # hand_vert
    obj_triangles,
    direction=torch.Tensor(
        [0.4395064455, 0.617598629942, 0.652231566745]).cuda(),
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
    # ray_origins.requires_grad = False
    # obj_triangles.requires_grad = False
    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :,
                               0], obj_triangles[:, :, 1], obj_triangles[:, :, 2]
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(
        1, 1, 3).expand(batch_size, triangle_nb, 3)

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
    not_parallel = parallel.logical_not()
    final_inter = v_correct * u_correct * not_parallel * t_pos
    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mes
    exterior = final_intersections.sum(2) % 2 == 0
    count = torch.sum(exterior < 1).item()

    return exterior


def masked_mean_loss(dists, mask):
    mask = mask.float()
    valid_vals = mask.sum()
    if valid_vals > 0:
        loss = (mask * dists).sum() / valid_vals
    else:
        loss = torch.Tensor([0]).cuda()
    return loss


# look like it calculate the distance between each point in x and each point in y
def batch_pairwise_dist(x, y, use_cuda=True):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    # print(y.size())
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    if use_cuda:
        dtype = torch.cuda.LongTensor
    else:
        dtype = torch.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = (
        xx[:, diag_ind_x, diag_ind_x]  # [barch,]
        .unsqueeze(1)
        .expand_as(zz.transpose(2, 1))
    )
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P


def batch_index_select(inp, dim, index):
    views = [inp.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(inp.shape))
    ]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


@lru_cache(maxsize=128)
def load_contacts(save_contact_paths="assets/contact_zones.pkl", display=False):
    with open(save_contact_paths, "rb") as p_f:
        contact_data = pickle.load(p_f)
    hand_verts = contact_data["verts"]
    return hand_verts, contact_data["contact_zones"]


def compute_contact_single(
        hand_verts_pt,
        obj_verts_pt,
        obj_faces
):
    hand = trimesh.PointCloud(hand_verts_pt)
    mesh = trimesh.Trimesh(vertices=obj_verts_pt, faces=obj_faces)
    # (mesh+hand).show()
    interior = mesh.ray.contains_points(hand_verts_pt)
    interior = torch.tensor(interior)
    index = torch.where(interior == True)
    print(index)


def compute_contact_loss(  # you have to remember, all the input is tensor with batchsize
    hand_verts_pt,
    obj_verts_pt,
    obj_faces,
    contact_thresh=5,  # maybe the value is too big!!! you have to change it
    contact_mode="dist_sq",
    collision_thresh=10,
    collision_mode="dist_sq",
    contact_target="all",
    contact_sym=False,
    contact_zones="all",
):
    # obj_verts_pt = obj_verts_pt.detach()
    # hand_verts_pt = hand_verts_pt.detach()
    # look like it calculate the distance between each point in hand and each point in obj euclidean distance

    dists = batch_pairwise_dist(hand_verts_pt, obj_verts_pt)
    # dists [b,x,y]
    # the dim means the dim you want to reduce

    mins12, min12idxs = torch.min(dists, 1)
    # the min dis between a point in hand with all points in obj
    mins21, min21idxs = torch.min(dists, 2)

    # Get obj triangle positions
    obj_triangles = obj_verts_pt[:, obj_faces]  # 按照面片组合在一起[b,face_num,3,3]

    exterior = batch_mesh_contains_points(  # 返回的是一个mask，表示在外部的点就是1
        hand_verts_pt.detach(), obj_triangles.detach()
    )

    penetr_mask = ~exterior
    results_close = batch_index_select(obj_verts_pt, 1, min21idxs)

    if contact_target == "all":  # calculate the vector between the hand and obj with the closet distances
        anchor_dists = torch.norm(results_close - hand_verts_pt, 2, 2)
    elif contact_target == "obj":
        anchor_dists = torch.norm(results_close - hand_verts_pt.detach(), 2, 2)
    elif contact_target == "hand":
        anchor_dists = torch.norm(results_close.detach() - hand_verts_pt, 2, 2)
    else:
        raise ValueError(
            "contact_target {} not in [all|obj|hand]".format(contact_target)
        )
    # choose one of the three mode to calculate the loss,the last two won't  cal the BP

    if contact_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            contact_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            contact_vals = ((results_close - hand_verts_pt.detach()) ** 2).sum(
                2
            )
        elif contact_target == "hand":
            contact_vals = ((results_close.detach() - hand_verts_pt) ** 2).sum(
                2
            )
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
        below_dist = mins21 < (contact_thresh ** 2)
    elif contact_mode == "dist":
        # Use distance to penalize contact
        contact_vals = anchor_dists
        below_dist = mins21 < contact_thresh
    elif contact_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        contact_vals = contact_thresh * torch.tanh(
            anchor_dists / contact_thresh
        )
        # All points are taken into account
        below_dist = torch.ones_like(mins21).byte()
    else:
        raise ValueError(
            "contact_mode {} not in [dist_sq|dist|dist_tanh]".format(
                contact_mode
            )
        )

    if collision_mode == "dist_sq":
        # Use squared distances to penalize contact
        if contact_target == "all":
            collision_vals = ((results_close - hand_verts_pt) ** 2).sum(2)
        elif contact_target == "obj":
            collision_vals = (
                (results_close - hand_verts_pt.detach()) ** 2
            ).sum(2)
        elif contact_target == "hand":
            collision_vals = (
                (results_close.detach() - hand_verts_pt) ** 2
            ).sum(2)
        else:
            raise ValueError(
                "contact_target {} not in [all|obj|hand]".format(
                    contact_target
                )
            )
    elif collision_mode == "dist":
        # Use distance to penalize collision
        collision_vals = anchor_dists
    elif collision_mode == "dist_tanh":
        # Use thresh * (dist / thresh) distances to penalize contact
        # (max derivative is 1 at 0)
        collision_vals = collision_thresh * torch.tanh(
            anchor_dists / collision_thresh
        )
    else:
        raise ValueError(
            "collision_mode {} not in "
            "[dist_sq|dist|dist_tanh]".format(collision_mode)
        )

    # begin to calculate the attr loss

    missed_mask = below_dist & exterior

    if contact_zones == "tips":
        tip_idxs = [745, 317, 444, 556, 673]  # 只给出了手指的五个点
        tips = torch.zeros_like(missed_mask)
        tips[:, tip_idxs] = 1
        missed_mask = missed_mask & tips
    elif contact_zones == "zones":
        _, contact_zones = load_contacts(
            "assets/contact_zones.pkl"
        )
        contact_matching = torch.zeros_like(missed_mask)
        for zone_idx, zone_idxs in contact_zones.items():
            min_zone_vals, min_zone_idxs = mins21[:, zone_idxs].min(1)
            cont_idxs = mins12.new(zone_idxs)[min_zone_idxs]
            # For each batch keep the closest point from the contact zone
            contact_matching[
                [torch.range(0, len(cont_idxs) - 1).long(), cont_idxs.long()]
            ] = 1
        missed_mask = missed_mask & contact_matching
    elif contact_zones == "all":
        missed_mask = missed_mask
    else:
        raise ValueError(
            "contact_zones {} not in [tips|zones|all]".format(contact_zones)
        )

    # Apply losses with correct mask
    # attr loss,just a scalar, it has sum all the loss together
    missed_loss = masked_mean_loss(contact_vals, missed_mask)
    penetr_loss = masked_mean_loss(collision_vals, penetr_mask)  # penetr loss
    if contact_sym:
        obj2hand_dists = torch.sqrt(mins12)
        sym_below_dist = mins12 < contact_thresh
        sym_loss = masked_mean_loss(obj2hand_dists, sym_below_dist)
        missed_loss = missed_loss + sym_loss
    # print('penetr_nb: {}'.format(penetr_mask.sum()))
    # print('missed_nb: {}'.format(missed_mask.sum()))

    max_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).max(1)[0].mean()
    )
    mean_penetr_depth = (
        (anchor_dists.detach() * penetr_mask.float()).mean(1).mean()
    )
    contact_info = {
        "attraction_masks": missed_mask,
        "repulsion_masks": penetr_mask,
        "contact_points": results_close,
        "min_dists": mins21,
    }
    metrics = {
        "max_penetr": max_penetr_depth,
        "mean_penetr": mean_penetr_depth,
    }
    return missed_loss, penetr_loss, contact_info, metrics


def transform_matrix_to_euler_angles_and_translation(matrix):
    # Convert rotation matrix to angle-axis
    quaternion = kornia.geometry.conversions.rotation_matrix_to_quaternion(
        matrix[:3, :3])

    # euler_angles = kornia.geometry.conversions.euler_from_quaternion(quaternion)
    euler_angles = kornia.geometry.conversions.euler_from_quaternion(
        quaternion[0], quaternion[1], quaternion[2], quaternion[3])

    # euler_angles = torch.Tensor(euler_angles).cuda()
    euler_angles_tensor = torch.stack(euler_angles, dim=-1)
    translation_tensor = matrix[:3, 3]
    euler_and_translation = torch.stack(
        (euler_angles_tensor, translation_tensor), dim=0).flatten()
    return euler_and_translation


def euler_angle_and_translation_to_transform_matrix(angle_and_translation):
    euler_angles = angle_and_translation[:3]
    translation = angle_and_translation[3:]
    quaternion = kornia.geometry.conversions.quaternion_from_euler(
        euler_angles[0], euler_angles[1], euler_angles[2])
    quaternion = torch.stack(quaternion, dim=-1)
    rotation_matrix = kornia.geometry.conversions.quaternion_to_rotation_matrix(
        quaternion)
    transform_matrix = torch.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation
    return transform_matrix


def optimize_object(angle_and_translation):  # all
    # 设备设置 (如果有CUDA)
    target_device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    folder = "/media/tony/T7/camera_data/test_object_position_optimize"
    folder = Path(folder)

    object_mesh_path = folder / \
        Path("input_model/1000times_simplifybanana.obj")
    hand_obj_path = folder / \
        Path("input_model/1000times_simplifyonly_hand_1015.obj")
    obj_mesh = pytorch3d.io.load_objs_as_meshes(
        [object_mesh_path], device=target_device)

    # hand_obj_IO = pytorch3d.io.IO()
    # hand_obj_data = hand_obj_IO.load_mesh(
    #     path=hand_obj_path, include_textures=False, device=target_device)
    hand_data = pytorch3d.io.load_obj(hand_obj_path)
    print(hand_data)
    hand_vert = hand_data[0].unsqueeze(0).to(target_device)

    # hand_vert and the obj_vert has to have the bachsize dim to use the obman func
    obj_verts = obj_mesh.verts_packed().to(target_device).unsqueeze(0)
    obj_faces = obj_mesh.faces_packed().to(target_device)

    angle_and_translation = angle_and_translation.to(
        target_device).requires_grad_()

    optimizer = torch.optim.Adam([angle_and_translation], lr=1e-2)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    camera_param_path = folder / Path("camera_params.json")

    original_hand_mesh_path = folder / "viz_model/1000times_only_hand_1015.ply"
    original_hand_mesh = o3d.io.read_triangle_mesh(
        str(original_hand_mesh_path))
    original_hand_mesh.compute_vertex_normals()
    original_obj_mesh_path = folder / "viz_model/1000times_banana.obj"
    original_obj_mesh = o3d.io.read_triangle_mesh(str(original_obj_mesh_path))
    original_obj_mesh.compute_vertex_normals()
    camera_params = o3d.io.read_pinhole_camera_parameters(
        str(camera_param_path))

    for iteration in np.arange(2000):
        optimizer.zero_grad()
        transform_matrix = euler_angle_and_translation_to_transform_matrix(
            angle_and_translation).to(target_device)
        print(transform_matrix)

        trans = Transform3d(
            matrix=transform_matrix.T, device=target_device)

        transformed_verts = trans.transform_points(obj_verts)
        # transformed_verts = transformed_verts_0
        transfomed_mesh = o3d.geometry.TriangleMesh()
        transfomed_mesh.vertices = o3d.utility.Vector3dVector(
            transformed_verts[0].detach().cpu())
        transfomed_mesh.triangles = o3d.utility.Vector3iVector(
            obj_faces.detach().cpu())
        transfomed_mesh.compute_vertex_normals()

        hand_vis = o3d.geometry.PointCloud()
        hand_vis.points = o3d.utility.Vector3dVector(
            hand_vert[0].detach().cpu())

        if iteration != 0:
            vis.clear_geometries()
        vis.add_geometry(original_hand_mesh)
        # vis.add_geometry((deepcopy(original_obj_mesh)).transform(
        #     transform_matrix.detach().cpu().numpy()))
        vis.add_geometry((deepcopy(transfomed_mesh)))
        vis.add_geometry(hand_vis)

        vis.get_view_control().convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        # with torch.no_grad():
        #     hand_verts = hand_vert[0].detach().cpu().numpy()
        #     obj_verts = transformed_verts[0].detach().cpu().numpy()
        #     obj_faces = obj_faces.detach().cpu().numpy()
        #     compute_contact_single(hand_verts, obj_verts, obj_faces)

        contact_loss, collision_loss, contact_info, metrics = compute_contact_loss(
            hand_vert, transformed_verts, obj_faces)

        loss = 0.6 * contact_loss + 0.8 * collision_loss

        loss.backward()

        optimizer.step()

        print(f"Iteration {iteration}: Loss {loss.item()}")
    vis.destroy_window()


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model_position_transform = torch.tensor(
        [
            [-0.591167, -0.479865, -0.648268, 1.473468],
            [0.458711, -0.861142, 0.219134, 0.010604],
            [-0.663405, -0.167823, 0.729198, 0.738428],
            [0.000000, 0.000000, 0.000000, 1.000000]], dtype=torch.float32)
    # 全部单位都改成mm
    model_position_transform[:3, 3] = model_position_transform[:3, 3] * 1000
    angle_and_translation = transform_matrix_to_euler_angles_and_translation(
        model_position_transform)
    optimize_object(angle_and_translation)


# import open3d as o3d
# import numpy as np

# # Let's assume original_obj_mesh is an instance of open3d.geometry.TriangleMesh
# # and transform_matrix is a NumPy array or a torch tensor with the transformation matrix.

# # Apply transform to the vertices
# transformed_vertices = np.dot(np.asarray(original_obj_mesh.vertices), transform_matrix[:3, :3].T) + transform_matrix[:3, 3]

# # Create a new mesh with the transformed vertices
# transformed_mesh = o3d.geometry.TriangleMesh()
# transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
# transformed_mesh.triangles = original_obj_mesh.triangles
# transformed_mesh.triangle_normals = original_obj_mesh.triangle_normals

# # Now, add the transformed_mesh to your visualizer
# vis.add_geometry(transformed_mesh)
