import torch
import numpy as np
import mano
from pytorch3d.structures import Meshes
from utils import utils_loss, utils_vis
import time

'''load predicted mano parma, obj xyz'''
B = 3  # just for repeating on batch dim to test batch operation
param_path = 'C:\\Users\\think\\server\\baseline_mano_cmap_ft_v2p\\7_param.npy'
#pc_path = 'C:\\Users\\think\\server\\baseline_mano_v2p2\\0.npy'
recon_param = np.load(param_path)  # [61]
recon_param = torch.tensor(recon_param).view(1, -1).repeat(B, 1)
obj = np.load(param_path.replace('_param.npy', '_obj.npy')).T

'''load mano model, get mano faces'''
with torch.no_grad():
    rh_mano = mano.load(model_path='../models/mano/MANO_RIGHT.pkl',
                        model_type='mano',
                        use_pca=True,
                        num_pca_comps=45,
                        batch_size=B,
                        flat_hand_mean=True)
rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).contiguous() # [1, 1538, 3], face triangle indexes

'''repeat on batch dim'''
obj = torch.tensor(obj).unsqueeze(0).repeat(B, 1, 1) # [N, 3000, 3]
rh_faces = rh_faces.repeat(B, 1, 1) # [N, 1538, 3]

'''get hand mesh using mano'''
hand_mesh = rh_mano(betas=recon_param[:, :10],
               global_orient=recon_param[:, 10:13],
               hand_pose=recon_param[:, 13:58],
               transl=recon_param[:, 58:])
vertices = hand_mesh.vertices # [N, 778, 3]

'''get hand mesh using pytorh3d mesh, as well as normal'''
mesh = Meshes(verts=vertices, faces=rh_faces)
hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)

'''visualize hand xyz and hand normals'''
# utils_vis.plt_plot_pc(vertices[0])
# utils_vis.plt_plot_normal(vertices[0], hand_normal[0])

t0 = time.time()
'''calculate exterior using normal inner product'''
nn_dist, nn_idx = utils_loss.get_NN(obj, vertices) # [B, 3000]
print(nn_idx.min(), nn_idx.max())
interior = utils_loss.get_interior(hand_normal, vertices, obj, nn_idx)
exterior1 = ~interior

penetr_dist = nn_dist[interior].sum()
print(penetr_dist)

t1 = time.time()
'''calculate exterior using ray intersection'''
batch_triangles = utils_loss.get_faces_xyz(rh_faces, vertices)
exterior2 = utils_loss.batch_mesh_contains_points(obj, batch_triangles)

t2 = time.time()

print('method 1 used {} ms, method 2 used {} ms'.format(
    int(round(t1 * 1000)) - int(round(t0 * 1000)), int(round(t2 * 1000)) - int(round(t1 * 1000))
))

'''show exterior'''
for i in range(1):
    utils_vis.show_pointcloud_objhand(vertices[i].detach().numpy(), obj[i].detach().numpy())
    utils_vis.show_exterior(exterior1[i].detach().numpy(), obj[i].detach().numpy())
    utils_vis.show_exterior(exterior2[i].detach().numpy(), obj[i].detach().numpy())

