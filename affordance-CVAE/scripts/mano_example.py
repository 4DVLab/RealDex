import torch
import mano
from mano.utils import Mesh
import numpy as np

model_path = 'PATH_TO_MANO_MODELS'
n_comps = 45
batch_size = 10
rhm_path = '../models/mano/MANO_RIGHT.pkl'

rh_model = mano.load(model_path=rhm_path,
                     is_right= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False,
                     return_verts=True,
                     return_tips=True
                     )

rh_model = mano.load(model_path='../models/mano/MANO_RIGHT.pkl',
                     model_type='mano',
                     num_pca_comps=45,
                     batch_size=batch_size,
                     flat_hand_mean=True)

print(dir(rh_model))
print('face shape', rh_model.faces.shape)
rh_faces = torch.from_numpy(rh_model.faces.astype(np.int32)).view(1, -1, 3)
print('max face idx', torch.max(rh_faces))
print('unique idx', torch.unique(rh_faces).size())

betas = torch.rand(batch_size, 10)*.1
pose = torch.rand(batch_size, n_comps)*.1
global_orient = torch.rand(batch_size, 3)
transl        = torch.rand(batch_size, 3)

output = rh_model(betas=betas,
                  global_orient=global_orient,
                  hand_pose=pose,
                  transl=transl)

print(dir(output))
print(output.vertices.size())
h_meshes = rh_model.hand_meshes(output)
j_meshes = rh_model.joint_meshes(output)
#
# #visualize hand mesh only
h_meshes[0].show()
# print(dir(h_meshes[0]))
# print(h_meshes[0].vertices.shape)
# print(h_meshes[0].edges.shape)
print(output.joints.shape)
#
# #visualize joints mesh only
j_meshes[0].show()
#print(j_meshes[0].vertices.shape)
#
# #visualize hand and joint meshes
hj_meshes = Mesh.concatenate_meshes([h_meshes[0], j_meshes[0]])
hj_meshes.show()
