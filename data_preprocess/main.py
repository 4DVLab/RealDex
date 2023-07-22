import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from vis_tools import draw_skeleton
from kintree import Kintree


data_file = "./out_json/frame_1687318023320834343.json"
tree = Kintree(data_file)
tree.forward_kinematic(base_pos=np.zeros(3), 
                       base_frame=Rotation.identity(),
                       node_name="ra_base_link_inertia")

base_link = tree.nodes['ra_wrist_1_link']
parent = base_link.parent

tree.forward_kinematic(base_pos=base_link.value['position'], 
                       base_frame=parent.value['orient'],
                       node_name="rh_wrist")


joints3D = {}
for name, node in tree.nodes.items():
    if len(node.value)==0:
        continue
    joints3D[name] = node.value['position']

fig = plt.figure(frameon=False)
ax = fig.add_subplot(111, projection='3d')

draw_skeleton(joints3D, tree.link_table, ax)

plt.show()

    
