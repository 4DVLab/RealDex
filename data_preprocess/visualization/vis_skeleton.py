import numpy as np
import os
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

from data_preprocess.ours.utils.kintree import Kintree
import json

def draw_skeleton(joints3D, link_table, ax=None, with_numbers=True):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    pos_mat = np.zeros((len(joints3D), 3))
    for id, j in enumerate(joints3D):
        pos_mat[id, :] = joints3D[j]


    for link in link_table:
        parent, child = link
        if parent not in joints3D or child not in joints3D:
            continue
        j0 = joints3D[parent]
        j1 = joints3D[child]
        ax.plot([j0[0], j1[0]],
                [j0[1], j1[1]],
                [j0[2], j1[2]],
                linestyle='-', linewidth=2, marker='o', markersize=5)
        
    # Fix aspect ratio
    max_range = np.linalg.norm(np.array([pos_mat[:, 0].max()-pos_mat[:, 0].min(), 
                        pos_mat[:, 1].max()-pos_mat[:, 1].min(),
                        pos_mat[:, 2].max()-pos_mat[:, 2].min()]))
    min_p = pos_mat.min(axis=0)
    ax.set_xlim(min_p[0] - max_range/2, min_p[0] + max_range)
    ax.set_ylim(min_p[1]- max_range/2, min_p[1] + max_range)
    ax.set_zlim(min_p[2]- max_range/2, min_p[2] + max_range)
    return ax


def test_skeleton():
    data_file = "./out_json/frame_1687318023320834343.json"
    tree = Kintree(data_file)
    if ONLY_HAND:
        tree.forward_kinematic(base_pos=np.zeros(3), 
                        base_frame=Rotation.identity(),
                        node_name="rh_palm")
    else:
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

def test_animation(ax, time):
    # N = 10
    file_list = list(json.load(open("data_preprocess/assets/file_filter.txt")))
    N = len(file_list)
    # file_list = file_list[:N]
    tree = Kintree(file_list[0])

    def gen_joints():
        tree.forward_kinematic(base_pos=np.zeros(3), 
                        base_frame=Rotation.identity(),
                        node_name="rh_palm")
        joints3D = {}
        for name, node in tree.nodes.items():
            if len(node.value)==0:
                continue
            joints3D[name] = node.value['position']

        return joints3D
    
    def update_joints(num):
        tree.update_joints(file_list[num])
        tree.forward_kinematic(base_pos=np.zeros(3), 
                        base_frame=Rotation.identity(),
                        node_name="rh_palm")
        for name, node in tree.nodes.items():
            if len(node.value)==0:
                continue
            joints3D[name] = node.value['position']


    # prepare for joints
    joints3D = gen_joints()
    skeleton_list = []
    for link in tree.link_table:
        parent, child = link
        if parent not in joints3D or child not in joints3D:
            continue
        j0 = joints3D[parent]
        j1 = joints3D[child]
        skeleton, = ax.plot([j0[0], j1[0]], [j0[1], j1[1]],[j0[2], j1[2]], 
                        linestyle='-', linewidth=2, marker='o', color='b',markersize=5)
        skeleton_list.append(skeleton)

    def init_func():
        pos_mat = np.zeros((len(joints3D.keys()), 3))
        for id, j in enumerate(joints3D):
            pos_mat[id, :] = joints3D[j]

        max_range = np.linalg.norm(np.array([pos_mat[:, 0].max()-pos_mat[:, 0].min(), 
                            pos_mat[:, 1].max()-pos_mat[:, 1].min(),
                            pos_mat[:, 2].max()-pos_mat[:, 2].min()]))
        min_p = pos_mat.min(axis=0)
        
        ax.set_xlim(min_p[0] - max_range/2, min_p[0] + max_range)
        ax.set_ylim(min_p[1]- max_range/2, min_p[1] + max_range)
        ax.set_zlim(min_p[2]- max_range/2, min_p[2] + max_range)

    def update(num):
        print(num, end="\r")
        update_joints(num)
        id = 0
        for link in tree.link_table:
            parent, child = link
            if parent not in joints3D or child not in joints3D:
                continue
            j0 = joints3D[parent]
            j1 = joints3D[child]
            skeleton_list[id].set_data([[j0[0], j1[0]],[j0[1], j1[1]]])
            skeleton_list[id].set_3d_properties([j0[2], j1[2]])
            id += 1
        time.set_text(str(num))

    
    ani = animation.FuncAnimation(fig, update, N, init_func, interval=5000/N, blit=False)
    # plt.show()
    return ani

if __name__ == "__main__":
    ONLY_HAND = True
    fig = plt.figure(frameon=True)
    ax = fig.add_subplot(projection='3d')
    axtext = fig.add_axes([0.0,0.95,0.1,0.05])
    axtext.axis("off")
    time = axtext.text(0.5,0.5, str(0), ha="left", va="top")
    ani = test_animation(ax, time)
    
    video_file = "./data_preprocess/video/skeleton.mp4"
    writervideo = animation.FFMpegWriter(fps=60) 
    ani.save(video_file, writer=writervideo)

    # test_skeleton()

