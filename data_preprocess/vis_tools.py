import matplotlib.pyplot as plt
import numpy as np

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
    
    # ax.set_xlim(-1, 0)
    # ax.set_ylim(-1, 0)
    # ax.set_zlim(-1, 0)

    max_range = np.array([pos_mat[:, 0].max()-pos_mat[:, 0].min(), 
                          pos_mat[:, 1].max()-pos_mat[:, 1].min(),
                          pos_mat[:, 2].max()-pos_mat[:, 2].min()]).max()
    min_p = pos_mat.min(axis=0)
    ax.set_xlim(min_p[0], min_p[0] + max_range)
    ax.set_ylim(min_p[1], min_p[1] + max_range)
    ax.set_zlim(min_p[2], min_p[2] + max_range)
    return ax
    # # For each 24 joint
    # for i in range(1, kintree_table.shape[1]):
    #     j1 = kintree_table[0][i]
    #     j2 = kintree_table[1][i]
    #     ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
    #             [joints3D[j1, 1], joints3D[j2, 1]],
    #             [joints3D[j1, 2], joints3D[j2, 2]],
    #             linestyle='-', linewidth=2, marker='o', markersize=5)
    #     if with_numbers:
    #         ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    # return ax


