import numpy as np
import torch
from PIL import Image
import cv2
import os
import open3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def tensor_to_cv2(img_tensor):
    '''
    :param img_tensor: [1,3,H,W]
    :return: numpy cv2 img [H,W,3]
    '''
    x = img_tensor[0]
    x = np.ascontiguousarray(255 * x.permute(1, 2, 0).numpy(), dtype=np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def vertices_reprojection(vertices, k):
    '''
    get reproject coordinate of 3D vertices on 2D image plane
    :param vertices: [N,3]
    :param k: camera intrinsic, [3,3]
    :return: [N, 2], point-wise reprojected coordinate on 2D image plane
    '''
    p = np.matmul(k, vertices)
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T

def visualize_contour(img, vertex, intrinsics):
    '''
    visualize the project of 3D vertices on 2D image plane, and get its contour
    :param img: [H, W, 3]
    :param vertex: [N, 3]
    :param intrinsics: [3, 3]
    '''
    H, W, _ = img.shape
    maskImg = np.zeros((H, W), np.uint8)
    vp = vertices_reprojection(vertex, intrinsics)
    for p in vp:
        if p[0] != p[0] or p[1] != p[1]:  # check nan
            continue
        maskImg = cv2.circle(maskImg, (int(p[0]), int(p[1])), 1, 1, -1)  # radius=1, color=1, thickness=-1
    kernel = np.ones((5, 5), np.uint8)
    maskImg = cv2.morphologyEx(maskImg, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(maskImg, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1]
    contourImg = cv2.drawContours(img, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
    return contourImg

def vis_reconstructed_hand_vertex(idx, recon, img_tensor, save_root, hand_xyz, obj_pc, recon_param, p=True):
    '''
    :param idx:
    :param recon: reconstructed hand xyz, [3, 778] or [778, 3]
    :param img_tensor: [H, W, 3]
    :param save_root:
    :param hand_xyz: gt hand xyz, [3, 778] or [778, 3]
    :param obj_pc: [4, 3000], [:3, :] are xyz, [3, :] is obj scale
    :param p: permutation status of hand xyz -> (778,3) as false and (3, 778) as true
    :param recon_param: reconstructed mano parameters, [61]
    '''
    img_tensor = img_tensor.cpu()
    obj_pc = obj_pc[:3, :].cpu().numpy()
    recon = recon.cpu()
    hand_xyz = hand_xyz.cpu()
    if not p:
        recon = torch.tensor(recon).permute(1,0).numpy()
        hand_xyz = torch.tensor(hand_xyz).permute(1,0).numpy() # [3,788]
    img = tensor_to_cv2(img_tensor) # [3,N]
    img = np.array(img, dtype=np.uint8)
    save_name = os.path.join(save_root, str(idx) + '.jpg')
    intrinsics_obman = np.array([[480., 0., 128.],
                                 [0., 480., 128.],
                                 [0., 0., 1.]]).astype(np.float32)
    contour_img = visualize_contour(img, recon, intrinsics_obman)
    #cv2.imwrite(save_name, contour_img)
    np.save(save_name.replace('jpg', 'npy'), np.hstack((hand_xyz, recon)))
    np.save(save_name.replace('.jpg', '_obj.npy'), obj_pc)
    np.save(save_name.replace('.jpg', '_param.npy'), recon_param.cpu().numpy())

def show_pointcloud_hand(preds):
    '''
    Draw ground truth and predicted hand xyz, gt in red, pred in blue
    :param preds: hand xyz, [3, 778*2], [:, :778] is ground truth, [:, 778:] is prediction, all in [3, 778]
    '''
    c_gt, c_recon = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_gt = np.repeat(c_gt, repeats=778, axis=0)
    c_recon = np.repeat(c_recon, repeats=778, axis=0) # [778, 3]
    c_all = np.vstack((c_gt, c_recon)) # [778*2, 3]

    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(preds)
    pc.colors = open3d.Vector3dVector(c_all)
    open3d.draw_geometries([pc])

def show_pointcloud_fingertips(preds):
    '''
    Draw all (red) and prior (blue) hand xyz, prior is finger or fingertip
    :param preds: hand xyz, [3, 778*2], [:, :778] is ground truth, [:, 778:] is prediction, can be in different size
    '''
    N, D = preds.shape
    c_gt, c_recon = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_gt = np.repeat(c_gt, repeats=778, axis=0)
    c_recon = np.repeat(c_recon, repeats=N-778, axis=0) # [778, 3]
    c_all = np.vstack((c_gt, c_recon))  # [N, 3]

    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(preds)
    pc.colors = open3d.Vector3dVector(c_all)
    open3d.draw_geometries([pc])

def show_pointcloud_objhand(hand, obj):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [3000, 3]
    '''
    handObj = np.vstack((hand, obj))
    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=778, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=3000, axis=0) # [3000,3]
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(handObj)
    pc.colors = open3d.Vector3dVector(c_hanObj)
    open3d.draw_geometries([pc])

def show_dist_objhand(handObj, dist):
    '''
    Draw hand and obj xyz at the same time
    :param hand: [778, 3]
    :param obj: [N, 3]
    :param dist: [N,]
    '''
    N = handObj.shape[0] - 778
    #handObj = np.vstack((hand, obj))
    cpoints_mask = np.where(dist < 0.005**2)
    # print(np.sum(cpoints_mask))
    c_hand, c_obj = np.array([[1, 0, 0]]), np.array([[0, 0, 1]]) # RGB
    c_hand = np.repeat(c_hand, repeats=778, axis=0) # [778,3]
    c_obj = np.repeat(c_obj, repeats=N, axis=0) # [3000,3]
    c_obj[cpoints_mask, 2] = 0
    c_obj[cpoints_mask, 1] = 1  # G for contact points
    c_hanObj = np.vstack((c_hand, c_obj)) # [778+3000, 3]

    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(handObj)
    pc.colors = open3d.Vector3dVector(c_hanObj)
    open3d.draw_geometries([pc])

def show_exterior(exterior, obj_xyz):
    '''
    Draw obj xyz, discriminate exterior points (in red) which intersect with hand mesh, and interior points (in blue)
    :param exterior: status of obj vertices in [N], 1 as exterior, 0 as interior
    :param obj_xyz: [N, 3]
    '''
    inner = np.where( exterior==0 )
    color = np.array([[1,0,0]])
    color = np.repeat(color, repeats=3000, axis=0) # [3000,3]
    color[inner, 0] = 0
    color[inner, 2] = 1 # blue for interior points

    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(obj_xyz)
    pc.colors = open3d.Vector3dVector(color)
    open3d.draw_geometries([pc])


def plt_plot_pc(xyz):
    '''
    Draw xyz with plt
    :param xyz: point cloud xyz [N, 3]
    '''
    xyz = xyz.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    ax.scatter(x, y, z, s=5, c='r', marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plt_plot_normal(xyz, normal):
    '''
    Draw xyz and vertice normal with plt
    :param xyz: point cloud xyz [N, 3]
    :param normal: point cloud vertice normal [N, 3]
    '''
    xyz = xyz.detach().cpu().numpy()
    normal = normal.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    dx, dy, dz = normal[...,0], normal[...,1], normal[...,2]
    ax.scatter(x, y, z, s=5, c='r', marker='.')
    ax.quiver(x, y, z, dx, dy, dz, length=0.005, normalize=True)
    plt.show()
