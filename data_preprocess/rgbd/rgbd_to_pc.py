import os
import open3d as o3d
import cv2

# Depth camera parameters:
FX_DEPTH = 5.8262448167737955e+02
FY_DEPTH = 5.8269103270988637e+02
CX_DEPTH = 3.1304475870804731e+02
CY_DEPTH = 2.3844389626620386e+02

def depth_to_pcd(depth_image):
    pcd = []
    height, width = depth_image.shape
    for i in range(height):
        for j in range(width):
            z = depth_image[i][j]
            x = (j - CX_DEPTH) * z / FX_DEPTH
            y = (i - CY_DEPTH) * z / FY_DEPTH
            pcd.append([x, y, z])
    pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    
    
    return pcd_o3d

def rgbd_to_pcd(depth_path, rgb_path):
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)
    print(rgbd_image)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        )
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd
    
    
def test_myself():
    depth_path = "/remote-home/liuym/data/rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_70_depth.png"
    out_path = "/remote-home/liuym/data/test/"
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    pcd = depth_to_pcd(depth_img)
    #save pcd
    o3d.io.write_point_cloud(os.path.join(out_path, "kitchen_small_1_70.ply"), pcd)
    # # Visualize:
    # o3d.visualization.webrtc_server.enable_webrtc() # enable web visualizer
    # o3d.visualization.draw_geometries([pcd])
    
def test_o3d():
    depth_path = "/remote-home/liuym/data/rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_70_depth.png"
    rgb_path = "/remote-home/liuym/data/rgbd-scenes/kitchen_small/kitchen_small_1/kitchen_small_1_70.png"
    out_path = "/remote-home/liuym/data/test/"
    
    pcd = rgbd_to_pcd(depth_path, rgb_path)
    o3d.io.write_point_cloud(os.path.join(out_path, "kitchen_small_1_70_colored.ply"), pcd)

if __name__ == '__main__':
    test_o3d()
    
    
    