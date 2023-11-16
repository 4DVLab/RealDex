import numpy as np
import cv2
import open3d as o3d
import json
from pathlib import Path 
import os
import shutil
# from typing import List




def undistort_image(image_path, camera_matrix, distortion_coeffs):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    h, w = image.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coeffs, None, None, (w, h), 5)
    undistorted_image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_NEAREST)

    return undistorted_image

def create_point_cloud_from_rgb_and_depth(undistorted_rgb, undistorted_depth, camera_inrisics,width,height):
    # Convert undistorted images to open3d format
    color_raw = o3d.geometry.Image(undistorted_rgb)
    depth_raw = o3d.geometry.Image(undistorted_depth)

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

    # Convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            width, height, camera_inrisics[0,0], camera_inrisics[1,1], camera_inrisics[0,2], camera_inrisics[1,2]
        )
    )
    
    # Return the point cloud
    return pcd

def get_camera_info(param_path):
    info_path = Path(param_path) / Path("camera_info") / Path("info.txt")
    with open(info_path,"r") as json_file:
        camera_param = json.load(json_file)

    camera_param["D"] = np.array(camera_param["D"]).reshape(-1)
    camera_param["K"] = np.array(camera_param["K"]).reshape((3,3))

    return camera_param


def find_time_closet(slot,time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index



def get_target_folders(root_folder:Path,folder_prefix:Path = "",folder_postfixs:list):
    


def gen_aligned_pc(bag_data_path:str,cam_nums = [0,1,2,3],img_nums = [num for num in range(0,2000,1)]):#set a range 
    cam_num = 4
    image_prefix_folder = ["depth_to_rgb","rgb"]
    for cam_index in cam_nums:
        cam_folder = Path(bag_data_path) / Path("cam" + str(cam_index))
        depth_msg_folder = Path(cam_folder) / Path(image_prefix_folder[0])
        rgb_msg_folder = Path(cam_folder) / Path(image_prefix_folder[1])
        
        depth_camera_param = get_camera_info(depth_msg_folder)
        rgb_camera_param = get_camera_info(rgb_msg_folder)

        depth_img_folder = depth_msg_folder / Path("image_raw") 
        rgb_img_folder = rgb_msg_folder / Path("image_raw")

        depth_stamp = np.loadtxt(depth_img_folder / Path("info.txt"))
        rgb_stamp = np.loadtxt(rgb_img_folder / Path("info.txt"))
        #以depth为主来生成点云
        for img_index in img_nums:
            rgb_index = find_time_closet(rgb_stamp[img_index], depth_stamp)

            rgb_img_path = rgb_img_folder / Path(str(rgb_index) + ".png")
            depth_img_path = depth_img_folder / Path(str(img_index) + ".png")

            undistorted_rgb = undistort_image(rgb_img_path, rgb_camera_param["K"], rgb_camera_param["D"])
            undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)

            undistorted_depth = undistort_image(depth_img_path, depth_camera_param["K"], depth_camera_param["D"])
            pcd = create_point_cloud_from_rgb_and_depth(undistorted_rgb, undistorted_depth, rgb_camera_param["K"],rgb_camera_param["width"],rgb_camera_param["height"])
            # o3d.visualization.draw_geometries([pcd])
            point_cloud_folder = cam_folder / Path("points2")
            os.makedirs(point_cloud_folder,exist_ok=True)

            o3d.io.write_point_cloud(str(point_cloud_folder / Path(str(img_index) + ".ply")), pcd)

        # 将depth_stamp当作是points_stramp,所有的stamp都是以主相机为准
        shutil.copy2(rgb_img_folder / Path("info.txt"),
                     cam_folder / Path("points2"))


def gen_one_pc_to_use_for_init_pose(bag_data_path:str):

    bag_name = path.split('/')[-1]
    cam_num = 4
    image_prefix_folder = ["depth_to_rgb","rgb"]
    for cam_index in np.arange(cam_num):
        cam_folder = Path(bag_data_path) / Path("cam" + str(cam_index))
        depth_msg_folder = Path(cam_folder) / Path(image_prefix_folder[0])
        rgb_msg_folder = Path(cam_folder) / Path(image_prefix_folder[1])
        
        depth_camera_param = get_camera_info(depth_msg_folder)
        rgb_camera_param = get_camera_info(rgb_msg_folder)

        depth_img_folder = depth_msg_folder / Path("image_raw") 
        rgb_img_folder = rgb_msg_folder / Path("image_raw")

        depth_stamp = np.loadtxt(depth_img_folder / Path("info.txt"))
        rgb_stamp = np.loadtxt(rgb_img_folder / Path("info.txt"))
        #以depth为主来生成点云

        img_index = 0
        rgb_index = find_time_closet(depth_stamp[img_index],rgb_stamp)

        rgb_img_path = rgb_img_folder / Path(str(rgb_index) + ".png")
        depth_img_path = depth_img_folder / Path(str(img_index) + ".png")

        undistorted_rgb = undistort_image(rgb_img_path, rgb_camera_param["K"], rgb_camera_param["D"])
        undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)

        undistorted_depth = undistort_image(depth_img_path, depth_camera_param["K"], depth_camera_param["D"])
        pcd = create_point_cloud_from_rgb_and_depth(undistorted_rgb, undistorted_depth, rgb_camera_param["K"],rgb_camera_param["width"],rgb_camera_param["height"])
        # o3d.visualization.draw_geometries([pcd])
        point_cloud_folder = Path(bag_data_path) / Path("tracking")
        os.makedirs(point_cloud_folder,exist_ok=True)

        o3d.io.write_point_cloud(str(point_cloud_folder / Path(f"{bag_name}_cam{cam_index}.ply")), pcd)








def genPC():
    # Provide your paths to the RGB and depth images
    rgb_image_path = '/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/utils/Kinect_genPC/rgb.png'
    depth_image_path = '/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/utils/Kinect_genPC/depth.png'

    # Assuming you have Kinect's intrinsic and distortion parameters (modify these values)
    intrinsics = {
        'width': 1920,  # image width
        'height': 1080,  # image height
        'fx': 922.2793389536421,  # focal length x
        'fy': 922.5350212900213,  # focal length y
        'cx': 959.0429664417595,  # center x
        'cy': 548.315639392366   # center y
    }

    # Kinect's distortion coefficients - modify these based on your actual values
    distortion_coeffs = np.array([0.08224517559026757, -0.02980258593309121, 0.002534489316579671, 0.0003794909833965187, 0.0])  # Example values

    camera_matrix = np.array([
        [intrinsics['fx'], 0, intrinsics['cx']],
        [0, intrinsics['fy'], intrinsics['cy']],
        [0, 0, 1]
    ])

    undistorted_rgb = undistort_image(rgb_image_path, camera_matrix, distortion_coeffs)
    undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)

    undistorted_depth = undistort_image(depth_image_path, camera_matrix, distortion_coeffs)

    # rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_UNCHANGED)
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    # depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

    pcd = create_point_cloud_from_rgb_and_depth(undistorted_rgb, undistorted_depth, intrinsics,intrinsics["width"],intrinsics["height"])
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/utils/Kinect_genPC/pcd.ply", pcd)

def gen_four_txt(path:str):
    bag_name = path.split('/')[-1]
    save_in_folder_path = Path(path) / Path("tracking")
    cam_num = 4
    for cam_index in np.arange(cam_num):
        file_name = f"{bag_name}_cam{cam_index}.txt"
        file_path = save_in_folder_path / file_name
        with open(file_path,"w") as file_writer:
            file_writer.write("")

if __name__ == "__main__":
    path = "/home/lab4dv/data/bags/dust_cleanning_spreyer/dust_cleanning_spreyer_6_20231105"
    
    gen_one_pc_to_use_for_init_pose(path)
    gen_four_txt(path)
    # gen_aligned_pc("/home/lab4dv/data/sda/duck/duck_4_20231024/")