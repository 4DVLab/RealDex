import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import struct
import ctypes
import cv2
import json
from pathlib import Path
import json
import copy
import threading
import shutil
import filecmp
from typing import Dict, List, TypedDict
from scipy.spatial.transform import Rotation

from collections import defaultdict


def seven_num2matrix(translation, roatation):  # translation x,y,z rotation x,y,z,w
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix


def find_time_closet(slot, time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index

def create_point_cloud_from_rgb_and_depth(undistorted_rgb, undistorted_depth, camera_inrisics, width, height):
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
            width, height, camera_inrisics[0, 0], camera_inrisics[1,
                                                                  1], camera_inrisics[0, 2], camera_inrisics[1, 2]
        )
    )

    # Return the point cloud
    return pcd


# so, later, we gen the point cloud from the four camera seperatly
# for every sequence, we have to gen a new class to gen the point cloud

# if you want to run this code,you have to have four cams intrisics and extrisics
class gen_merge_pcd_ply:
    def __init__(self, 
                 bag_folder, 
                 four_cam_intrisics_extrisics_save_folder,
                 intrisics_use_ros_data = False):
        """
        use four_cam_transform_folder to transform the point cloud
        """
        self.bag_folder = Path(bag_folder)
        self.intrisics, self.extrics = self.get_four_cam_intrisics_extrisics(
            four_cam_intrisics_extrisics_save_folder,intrisics_use_ros_data)
        self.cam0_transform_to_world = self.load_cam0_transform_to_world(
            bag_folder)

        self.four_cams_to_world_frame = self.get_all_cams_to_world_frame(
            self.extrics, self.cam0_transform_to_world)
        self.merge_pcd_save_folder = self.bag_folder / "merged_pcd_filter"
        os.makedirs(self.merge_pcd_save_folder, exist_ok=True)

        self.all_cams_time_stamp_index = []

        # self.merge_pcd_and_filter()

    def read_intrisics_from_ros_data(self,cam_index):
        intrisics_path = self.bag_folder / f"cam{cam_index}/rgb/camera_info"
        intrisics = {}
        with open(intrisics_path,"r") as json_reader:
            camera_data = json.load(json)
            
            intrisics["matrix"] = np.array(camera_data["K"]).reshape((3, 3))
            intrisics["width"] = camera_data["width"]
            intrisics["height"] = camera_data["height"]
            intrisics["distortion"] = np.array(camera_data["D"]).flatten(-1)
        return intrisics

    def undistort_image(self, image_path, camera_matrix, distortion_coeffs):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        h, w = image.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, distortion_coeffs, None, None, (w, h), 5)
        undistorted_image = cv2.remap(
            image, mapx, mapy, interpolation=cv2.INTER_NEAREST)

        return undistorted_image


    def gen_pcd_with_depth_and_rgb_paths(
        self,
        rgb_img_index,
        cam_index
    ):

        rgb_img_path = self.bag_folder / \
            Path(f"cam{cam_index}/rgb/image_raw/{str(rgb_img_index)}.png")
        depth_img_path = self.bag_folder / \
            Path(f"cam{cam_index}/depth_to_rgb/image_raw/{str(rgb_img_index)}.png")

        undistorted_rgb = self.undistort_image(
            rgb_img_path, self.intrisics[cam_index]["matrix"], self.intrisics[cam_index]["distortion"])
        undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)

        undistorted_depth = self.undistort_image(
            depth_img_path, self.intrisics[cam_index]["matrix"], self.intrisics[cam_index]["distortion"])
        pcd = create_point_cloud_from_rgb_and_depth(
            undistorted_rgb, undistorted_depth, self.intrisics[cam_index]["matrix"], self.intrisics[cam_index]["width"], self.intrisics[cam_index]["height"])
        return pcd

    def gen_and_check_cams_RGBD(self):
        pass

    def gen_cams_time_stamp(self):
        all_cams_time_stamp_list = []
        for cam_index in np.arange(4):
            cam_time_stamp = np.loadtxt(
                self.bag_folder / Path(f"cam{cam_index}/rgb/image_raw/info.txt"))
            all_cams_time_stamp_list.append(cam_time_stamp)
        all_cams_align_to_cam0_rgb_time_index = [
            list(range(all_cams_time_stamp_list[0].shape[0])), [], [], []]
        for cam_index in np.arange(1, 4):
            for cam_time_stamp in all_cams_time_stamp_list[0]:
                index = find_time_closet(
                    cam_time_stamp, all_cams_time_stamp_list[cam_index])
                all_cams_align_to_cam0_rgb_time_index[cam_index].append(index)

        return all_cams_align_to_cam0_rgb_time_index

    def merge_pcd_and_filter(self,constrain_bound):
        constrain_bound[1] += 1
        if constrain_bound is None:
            constrain_bound = [0,np.inf]
        self.all_cams_time_stamp_index = self.gen_cams_time_stamp()
        merge_pcd = o3d.geometry.PointCloud()

        time_index_length = len(self.all_cams_time_stamp_index[0])
        # for time_index in self.all_cams_time_stamp_index[0]:
        for index in np.arange(constrain_bound[0],constrain_bound[1]):
            if index >= time_index_length:
                return
            time_index = self.all_cams_time_stamp_index[0][index]

            for cam_index in np.arange(4):
                pcd = self.gen_pcd_with_depth_and_rgb_paths(
                    self.all_cams_time_stamp_index[cam_index][time_index], cam_index)
                
                pcd.transform(self.four_cams_to_world_frame[cam_index])
                
                pcd = self.filter_pcd(pcd)

                
                # o3d.io.write_point_cloud(str(
                #     self.merge_pcd_save_folder / f"pcd{cam_index}.ply"), pcd, write_ascii=False, compressed=False, print_progress=True)
                merge_pcd += pcd
                merge_pcd.remove_duplicated_points()
            # o3d.visualization.draw_geometries([merge_pcd])

            o3d.io.write_point_cloud(str(
                self.merge_pcd_save_folder / f"merge_pcd_{time_index}.ply"), merge_pcd, write_ascii=False, compressed=False, print_progress=True)

    def init_merge_pcd_timestamp(self):
        shutil.copy2(
            self.bag_folder / Path("cam0/rgb/image_raw/info.txt"), self.merge_pcd_save_folder)
        os.makedirs(self.merge_pcd_save_folder, exist_ok=True)

    def get_all_cams_to_world_frame(self, cams_inter_transform, cam0_to_world_transform):
        four_cams_to_world_frame = [cam0_to_world_transform @
                                    cams_inter_transform[f"{cam_index}"] for cam_index in np.arange(1, 4, 1)]

        four_cams_to_world_frame.insert(
            0, cam0_to_world_transform)

        return four_cams_to_world_frame

    def load_cam0_transform_to_world(self, bag_folder):
        global_posistion_path = bag_folder / "global_name_position/0.txt"
        with open(global_posistion_path, "r") as json_reader:
            json_data = json.load(json_reader)
            return np.array(json_data["cam0_rgb_camera_link"]).reshape((4, 4))

    def load_one_camera_intrisics(self, cam_intrisics_folder: Path, cam_num: Path) -> Dict[str, int]:
        cam_intrisics_path = cam_intrisics_folder / f"cam{cam_num}_rgb.txt"
        intrisics = {"matrix": None, "width": None,
                     "height": None, "distortion": None}
        with open(cam_intrisics_path, "r") as json_reader:
            camera_data = json.load(json_reader)
            intrisics["matrix"] = np.array(camera_data["K"]).reshape((3, 3))
            intrisics["width"] = camera_data["width"]
            intrisics["height"] = camera_data["height"]
            intrisics["distortion"] = np.array(camera_data["D"]).reshape(-1)
        return intrisics

    def load_internal_camera_extrisics(self, cam_extrisics_folder: Path, cam_num: Path):
        cam_extrisics_path = cam_extrisics_folder / f"cali0{cam_num}.json"
        extrisics = None
        with open(cam_extrisics_path, "r") as json_reader:
            camera_data = json.load(json_reader)

            rotation = np.array(list(camera_data["value0"]["rotation"].values())).flatten()[
                [1, 2, 3, 0]]
            translation = np.array(
                list(camera_data["value0"]["translation"].values())).flatten()
            extrisics = seven_num2matrix(translation, rotation)
            # extrisics = np.linalg.inv(extrisics)
        return extrisics

    def get_four_cam_intrisics_extrisics(self, four_cam_intrisics_extrisics_save_folder,intrisics_use_ros_data) -> dict:
        cam_intrisics = []
        if intrisics_use_ros_data:
            cam_intrisics = [self.read_intrisics_from_ros_data(
                cam_index) for cam_index in np.arange(4)]
        else:
            cam_intrisics = [self.load_one_camera_intrisics(
            four_cam_intrisics_extrisics_save_folder, cam_index) for cam_index in np.arange(4)]
        cam_extrisics = {f"{cam_index}": self.load_internal_camera_extrisics(
            four_cam_intrisics_extrisics_save_folder, cam_index) for cam_index in np.arange(1, 4, 1)}
        return cam_intrisics, cam_extrisics

    def filter_pcd(self, pcd):  # for every pcd, we have to filter the point cloud
        # origin, she process the filter in the world frame

        _, pt_map = pcd.hidden_point_removal([0, 0, 0], 10000)
        pcd = pcd.select_by_index(pt_map)

        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            np.array([-0.5, -0.8, 0], np.float64), np.array([2, 0.8, 1.5], np.float64)))

        # filter
        pcd, ind = pcd.remove_statistical_outlier(30, 1.5)

        return pcd


def gen_severa_annotate_pcd(bag_folder, four_cam_intrisics_extrisics_save_folder, constrain_bound):

    gen_merge_pcd = gen_merge_pcd_ply(
        bag_folder, four_cam_intrisics_extrisics_save_folder)

    gen_merge_pcd.merge_pcd_and_filter(constrain_bound)


def gen_pcd_for_annotate(root_path: Path, four_cam_intrisics_extrisics_save_folder: Path, constrain_bound):

    try:
        if "TF" in os.listdir(root_path):
            print(root_path)
            gen_severa_annotate_pcd(
                root_path, four_cam_intrisics_extrisics_save_folder, constrain_bound)
            return 
        for sub_file_path in os.listdir(root_path):
            if os.path.isdir(root_path / sub_file_path):
                gen_pcd_for_annotate(root_path / sub_file_path,
                                    four_cam_intrisics_extrisics_save_folder)
    except PermissionError:
        print(root_path)
        return

def mint():
    bag_folder = Path("/home/lab4dv/data/sda/banana/original/banana_1_20231027")
    four_cam_intrisics_extrisics_save_folder = Path(
        "/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data")
    gen_merge_pcd = gen_merge_pcd_ply(
        bag_folder, four_cam_intrisics_extrisics_save_folder)
    gen_merge_pcd.merge_pcd_and_filter(2)



if __name__ == "__main__":
    four_cam_intrisics_extrisics_save_folder = Path(
        "/home/tony/mine/Projects/ArmHandVis/git_hand_arm/IntelligentHand/calibration_ws/calibration_process/data")
    
    root_path = Path("/media/tony/新加卷/test_data/test/test_1")

    
    
    constrain_bound = [0,1]
    gen_pcd_for_annotate(
        root_path, four_cam_intrisics_extrisics_save_folder, constrain_bound)


