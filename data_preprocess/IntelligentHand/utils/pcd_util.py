import os
# from sensor_msgs.msg import PointCloud2
# from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import cv2
import json
from pathlib import Path
import json
from typing import Dict, List, TypedDict
from scipy.spatial.transform import Rotation

from tqdm import tqdm


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
            width, height, camera_inrisics[0, 0], camera_inrisics[1,1], camera_inrisics[0, 2], camera_inrisics[1, 2]
        )
    )

    # Return the point cloud
    return pcd


# so, later, we gen the point cloud from the four camera seperatly
# for every sequence, we have to gen a new class to gen the point cloud

# if you want to run this code,you need four cams intrisics and extrisics
class PCDGenerator:
    def __init__(self, 
                bag_folder, 
                cam_param_dir,
                intrisics_use_ros_data = False):
        """
        use four_cam_transform_folder to transform the point cloud
        """
        self.bag_folder = Path(bag_folder)
        self.intrisics, self.extrics = self.get_four_cam_intrisics_extrisics(
            cam_param_dir,intrisics_use_ros_data)
        self.cam0_transform_to_world = self.load_cam0_transform_to_world(
            bag_folder)

        self.four_cams_to_world_frame = self.get_all_cams_to_world_frame(
            self.extrics, self.cam0_transform_to_world)

        self.cam_time_aligned_list = []
        self.gen_cams_time_stamp()

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

        rgb_img_path = os.path.join(self.bag_folder, f"cam{cam_index}/rgb/image_raw/{str(rgb_img_index)}.png")
        depth_img_path = os.path.join(self.bag_folder, f"cam{cam_index}/depth_to_rgb/image_raw/{str(rgb_img_index)}.png")

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
        cams_time_list = []
        for cam_index in np.arange(4):
            cam_time_stamp = np.loadtxt(os.path.join(self.bag_folder, f"cam{cam_index}/rgb/image_raw/info.txt"))
            cams_time_list.append(cam_time_stamp)
        cam_time_aligned_list = [
            list(range(cams_time_list[0].shape[0])), [], [], []]
        for cam_index in np.arange(1, 4):
            for cam_time_stamp in cams_time_list[0]:
                index = find_time_closet(
                    cam_time_stamp, cams_time_list[cam_index])
                cam_time_aligned_list[cam_index].append(index)
        self.cam_time_aligned_list = cam_time_aligned_list
        return cam_time_aligned_list

    def gen_pcd(self, index, cam_index=0, export=True):
        time_index = self.cam_time_aligned_list[0][index]
        pcd = self.gen_pcd_with_depth_and_rgb_paths(self.cam_time_aligned_list[cam_index][time_index], cam_index)
        pcd.transform(self.four_cams_to_world_frame[cam_index])
        pcd = self.filter_pcd(pcd)
        if export:
            out_dir = os.path.join(self.bag_folder, f"cam{cam_index}", "pcd")
            os.makedirs(out_dir, exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(out_dir, f"{index}.ply"), 
                                    pcd, write_ascii=False, compressed=False, print_progress=True)
        return pcd

    def get_all_cams_to_world_frame(self, cams_inter_transform, cam0_to_world_transform):
        four_cams_to_world_frame = [cam0_to_world_transform @
                                    cams_inter_transform[f"{cam_index}"] for cam_index in np.arange(1, 4, 1)]

        four_cams_to_world_frame.insert(
            0, cam0_to_world_transform)

        return four_cams_to_world_frame

    def load_cam0_transform_to_world(self, bag_folder):
        global_posistion_path = os.path.join(bag_folder, "global_name_position/0.txt")
        with open(global_posistion_path, "r") as json_reader:
            json_data = json.load(json_reader)
            return np.array(json_data["cam0_rgb_camera_link"]).reshape((4, 4))

    def load_one_camera_intrisics(self, cam_intrisics_folder: Path, cam_num: Path) -> Dict[str, int]:
        cam_intrisics_path = os.path.join(cam_intrisics_folder, f"cam{cam_num}_rgb.txt")
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
        cam_extrisics_path = os.path.join(cam_extrisics_folder, f"cali0{cam_num}.json")
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

    def get_four_cam_intrisics_extrisics(self, cam_param_dir,intrisics_use_ros_data) -> dict:
        cam_intrisics = []
        if intrisics_use_ros_data:
            cam_intrisics = [self.read_intrisics_from_ros_data(
                cam_index) for cam_index in np.arange(4)]
        else:
            cam_intrisics = [self.load_one_camera_intrisics(
            cam_param_dir, cam_index) for cam_index in np.arange(4)]
        cam_extrisics = {f"{cam_index}": self.load_internal_camera_extrisics(
            cam_param_dir, cam_index) for cam_index in np.arange(1, 4, 1)}
        return cam_intrisics, cam_extrisics

    def filter_pcd(self, pcd):
        
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            np.array([-0.3, -0.6, 0.5], np.float64), np.array([2, 0.5, 1.4], np.float64)))
        
        _, pt_map = pcd.hidden_point_removal([0, 0, 0], 10000)
        pcd = pcd.select_by_index(pt_map)  
        pcd = pcd.voxel_down_sample(0.002)

        return pcd
    
    @staticmethod
    def crop_pcd(pcd, bb_min, bb_max):
        cropped_pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max))
        return cropped_pcd
    
    @staticmethod     
    def random_subsample(pcd, num_points=8000):
        """
        Randomly subsample a point cloud to a fixed number of points.

        Parameters:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        num_points (int): The number of points to sample.

        Returns:
        open3d.geometry.PointCloud: The subsampled point cloud.
        """
        # Get the total number of points in the point cloud
        total_points = np.asarray(pcd.points).shape[0]
        
        # Check if the point cloud has enough points
        if num_points >= total_points:
            return pcd
        
        # Randomly select indices
        indices = np.random.choice(total_points, num_points, replace=False)
        
        # Select points based on indices
        selected_points = np.asarray(pcd.points)[indices, :]
        
        # Create a new point cloud based on the selected points
        subsampled_pcd = o3d.geometry.PointCloud()
        subsampled_pcd.points = o3d.utility.Vector3dVector(selected_points)
        
        # If the original point cloud has colors or normals, also subsample them
        if pcd.has_colors():
            subsampled_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices, :])
        if pcd.has_normals():
            subsampled_pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indices, :])
    
        return subsampled_pcd
    
    def gen_merged_pcd(self, index, out_dir, bb_min=None, bb_max=None, export=True):
        if bb_min is None or bb_max is None:
            '''box around the object point cloud'''
            bb_min = np.array([0.78, -0.6, 0.73], np.float64)
            bb_max = np.array([2, 0.5, 1.06], np.float64)
        
        pcd_list = []
        for cam_index in range(4):
            pcd = self.gen_pcd(index, cam_index, export=export)
            pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(bb_min, bb_max))
            pcd_list.append(pcd)
        
        target = pcd_list[0]
        max_dist= 0.001
        registered_pc = [pcd_list[0]]
        for i in range(1, 4):
            source = pcd_list[i]
            icp_fine = o3d.pipelines.registration.registration_icp(
                source, target, max_dist, np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10))
            transformation_icp = icp_fine.transformation
            source.transform(transformation_icp)
            registered_pc.append(source)
            
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in registered_pc:
            combined_pcd += pcd
            
        # combined_pcd = combined_pcd.voxel_down_sample(0.001)
        _, ind = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # Select the inlier points
        inlier_cloud = combined_pcd.select_by_index(ind)
        
        out_path = os.path.join(out_dir, f"{index}.ply")
        
        if export:
            o3d.io.write_point_cloud(out_path, inlier_cloud)
        return inlier_cloud


def gen_pcd_for_annotate(root_path, cam_param_dir, start, end=None):

    try:
        if "TF" in os.listdir(root_path):
            print(root_path)
            gen_merge_pcd = PCDGenerator(root_path, cam_param_dir)
            gen_merge_pcd.gen_pcd(start, end, cam_index=3)
            return 
    except PermissionError:
        print(root_path)
        return


# def down_sample_pcd(ply_files, voxel_size=0.002):
#     for ply_file in ply_files:
#         file_path = os.path.join(pcd_dir, ply_file)
#         point_cloud = o3d.io.read_point_cloud(file_path)

#         downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)

#         o3d.io.write_point_cloud(file_path, downsampled_point_cloud)


if __name__ == "__main__":
    cam_param_dir = "../../calibration_ws/calibration_process/data"
    
    # root_path = "/Users/yumeng/Working/data/CollectedDataset/sprayer_1_20231209/"
    root_path = "/Users/yumeng/Working/data/CollectedDataset/yogurt/yogurt_1_20231207"

    gen_pcd_for_annotate(root_path, cam_param_dir, start=0, end=None)


