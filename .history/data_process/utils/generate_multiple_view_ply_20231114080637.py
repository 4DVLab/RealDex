import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import struct
import ctypes
from cv_bridge import CvBridge
import cv2
import json
from pathlib import Path
import json
import copy
import threading
import shutil
import filecmp
from typing import Dict, List, TypedDict
from genPC import genPC_use_o3d
from common import common
from collections import defaultdict





# so, later, we gen the point cloud from the four camera seperatly
# for every sequence, we have to gen a new class to gen the point cloud

#if you want to run this code,you have to have four ca
class gen_merge_pcd_ply:
    def __init__(self,bag_folder,four_cam_intrisics_extrisics_save_folder) -> List[str,Dict[str,int]]:
        """
        use four_cam_transform_folder to transform the point cloud
        """
        self.bag_folder = Path(bag_folder)
        self.intrisics,self.extrics = self.get_four_cam_intrisics_extrisics(four_cam_intrisics_extrisics_save_folder)
        self.cam0_transform_to_world = self.load_cam0_transform_to_world(bag_folder)
        self.four_cams_to_world_frame = self.get_all_cams_to_world_frame(self.extrics,self.cam0_transform_to_world)
        self.merge_pcd_save_folder = self.bag_folder / "merged_pcd_filter"
        self.all_cams_time_stamp_list = []
        self.init_merge_pcd_timestamp_and_folder()
        self.merge_pcd_and_filter()
        
    def gen_and_check_cams_RGBD(self,):
        pass
        

    def gen_cams_time_stamp(self,):
        

    def merge_pcd_and_filter(self):



    def init_merge_pcd_timestamp(self):
        shutil.copy2(
            self.bag_folder / Path("cam0/rgb/image_raw/info.txt"), self.merge_pcd_save_folder)
        os.makedirs(self.merge_pcd_save_folder, exist_ok=True)

    def get_all_cams_to_world_frame(self,cams_inter_transform,cam0_to_world_transform):
        four_cams_to_world_frame = [cam0_to_world_transform @
                                    cams_inter_transform[f"{cam_index}"] for cam_index in np.arange(1, 4, 1)]
        four_cams_to_world_frame = four_cams_to_world_frame.insert(0, cam0_to_world_transform)
        return four_cams_to_world_frame

    def load_cam0_transform_to_world(self,bag_folder):
        global_posistion_path = bag_folder / "global_name_position/0.txt" 
        with open(global_posistion_path,"r") as json_reader:
            json_data = json.load(json_reader)
            return np.array(json_data["cam0_rgb_camera_link"]).reshape((3,3))


    def load_one_camera_intrisics(self,cam_intrisics_folder:Path,cam_num:Path)->Dict[str,int]:
        cam_intrisics_path = cam_intrisics_folder / f"cam{cam_num}_rgb.txt"
        intrisics = {"matrix":None,"width":None,"height":None,"distortion":None}
        with open(cam_intrisics_path,"r") as json_reader:
            camera_data = json.load(json_reader)
            intrisics["matrix"] = np.array(camera_data["K"]).reshape(3,3)
            intrisics["width"] = camera_data["width"]
            intrisics["height"] = camera_data["height"]
            intrisics["distortion"] = np.array(camera_data["D"]).flatten()
        return intrisics


    def load_internal_camera_extrisics(self,cam_extrisics_folder:Path,cam_num:Path):
        cam_extrisics_path = cam_extrisics_folder / f"cali0{cam_num}.json"
        extrisics = None
        with open(cam_extrisics_path,"r") as json_reader:
            camera_data = json.load(json_reader)
            rotation = np.array(camera_data["rotation"]).flatten()[3,0,1,2]
            translation = np.array(camera_data["translation"]).flatten()
            extrisics = common.seven_num2matrix(translation,rotation)
        return extrisics
    

    def get_four_cam_intrisics_extrisics(self,four_cam_intrisics_extrisics_save_folder)->dict:

        cam_intrisics = [self.load_one_camera_intrisics(
            four_cam_intrisics_extrisics_save_folder,cam_index) for cam_index in np.arange(4)]
        cam_extrisics = {f"{cam_index}":self.load_internal_camera_extrisics(
            four_cam_intrisics_extrisics_save_folder,cam_index) for cam_index in np.arange(1, 4, 1)}
        return cam_intrisics,cam_extrisics

    def filter_pcd(self,pcd):#for every pcd, we have to filter the point cloud
        #origin, she process the filter in the world frame

        _, pt_map = pcd.hidden_point_removal([0, 0, 0], 200)
        pcd = pcd.select_by_index(pt_map)

        pcd.transform(self.cam_transform_list[cam])
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(
            np.array([-0.5, -0.8, 0], np.float64), np.array([2, 0.8, 1.5], np.float64)))

        # filter
        pcd, ind = pcd.remove_statistical_outlier(30, 0.08)

        return pcd





#you have to change the code use the code as follow to read the file of extrisics to gen the multiple view ply
#two mode, input the root path,gen the point cloud

# 1. i can input the list of the number of pcd that i want to  gen the point cloud
class multiple_view_ply:

    def __init__(self, root_path, middle_path,four_cam_transform_folder):

        self.four_cam_transform_folder = Path(four_cam_transform_folder)
        self.read_from_ply = False
       
        self.middle_path = Path(middle_path)
        self.root_path = Path(root_path) / Path(middle_path) 
        print(self.root_path)
        self.merged_path = Path(self.root_path)/Path("merged_pcd_filter")
        if not self.merged_path.exists():
            os.mkdir(self.merged_path)
        
        shutil.copy2(self.root_path / Path("cam0/depth_to_rgb/image_raw/info.txt"), self.merged_path )

        self.cam_transform_list = []
        # calibration value read from local file 

        # for the wrong extrinsics, read from calibration file then calculation is needed
        # extrinsics  

        # for i in range(4):
        #     # with error calibration value, read form calibration file then calculation is needed
            
        #     if i == 0:
        #          self.cam_transform_list.append(cam0_to_world)
        #     else:
        #          with open("/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data/cali0"+str(i)+".json") as cal_f:
        #             cal_data = json.load(cal_f)
        #          cami_to_cam0 = np.array( common.seven_num2matrix(np.array([cal_data["value0"]["translation"]["x"], cal_data["value0"]["translation"]["y"], cal_data["value0"]["translation"]["z"]])
        #                                                                , np.array([cal_data["value0"]["rotation"]["x"], cal_data["value0"]["rotation"]["y"], cal_data["value0"]["rotation"]["z"], cal_data["value0"]["rotation"]["w"]])))
                
        #          self.cam_transform_list.append(np.dot(cam0_to_world,cami_to_cam0))



       # calibration value read from corresponding files

            
        if self.read_from_ply:
            self.cam_path_list = []
            self.timestamp_list = []
            for i in range(4):
                self.cam_path_list.append(self.root_path / Path("cam" + str(i) + "/points2"))
                timestamp_one_list = np.loadtxt(self.cam_path_list[i] / Path("info.txt"), dtype=np.float128)
                self.timestamp_list.append(timestamp_one_list)
                # print("the number of frames of cam", i, len(timestamp_one_list)) 
         
        else :
            self.depth_camera_param_list = []
            self.rgb_camera_param_list = []
            self.depth_stamp_list = []
            self.rgb_stamp_list = []

            for i in np.arange(4):
                self.cam_folder = Path(self.root_path) / Path("cam" + str(i))
                self.depth_stamp_list.append(np.loadtxt(Path(self.cam_folder)  / Path("depth_to_rgb") / Path("image_raw")/ Path("info.txt")))
                self.rgb_stamp_list.append(np.loadtxt( Path(self.cam_folder) / Path("rgb") / Path("image_raw")/ Path("info.txt")))
            self.timestamp_list = self.depth_stamp_list        
        
        # 15Hz for 0.0667s interval, 30Hz for 0.033 interval 
                # test 15Hz or 30Hz , pick frame 10 and frame 19 for stable and random
        # if (self.timestamp_list[0][10]- self.timestamp_list[0][9]) and (self.timestamp_list[0][19]- self.timestamp_list[0][18]) < 4e+7 :
        #             self.interval = 3.333e+7
        #             print("30 Hz")
        # else:
        #             self.interval = 6.667e+7
        #             print("15 Hz")


        self.read_intrinsics_from_local()



    def read_intrinsics_from_local(self):
        with open(self.root_path / Path("global_name_position/0.txt"), "r") as f:
            data= json.load(f)


        for i in range(4):
            self.cam_transform_list.append(np.array(data["cam"+str(i)+"_rgb_camera_link"]))

            self.depth_camera_param_list.append(genPC_use_o3d.get_camera_info(Path(self.cam_folder) / Path("depth_to_rgb")))
            self.rgb_camera_param_list.append(genPC_use_o3d.get_camera_info(Path(self.cam_folder) / Path("rgb")))

        
    def read_single_view(self, cam:int, num:int):


        
        depth_camera_param = self.depth_camera_param_list[cam]
        rgb_camera_param = self.rgb_camera_param_list[cam]

        depth_stamp = self.depth_stamp_list[cam]
        rgb_stamp = self.rgb_stamp_list[cam]

        depth_img_folder = self.root_path / Path("cam" + str(cam)) /  Path("depth_to_rgb") / Path("image_raw")
        rgb_img_folder = self.root_path / Path("cam" + str(cam)) /  Path("rgb") / Path("image_raw")

        
   
        rgb_index = common.find_time_closet(depth_stamp[num],rgb_stamp)

        rgb_img_path = rgb_img_folder / Path(str(rgb_index) + ".png")
        depth_img_path = depth_img_folder / Path(str(num) + ".png")
        undistorted_rgb = genPC_use_o3d.undistort_image(rgb_img_path, rgb_camera_param["K"], rgb_camera_param["D"])
        undistorted_rgb = cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB)
        undistorted_depth = genPC_use_o3d.undistort_image(depth_img_path, depth_camera_param["K"], depth_camera_param["D"])
        pcd = genPC_use_o3d.create_point_cloud_from_rgb_and_depth(undistorted_rgb, undistorted_depth, rgb_camera_param["K"],rgb_camera_param["width"],rgb_camera_param["height"])
        
        return pcd


    def process_single_view(self, cam:int , num:int):#crop and filter the point cloud 
        # read from file 

        if self.read_from_ply:
            ply = o3d.io.read_point_cloud(str(self.cam_path_list[cam] / Path(str(num)+".ply")))

        # read from images

        else:
            ply = self.read_single_view(cam, num)

        # o3d.visualization.draw_geometries([ply])

        # hide unvisitable points
        _, pt_map = ply.hidden_point_removal([0, 0, 0], 200) 
        ply = ply.select_by_index(pt_map)

        # rotation to world frame
     
        ply.transform(self.cam_transform_list[cam])
        # o3d.visualization.draw_geometries([ply])

        # crop
        # x forward, y left , z up
        # ply= ply.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([1, -0.1, 0], np.float64), np.array([1.5, 0.3, 1.5], np.float64)))
        ply= ply.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.5, -0.8, 0], np.float64), np.array([2, 0.8, 1.5], np.float64)))

        # filter
        ply, ind =ply.remove_statistical_outlier(30, 0.08)

        return ply
       # o3d.visualization.draw_geometries([ply])
    
    def read_multiple_view(self):
        
        # align timestamps from 4 different view
        # iteration = [180, 176, 181, 180]
        # iteration = [84, 84, 84, 84]
        iteration = [0, 0, 0, 0]
        good_merge_num = 0
        while(iteration[0] < len(self.timestamp_list[0])):
            
            # main_timestamp = self.timestamp_list[0][iteration[0]]
            # print(main_timestamp)
            main_ply = self.process_single_view(0, iteration[0])
            # previous_ply = copy.deepcopy(main_ply)
            merge_ply = main_ply
            miss_view = []
            for i in range(1, 4):
                if iteration[i] >= len(self.timestamp_list[i]):
                    miss_view.append((i,iteration[i], "^"))
                    break
               
                # if iteration[i] - 1 >=0 and abs (self.timestamp_list[i][iteration[i] - 1 ] - main_timestamp) < abs( self.timestamp_list[i][iteration[i]]- main_timestamp)  :
                #     miss_view.append((i, iteration[i], "-"))
                #     iteration[i] = iteration[i] - 1
                # elif iteration[i] + 1 < len(self.timestamp_list[i]) and abs( self.timestamp_list[i][iteration[i]+1] - main_timestamp) < abs(self.timestamp_list[i][iteration[i]]- main_timestamp):
                #         miss_view.append((i, iteration[i], "+"))
                #         iteration[i] = iteration[i] + 1
                # iteration[i] = common.find_time_closet(main_timestamp, self.timestamp_list[i])  
                # if abs(self.timestamp_list[i][iteration[i]] - main_timestamp) >  self.interval:
                #      miss_view.append(i, iteration[0],iteration[i])
                #      continue
                        
                # print( self.timestamp_list[i][iteration[i]])
                iteration[i] = common.find_time_closet(self.timestamp_list[i], self.timestamp_list[i][iteration[0]])
                # print(iteration[i])
                cam_ply = self.process_single_view(i, iteration[i])

                # try some registration method
                # color_icp = o3d.pipelines.registration.registration_colored_icp(cam_ply, merge_ply, 0.005, np.identity(4)
                                                                                # , o3d.pipelines.registration.TransformationEstimationForColoredICP(0.9),
                                                                                # o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                                # relative_fitness=1e-4, relative_rmse=1e-4, max_iteration=5) )
             
                # print(type(color_icp))
                # print(type(merge_ply))
                # cam_ply.transform(color_icp.transformation)
                merge_ply = merge_ply + cam_ply

                merge_ply = merge_ply.remove_duplicated_points()
                
                
                # previous_ply = copy.deepcopy(cam_ply)
                iteration[i] = iteration[i] + 1
            if len(miss_view) != 0:
               print("miss view after", good_merge_num," good merge: ", miss_view)
            else:
                # print(iteration)
                # o3d.visualization.draw_geometries([merge_ply])
                o3d.io.write_point_cloud(str(self.merged_path/ Path(str(self.middle_path)+"_"+str(iteration[0])+".ply")), merge_ply, write_ascii=False, compressed=False, print_progress=True)
                good_merge_num = good_merge_num + 1
            
            iteration[0] = iteration[0] + 1
        print(" there are", good_merge_num, "good merge\n")


def generate_multiple_view_ply(root_path, middle_path):
    mvp = multiple_view_ply(root_path, middle_path)
    
    mvp.read_multiple_view()





root_folder_list = ["/home/lab4dv/data/bags/", "/home/lab4dv/data/ssd/", "/home/lab4dv/data/sda/" ]
bags_folder_list = ["ramen_noodles", "wd40"]
ssd_folder_list = ["castle", "cream_cake", "duck_toy", "flower_bread", "light", "xbox", "yibu"]
sda_folder_list =["banana", "girl_toy", "goji_jar", "instance_noodles", "light", "sprayer", "thunder_toy", "yogurt"]


# generate_multiple_view_ply("/home/lab4dv/data/bags/apple", "apple_5_20231105")
generate_multiple_view_ply(root_folder_list[1]+ssd_folder_list[5], "xbox_6_20231105")




