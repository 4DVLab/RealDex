# %%
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
from common import common
import copy


# %%

class multiple_view_ply:

    def __init__(self, root_path):
        
        self.root_path = Path(root_path)

        self.cam_path_list = []
        self.time_stamp_txt = Path("info.txt")

        # x y z,  x y z w
        self.cam_transform_list = []
        with open(self.root_path / Path("global_name_position/0.txt"), "r") as f:
            data= json.load(f)
            cam0_to_world = np.array(data["cam"+str(0)+"_rgb_camera_link"])

            cam1_to_world = np.array(data["cam"+str(1)+"_rgb_camera_link"])

            # print(np.dot(np.linalg.inv(cam0_to_world),cam1_to_world))

        for i in range(4):
            self.cam_path_list.append(root_path / Path("cam" + str(i) + "/points2"))
            # with correct calibration value, just read from one file

            self.cam_transform_list.append(np.array(data["cam"+str(i)+"_rgb_camera_link"]))
            # with error calibration value, read form calibration file then calculation is needed
            
            # if i == 0:
            #      self.cam_transform_list.append(cam0_to_world)
            # else:
            #      with open("/home/lab4dv/IntelligentHand/calibration_ws/calibration_process/data/cali0"+str(i)+".json") as cal_f:
            #         cal_data = json.load(cal_f)
            #      cami_to_cam0 = np.array( common.seven_num2matrix(np.array([cal_data["value0"]["translation"]["x"], cal_data["value0"]["translation"]["y"], cal_data["value0"]["translation"]["z"]])
            #                                                            , np.array([cal_data["value0"]["rotation"]["x"], cal_data["value0"]["rotation"]["y"], cal_data["value0"]["rotation"]["z"], cal_data["value0"]["rotation"]["w"]])))
                 
            #      self.cam_transform_list.append(np.dot(cam0_to_world,cami_to_cam0))

        self.timestamp_list = []
        for i in range(4):
            timestamp_one_list = np.loadtxt(self.cam_path_list[i] / self.time_stamp_txt, dtype=np.float128)
            self.timestamp_list.append(timestamp_one_list)
            print("the number of frames of cam", i, len(timestamp_one_list)) 
         # 15Hz for 0.0667s interval, 30Hz for 0.033 interval 
                # test 15Hz or 30Hz , pick frame 10 and frame 19 for stable and random
        if (self.timestamp_list[0][10]- self.timestamp_list[0][9]) and (self.timestamp_list[0][19]- self.timestamp_list[0][18]) < 4e+7 :
                    self.interval = 3.333e+7
                    # print("30 Hz")
        else:
                    self.interval = 6.667e+7
                    # print("15 Hz")
                
        # print(self.cam_transform_list[0], self.cam_transform_list[1], self.cam_transform_list[2])       
        

    def read_single_view(self, cam , num):
        # read from file or from other function
        ply = o3d.io.read_point_cloud(str(self.cam_path_list[cam] / Path(str(num)+".ply")))

        # hide unvisitable points
        _, pt_map = ply.hidden_point_removal([0, 0, 0], 200) 
        ply = ply.select_by_index(pt_map)

        # rotation to world frame
     
        ply.transform(self.cam_transform_list[cam])

        # crop
        # x forward, y left , z up
        # ply= ply.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([1, -0.1, 0], np.float64), np.array([1.5, 0.3, 1.5], np.float64)))
        ply= ply.crop(o3d.geometry.AxisAlignedBoundingBox(np.array([-0.5, -0.8, 0], np.float64), np.array([2, 0.8, 1.5], np.float64)))

        # filter
        # ply, ind =ply.remove_statistical_outlier(30, 0.08)

        # estimate normal

        # ply.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30), True)

        #temp write
        o3d.io.write_point_cloud("/home/lab4dv/data/bags/cam"+str(cam)+"_"+str(num)+".ply", ply, write_ascii=False, compressed=False, print_progress=False)

        return ply
       # o3d.visualization.draw_geometries([ply])
    
    def read_multiple_view(self):
        
        # align timestamps from 4 different view
        iteration = [180, 180, 180, 181]
        good_merge_num = 0
        while(iteration[0] != len(self.timestamp_list[0])):
            
            main_timestamp = self.timestamp_list[0][iteration[0]]
            # print(main_timestamp)
            main_ply = self.read_single_view(0, iteration[0])
            previous_ply = copy.deepcopy(main_ply)
            merge_ply = main_ply
            miss_view = []
            for i in range(1, 4):
                # if iteration[i] >= len(self.timestamp_list[i]):
                #     miss_view.append((i,iteration[i], "^"))
                #     continue
               
                # if iteration[i] - 1 >=0 and abs (self.timestamp_list[i][iteration[i] - 1 ] - main_timestamp) < abs( self.timestamp_list[i][iteration[i]]- main_timestamp)  :
                #     miss_view.append((i, iteration[i], "-"))
                #     iteration[i] = iteration[i] - 1
                # elif iteration[i] + 1 < len(self.timestamp_list[i]) and abs( self.timestamp_list[i][iteration[i]+1] - main_timestamp) < abs(self.timestamp_list[i][iteration[i]]- main_timestamp):
                #         miss_view.append((i, iteration[i], "+"))
                #         iteration[i] = iteration[i] + 1
                # iteration[i] = common.find_time_closet(main_timestamp, self.timestamp_list[i])  
                if abs(self.timestamp_list[i][iteration[i]] - main_timestamp) >  self.interval:
                     miss_view.append(i, iteration[0],iteration[i])
                     continue
                        
                # print( self.timestamp_list[i][iteration[i]])

                cam_ply = self.read_single_view(i, iteration[i])

                # try some registration method
                # color_icp = o3d.pipelines.registration.registration_colored_icp(cam_ply, merge_ply, 0.01, np.identity(4)
                #                                                                 , o3d.pipelines.registration.TransformationEstimationForColoredICP(0.8),
                #                                                                 o3d.pipelines.registration.ICPConvergenceCriteria(
                #                                                                 relative_fitness=1e-4, relative_rmse=1e-4, max_iteration=50) )
             
                # print(color_icp)
                # cam_ply.transform(color_icp.transformation)
                merge_ply = merge_ply + cam_ply

                merge_ply = merge_ply.remove_duplicated_points()
                
                
                previous_ply = copy.deepcopy(cam_ply)
                # iteration[i] = iteration[i] + 1
            if len(miss_view) != 0:
               print("miss view after", good_merge_num," good merge: ", miss_view)
            else:
                print(iteration)
                o3d.visualization.draw_geometries([merge_ply])
                good_merge_num = good_merge_num + 1
            
            iteration[0] = iteration[0] + 1
        print(" there are", good_merge_num, "good merge\n")


def generate_multiple_view_ply(root_path):
    mvp = multiple_view_ply(root_path)
    
    mvp.read_multiple_view()

generate_multiple_view_ply("/home/lab4dv/data/bags/apple/apple_1_20231030")





