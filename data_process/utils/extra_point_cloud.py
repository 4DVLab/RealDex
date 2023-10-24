import rosbag
import os
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import struct,ctypes

def num2color(num):
    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', num)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
    rgb = np.array([r, g, b]) / 255.0
    return rgb

def ros_pointcloud2_to_open3d(pointcloud2_msg):
    # 使用point_cloud2模块从PointCloud2消息中提取点
    pc2_data = point_cloud2.read_points(pointcloud2_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    points = []
    colors = []
    for p in pc2_data:
        points.append([p[0], p[1], p[2]])

        colors.append(num2color(p[3]))
    # 创建Open3D的点云对象并填充数据
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

folder = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/"
bag_path = folder + "banana.bag"


output_dirs = [
    "/cam0",
    "/cam1",
    "/cam2",
    "/cam3",
]
output_dirs = [folder + "/bag_point_cloud/" + d for d in output_dirs]
# 创建输出文件夹，如果它们不存在
for dir_path in output_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

bag_topics = [
    '/cam0/points2',
    '/cam1/points2',
    '/cam2/points2',
    '/cam3/points2'
]
start_time = np.int64(1695281121856355352)

# for index in np.arange(4):
index = 3
timestamp = []
num = 0
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[bag_topics[index]]):
        t = np.int64(str(t))
        t = t - start_time
        timestamp.append(t)
        output_path = os.path.join(output_dirs[index], str(num) + ".ply")
        cloud = ros_pointcloud2_to_open3d(msg)
        o3d.io.write_point_cloud(output_path, cloud)
        num += 1
with open(output_dirs[index] + "/timestamp.txt", "w+") as f:
    for t in timestamp:
        f.write(str(t) + "\n")