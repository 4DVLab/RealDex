import rosbag
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
# from utils.genPC.genPC_use_o3d import gen_aligned_pc
def camera_info2dict(msg):
    camera_info_dict = {
            'header': {
                'seq': msg.header.seq,
                'stamp': {
                    'secs': msg.header.stamp.secs,
                    'nsecs': msg.header.stamp.nsecs
                },
                'frame_id': msg.header.frame_id
            },
            'height': msg.height,
            'width': msg.width,
            'distortion_model': msg.distortion_model,
            'D': msg.D,
            'K': msg.K,
            'R': msg.R,
            'P': msg.P,
            'binning_x': msg.binning_x,
            'binning_y': msg.binning_y,
            'roi': {
                'x_offset': msg.roi.x_offset,
                'y_offset': msg.roi.y_offset,
                'height': msg.roi.height,
                'width': msg.roi.width,
                'do_rectify': msg.roi.do_rectify
            }
        }
    return camera_info_dict



class path_data_class:
    def __init__(self):
        self.bag_name = Path("banana")
        self.folder_path = Path(
            "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF")
        self.bag_path = self.folder_path / \
            self.bag_name / Path(str(self.bag_name)+".bag")
        self.transform_json_path = self.folder_path / self.bag_name / \
            Path("global_name_position") / Path("0.txt")
        self.output_path = self.folder_path / self.bag_name
        self.cams = [
            '/cam0',
            '/cam1',
            '/cam2',
            '/cam3']
        self.bag_topics = [
            # '/depth/camera_info',
            # '/depth/image_raw',
            '/depth_to_rgb/camera_info',
            '/depth_to_rgb/image_raw',
            # '/points2',
            '/rgb/camera_info',
            '/rgb/image_raw'
        ]

    def update(self):
        self.bag_name = Path(self.bag_name)
        self.folder_path = Path(self.folder_path)
        self.output_path = self.folder_path / self.bag_name
        self.bag_path = self.folder_path / \
            self.bag_name / Path(str(self.bag_name)+".bag")
        self.transform_json_path = self.folder_path / self.bag_name / \
            Path("global_name_position") / Path("0.txt")


class param_collect_class:
    def __init__(self):

        self.cam_transform_data = {
            "cam0": None,

            "cam1": None,

            "cam2": None,

            "cam3": None
        }


def init_cam_transform(cam_transform, json_path):
    with open(json_path, 'r') as file:
        loaded_data = json.load(file)
    loaded_data = {key: np.array(value) if isinstance(
        value, list) else value for key, value in loaded_data.items()}
    for index in np.arange(4):
        cam_transform["cam" + str(index)] = loaded_data["cam" +
                                                        str(index) + "_depth_camera_link"]


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
    pc2_data = point_cloud2.read_points(
        pointcloud2_msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
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


def transform_mesh_with_matrix(transform_matrix, mesh):
    vertices = np.asarray(mesh.points).copy()
    vertices = np.hstack(
        (vertices, np.ones((vertices.shape[0], 1))), dtype=float).T
    transformed_vertices = np.dot(transform_matrix[:3, :], vertices).T
    mesh.points = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh


def DepthFromBuffer(depthimage_msg):
    DepthImageFromBuffer = np.frombuffer(depthimage_msg.data, dtype=np.uint8).reshape(
        depthimage_msg.height, depthimage_msg.width, -1)
    if DepthImageFromBuffer.shape[-1] == 2:
        DepthImageFromBuffer0 = DepthImageFromBuffer[:, :, 0].copy()
        DepthImageFromBuffer1 = DepthImageFromBuffer[:, :, 1].copy()
        DepthImageFromBuffer = DepthImageFromBuffer.astype(np.uint16)
        DepthImageFromBuffer = DepthImageFromBuffer0 + 2**8 * DepthImageFromBuffer1
        # print("it has 2 chennels")
    else:
        print("it has only one chennel")
    return DepthImageFromBuffer


def write_camera_info(folder, bag_data):
    _, msg, _ = next(bag_data)
    file_path = folder / Path("info.txt")
    msg = camera_info2dict(msg)
    json.dump(msg,open(file_path, 'w+'))
    # with open(file_path, 'w+') as f:
    #     f.write(str(msg))


def write_rgb_image(folder_path, bag_data):  # 也会写入depth
    time_stamp = []
    num = 0
    for _, msg, t in bag_data:
        time_stamp.append(np.int64(str(t)))
        rgb_cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")[
            :, :, :3]  # 必须去掉最后一维透明通道，不然tracking 算法会有问题\
        cv2.imwrite(str(folder_path / Path(str(num) + ".png")), rgb_cv_image)
        num += 1
    np.savetxt(str(folder_path / Path("info.txt")), time_stamp)


def write_depth_image(folder_path, bag_data):
    time_stamp = []
    num = 0
    for _, msg, t in bag_data:
        time_stamp.append(np.int64(str(t)))
        depth_image = DepthFromBuffer(msg)
        cv2.imwrite(str(folder_path / Path(str(num) + ".png")), depth_image)
        num += 1
    np.savetxt(str(folder_path / Path("info.txt")), time_stamp)


def write_pc(folder_path, bag_data, cam_transform_data):
    cam = [key for key in cam_transform_data if key in str(folder_path)][0]
    time_stamp = []
    num = 0
    for _, msg, t in bag_data:
        time_stamp.append(np.int64(str(t)))
        pc = ros_pointcloud2_to_open3d(msg)
        pc = transform_mesh_with_matrix(cam_transform_data[cam], pc)
        o3d.io.write_point_cloud(
            str(folder_path / Path(str(num) + ".ply")), pc)
        num += 1

    np.savetxt(str(folder_path / Path("info.txt")), time_stamp)


def bag_data_extract(bag_path, cams, bag_topics, output_folder, cam_transform_data):
    with rosbag.Bag(bag_path, 'r') as bag_data:
        for cam in cams:
            for topic in bag_topics:
                data_write_folder = output_folder / \
                    Path(cam[1:]) / Path(topic[1:])
                topic = cam + topic
                if not os.path.exists(data_write_folder):
                    os.makedirs(data_write_folder)
                bag_topic_data = bag_data.read_messages(topics=[topic])

                if topic.endswith('camera_info'):
                    write_camera_info(data_write_folder, bag_topic_data)
                    continue

                elif topic.endswith('/rgb/image_raw'):
                    write_rgb_image(data_write_folder, bag_topic_data)

                elif topic.endswith('depth/image_raw') or topic.endswith('depth_to_rgb/image_raw'):
                    write_depth_image(data_write_folder, bag_topic_data)

                elif topic.endswith('points2'):
                    write_pc(data_write_folder, bag_topic_data,
                             cam_transform_data)

                else:
                    print("nothing")
    # gen_aligned_pc(output_folder)

# folder是比专门的bag文件夹更高一级的
def extract_everything_from_bag(folder_path: str, bag_name: str):
    print("extract everything from bag")
    path_data = path_data_class()
    path_data.folder_path = folder_path
    path_data.bag_name = bag_name
    path_data.update()

    param_collect = param_collect_class()
    # init_cam_transform(param_collect.cam_transform_data,
    #                    path_data.transform_json_path)

    bag_data_extract(path_data.bag_path,
                     path_data.cams,
                     path_data.bag_topics,
                     path_data.output_path,
                     param_collect.cam_transform_data)
