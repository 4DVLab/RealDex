import rosbag
import xml.etree.ElementTree as ET
from pathlib import Path
from cv_bridge import CvBridge
import cv2
import re
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pylab as plt
import open3d as o3d
from tqdm import tqdm
import math
import copy
import os
from numba import jit
import scipy
from scipy.spatial.transform import Rotation
import pywavefront
import sensor_msgs.point_cloud2 as pc2
import ctypes
import time
import struct
from collections import defaultdict
import copy
import json
from pprint import pprint


def camera_param_init():
    camera_param = {
        "intrisics": None,
        "extrisics": None,
        "width": None,
        "height": None,
        "distortion": None
    }
    return camera_param


def ros_camera_info_to_camera_param(camera_info):
    camera_param = camera_param_init()
    camera_param["intrinsic"] = camera_info["K"]
    camera_param["width"] = camera_info["width"]
    camera_param["height"] = camera_info["height"]
    camera_param["distortion"] = camera_info["D"]
    return camera_param


def load_camera_intrics(folder_path, cam_index=0):
    camera_param = camera_param_init()
    camera_info_path = Path(folder_path) / \
        Path(f"cam{cam_index}/rgb/camera_info/info.txt")
    with open(camera_info_path, "r") as json_file:
        camera_info = json.load(json_file)
    camera_param = ros_camera_info_to_camera_param(camera_info)
    return camera_param


def load_camera_extrinsic(folder_path):
    global_name_postion_path = Path(
        folder_path) / Path("global_name_position/0.txt")
    global_name_postion = None
    with open(global_name_postion_path, "r") as json_file:
        global_name_postion = json.load(json_file)
    return global_name_postion["cam0_rgb_camera_link"]


def load_camera_param(folder_path, cam_index=0):
    # load_intrisics
    camera_param = load_camera_intrics(folder_path, cam_index)
    camera_param["extrinsic"] = np.linalg.inv(
        load_camera_extrinsic(folder_path))
    return camera_param


def euler_translation2transform(mesh_rpy, mesh_xyz):  # 这里面的xyz是以米为单位的吗
    mesh_transform = np.identity(4)
    mesh_rotation = Rotation.from_euler(
        'xyz', mesh_rpy, degrees=False).as_matrix()
    mesh_transform[:3, :3] = mesh_rotation
    mesh_transform[:3, 3] = (np.array(mesh_xyz)).reshape(1, 3)
    return mesh_transform


def ros_path2rosolute_path(ros_path, ros_path_prefix):  # ros_path 是记录在urdf中的ros_path
    parts = ros_path.split('/')
    parts[-1] = parts[-1].replace("dae", "obj")
    parts = parts[2:]
    rosolute_path = ros_path_prefix / Path('/'.join(parts))
    return rosolute_path


def scale_inittransform_read_obj(attrib, mesh_origin, ros_path_prefix):
    # scale:np.array((1x3))
    file_path = ros_path2rosolute_path(attrib['filename'], ros_path_prefix)
    mesh = o3d.io.read_triangle_mesh(str(file_path))
    mesh.compute_vertex_normals()  # recalculate the normal vector to reder in the open3d
    vertices = np.asarray(mesh.vertices).copy()
    if 'scale' in attrib.keys():
        scale = np.array([float(item) for item in attrib['scale'].split(' ')]).reshape(
            1, 3).squeeze(0)
        vertices = vertices * scale

    if mesh_origin is not None:
        origin_rpy = mesh_origin.attrib['rpy'].split(' ')
        origin_xyz = mesh_origin.attrib['xyz'].split(' ')
        origin_rpy = [float(item) for item in origin_rpy]
        origin_xyz = [float(item) for item in origin_xyz]
        mesh_transform = euler_translation2transform(
            origin_rpy, origin_xyz)[:3, :]  # 不需要齐次的最后一行
        vertices = mesh_transform @ np.hstack(
            (vertices, np.ones((vertices.shape[0], 1)))).T
        vertices = vertices.T
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def get_mesh_from_urdf(urdf_path, ros_names: set, ros_path_prefix):
    # the rosname is the name that in ros you have to find them all in urdf
    urdf_tree = ET.parse(urdf_path)
    # names:mesh + position
    root = urdf_tree.getroot()
    links = root.findall('link')
    link_names = set(link.attrib['name'] for link in links)
    mesh_list = {}
    for link in links:  # 少加载了一个link，就是mouting
        name = link.attrib['name']
        visual = link.find('visual')
        if name in ros_names and visual is not None:
            geometry = visual.find('geometry')
            mesh_origin = visual.find('origin')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    mesh_list[name] = scale_inittransform_read_obj(
                        mesh.attrib, mesh_origin, ros_path_prefix)

        link_visuals = link.findall('visual')
        if len(link_visuals) > 1:
            visual = link_visuals[1]
            if name in ros_names and visual is not None:
                geometry = visual.find('geometry')
                mesh_origin = visual.find('origin')
                if geometry is not None:
                    name = geometry.attrib['name']
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_list[name] = scale_inittransform_read_obj(
                            mesh.attrib, mesh_origin, ros_path_prefix)
    # 得到所有带有名字的mesh
    return mesh_list, link_names  # 这个link_name没有用，因为它记录的是所有的urdf中的name,但是mesh_list中的是有用的


def get_urdf_mesh_list(urdf_folder_path, all_names_in_rosbag, ros_path_prefix):
    urdf_path = Path(urdf_folder_path) / Path("bimanual_srhand_ur.urdf")
    mesh_list, _ = get_mesh_from_urdf(
        urdf_path, all_names_in_rosbag, ros_path_prefix)
    return mesh_list


def get_global_position_with_timeslot(time_index, bag_folder):
    position_path = bag_folder / Path(f'global_name_position/{time_index}.txt')
    with open(position_path, 'r') as json_file:
        position_dict = json.load(json_file)
    return position_dict


def get_all_names_in_ros(bag_folder):

    position_dict = get_global_position_with_timeslot(0, bag_folder)
    names = set([key for key in position_dict.keys()])
    return names


def get_arm_hand_mesh(time_index):
    bag_folder = Path('/media/tony/新加卷/camera_data/banana/')
    ros_prefix = Path('/media/tony/新加卷/hand_arm_urdf_mesh/')

    names_in_ros = get_all_names_in_ros(bag_folder)

    mesh_list, _ = get_mesh_from_urdf(
        bag_folder / Path('bimanual_srhand_ur.urdf'), names_in_ros, ros_prefix)

    position_dict = get_global_position_with_timeslot(time_index, bag_folder)
    for name, mesh in mesh_list.items():
        mesh.transform(position_dict[name])

    merge_arm_hand_mesh = o3d.geometry.TriangleMesh()

    for mesh in mesh_list.values():
        merge_arm_hand_mesh += mesh

    merge_arm_hand_mesh_write_folder = bag_folder / Path('arm_hand_mesh')
    os.makedirs(merge_arm_hand_mesh_write_folder, exist_ok=True)

    o3d.io.write_triangle_mesh(str(
        merge_arm_hand_mesh_write_folder / Path(f"{time_index}.ply")), merge_arm_hand_mesh)


def build_TFtree_without_bag(bag_folder) -> dict:

    tf_folder = bag_folder / Path('TF')
    global_name_postion = {}
    TF_tree = defaultdict(set)
    link_names = set()

    for filename in os.listdir(tf_folder):
        match = re.match(r"(.+)-(.+)\.txt", filename)
        link1 = match.group(1)
        link2 = match.group(2)
        TF_tree[link1].add(link2)
        link_names.add(link1)
        link_names.add(link2)

    global_name_postion = {name: None for name in link_names}
    return TF_tree, global_name_postion


def mount_data2TFtree(TF_tree: dict, rosbag_folder_path):
    TF_path = rosbag_folder_path / Path('TF')
    for key in TF_tree.keys():
        temp_set = TF_tree[key]
        TF_tree[key] = {item: [] for item in temp_set}
    folder = TF_path
    for filename in os.listdir(folder):
        match = re.match(r"(.+)-(.+)\.txt", filename)
        link1 = match.group(1)
        link2 = match.group(2)
        # print(link1," ",link2)
        # print(type(TF_tree[link1]))
        transform = np.loadtxt(folder / Path(filename),
                               dtype=np.float128)  # x,y,z,qx,qy,qz,qw
        if transform.shape[0] == 8:
            transform = transform.reshape(1, 8)
        TF_tree[link1][link2] = transform
    # TF_tree["ra_base_link"]["cam0_camera_base"] = np.array([[999,1.95177861 ,-0.14322201 , 0.55406896,-0.31840981 , 0.12287842 , 0.93084188 , 0.13057362]])
    # 专门添加的机械臂到相机之间的transoform
    # 这里添加的这个，是会永久改变，后面不刷新的
    # 如果后面tree的挂载出了问题，那么就把整个tree给画出来

    return TF_tree


def dfs_position(TF_tree, global_name_postion, time_slot, output_time_record=False, output_tf=False):
    global_name_postion['world'] = np.identity(4)
    time_record = {}
    if not output_time_record:
        time_record = False
    all_links_tf = {}
    if not output_tf:
        all_links_tf = False

    dfs_position_update(TF_tree, global_name_postion,
                        'world', time_slot, time_record, all_links_tf)
    # 这里面有一个安装盘，这个安装盘的位置是跟机械化艘的手腕的位置是一样的，特殊设置
    global_name_postion["rh_mounting_plate"] = global_name_postion["rh_forearm"]
    return time_record, all_links_tf


def find_time_closet(slot, time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index


def seven_num2matrix(translation, roatation):  # translation x,y,z rotation x,y,z,w
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3, 3] = translation
    return transform_matrix


def dfs_position_update(TF_tree, global_name_postion, name, time_slot, time_record, all_links_tf):
    if name in TF_tree.keys():  # 叶子节点不会在keys里面
        for child_name in TF_tree[name].keys():
            # 有些TF只有一个，是static TF
            child_time_and_transform = TF_tree[name][child_name]
            time_index = find_time_closet(
                time_slot, child_time_and_transform[:, 0])
            if time_record and child_name.startswith("rh_"):
                time_record[child_name] = child_time_and_transform[time_index, 0]
                print(child_name, child_time_and_transform[time_index, 0])

            senven_num_transform = child_time_and_transform[time_index, 1:]
            child_transform = seven_num2matrix(
                translation=senven_num_transform[:3], roatation=senven_num_transform[3:])

            if all_links_tf is not None:
                all_links_tf[child_name] = child_transform

            child_transform_position = np.dot(
                global_name_postion[name], child_transform)
            global_name_postion[child_name] = child_transform_position
            dfs_position_update(TF_tree, global_name_postion,
                                child_name, time_slot, time_record, all_links_tf)


def output_time_stamps(TF_tree, global_name_postion):
    time_record = dfs_position(
        TF_tree, global_name_postion, 1698133703698676224.000000)
    sorted_dict = dict(sorted(time_record.items(), key=lambda item: item[1]))

    max_key_length = max(len(key) for key in sorted_dict.keys())
    for key, value in sorted_dict.items():
        print(f"{key:<{max_key_length}}: {value:,.2f}")
    times = [value for key, value in sorted_dict.items()]
    for time in times:
        print(f"{time:,.2f}")
    times = sorted(times, reverse=False)
    diff = [times[index] - times[index - 1] for index in range(1, len(times))]
    pprint(diff)


def transform_mesh_with_matrix(transform_matrix, mesh):
    vertices = np.asarray(mesh.vertices).copy()
    vertices = np.hstack(
        (vertices, np.ones((vertices.shape[0], 1))), dtype=float).T
    transformed_vertices = np.dot(transform_matrix[:3, :], vertices).T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh


def output_merge_mesh(meshes, bag_folder, index, simplify=True):
    output_folder = bag_folder / Path('arm_hand_mesh')
    os.makedirs(output_folder, exist_ok=True)
    merged_mesh = o3d.geometry.TriangleMesh()
    for key, value in meshes.items():
        # if key.startswith("rh"):
        if simplify:
            value = value.simplify_quadric_decimation(
                int((1-0.9) * len(value.triangles)))
        merged_mesh += value
    o3d.io.write_triangle_mesh(
        str(output_folder / Path(f"{index}.ply")), merged_mesh)


class KeyCallback:
    def __init__(self) -> None:
        self.current_num = None
        self.nums_to_save = []

    def on_key_A(self, vis):
        print(self.current_num)
        self.nums_to_save.append(self.current_num)
        print("num be saved")

    def on_key_D(self, vis):
        print(self.current_num)
        if len(self.nums_to_save) == 0:
            return
        self.nums_to_save.remove(self.nums_to_save[-1])
        print("num be removed")

    def set_current_num(self, num):
        self.current_num = num


def save_seg_index(seg_index, bag_folder):
    seg_index_folder = bag_folder / Path('seg_index.txt')
    with open(seg_index_folder, 'w') as f:
        for index in seg_index:
            f.write(f"{index}\n")


# you should seperate the path, as the bag you process and the urdf and the mesh you should put in the other folder

def gen_video_rm_images(image_save_folder, bag_folder):
    video_save_path = bag_folder / Path('arm_hand_motion.mp4')
    command = f"ffmpeg -framerate 15 -i {image_save_folder}/%d.png -c:v libx264 -crf 51 -pix_fmt yuv420p {video_save_path}"
    os.system(command)
    # os.system(f"rm -r {image_save_folder}")


def chang_trinsic(trinsic):
    length = np.sqrt(len(trinsic))
    length = round(length)
    trinsic = np.array(trinsic).reshape((length, length))
    trinsic = trinsic.T
    trinsic = trinsic.flatten()
    trinsic = trinsic.tolist()
    return trinsic


def change_o3d_camera_param(o3d_camera_param_path, mine_camera_param):

    o3d_camera_param = json.load(open(o3d_camera_param_path, "r"))
    extrisic = [item for list_item in mine_camera_param["extrinsic"]
                for item in list_item]
    o3d_camera_param["extrinsic"] = chang_trinsic(extrisic)
    # mine_camera_param["intrinsic"][2] = mine_camera_param["width"] / 2 -0.5
    # mine_camera_param["intrinsic"][5] = mine_camera_param["height"] / 2 -0.5
    o3d_camera_param["intrinsic"]["intrinsic_matrix"] = chang_trinsic(
        mine_camera_param["intrinsic"])
    o3d_camera_param["intrinsic"]["width"] = mine_camera_param["width"]
    o3d_camera_param["intrinsic"]["height"] = mine_camera_param["height"]
    return o3d_camera_param



def viz_project_object_to_2d():

    vis = o3d.visualization.Visualizer()
    vis.create_window()



    
# 这个path在bag文件夹之上，equal to
def viz_arm_hand_mesh_without_bag(config_folder, bag_folder, rosbag_predix, base_frame="world", output_jud=False, my_cam=True):

    TF_tree, global_name_postion = build_TFtree_without_bag(bag_folder)

    all_names_in_rosbag = get_all_names_in_ros(bag_folder)
    TF_tree = mount_data2TFtree(TF_tree, bag_folder)

    cam0_rgb_time_stamp = np.loadtxt(
        bag_folder / Path('rgbimage_timestamp.txt'), dtype=np.float128)

    mesh_list = get_urdf_mesh_list(
        config_folder, all_names_in_rosbag, rosbag_predix)
    # o3d.visualization.draw_geometries([mesh_list['rh_palm']], window_name="Open3D1")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis = o3d.visualization.VisualizerWithKeyCallback()

    # vis.register_key_callback(ord("A"), KeyCallback_object.on_key_A)
    # vis.register_key_callback(ord("D"), KeyCallback_object.on_key_D)
    camera_param_path = config_folder / \
        Path("project_arm_hand_to_2d/camera_params.json")
    print(camera_param_path)

    if my_cam:
        camera_param = load_camera_param(bag_folder)
        o3d_params = change_o3d_camera_param(camera_param_path, camera_param)
        json.dump(o3d_params, open(camera_param_path, "w"), indent=4)

    camera_params = o3d.io.read_pinhole_camera_parameters(
        str(camera_param_path))
    # vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
    # print(camera_params.intrinsic.width,camera_params.intrinsic.height)

    image_save_folder = bag_folder / Path('cam0_arm_hand_images')
    os.makedirs(image_save_folder, exist_ok=True)

    for num, slot in tqdm(enumerate(cam0_rgb_time_stamp)):
        dfs_position(TF_tree, global_name_postion, slot)
        # save_global_position(global_name_postion,num,gobal_position_folder)
        mesh_show = copy.deepcopy(mesh_list)
        for mesh_name in mesh_show.keys():
            mesh_show[mesh_name] = transform_mesh_with_matrix(
                global_name_postion[mesh_name], mesh_show[mesh_name])
        if base_frame != 'world':
            for mesh_name in mesh_show.keys():
                mesh_show[mesh_name] = transform_mesh_with_matrix(
                    global_name_postion[base_frame], mesh_show[mesh_name])
        # the code for show point cloud sequence
        if num == 0:
            for mesh_item in mesh_show.values():
                vis.add_geometry(mesh_item)
        else:
            vis.clear_geometries()
            for mesh_item in mesh_show.values():
                vis.add_geometry(mesh_item)
        vis.get_view_control().convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(image_save_folder /
                   Path(f"{num}.png"), np.asarray(image), dpi=1)
        # vis.run()  # 这将显示mesh并允许交互直到用户按'q' 用来与用户交互的
        if output_jud:
            output_merge_mesh(mesh_show, bag_folder, num)

    vis.destroy_window()

    # gen_video_rm_images(image_save_folder,bag_folder)
    # get_arm_hand_mesh(123)


def gen_hand_arm_move_video(config_folder, root_folder, rosbag_predix=Path('/media/tony/新加卷/configuration')):

    config_folder = Path(config_folder)

    for file in os.listdir(root_folder):
        temp_path = Path(root_folder) / Path(file)
        if "TF" in file and os.path.isdir(temp_path):
            print(root_folder)
            # if you  code in this way, other people would easily understand it
            bag_folder = Path(root_folder)
            viz_arm_hand_mesh_without_bag(
                config_folder, bag_folder, rosbag_predix, base_frame="world", output_jud=False)
            return  # 因为一个bag_folder只会有一个TF folder
        elif os.path.isdir(temp_path):
            gen_hand_arm_move_video(config_folder, temp_path, rosbag_predix)


def simplify_mesh(mesh, proportion=0.5):
    mesh = mesh.simplify_quadric_decimation(
        int(proportion * len(mesh.triangles)))
    mesh = mesh.remove_unreferenced_vertices()
    return mesh


def simplify_mesh_list(mesh_list, proportion=0.5):
    for key, mesh in mesh_list.items():
        mesh_list[key] = simplify_mesh(mesh, proportion)
    return mesh_list


def gen_on_stamp_hand_arm_mesh(config_folder, bag_folder, rosbag_predix, stamp_index_begin, stamp_index_end=None, base_frame="world"):

    TF_tree, global_name_postion = build_TFtree_without_bag(bag_folder)
    all_names_in_rosbag = get_all_names_in_ros(bag_folder)
    TF_tree = mount_data2TFtree(TF_tree, bag_folder)
    cam0_rgb_time_stamp = np.loadtxt(
        bag_folder / Path('rgbimage_timestamp.txt'), dtype=np.float128)

    mesh_list = get_urdf_mesh_list(
        config_folder, all_names_in_rosbag, rosbag_predix)

    mesh_list = simplify_mesh_list(mesh_list, proportion=0.2)
    # vis.create_window()

    if stamp_index_end is None:
        stamp_index_end = stamp_index_begin + 1
    elif stamp_index_end > cam0_rgb_time_stamp.shape[0]:
        stamp_index_end = cam0_rgb_time_stamp.shape[0]
    else:
        stamp_index_end += 1

    for stamp_index in np.arange(stamp_index_begin, stamp_index_end):
        slot = cam0_rgb_time_stamp[stamp_index]
        dfs_position(TF_tree, global_name_postion, slot)
        # save_global_position(global_name_postion,num,gobal_position_folder)
        mesh_show = copy.deepcopy(mesh_list)
        for mesh_name in mesh_show.keys():
            mesh_show[mesh_name] = transform_mesh_with_matrix(
                global_name_postion[mesh_name], mesh_show[mesh_name])
        if base_frame != 'world':
            for mesh_name in mesh_show.keys():
                mesh_show[mesh_name] = transform_mesh_with_matrix(
                    global_name_postion[base_frame], mesh_show[mesh_name])
        output_merge_mesh(mesh_show, bag_folder, stamp_index, simplify=False)

    # vis.run()  # 这将显示mesh并允许交互直到用户按'q' 用来与用户交互的


def gen_one_stamp_all_tf(bag_folder, stamp_index_begin, stamp_index_end=None):
    TF_tree, global_name_postion = build_TFtree_without_bag(bag_folder)
    all_names_in_rosbag = get_all_names_in_ros(bag_folder)
    TF_tree = mount_data2TFtree(TF_tree, bag_folder)
    cam0_rgb_time_stamp = np.loadtxt(
        bag_folder / Path('rgbimage_timestamp.txt'), dtype=np.float128)

    if stamp_index_end is None:
        stamp_index_end = stamp_index_begin + 1
    elif stamp_index_end > cam0_rgb_time_stamp.shape[0]:
        stamp_index_end = cam0_rgb_time_stamp.shape[0]
    else:
        stamp_index_end += 1

    for stamp_index in np.arange(stamp_index_begin, stamp_index_end):
        slot = cam0_rgb_time_stamp[stamp_index]
        _, all_tf_links = dfs_position(
            TF_tree, global_name_postion, slot, output_tf=True)
        # save_global_position(global_name_postion,num,gobal_position_folder)
    all_tf_links = {key: value.tolist() for key, value in all_tf_links.items()}

    with open("/media/tony/T7/yumeng/TF/all_links_tf.json", "w+") as json_writer:
        json.dump(all_tf_links, json_writer, indent=4)

    # vis.run()  # 这将显示mesh并允许交互直到用户按'q' 用来与用户交互的


# 也需要只有一个arm_hand_mesh的
if __name__ == "__main__":

    ros_prefix_path = "/media/tony/T7/camera_data/configuration/hand_arm_mesh"
    configuration_path = "/media/tony/T7/camera_data/configuration"
    bag_folder = "/media/tony/T7/yyx_tmp/for_dust_cleanning_sprayer_tracking/dust_cleanning_spreyer/dust_cleanning_spreyer_1_20231105"

    gen_hand_arm_move_video()
