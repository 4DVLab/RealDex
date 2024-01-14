import rosbag
import xml.etree.ElementTree as ET
from pathlib import Path
from cv_bridge import CvBridge
import cv2,re
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pylab as plt
import open3d as o3d
from tqdm import tqdm
import math,copy
import os
from numba import jit
import scipy
from scipy.spatial.transform import Rotation
import pywavefront
import sensor_msgs.point_cloud2 as pc2
import ctypes,time
import struct
from collections import defaultdict
import copy,json


#所有的数据都集中在这里，或许用配置文件会更好？
#path 还有os.path中的数据，如果有/开头的路径，会直接去掉前面的所有路径
class path_data_class:
    def __init__(self):
        self.bag_name = Path("banana")
        self.root_path = Path("/home/lab4dv/bags")
        self.folder_path = Path("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF")
        self.urdf_path = self.root_path/ Path("urdf") / Path("bimanual_srhand_ur.urdf")
        self.bag_path = self.folder_path / Path(self.bag_name) / Path(str(self.bag_name) + ".bag")
        self.output_folder = self.folder_path/ Path(self.bag_name)  / Path("output")
        self.ros_path_prefix = Path("/home/lab4dv/data/hand_arm_mesh/")#如果搬运到另一台电脑上，要修改这个文件名
        self.rgb_timestamp_path = self.folder_path / Path(self.bag_name) 
        self.merge_mesh_output_folder = self.folder_path/ Path(self.bag_name)  / Path("merge_mesh/")
        self.gobal_position_folder = self.folder_path / Path(self.bag_name)  / Path("global_name_position/")
        self.pinhole_camera_parameters_path = Path("/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/camera_params.json")
        self.TF_path = self.folder_path / Path(self.bag_name)  / Path("TF/")
        self.baginfo_path = self.folder_path / Path(self.bag_name) / Path("bag_info.txt")
    
    def update_path(self):
        self.root_path = Path(self.root_path) 
        self.folder_path = Path(self.folder_path)
        # self.urdf_path = self.root_path / Path("urdf") / Path("bimanual_srhand_ur.urdf")
        self.urdf_path = Path("/home/lab4dv/data/hand_arm_mesh/bimanual_srhand_ur.urdf")
        self.bag_path = self.folder_path / Path(self.bag_name) / Path(str(self.bag_name) + ".bag")
        self.output_folder = self.folder_path/ Path(self.bag_name)  / Path("output")
        self.rgb_timestamp_path = self.folder_path / Path(self.bag_name) 
        self.merge_mesh_output_folder = self.folder_path/ Path(self.bag_name)  / Path("merge_mesh/")
        self.gobal_position_folder = self.folder_path / Path(self.bag_name)  / Path("global_name_position/")
        self.TF_path = self.folder_path / Path(self.bag_name)  / Path("TF/")
        self.baginfo_path = self.folder_path / Path(self.bag_name) / Path("bag_info.txt")




class param_collect_class:
    def __init__(self):
        pass


def build_TFtree(bag_data)->dict:
    global_name_postion =  {}
    TF_tree = defaultdict(set)
    link_names = set()
    num = 0
    tf_topics = ['/tf', '/tf_static']#有些TF，在TF中是存在的，但是
    for _,msg,_ in bag_data.read_messages(topics=tf_topics):
        for transform in msg.transforms:
            frame_id = transform.header.frame_id.replace('/','')
            chind_frame_id = transform.child_frame_id.replace('/','')
            link_names.add(frame_id)
            link_names.add(chind_frame_id)
            TF_tree[frame_id].add(chind_frame_id)
        num += 1
        if num >= 2000:#500个msg以上就能构建tftree 这里的值设的不够大，有可能会影响后面
            break
    global_name_postion = {name:None for name in link_names}
    return TF_tree,global_name_postion



#如果后面tree的挂载出了问题，那么就把整个tree给画出来
def mount_data2TFtree(TF_tree:dict,TF_path):
    for key in TF_tree.keys():
        temp_set = TF_tree[key]
        TF_tree[key] = {item:[] for item in temp_set}
    folder = TF_path
    for filename in os.listdir(folder):
        match = re.match(r"(.+)-(.+)\.txt",filename)
        link1 = match.group(1)
        link2 = match.group(2)
        #print(link1," ",link2)
        #print(type(TF_tree[link1]))
        transform = np.loadtxt(folder / Path(filename),dtype= np.float128)#x,y,z,qx,qy,qz,qw
        if transform.shape[0] == 8:
            transform = transform.reshape(1,8)
        TF_tree[link1][link2] = transform
    # TF_tree["ra_base_link"]["cam0_camera_base"] = np.array([[999,1.95177861 ,-0.14322201 , 0.55406896,-0.31840981 , 0.12287842 , 0.93084188 , 0.13057362]])
    #专门添加的机械臂到相机之间的transoform
    #这里添加的这个，是会永久改变，后面不刷新的
    #如果后面tree的挂载出了问题，那么就把整个tree给画出来

    return TF_tree



#init all the mesh in the urdf according to the right position in urdf
#urdf找到在TF tree中有节点的mesh
#以后obj的全部操作都由open3d进行
#i found that, not erery function support the path varivable

def euler_translation2transform(mesh_rpy,mesh_xyz):#这里面的xyz是以米为单位的吗
    mesh_transform = np.identity(4)
    mesh_rotation = Rotation.from_euler('xyz', mesh_rpy, degrees=False).as_matrix()
    mesh_transform[:3, :3] = mesh_rotation
    mesh_transform[:3, 3] = (np.array(mesh_xyz)).reshape(1,3)
    return mesh_transform

def ros_path2rosolute_path(ros_path,ros_path_prefix):#ros_path 是记录在urdf中的ros_path
    parts = ros_path.split('/')
    parts[-1] = parts[-1].replace("dae", "obj")
    parts = parts[2:]
    rosolute_path = ros_path_prefix / Path('/'.join(parts))
    return rosolute_path

def scale_inittransform_read_obj(attrib,mesh_origin,ros_path_prefix):
    #scale:np.array((1x3))
    file_path = ros_path2rosolute_path(attrib['filename'],ros_path_prefix)
    mesh = o3d.io.read_triangle_mesh(str(file_path))
    mesh.compute_vertex_normals()#recalculate the normal vector to reder in the open3d
    vertices = np.asarray(mesh.vertices).copy()
    if 'scale' in attrib.keys():
        scale = np.array([float(item) for item in attrib['scale'].split(' ')]).reshape(1,3).squeeze(0)
        vertices = vertices * scale
        
    if mesh_origin is not None:
        origin_rpy = mesh_origin.attrib['rpy'].split(' ')
        origin_xyz = mesh_origin.attrib['xyz'].split(' ')
        origin_rpy = [float(item) for item in origin_rpy]   
        origin_xyz = [float(item) for item in origin_xyz]   
        mesh_transform = euler_translation2transform(origin_rpy,origin_xyz)[:3,:]#不需要齐次的最后一行
        vertices = mesh_transform @ np.hstack((vertices,np.ones((vertices.shape[0],1)))).T
        vertices = vertices.T
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh




def get_mesh_from_urdf(urdf_path,ros_names:set,ros_path_prefix):
    urdf_tree = ET.parse(urdf_path)
    #names:mesh + position
    root = urdf_tree.getroot()
    links = root.findall('link')
    link_names = set(link.attrib['name'] for link in links)
    mesh_list = {}
    for link in links:#少加载了一个link，就是mouting
        name = link.attrib['name']
        visual = link.find('visual')
        if name in ros_names and visual is not None:
            geometry = visual.find('geometry')
            mesh_origin = visual.find('origin')
            if geometry is not None:
                mesh = geometry.find('mesh')
                if mesh is not None:
                    mesh_list[name] = scale_inittransform_read_obj(mesh.attrib,mesh_origin,ros_path_prefix)
        
        link_visuals = link.findall('visual')
        if len(link_visuals) > 1 :
            visual = link_visuals[1]
            if name in ros_names and visual is not None:
                geometry = visual.find('geometry')
                mesh_origin = visual.find('origin')
                if geometry is not None:
                    name = geometry.attrib['name']
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        mesh_list[name] = scale_inittransform_read_obj(mesh.attrib,mesh_origin,ros_path_prefix)
    #得到所有带有名字的mesh
    return  mesh_list,link_names#这个link_name没有用，因为它记录的是所有的urdf中的name,但是mesh_list中的是有用的


def seven_num2matrix(translation,roatation):#translation x,y,z rotation x,y,z,w
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix


def transform_mesh_with_matrix(transform_matrix,mesh):
    vertices = np.asarray(mesh.vertices).copy() 
    vertices = np.hstack((vertices,np.ones((vertices.shape[0],1))),dtype=float).T
    transformed_vertices = np.dot(transform_matrix[:3,:],vertices).T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh



def find_time_closet(slot,time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index


def dfs_position(TF_tree,global_name_postion,time_slot):
    global_name_postion['world'] = np.identity(4)
    dfs_position_update(TF_tree,global_name_postion,'world',time_slot)
    global_name_postion["rh_mounting_plate"] = global_name_postion["rh_forearm"]#这里面有一个安装盘，这个安装盘的位置是跟机械化艘的手腕的位置是一样的，特殊设置
    
    
def transform_mesh_with_quater(senven_num_transform,mesh):
    child_transform = seven_num2matrix(senven_num_transform[:3],senven_num_transform[3:])[:3,:]
    vertices = np.asarray(mesh.vertices).copy() 
    vertices = np.hstack([vertices,np.ones((vertices.shape[0],1))],dtype=float).T
    transformed_vertices = np.dot(child_transform,vertices).T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh


def transform_mesh_with_matrix(transform_matrix,mesh):
    vertices = np.asarray(mesh.vertices).copy() 
    vertices = np.hstack((vertices,np.ones((vertices.shape[0],1))),dtype=float).T
    transformed_vertices = np.dot(transform_matrix[:3,:],vertices).T
    mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    return mesh


def dfs_position_update(TF_tree,global_name_postion,name,time_slot):
    if name in TF_tree.keys():#叶子节点不会在keys里面
        for child_name in TF_tree[name].keys():
            child_time_and_transform = TF_tree[name][child_name]#有些TF只有一个，是static TF
            time_index = find_time_closet(time_slot,child_time_and_transform[:,0])
            senven_num_transform = child_time_and_transform[time_index,1:]
            child_transform = seven_num2matrix(translation=senven_num_transform[:3],roatation=senven_num_transform[3:])
            child_transform_position = np.dot(global_name_postion[name],child_transform)
            global_name_postion[child_name] = child_transform_position
            dfs_position_update(TF_tree,global_name_postion,child_name,time_slot)


def show_meshes(meshes):
    o3d.visualization.draw_geometries(meshes)

def output_merge_mesh(meshes,output_path):
    merged_mesh = o3d.geometry.TriangleMesh()
    for key,value in meshes.items():
        # if key.startswith("rh"):
        merged_mesh += value.simplify_quadric_decimation(int((1-0.9) * len(value.triangles)))
    # merged_mesh.simplify_quadric_decimation(int((1-0.8) * len(merged_mesh.triangles)))
    o3d.io.write_triangle_mesh(str(output_path), merged_mesh)

def save_global_position(global_name_postion,num,gobal_position_folder):# attention some item's transform is None
    if not os.path.exists(gobal_position_folder):
        os.makedirs(gobal_position_folder)
    with open(gobal_position_folder / Path(f"{num}.txt"), "w+") as position_write:
        global_name_postion = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in global_name_postion.items()}
        json.dump(global_name_postion, position_write, indent=4)

def show_meshposition(rgb_timestamp,mesh_list,TF_tree,global_name_postion,output_folder,base_frame,pinhole_camera_parameters_path,gobal_position_folder,output_jud,positon_vis = False):#最后的参数表示，我们这个展示的点云是要在哪个坐标系下表示
    #1.可视化 2.save global positon 还有 3.输出obj
    print("begin to extract the arm hand obj")
    vis, camera_params, global_position_jud = None,  None, False
    if not os.path.exists(gobal_position_folder):
        os.makedirs(gobal_position_folder)
        global_position_jud = True
    if not global_position_jud and not positon_vis and not output_jud:
        return

    if positon_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        camera_params = o3d.io.read_pinhole_camera_parameters(str(pinhole_camera_parameters_path))
    for num,slot in tqdm(enumerate(rgb_timestamp)):
        dfs_position(TF_tree,global_name_postion,slot)
        save_global_position(global_name_postion,num,gobal_position_folder)
        mesh_show = copy.deepcopy(mesh_list)
        for mesh_name in mesh_show.keys():
            mesh_show[mesh_name] = transform_mesh_with_matrix(global_name_postion[mesh_name],mesh_show[mesh_name])
        if base_frame != 'world':
            for mesh_name in mesh_show.keys():
                mesh_show[mesh_name] = transform_mesh_with_matrix(global_name_postion[base_frame],mesh_show[mesh_name])
        # the code for show point cloud sequence
        if positon_vis:
            if num == 0:
                for mesh_item in mesh_show.values():
                    vis.add_geometry(mesh_item)
            else:
                vis.clear_geometries()
                for mesh_item in mesh_show.values():
                    vis.add_geometry(mesh_item)
                vis.get_view_control().convert_from_pinhole_camera_parameters(camera_params)
                vis.poll_events()
                vis.update_renderer()
            vis.run()  # 这将显示mesh并允许交互直到用户按'q'

        if output_jud:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_merge_mesh(mesh_show,output_folder / Path(base_frame+"_"+ str(num) + ".obj"))#没有这个文件夹再生成
    # vis.destroy_window()
        # show_meshes([value for value in mesh_show.values()])


#修改观看视角将导致之后的机械手有可能动不了

def get_rgbimage_timestamp(bag_data,path):
    if not os.path.exists(path):
        os.makedirs(path)
    path = path / "rgbimage_timestamp.txt"
    if os.path.exists(path): 
        return np.loadtxt(path,dtype=np.int64)
    else:
        time_topics = ['/cam0/rgb/image_raw']
        rgb_timestamp = []

        for _,msg,t in bag_data.read_messages(topics=time_topics):
            t = np.float128(str(t))
            rgb_timestamp.append(t)
        rgb_timestamp = np.array(rgb_timestamp)
        np.savetxt(path,rgb_timestamp,fmt="%f")
        return rgb_timestamp



def extract_arm_hand_obj(folder_path:str,bag_name:str, root_path:str):
    path_data = path_data_class()
    path_data.bag_name = bag_name
    path_data.folder_path = folder_path
    path_data.root_path = root_path
    path_data.update_path()

    param_collect = param_collect_class()

    setattr(param_collect,"bag_data",rosbag.Bag(path_data.bag_path))
    setattr(param_collect,"rgb_timestamp",
    get_rgbimage_timestamp(param_collect.bag_data,path_data.rgb_timestamp_path))
    setattr(param_collect,"TF_tree",None)
    setattr(param_collect,"global_name_postion",None)

    param_collect.TF_tree,param_collect.global_name_postion = build_TFtree(param_collect.bag_data)
    param_collect.TF_tree = mount_data2TFtree(param_collect.TF_tree,path_data.TF_path)
    setattr(param_collect,"mesh_list",None)
    setattr(param_collect,"link_names",None)
    param_collect.mesh_list, param_collect.link_names = get_mesh_from_urdf(path_data.urdf_path,param_collect.global_name_postion.keys(),path_data.ros_path_prefix)
    show_meshposition(rgb_timestamp = param_collect.rgb_timestamp,
                  mesh_list = param_collect.mesh_list,
                  TF_tree = param_collect.TF_tree,
                  global_name_postion = param_collect.global_name_postion,
                  output_folder = path_data.merge_mesh_output_folder,
                  base_frame = 'world',
                  pinhole_camera_parameters_path = path_data.pinhole_camera_parameters_path,
                  gobal_position_folder = path_data.gobal_position_folder,
                  output_jud = False)

