import rosbag
from pprint import pprint
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pylab as plt
import open3d as o3d
from tqdm import tqdm
import math
import os
from numba import jit
import scipy
from scipy.spatial.transform import Rotation
import pywavefront
import sensor_msgs.point_cloud2 as pc2
import ctypes
import struct
from pathlib import Path
# if you use the square depth image to generate pc  ,it pc is distorted
# undistorted_coeffient =  [k1,k2,p1,p2,k3]
# you have to remember,the input image is cv:mat,which means,the x,y is reverse
#matlotlib coordinate system is same as the opencv
#but its color system is not same 
# 之后尽可能用einstan 来算，不然很容易出错

#把功能给单独包装成一个函数，这样就能单独测试每一个功能

#can use KDtree to do interplot when you project the 3D point to the plane


def Rotation2quat(matrix):
    rotation = Rotation.from_matrix(matrix)
    return rotation.as_quat()


def quat2Rotation(quat):
    rotation = Rotation.from_quat(quat)
    return rotation.as_matrix()

def TransformPC(vertices, TransformMatrix):
    vertices = np.hstack(
        (vertices, np.ones((vertices.shape[0], 1), dtype=np.float64)))
    vertices = np.dot(TransformMatrix, vertices.T)
    vertices = vertices.T
    return vertices


def transform_body2target_pose(file_path, output_path, transform_matrix):
    mesh = pywavefront.Wavefront(
        file_path, collect_faces=True, create_materials=True)
    TransformMatrix = transform_matrix
    vertices = TransformPC(np.array(mesh.vertices), TransformMatrix)

    faces = np.array(mesh.mesh_list[0].faces, dtype=np.int32)
    faces = faces + np.ones_like(faces, dtype=np.int8)

    path = output_path
    with open(path, 'a+') as file:
        np.savetxt(
            file, vertices[:, :3], fmt="v %.6f %.6f %.6f", delimiter=" ")
    with open(path, 'a+') as file:
        np.savetxt(
            file, faces, fmt="f %d %d %d", delimiter=" ")

def ScalePCFromFile(InputPath, OutputPath, scale):
    # "/home/tony/mine/Projects/ProjectRecorded8_28_2023/ICG+_TestOnOurData/body/geometrycenter_duck.obj"
    # "/home/tony/mine/Projects/ProjectRecorded8_28_2023/ICG+_TestOnOurData/body/scale_geometrycenter_duck.obj"
    vertices = []
    faces = []
    with open(InputPath, "r") as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                faces.append(list(map(int, line.split()[1:])))

    with open(OutputPath, "a+") as file:
        for line in vertices:
            line = [item / scale for item in line]
            file.write("v " + str(line[0]) + " " +
                       str(line[1]) + " " + str(line[2]) + "\n")
        for line in faces:
            file.write("f " + str(line[0]) + " " +
                       str(line[1]) + " " + str(line[2]) + "\n")


def ScalePC(OutputPath, PC, scale):
    PC = PC / scale
    np.savetxt(OutputPath, PC, fmt="v %.6f %.6f %.6f", delimiter=" ")
def SavePC(path, folder, FileName, PaintedPC):
    FilePath = f"{path}/{folder}/{FileName}.obj"
    np.savetxt(FilePath, PaintedPC[:, :3], fmt="v %.6f %.6f %.6f", delimiter=" ")
    
def Rotation2quat(matrix):
    rotation = Rotation.from_matrix(matrix)
    return rotation.as_quat()[[3,0,1,2]]

def SaveMatrix(Matrix,FileName):#you should change it by your self
    with open("/home/tony/test_mkv/" + FileName +".txt","w") as file:
        for row_index in np.arange(Matrix.shape[0]):
            for colomn_index in np.arange(Matrix.shape[1]):
                # for item in Matrix[row_index,colomn_index]:
                file.write(str(Matrix[row_index,colomn_index]) + " ")
            file.write("\n")
#you can use savetxt

def GetKD_FromMsg(msg):
    # get the K matrix
    K_matrix = np.array(msg.K).reshape((3, 3))
    # get the D matrix
    D_matrix = np.array(msg.D).reshape((1, -1))
    return K_matrix, D_matrix.squeeze()#you have to remember the D you read from the bag you have to squeeze it   


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


def Depth2PC(DepthIntrisics, DepthImage,scale = 1):
    # when you  transform the depth image to point cloud, you can't use a np.zero to initial
    tolerance = 1e-6
    DepthPC = np.ones((DepthImage.shape[0] * DepthImage.shape[1], 3),dtype = np.float64)
    TempIndex_y,TempIndex_x = np.indices((DepthImage.shape[0], DepthImage.shape[1]))
    DepthPC[:, 1], DepthPC[:, 0] = TempIndex_y.flatten('C'),TempIndex_x.flatten('C')
    DepthPC = DepthPC * DepthImage.flatten('C').reshape((-1,1))
    DepthPC[:, :2] = DepthPC[:, :2] * scale
    DepthIntrisicsReverse = np.linalg.inv(DepthIntrisics)
    DepthPC = np.dot(DepthIntrisicsReverse, DepthPC.T).T
    DepthPC[:, 2] = DepthImage.flatten('C')#not run 原地操作
    DepthPC = DepthPC[DepthImage.flatten('C') > tolerance]
    return DepthPC

def ShowImage(image):
    plt.imshow(image)
    plt.show()
    plt.close()

def ShowPC(PC):#pc 是在一个[]中
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PC)
    o3d.visualization.draw_geometries([pcd])

def ReadBagMsg(BagPath,*msg):
    # info = bag_data.get_type_and_topic_info()
    BagData = rosbag.Bag(BagPath, "r")
    BagMsgs = []
    for item in msg:
        BagMsgs.append(BagData.read_messages(item))
    return BagData,*BagMsgs
        
def ProgramInit():
    np.set_printoptions(threshold=np.inf)
    

#由于我直接是用的去掉小数部分，所以会有一些误差
def DepthPC2Plane(RGBK_matrix,RGBCoordinateDepthPC,RGBImageWidth,RGBImageHeight,scale,DepthCoordinateDepthPC,UndistortedRGB = None):#scale is used to scale the Depth image whic project to the RGB plane
    #flag = 1 means use origin depth from DepthIMage to fill the RGBDepthPlane
    #要把depthimage还有rgbimage的坐标都统一到rgbcoordiate下
    _DepthCoordinateDepthPC = DepthCoordinateDepthPC.copy()
    _RGBCoordinateDepthPC = RGBCoordinateDepthPC.copy()
    
    tolerance = 1e-6
    ConditionOne = (
        (np.abs(_RGBCoordinateDepthPC[:,2]) > tolerance) &
        (~np.isnan(_RGBCoordinateDepthPC[:, 1])) & 
        (~np.isnan(_RGBCoordinateDepthPC[:, 2]))
    )
    _RGBCoordinateDepthPC = _RGBCoordinateDepthPC[ConditionOne]#filter the z close to 0
    _DepthCoordinateDepthPC = _DepthCoordinateDepthPC[ConditionOne]

    DepthImageWidth = RGBImageWidth / scale
    DepthImageHeight = RGBImageHeight / scale

    TempRgbCoordinateDepthImage = np.dot(RGBK_matrix, _RGBCoordinateDepthPC.T).T  # nx3

    RgbCoordinate_DepthImageUsedRGBPCDepth = np.zeros((round(DepthImageHeight), round(DepthImageWidth)), dtype=np.float32)
    RgbCoordinate_DepthImageUsedDepthPCDepth = np.zeros((round(DepthImageHeight), round(DepthImageWidth)), dtype=np.float32)

    TempRgbCoordinateDepthImage /= (TempRgbCoordinateDepthImage[:,2]* scale).reshape((-1,1))#最后的真depth不变

    ConditionTwo = (
        (TempRgbCoordinateDepthImage[:, 0] >= 0) &
        (TempRgbCoordinateDepthImage[:, 0] < DepthImageWidth) &
        (TempRgbCoordinateDepthImage[:, 1] >= 0) &
        (TempRgbCoordinateDepthImage[:, 1] < DepthImageHeight)
    )
    _RGBCoordinateDepthPC = _RGBCoordinateDepthPC[ConditionTwo]
    TempRgbCoordinateDepthImage = TempRgbCoordinateDepthImage[ConditionTwo]
    _DepthCoordinateDepthPC = _DepthCoordinateDepthPC[ConditionTwo]

    PaintedPC = np.zeros((_DepthCoordinateDepthPC.shape[0],6),dtype= np.float32)
    PaintedPC[:,:3] = _RGBCoordinateDepthPC[:,:3]#不管下面做什么操作，这里都是要用RGB的点云，跟DepthPC中的深度无关
    #astype不够大，就会发生截断
    x = TempRgbCoordinateDepthImage[:, 0].astype(np.int16)
    y = TempRgbCoordinateDepthImage[:, 1].astype(np.int16)
    RgbCoordinate_DepthImageUsedDepthPCDepth[y,x] = _DepthCoordinateDepthPC[:, 2]
    RgbCoordinate_DepthImageUsedRGBPCDepth[y,x] = _RGBCoordinateDepthPC[:, 2]

    if UndistortedRGB is not None:
        _UndistortedRGB = cv2.cvtColor(UndistortedRGB, cv2.COLOR_BGR2RGB)
        PaintedPC[:,3:] = _UndistortedRGB[y,x,:3]

    return RgbCoordinate_DepthImageUsedDepthPCDepth,RgbCoordinate_DepthImageUsedRGBPCDepth,PaintedPC


def SavePaintedPC(path, folder, FileName, PaintedPC):
    FilePath = f"{path}/{folder}/{FileName}.obj"
    np.savetxt(FilePath, PaintedPC[:, :6], fmt="v %.6f %.6f %.6f %.6f %.6f %.6f", delimiter=" ")
        


def DepthPC2RGBDepthPC(DepthPC):
    Depth2RGBPCExtisics = np.ones(shape=(3, 4), dtype=np.float64)
    Depth2RGBPCExtisics[:3, :3] = np.array([[0.999999    ,  0.000993896, -0.000302358],
                                            [-0.000963703,  0.996181   ,  0.0873081  ],
                                            [0.000387978 , -0.0873078  ,  0.996181   ]], dtype=np.float64)
    Depth2RGBPCExtisics[:, 3] = np.array([-0.0319602, -0.00235863, 0.00389216], dtype=np.float64)
    ones = np.ones((DepthPC.shape[0], 1), dtype=np.float64)
    DepthPC = np.hstack((DepthPC, ones)).T  # 4xn
    RGBDepthPC = np.dot(Depth2RGBPCExtisics, DepthPC).T  # nx4 musr preserve the shape of the pc is nx3
    return RGBDepthPC

def SaveImage(path,folder,FileName,OutputImage):
    cv2.imwrite(path + "/" + folder + "/" + FileName, OutputImage)


def DepthImageUndistortNotInterplot(DepthImage,Intrisics,distortion):
    DepthPC = Depth2PC(Intrisics, DepthImage)
    DepthPCNormailized = DepthPC / DepthPC[:,2].reshape(-1,1)

    X = DepthPCNormailized[:,0].reshape(-1,1)
    Y = DepthPCNormailized[:,1].reshape(-1,1)
    XSquare = np.power(X, 2).reshape(-1,1)
    YSquare = np.power(Y, 2).reshape(-1,1)
    Rdistance = (np.sqrt(XSquare + YSquare)).reshape(-1,1)
    ParaOne = np.ones((Rdistance.shape[0],1)) + distortion[0] * np.power(Rdistance ,2) + distortion[1] * np.power(Rdistance ,4) + distortion[4] * np.power(Rdistance ,6)
    ParaOne = ParaOne.reshape(-1,1)

    XY = (X * Y).reshape(-1,1)
    UndistortedXY = np.ones((X.shape[0],3),dtype=np.float64)

    UndistortedXY[:,0] = (X * ParaOne + 2 * distortion[2] * XY + distortion[3] * (np.power(Rdistance ,2)+ 2 * np.power(X ,2))).flatten()
    UndistortedXY[:,1] = (Y * ParaOne + distortion[2] * (np.power(Rdistance ,2) + 2 * np.power(Y ,2)) + 2 * distortion[3] * XY).flatten()
    UndistortedXY = np.dot(Intrisics , UndistortedXY.T).T

    height, width = DepthImage.shape[0],DepthImage.shape[1]
    UndistortedDepthImage = np.zeros((height,width),dtype=np.uint16)

    condition = (
        (UndistortedXY[:,1] >=0) &
        (UndistortedXY[:,1] < height) &
        (UndistortedXY[:,0] >=0) &
        (UndistortedXY[:,0] < width)
    )
    UndistortedXY = UndistortedXY[condition]
    DepthPC = DepthPC[condition]
    for index in np.arange(int(UndistortedXY.shape[0])):
        UndistortedDepthImage[int(UndistortedXY[index,1]),int(UndistortedXY[index,0])] = DepthPC[index,2]
    return UndistortedDepthImage

def DepthImageUndistortWithInterplot(DepthImage,Intrisics,distortion):
    h, w = DepthImage.shape[:2]
    #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(Intrisics, distortion, (w, h), 1, (w, h))
    #you don't use the upper thing, the code will be correct
    mapx, mapy = cv2.initUndistortRectifyMap(Intrisics, distortion, None, None, (w, h), 5)
    depth_undistorted = cv2.remap(DepthImage, mapx, mapy, interpolation=cv2.INTER_NEAREST)
    # x, y, w, h = roi
    # depth_undistorted = depth_undistorted[y:y + h, x:x + w]
    return depth_undistorted


def BagDataSave(BagPath,output_path, folder_index: str, topics):

    BagData, depth_image_info, depth_image_raw, rgb_image_info, rgb_image_raw = ReadBagMsg(
        BagPath, *topics)

    image_num = 1  # use for save image,it means saved image index
    OutputPath = output_path + "/" + folder_index + "/"
    for (depthinfo_topic, depthinfo_msg, depthinfo_t),\
            (depthimage_topic, depthimage_msg, depthimage_t),\
            (rgbinfo_topic, rgbinfo_msg, rgbinfo_t),\
            (rgbimage_topic, rgbimage_msg, rgbimage_t) \
                in tqdm(zip(depth_image_info, depth_image_raw, rgb_image_info, rgb_image_raw)):
            scale = 1.0  # 缩小四倍，这样跟原来的depth就会比较接近
            

            # rgb process
            rgbK_matrix, rgbD_matrix = GetKD_FromMsg(rgbinfo_msg)
            rgb_cv_image = CvBridge().imgmsg_to_cv2(
                rgbimage_msg, desired_encoding="passthrough")
            rgbimage_width, rgbimage_height = rgb_cv_image.shape[1], rgb_cv_image.shape[0]
            undistorted_rgb_image = cv2.undistort(
                src=rgb_cv_image, cameraMatrix=rgbK_matrix, distCoeffs=rgbD_matrix)
            undistorted_rgb_image = cv2.resize(src=undistorted_rgb_image, dsize=(
                round(rgbimage_width / scale), round(rgbimage_height / scale)), interpolation=cv2.INTER_AREA)  # 缩放到原来的1/4，也就是1024x768
            # undistorted_rgb_image = cv2.cvtColor(undistorted_rgb_image, cv2.COLOR_BGR2RGB)
            # rgb_cv_image = cv2.cvtColor(rgb_cv_image, cv2.COLOR_BGR2RGB)
            # rgb process done
            if image_num == 1:
                with open(output_path + "/metafile.txt", "+a") as metafile:
                    metafile.write("rgbK_matrix\n")
                    np.savetxt(metafile, rgbK_matrix, fmt="%f", delimiter=",")
                    metafile.write("rgb height width: {} {}\n".format(rgbimage_height, rgbimage_width))
            depthK_matrix, depthD_matrix = GetKD_FromMsg(depthinfo_msg)
            DepthImageFromBuffer = DepthFromBuffer(depthimage_msg)
            depthimage_width, depthimage_height = DepthImageFromBuffer.shape[
                1], DepthImageFromBuffer.shape[0]
            UndistoredDepthImageWithInterplot = DepthImageUndistortWithInterplot(
                DepthImageFromBuffer, depthK_matrix, depthD_matrix)
            DepthPC = Depth2PC(depthK_matrix, UndistoredDepthImageWithInterplot)

            RGBCoordinateDepthPC = DepthPC2RGBDepthPC(DepthPC)

            DepthUsedFlag = 1

            RgbCoordinate_DepthImageUsedDepthPCDepth, RgbCoordinate_DepthImageUsedRGBPCDepth, PaintedPC \
                = DepthPC2Plane(rgbK_matrix.astype(np.float32), RGBCoordinateDepthPC.astype(np.float32), rgbimage_width, rgbimage_height, scale, DepthPC.astype(np.float32), undistorted_rgb_image)
            # the depth X 10
            RgbCoordinate_DepthImageUsedRGBPCDepth = RgbCoordinate_DepthImageUsedRGBPCDepth
            RgbCoordinate_DepthImageUsedDepthPCDepth = RgbCoordinate_DepthImageUsedDepthPCDepth
            if image_num == 1:
                with open(output_path+"/0/metafile.txt", "+a") as metafile:
                    metafile.write("depthK_matrix\n")
                    np.savetxt(metafile, depthK_matrix, fmt="%f", delimiter=",")
                    metafile.write("rgb height width: {} {}\n".format(depthimage_height, depthimage_width))
            SavePaintedPC(OutputPath, "PaintedPC",
                        "{:06d}-PaintedPC".format(image_num), PaintedPC)
            SaveImage(OutputPath, "ColorUndistort",
                    "{:06d}-color.png".format(image_num), undistorted_rgb_image[:,:,:3])
            SaveImage(OutputPath, "colorDistort",
                      "{:06d}-DistortedRGBImage.png".format(image_num), rgb_cv_image[:, :, :3])
            SaveImage(OutputPath, "depthUsedDpethPCDepth", "{:06d}-DepthUsedDpethPCDepth.png".format(
                image_num), RgbCoordinate_DepthImageUsedDepthPCDepth.astype(np.uint16))
            SaveImage(OutputPath, "depthUsedRGBPCDepth", "{:06d}-depth.png".format(
                image_num), RgbCoordinate_DepthImageUsedRGBPCDepth.astype(np.uint16))
            image_num += 1

    BagData.close()
    

def RosBagSaveTime(BagPath, output_path, folder_index: str, topics):
    BagData,time_stamp = ReadBagMsg(BagPath, *topics)
    print()
    OutputPath = output_path + "/" + folder_index + "/" + "time.txt"
    image_num = 1  # use for save image,it means saved image index
    with open(OutputPath, 'a+') as file:
        for (topic, msg, t) in tqdm(time_stamp):
            file.write(f"{t}\n")
    BagData.close()


def ShowBagInfo(bag_path):
    BagData = rosbag.Bag(bag_path)
    print(BagData.get_type_and_topic_info())
    BagData.close()
#use time stamp to paint point cloud


def num2color(num):
    # cast float32 to int so that bitwise operations are possible
    s = struct.pack('>f', num)
    i = struct.unpack('>l', s)[0]
    # you can get back the float value by the inverse operations
    pack = ctypes.c_uint32(i).value
    r = (pack & 0x00FF0000) >> 16
    g = (pack & 0x0000FF00) >> 8
    b = (pack & 0x000000FF)
    return np.array([r, g, b])


def SavePCFromRosBag(bag_path,output_path,camera_num):

    for index in np.arange(camera_num):
        PC_num = 0
        topics = f'/cam{index}/points2'
        # 读取rosbag文件
        bag = rosbag.Bag(bag_path)
        # 设置保存点云的文件名和格式
        output_file = output_path + "/" + str(index) + "/" + f'camera{index}index'

        # 创建Open3D点云对象
        point_cloud = o3d.geometry.PointCloud()
        # 遍历rosbag中的消息
        for topic, msg, t in bag.read_messages(topics):
            # 检查消息类型是否为sensor_msgs/PointCloud2
            if msg._type == 'sensor_msgs/PointCloud2':
                # 解包点云数据
                pc = pc2.read_points(msg, skip_nans=True)
                points = np.array(list(pc))
                points = remove_nan_rows(points)
                # print(points.shape)
                # 添加点云数据到Open3D点云对象
                point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
                color = np.zeros((points.shape[0], 3))
                for index in np.arange(points.shape[0]):
                    color[index] = num2color(points[index][3])
                point_cloud.colors = o3d.utility.Vector3dVector(color / 255.0)
                o3d.io.write_point_cloud(output_file + str(PC_num) + ".ply", point_cloud)
                PC_num += 1

        bag.close()


def remove_nan_rows(arr):
    # 检测数组中的NaN值
    nan_mask = np.isnan(arr).any(axis=1)

    # 使用布尔索引过滤数组，去掉带有NaN值的行
    arr_without_nan = arr[~nan_mask]

    return arr_without_nan



if __name__ == "__main__":

    ProgramInit()
    rgb_img_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/banana/cam0/rgb/image_raw/0.png"
    depth_img_path = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/banana/cam0/depth_to_rgb/image_raw/0.png"
    depth_img = cv2.imread(depth_img_path,cv2.IMREAD_UNCHANGED)
    RGB_img = cv2.imread(rgb_img_path,cv2.IMREAD_UNCHANGED)
    rgbimage_width, rgbimage_height = RGB_img.shape[1], RGB_img.shape[0]

    undistorted_rgb_image = cv2.undistort(
        src=RGB_img, cameraMatrix=rgbK_matrix, distCoeffs=rgbD_matrix)
    undistorted_rgb_image = cv2.resize(src=undistorted_rgb_image, dsize=(
        round(rgbimage_width / scale), round(rgbimage_height / scale)), interpolation=cv2.INTER_AREA)  # 缩放到原来的1/4，也就是1024x768



    # folder = "/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF"
    # bagpath = folder + "/" +  "meal_spoon_0_20230921.bag"
    # outputpath = folder + "/" + "RGBD_PC"
    # # bagpath = "/media/tony/新加卷/test/test_duck_20230914.bag"
    # # outputpath = "/media/tony/新加卷/test/"
    # predix = "/cam"
    # post = ['/depth/camera_info', '/depth/image_raw',
    #         '/rgb/camera_info', '/rgb/image_raw']
    # # for index in np.arange(0, 4):
    # topics = []
    # index = 3
    # # for index in np.arange(4):
    # topics = [predix + str(index) + item for item in post]
    # BagDataSave(bagpath,outputpath,str(index),topics)


    # ProgramInit()
    # output_path = "/media/tony/新加卷/Bags2InitialPose/NewData/"
    # post_name = '/points2'
    # predix = '/cam'
    # cam_num = 4
    # BagPath = "/media/tony/新加卷/Bags2InitialPose/test_flowerBread__20230912.bag"
    # SavePCFromRosBag(BagPath, output_path,cam_num)
    # for index in np.arange(cam_num):
    #     topics = [predix + str(index) + post_name]
    #     RosBagSaveTime(BagPath, output_path, str(index), topics)

    # quaternion = np.array([-0.3203322686471465,  -0.017645953375179612,
    #                        0.9138836939413971, 0.24878193652449973])
    # translation = [0.21404719352722168,  -
    #                0.18618075549602509, 1.1920537948608398]
