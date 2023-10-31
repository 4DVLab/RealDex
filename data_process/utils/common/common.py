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


def seven_num2matrix(translation,roatation):#translation x,y,z rotation x,y,z,w
    transform_matrix = np.identity(4)
    transform_matrix[:3,:3] = Rotation.from_quat(roatation).as_matrix()
    transform_matrix[:3,3] = translation
    return transform_matrix

def find_time_closet(slot,time_stamps):
    diff = np.abs(time_stamps - slot)
    index = np.argmin(diff)
    return index