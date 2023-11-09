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


def sequence_split(root_path, split_file):
    root_path = Path(root_path)
    split_file = Path(split_file)
    for folder in os.listdir(root_path):
        with open(split_file ) as f:
            




    