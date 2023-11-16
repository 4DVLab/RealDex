import rosbag
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os




def convert_dict_values_to_float(dictionary):
    for key in dictionary.keys():
        dictionary[key] = [[np.float128(value) for value in list_item] for list_item in dictionary[key]]


def write_dict_to_files(dictionary,write_path):

    for key, value in dictionary.items():
        filename = write_path + str(key) + ".txt"  # 以键名作为文件名，添加.txt扩展名
        with open(filename, 'w+') as file:
            for sublist in value:
                line = ' '.join(map(str, sublist))  # 将第二层列表中的元素转换为字符串，并以空格分隔
                file.write(line + '\n')  # 写入文件，每行末尾添加换行符



def extract_rosbag_tf(folder_path,bag_name):
    print("begin to extract the rosbag tf")
    folder_path = folder_path + "/" + bag_name
    bag_path = folder_path + "/" + bag_name + ".bag"
    bag_data = rosbag.Bag(bag_path, "r")
    output_path =  folder_path + "/TF/"
    if not os.path.exists(output_path) :#如果这个目录已经生成过一次，那么就不要再生成了
        os.mkdir(output_path)  # 创建目录
        #info = bag_data.get_type_and_topic_info()
        bag_data = bag_data.read_messages(topics=['/tf', '/tf_static'])
        data_write = defaultdict(list)
        link_names = set()
        #base on the topic cam0/rgb_image raw
        for topic, msg, t in tqdm(bag_data):
            t = np.int64(str(t))
            for transform in msg.transforms:

                frame_id = transform.header.frame_id.replace('/','')
                chind_frame_id = transform.child_frame_id.replace('/','')
                link_names.add(frame_id)
                link_names.add(chind_frame_id)
                dict_key = frame_id + '-' + chind_frame_id
                data_write[dict_key].append([            
                    t,
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,])
        convert_dict_values_to_float(data_write)
        print(output_path)
        write_dict_to_files(data_write,output_path)


