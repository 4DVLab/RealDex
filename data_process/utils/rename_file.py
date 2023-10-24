import os,cv2

def rename_files(folder_path):
    file_list = os.listdir(folder_path)
    for filename in file_list:
        full_path = os.path.join(folder_path, filename)
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)[:,:,:3]
        cv2.imwrite(full_path, image)
        new_name = '{:06d}-color.png'.format(int(os.path.splitext(filename)[0]))
        new_path = os.path.join(folder_path, new_name)
        os.rename(full_path, new_path)

# 指定文件夹路径
folder_path = '/home/tony/mine/Projects/ArmHandVis/HandVersion/HandArmFiles/ARM_HAND_URDF/banana/just_get_objct_transform/data/color/'

# 调用函数进行文件重命名
rename_files(folder_path)