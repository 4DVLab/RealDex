import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import os


cam_index = 0
folder = "/media/tony/新加卷/test_data/test/test_1"
folder = Path(folder)
rgb_folder = folder / Path(f"cam{cam_index}/rgb/image_raw")
depth_folder = folder / Path(f"cam{cam_index}/depth_to_rgb/image_raw")
# fig = plt.figure()
#要么就用这种办法，一次在整个画布上画一幅画，要么就是用subplot创建一个画布还有很多个绘画空间，在各个空间上单独画画

object_folder = folder / Path(f"icg_capture_image")

save_folder = folder / Path("objecct_merge_with_rgb/depth_merge")
os.makedirs(save_folder,exist_ok=True)
for index in np.arange(0,100):
    depth_img_path = rgb_folder / Path(f"{index}.png")
    object_img_path = object_folder / Path(f"{index}.png")
    rgb_img = cv2.imread(str(depth_img_path), -1)
    # rgb_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2RGB)

    hand_arm_img = cv2.imread(str(object_img_path), -1)
    hand_arm_img = cv2.cvtColor(hand_arm_img,cv2.COLOR_BGR2RGB)

    result = cv2.addWeighted(rgb_img, 0.6, hand_arm_img, 0.4, 0.0)

    save_path = save_folder / Path(f"icg_result_rgb_{index}.png")
    plt.imsave(save_path,result)
    

#     plt.imshow(result)
#     plt.pause(0.001)  
#     # plt.show()#使用这个的话就一次只能出一个窗口，要看下一个，就要重新开一个
#     fig.clf()

# plt.close(fig)