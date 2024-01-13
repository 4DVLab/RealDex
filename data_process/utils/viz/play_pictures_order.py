import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from PIL import Image

# 图片存储位置和格式
image_folder = '/home/lab4dv/data/bags/apple/apple_1_20231207'  # 图片所在文件夹
image_format = '{}.png'  # 图片命名格式

post_fix_path = "cam0/rgb/image_raw"


# 创建figure对象
fig = plt.figure(figsize=(20,20))

# 初始化一个空的图片对象
im = plt.imshow(Image.open(os.path.join(
    image_folder,post_fix_path, image_format.format(1))), animated=True)

# 播放状态和当前帧变量
playing = True
current_frame = 0


def update(frame):
    global current_frame
    if frame > current_frame:
        frame = current_frame
    if playing:
        try:
            img = Image.open(os.path.join(
                image_folder, post_fix_path, image_format.format(frame)))
            im.set_array(img)
        except FileNotFoundError:
            print(f"图片 {image_format.format(frame)} 未找到")
    else:
        current_frame = frame
    return im,


def on_press(event):
    global playing
    # 如果按下的是空格键，则切换播放状态
    if event.key == 'space':
        playing = not playing


# 创建动画
ani = FuncAnimation(fig, update, frames=range(0, 101), blit=True, interval=1)

# 连接键盘事件
fig.canvas.mpl_connect('key_press_event', on_press)

# 显示动画
plt.show()
