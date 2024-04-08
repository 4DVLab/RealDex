import cv2
import dlib
import os
import re
from tqdm import tqdm
def apply_mosaic(image, bbox, intensity=15):
    (x, y, w, h) = bbox
    # 裁剪人脸区域
    face = image[y:y + h, x:x + w]
    # 降低人脸区域的分辨率
    if w < intensity or h < intensity:
        return image
    face = cv2.resize(face, (w // intensity + 1, h // intensity + 1), interpolation=cv2.INTER_LINEAR)
    # 放大回原来的尺寸
    face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
    # 替换原图中的人脸区域
    image[y:y + h, x:x + w] = face
    return image

def mosaic_face(image_path):
    image = cv2.imread(image_path)

    detector = dlib.get_frontal_face_detector()
    faces = detector(image, 1)

    for face in faces:
        x, y = face.left(), face.top()
        w, h = face.right() - x, face.bottom() - y
        image = apply_mosaic(image, (x, y, w, h))
    
    cv2.imwrite(image_path, image)
    
    if len(faces) == 0:
        return False
    else:
        return True
    
if __name__ == "__main__":
    base_dir = "/storage/group/4dvlab/youzhuo/bags"
    model_name_list = os.listdir(base_dir)
    
    model_name_list = ["sprayer", "yogurt",
                       "toilet_cleaning_sprayer", "elephant_watering_can"]
    
    for model_name in model_name_list:
        path = os.path.join(base_dir, model_name)
        if not os.path.isdir(path):
            continue
        for exp_code in os.listdir(path):
            subpath = os.path.join(path, exp_code)
            if os.path.isdir(subpath) and re.match(rf"{model_name}_\d+", exp_code):
                data_dir = os.path.join(base_dir, model_name, exp_code)
                for folder_name in ["cam2"]:
                    image_dir = os.path.join(data_dir, folder_name, "rgb/image_raw")
                    print(image_dir)
                    for file in tqdm(os.listdir(image_dir)):
                        if not file.endswith("jpg"):
                            continue
                        exist_face = mosaic_face(image_path=os.path.join(image_dir, file))
                        # if not exist_face:
                        #     break