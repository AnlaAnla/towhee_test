import towhee
import torch
import os
import glob
from PIL import Image, ImageOps
from MyModel import MyModel
import numpy as np

vec_num = 0
yolo_model = torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master", 'custom',
                            path="yolov5s.pt", source='local')

dataset_path = [r"D:\Code\ML\images\test02\test(mosaic,pz)\*\*\*\*"]
yolo_dataset_dir = r"D:\Code\ML\images\test02\test(mosaic,pz)_yolo"


def get_save_dir(save_dir, source_path):
    path02, path01 = os.path.split(source_path)
    path03, path02 = os.path.split(path02)
    path04, path03 = os.path.split(path03)
    path05, path04 = os.path.split(path04)

    return os.path.join(save_dir, path04, path03, path02, path01)


def yolo_detect(img_path):
    dest_path = get_save_dir(yolo_dataset_dir, img_path)
    save_dir = os.path.split(dest_path)[0]

    # 如果已经存在这个yolo检测后的图片
    if os.path.exists(dest_path):
        print("----已经存在 ", dest_path)
        return

    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)
    results = yolo_model(img)

    pred = results.pred[0][:, :4].cpu().numpy()
    boxes = pred.astype(np.int32)

    max_img = get_object(img_path, boxes)



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    max_img.save(dest_path)
    print(dest_path)


def get_object(img, boxes):
    if isinstance(img, str):
        img = Image.open(img)

    if len(boxes) == 0:
        return img

    max_area = 0

    # 选出最大的框
    x1, y1, x2, y2 = 0, 0, 0, 0
    for box in boxes:
        temp_x1, temp_y1, temp_x2, temp_y2 = box
        area = (temp_x2 - temp_x1) * (temp_y2 - temp_y1)
        if area > max_area:
            max_area = area
            x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2

    max_img = img.crop((x1, y1, x2, y2))
    return max_img


data = (towhee.glob['path'](*dataset_path)
        .runas_op['path', ''](yolo_detect)
        )

print('end')
