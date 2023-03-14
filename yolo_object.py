import towhee
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps

yolo_model = torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master", 'custom', path="yolov5s.pt", source='local')
# yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s")

def yolo_detect(img):
    results = yolo_model(img)

    pred = results.pred[0][:, :4].cpu().numpy()
    boxes = pred.astype(np.int32)

    max_img = get_object(img, boxes)
    return max_img


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


img = Image.open(r"D:\Code\ML\images\Mywork3\train_data\train\1\IMG_5024.JPG").convert("RGB")
img = ImageOps.exif_transpose(img)
results = yolo_model(img)
results.show()
# print(img)
# result = yolo_detect(img)

# result.show()

dc = (
    towhee.glob['path'](r"D:\Code\ML\images\test02\test\prizm\26-1.jpg")
    .runas_op['path', "object"](yolo_detect)

)

print(dc)

