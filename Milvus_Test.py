import towhee
import cv2
from towhee._types.image import Image
import os
import PIL.Image as Image
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from MyModel import MyModel
from MyEfficientNet import MyEfficient
import torch
from transformers import ViTFeatureExtractor, ViTModel
from towhee.types.image_utils import to_image_color

connections.connect(host='127.0.0.1', port='19530')
dataset_path = ["D:\Code\ML\images\Mywork3\card_database_yolo/*/*/*/*"]

img_id = 0
vec_num = 0
myModel = MyModel(r"D:\Code\ML\model\card_cls\res_card_out764_freeze4.pth", out_features=764)
# myModel = MyModel(r"C:\Users\Administrator\.cache\torch\hub\checkpoints\resnet50-0676ba61.pth", out_features=1000)

# myModel = MyEfficient('')


yolo_model = torch.hub.load(r"C:\Users\Administrator\.cache\torch\hub\ultralytics_yolov5_master", 'custom',
                            path="yolov5s.pt", source='local')


# yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s")


# 生成ID
def get_id(param):
    global img_id
    img_id += 1
    return img_id


# def eff_enbedding(img):
#     global vec_num
#     vec_num += 1
#     print('vec: ', vec_num)
#     return myModel.predict(img)

# 生成向量
def img2vec(img):
    global vec_num
    vec_num += 1
    print('vec: ', vec_num)
    return myModel.predict(img)


# 生成信息
path_num = 0


def get_info(path):
    path = os.path.split(path)[0]

    path, num_and_player = os.path.split(path)
    num = num_and_player.split(' ')[0]
    player = ' '.join(os.path.split(num_and_player)[-1].split(' ')[1:])
    path, year = os.path.split(path)
    series = os.path.split(path)[1]
    rtn = "{} {} {} #{}".format(series, year, player, num)

    global path_num
    path_num += 1
    print(path_num, " loading " + rtn)
    return rtn


def read_imgID(results):
    imgIDs = []
    for re in results:
        # 输出结果图片信息
        print('---------', re)
        imgIDs.append(re.id)
    return imgIDs


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


# 创建向量数据库
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='img_id', dtype=DataType.INT64, is_primary=True),
        FieldSchema(name='path', dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="info", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, descrition='image embedding vectors', dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='reverse image search')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': 'L2',
        'index_type': "IVF_FLAT",
        'params': {"nlist": dim}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


# 判断是否加载已有数据库，或新创建数据库
def is_creat_collection(have_coll, collection_name):
    if have_coll:
        # 连接现有的数据库
        collection = Collection(name=collection_name)
    else:
        # 新建立数据库
        collection = create_milvus_collection(collection_name, 2048)
        dc = (
            towhee.glob['path'](*dataset_path)
            .runas_op['path', 'img_id'](func=get_id)
            .runas_op['path', 'info'](func=get_info)
            # .image_decode['path', 'img']()
            # .runas_op['path', "object"](yolo_detect)
            .runas_op['path', 'vec'](func=img2vec)
            .tensor_normalize['vec', 'vec']()
            # .image_embedding.timm['img', 'vec'](model_name='resnet50')
            .ann_insert.milvus[('img_id', 'path', 'info', 'vec'), 'mr'](collection=collection)
        )

    print('Total number of inserted data is {}.'.format(collection.num_entities))
    return collection


# 通过ID查询
def query_by_imgID(collection, img_id, limit=1):
    expr = 'img_id == ' + str(img_id)
    res = collection.query(expr, output_fields=["path", "info"], offset=0, limit=limit, timeout=2)
    return res


def from_path_get_series(path):
    for i in range(3):
        path = os.path.split(path)[0]
    series = os.path.split(path)[-1]

    return series


if __name__ == '__main__':
    print('start')

    # 是否存在数据库
    have_coll = False

    # 默认模型
    # collection = is_creat_collection(have_coll=have_coll, collection_name="reverse_image_search")
    # 自定义模型
    collection = is_creat_collection(have_coll=have_coll, collection_name="reverse_image_search_myModel")

    # 测试的图片路径
    img_path = ["D:/Code/ML/images/test02/test2/*/*/*/*"]

    data = (towhee.glob['path'](*img_path)
            # image_decode['path', 'img']().
            .runas_op['path', "object"](yolo_detect)
            .runas_op['object', 'vec'](func=img2vec)
            .tensor_normalize['vec', 'vec']()
            # image_embedding.timm['img', 'vec'](model_name='resnet50').
            .ann_search.milvus['vec', 'result'](collection=collection, limit=3)
            .runas_op['result', 'result_imgID'](func=read_imgID)
            .select['path', 'result_imgID', 'vec']()
            )

    print(data)

    collection.load()
    # res = query_by_imgID(collection, data[0].result_imgID[0])
    #
    # print(res[0])


    top3_num = 0
    top1_num = 0
    test_img_num = len(list(data))

    # 查询所有测试图片
    for i in range(test_img_num):
        top3_flag = False

        # 获取图片真正的系列
        source_card_series = from_path_get_series(data[i].path)
        # 获取图片真正的编号
        source_num = os.path.split(os.path.split(data[i].path)[0])[-1].split('#')[-1]

        # 每个测试图片返回三个最相似的图片ID，一一测试
        for j in range(3):
            res = query_by_imgID(collection, data[i].result_imgID[j])

            # 获取预测的图片的系列
            result_card_series = from_path_get_series(res[0]['path'])
            # 获取预测的图片的编号
            result_num = os.path.split(os.path.split(res[0]['path'])[0])[-1].split(' ')[0].split('#')[-1]

            # 判断top1是否正确
            if j == 0 and source_num == result_num and source_card_series == result_card_series:
                top1_num += 1

            # top3中有一个正确的标记为正确
            if source_num == result_num and source_card_series == result_card_series:
                top3_flag = True

            # 日志
            if j == 0 and source_num == result_num and source_card_series == result_card_series:
                print(top1_num)
            elif j == 0:
                print('top_1 错误')
            print("series: {}, num: {} === result - series: {}, num: {}".format(
                source_card_series, source_num, result_card_series, result_num
            ))

        if top3_flag:
            top3_num += 1

        print("====================================")

    print("测试图片共: ", test_img_num)
    top1_accuracy = (top1_num / test_img_num) * 100
    top3_accuracy = (top3_num / test_img_num) * 100

    print("top3 准确率:{} % \n top1 准确率: {} %".
          format(top3_accuracy, top1_accuracy))

'''
 测试图片共:  168
 自定义resnet50_freeze_out421 + yolo + normalize
top3 准确率:96.42857142857143 % 
 top1 准确率: 95.23809523809523 %


测试图片: 773, 数据库图片: 5848
自定义resnet50_freeze_out421 + yolo + normalize
测试图片共:  773
top3 准确率:96.63648124191462 % 
 top1 准确率: 95.60155239327295 %

 
 测试图片: 773, 数据库图片: 5848
 自定义resnet50_out764_freeze + yolo + normalize
top3 准确率:96.76584734799482 % 
 top1 准确率: 96.50711513583441 %
'''
