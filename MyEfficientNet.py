import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import timm


class MyEfficient:
    def __init__(self, model_dict_path, out_features=2560):
        self.out_features = out_features
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.efficientnet_b7(pretrained=True)
        # self.model.fc = torch.nn.Linear(in_features=2048, out_features=self.out_features)
        # self.model.load_state_dict(torch.load(model_dict_path))

        # 自定义模型
        # print(list(self.model.children()))
        features = list(self.model.children())[:-1]  # 去掉最后一部分
        self.model = torch.nn.Sequential(*features).to(self.device)

        self.model.eval()
        # self.model.to(self.device)

    def inference_transform(self):
        inference_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        return inference_transform

    def img_transform(self, img_rgb, transform=None):
        # 将数据转换为模型读取的形式
        if transform is None:
            raise ValueError("找不到transform！必须有transform对img进行处理")

        img_t = transform(img_rgb)
        return img_t

    def get_model(self):
        return self.model

    # 输出图片路径或者cv2格式的图片数据
    def predict(self, img):
        if type(img) == type('path'):
            img = Image.open(img).convert('RGB')

        transform = self.inference_transform()

        img_tensor = transform(img)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(self.device)
        # print(img.shape)

        with torch.no_grad():
            outputs = self.model(img_tensor)
        return outputs.reshape(2560).cpu().numpy()
