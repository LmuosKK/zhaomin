import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import joblib

# 定义MLP模型结构
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载模型权重
def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model.eval()

# 转换图像到模型输入所需的格式
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

# 主函数，定义Streamlit应用
def main():
    st.title('AI图像分类应用')

    # 文件上传器
    uploaded_file = st.file_uploader("请选择图像文件", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # 显示上传的图像
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的图像', use_column_width=True)

        # 加载模型
        model_path_mlp = "E:\\cifar-10-python\\mlp_model.pth"
        model_path_knn = "D:\\pythonProject7\\best_HOG_classifier_knn.pkl"

        # 根据用户选择加载相应的模型
        model = MLPModel()
        model = load_model(model_path_mlp, model)

        # 图像预处理
        image_tensor = transform_image(uploaded_file)
        with torch.no_grad():
            # 使用模型进行预测
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # 输出分类结果
        st.write(f"模型预测的类别索引: {predicted.item()}")

if __name__ == "__main__":
    main()