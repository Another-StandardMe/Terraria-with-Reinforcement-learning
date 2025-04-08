from rembg import remove
import cv2
import numpy as np
import os

# 输入和输出文件夹
input_folder = "D:/RL_Terraria/ultralytics-main/dataset/images/train_resized/"
output_folder = "D:/RL_Terraria/ultralytics-main/dataset/images/train_bg0/"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)

        # 读取图片
        image = cv2.imread(image_path)
        with open(image_path, "rb") as f:
            output = remove(f.read())  # 去除背景

        # 保存去背景后的图片
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "wb") as f:
            f.write(output)

print("✅ 背景已移除，图片保存在:", output_folder)
