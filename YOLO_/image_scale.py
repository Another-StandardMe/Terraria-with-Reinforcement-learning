import cv2
import os
import numpy as np

def letterbox_resize(image, target_size=(224, 224)):  # 这里可以改成 224x224 或 256x256
    """使用 letterbox 方式缩放图片，保持比例，并填充到固定大小"""
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)  # 计算缩放比例
    nh, nw = int(h * scale), int(w * scale)  # 缩放后的尺寸
    image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # 创建新的黑色背景（默认填充黑色，你可以改成白色）
    canvas = np.full((target_size[0], target_size[1], 3), (128, 128, 128), dtype=np.uint8)

    # 计算填充区域
    top = (target_size[0] - nh) // 2
    left = (target_size[1] - nw) // 2

    # 将缩放后的图片放到 canvas 中
    canvas[top:top+nh, left:left+nw] = image_resized
    return canvas

# 批量处理图片
input_folder = "D:/RL_Terraria/ultralytics-main/dataset/images/train/"
output_folder = "D:/RL_Terraria/ultralytics-main/dataset/images/train_resized/"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        resized_image = letterbox_resize(image, target_size=(224, 224))  # 调整到 224x224
        cv2.imwrite(os.path.join(output_folder, filename), resized_image)

print("✅ 训练图片已全部调整到统一大小！")
