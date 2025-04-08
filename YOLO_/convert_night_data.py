import cv2
import numpy as np
import albumentations as A
import os

night_augmentations = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), contrast_limit=(-0.1, 0.1), p=1.0),  # 适当降低亮度
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=-30, val_shift_limit=-50, p=0.8),  # 适当调整颜色

])

# 处理单张图片
def simulate_night(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 处理颜色格式
    augmented = night_augmentations(image=image)['image']
    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, augmented)

# 处理整个训练集
input_folder = "D:/RL_Terraria/ultralytics-main/dataset/images/train_bg0/"
output_folder = "D:/RL_Terraria/ultralytics-main/dataset/images/train_night/"

os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        simulate_night(os.path.join(input_folder, filename), os.path.join(output_folder, filename))

print("✅ 夜晚增强数据已生成！")
