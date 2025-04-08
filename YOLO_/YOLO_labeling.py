# This code converts YOLOv11 detection results into YOLO-format TXT labels.
# It processes images from a dataset, detects objects, and saves normalized bounding box coordinates.
# The output labels are stored in the specified directory for training YOLO models.

from ultralytics import YOLO
import cv2
import os

model = YOLO("E:/terraria_project/after_training_weight/7000best.pt")

class_mapping = {
    "Eye_of_Cthulhu": 0,
    "King_Slime": 1,
    "Player": 2,
    "Cursor": 3
}

input_folder = "E:/terraria_project/test"
output_folder = "E:/terraria_project/test"

# input_folder = "E:/terraria_project/GAME_ENV_DATASET/images/train"
# output_folder = "E:/terraria_project/GAME_ENV_DATASET/labels/train"

# input_folder = "E:/terraria_project/GAME_ENV_DATASET/images/val"
# output_folder = "E:/terraria_project/GAME_ENV_DATASET/labels/val"

# input_folder = "E:/terraria_project/daytime/all_img"
# output_folder = "E:/terraria_project/daytime/all_labels"


os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    if img_name.endswith(".png"):
        img_path = os.path.join(input_folder, img_name)
        results = model(img_path)

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        txt_filename = os.path.join(output_folder, img_name.replace(".png", ".txt"))

        with open(txt_filename, "w") as f:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())

                    if cls_id in class_mapping.values():
                        x_center = (x1 + x2) / 2 / img_w
                        y_center = (y1 + y2) / 2 / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h

                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        print(f"{img_name} processed!")

print("completed!")


