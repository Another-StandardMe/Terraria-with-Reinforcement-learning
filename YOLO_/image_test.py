from ultralytics import YOLO
import os
from PIL import Image

def save_yolo_labels(result, image_path, label_dir, index):
    """
    保存 YOLO 检测框为 txt 标签格式
    """
    boxes = result.boxes
    img_name = os.path.basename(image_path)
    base_name = os.path.splitext(img_name)[0]
    txt_name = f"{base_name}_det{index}.txt"
    txt_path = os.path.join(label_dir, txt_name)

    w, h = Image.open(image_path).size

    with open(txt_path, 'w') as f:
        for box in boxes:
            cls = int(box.cls[0])
            xywh = box.xywh[0]
            x_center, y_center, width, height = xywh.tolist()
            x_center /= w
            y_center /= h
            width /= w
            height /= h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == '__main__':
    model = YOLO(model='E:/terraria_project/test/best.pt')
    image_path = 'E:/terraria_project/test/YOLO1.png'

    save_dir = 'E:/terraria_project/test'
    label_dir = os.path.join(save_dir, 'labels1')
    image_out_dir = os.path.join(save_dir, 'images1')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    for i in range(10):
        results = model(image_path)

        for result in results:
            # 保存 txt 标注
            save_yolo_labels(result, image_path, label_dir, i)

            # 保存图像（加编号）
            img_name = os.path.basename(image_path)
            base_name, ext = os.path.splitext(img_name)
            save_path = os.path.join(image_out_dir, f"{base_name}_det{i}{ext}")
            result.save(filename=save_path)
