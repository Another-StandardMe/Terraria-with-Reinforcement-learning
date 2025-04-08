# This code implements an aimbot for Terraria using YOLO-based object detection.
# It captures screen images, detects bosses, and returns their coordinates for targeting.
# The system is optimized for real-time performance with high-priority processing.

import time
import os
import psutil
import dxcam
from ultralytics import YOLO
import cv2


class TerrariaAimbot:
    def __init__(self, model_path, region=(10, 120, 1350,1120), target_classes=None):
        """
        初始化 Terraria 目标检测
        :param model_path: YOLOv11 模型路径
        :param region: 截图区域 (left, top, right, bottom)
        :param target_classes: 目标检测类别
        """
        self.camera = dxcam.create(output_color="BGR")  # 直接返回 OpenCV 兼容格式 (BGR)
        self.region = region
        self.model = YOLO(model_path)
        self.target_classes = target_classes or ['Eye_of_Cthulhu', 'King_Slime']
        self.last_boss_pos = None  # 记录上一次的 BOSS 位置

        # 提高 Python 进程优先级，减少 CPU 调度延迟
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)

    def grab_screen(self):
        """使用 dxcam 截取屏幕，返回 OpenCV 格式的图像"""
        img = self.camera.grab(self.region)
        if img is None:
            print("截图失败！")
            return None
        return img

    def detect_objects(self, img):
        """
        运行 YOLOv11 目标检测，返回检测框
        :param img: 截图
        :return: [(类别, x1, y1, x2, y2, 置信度)]
        """
        results = self.model(img)
        boxes = []

        if not results:  # 如果结果为空
            return boxes

        for r in results:
            if not hasattr(r, "boxes"):  # 确保 `boxes` 存在
                continue

            for box in r.boxes:
                cls = r.names[int(box.cls)]  # 获取类别名
                if cls in self.target_classes:  # 只保留指定目标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()  # 置信度
                    boxes.append((cls, x1, y1, x2, y2, conf))

        return boxes

    def boss_pos(self, img):
        """
        获取 BOSS 位置坐标 (中心点)
        :param img: 传入的截图
        :return: (target_x, target_y) 或 None
        """
        if img is None:
            return None

        boxes = self.detect_objects(img)
        if not boxes:
            return None  # 没有检测到目标

        # 获取第一个目标的中心点
        cls, x1, y1, x2, y2, conf = boxes[0]
        target_x = self.region[0] + (x1 + x2) // 2
        target_y = self.region[1] + (y1 + y2) // 2
        self.last_boss_pos = (target_x, target_y)  # 记录上次位置
        return target_x, target_y


if __name__ == "__main__":
    aimbot = TerrariaAimbot("E:/terraria_project/after_training_weight/10000best.pt")

    while True:
        start_time = time.time()

        # 只调用一次截图，提高 FPS
        img = aimbot.grab_screen()
        boss_coords = aimbot.boss_pos(img)



        if boss_coords:
            print(f"BOSS 位置: {boss_coords}")

        elapsed_time = time.time() - start_time
        print(f"time: {elapsed_time:.6f} seconds")

        if img is not None:  # 防止 cv2.imshow() 崩溃
            cv2.imshow("YOLO Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.005)  # 防止 CPU 100% 占用

    cv2.destroyAllWindows()
