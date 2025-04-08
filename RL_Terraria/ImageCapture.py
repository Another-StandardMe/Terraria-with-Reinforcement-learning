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
    def __init__(self, region=(20, 120, 1340, 1300), target_classes=None):
        """
        初始化 Terraria 目标检测  190, 30, 1140,980
        :param model_path: YOLOv11 模型路径
        :param region: 截图区域 (left, top, right, bottom)
        :param target_classes: 目标检测类别
        """
        # self.model = YOLO(model_path)
        # self.target_classes = ['King_Slime']
        # self.last_boss_pos = None  # 记录上一次的 BOSS 位置
        """
        初始化 dxcam Aimbot（非线程模式，更安全）
        :param region: 截图区域 (left, top, right, bottom)
        """
        self.region = region
        self.camera = None

        self._init_camera()

        # 提高进程优先级（可选）
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)

    def _init_camera(self):
        try:
            self.camera = dxcam.create(output_color="BGR")
            self.camera.start(target_fps=60)
            print("✅ dxcam 初始化成功（线程模式）")
        except Exception as e:
            print(f"❌ dxcam 初始化失败: {e}")
            self.camera = None

    def grab_screen(self):
        """
        获取最新帧并裁剪到指定区域
        """
        if not self.camera:
            return None
        img = self.camera.get_latest_frame()
        if img is None:
            return None
        left, top, right, bottom = self.region
        return img[top:bottom, left:right]

    # def detect_objects(self, img):
    #     """
    #     运行 YOLOv11 目标检测，返回检测框
    #     :param img: 截图
    #     :return: [(类别, x1, y1, x2, y2, 置信度)]
    #     x2 - x1 = y2 - y1 = 940
    #     """
    #     results = self.model(img)
    #     boxes = []
    #
    #     if not results:  # 如果结果为空
    #         return boxes
    #
    #     for r in results:
    #         if not hasattr(r, "boxes"):  # 确保 `boxes` 存在
    #             continue
    #
    #         for box in r.boxes:
    #             cls = r.names[int(box.cls)]  # 获取类别名
    #             if cls in self.target_classes:  # 只保留指定目标
    #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
    #                 conf = box.conf[0].item()  # 置信度
    #                 boxes.append((cls, x1, y1, x2, y2, conf))
    #
    #     return boxes
    #
    # def boss_pos(self, img):
    #     """
    #     获取 BOSS 位置坐标 (中心点)
    #     :param img: 传入的截图
    #     :return: (target_x, target_y) 或 None
    #     """
    #     if img is None:
    #         return None
    #
    #     boxes = self.detect_objects(img)
    #     if not boxes:
    #         return None  # 没有检测到目标
    #
    #     # 获取第一个目标的中心点
    #     cls, x1, y1, x2, y2, conf = boxes[0]
    #     target_x = self.region[0] + (x1 + x2) // 2
    #     target_y = self.region[1] + (y1 + y2) // 2
    #     self.last_boss_pos = (target_x, target_y)  # 记录上次位置
    #     return target_x, target_y

if __name__ == "__main__":
    aimbot = TerrariaAimbot()

    while True:
        img = aimbot.grab_screen()
        if img is None:
            print("⚠️ 未获取到图像帧")
            continue

        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, None, fx=0.3, fy=0.3)
        cv2.imshow("Terraria View", frame_resized)
        print(f"Frame size: {frame_resized.shape}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
