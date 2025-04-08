import cv2
from ultralytics import YOLO
import numpy as np


def adjust_brightness(image, factor=1.5):
    """增加亮度，factor > 1 增强亮度，factor < 1 降低亮度"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# "E:/terraria_project/after_training_weight/king_slime_single_target.pt"
#"E:/terraria_project/after_training_weight/epoch_1.pt"
model = YOLO("E:/terraria_project/after_training_weight/10000best.pt")

# "E:/terraria_project/Eye_of_Cthulhu_1.mp4" king_slime_video
video_path = "E:/terraria_project/terraria_daytime.mp4"
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "E:/terraria_project/additional_2.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    adjusted_frame = adjust_brightness(frame, factor=1)

    results = model(adjusted_frame)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
