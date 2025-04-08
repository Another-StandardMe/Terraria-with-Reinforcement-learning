# This code performs object detection and tracking using YOLO and SORT.
# It detects objects in a video and assigns unique tracking IDs to each detected instance.
# The output is displayed in real-time with bounding boxes and object IDs.

from ultralytics import YOLO
import cv2
from sort import Sort
import numpy as np

model = YOLO("E:/terraria_project/after_training_weight/player_cursor_Cthulhu_env.pt")

tracker = Sort()

video_path = "E:/terraria_project/Eye_of_Cthulhu_1.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    if len(detections) == 0:
        detections = np.empty((0, 5))

    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(track_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
