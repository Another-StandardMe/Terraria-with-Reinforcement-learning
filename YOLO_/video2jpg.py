# This code extracts frames from a video and saves them as images.
# It processes the video by skipping frames to reduce redundancy and stores the extracted images in a specified folder.
# The output frames are saved sequentially with a formatted filename.

import cv2
import os

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    frame_skip = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # if frame_count % frame_skip == 0:
        cv2.imwrite(os.path.join(output_folder, f"additional_4_{frame_count:04d}.jpg"), frame)
        frame_count += 1

    cap.release()
    print(f"Extraction completed! Total frames processed: {frame_count}, saved in {output_folder}")


#E:\\terraria_project\\Eye_of_Cthulhu_2.mp4
#"E:\\terraria_project\\Eye_of_Cthulhu_1.mp4"
#E:\terraria_project night_clean  , // focus.ONplayer_night = focus_ONplayer_daytime
video_path = "E:\\terraria_project\\additional_1.mp4"
output_folder = "E:\\terraria_project\\test"

extract_frames(video_path, output_folder)
