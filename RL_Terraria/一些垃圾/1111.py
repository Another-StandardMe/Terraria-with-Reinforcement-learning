# reset test

from dxcam_capture import TerrariaAimbot
from memory_read import MemoryReader
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button
import time
import cv2
import threading

aim_bot = TerrariaAimbot("E:/terraria_project/after_training_weight/10000best.pt")
reader = MemoryReader()

keyboard = KeyboardController()
mouse = MouseController()

def call_boss():
    print("召唤 BOSS: 按键 6")
    keyboard.press('6')
    time.sleep(0.05)
    keyboard.release('6')

    time.sleep(0.1)
    mouse.click(Button.left)
    time.sleep(0.1)

while True:
    start_time = time.time()

    img = aim_bot.grab_screen()

    if img is None:
        print("截图失败，跳过当前帧...")
        continue

    boss_coord = aim_bot.boss_pos(img)
    result = reader.read_memory()

    if result is None or len(result) != 6:
        print("读取游戏数据失败，跳过...")
        continue

    player_hp, boss_hp, player_x, player_y, velocity_x, velocity_y = result

    print(f"boss hp: {boss_hp}")

    if boss_hp == -1:
        threading.Thread(target=call_boss).start()

    if boss_coord:
        print(f"BOSS 位置: {boss_coord}")

    elapsed_time = time.time() - start_time
    print(f"time: {elapsed_time:.6f} seconds")

    if img is not None:
        cv2.imshow("YOLO Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.005)

cv2.destroyAllWindows()
