from MemoryRead import MemoryReader
import math
import pyautogui
import keyboard
import time
import threading

# 读取内存
reader = MemoryReader()

def calculate_angle(player_x, player_y, boss_x, boss_y):
    """ 计算从玩家到Boss的角度（弧度） """
    dx = boss_x - player_x
    dy = boss_y - player_y
    angle_rad = math.atan2(dy, dx)  # 计算弧度
    return angle_rad

def move_and_click(angle_rad):
    """ 根据计算出的角度移动鼠标到目标位置 """
    # 以 (685, 720) 为中心
    target_x = 685 + math.cos(angle_rad) * 100  # 100 是移动距离
    target_y = 720 + math.sin(angle_rad) * 100

    # 移动鼠标
    pyautogui.moveTo(target_x, target_y, duration=0.05)


# 将鼠标移到中间位置（685，720）
pyautogui.moveTo(685, 720, duration=0.02)
pyautogui.mouseDown(button="left")
while True:
    start_time = time.time()

    # 读取游戏内存
    result = reader.read_memory()
    if result:
        player_hp, boss_hp, player_x, player_y, boss_x, boss_y = result

        # 计算角度
        angle_rad = calculate_angle(player_x, player_y, boss_x, boss_y)

        # 使用多线程移动鼠标
        threading.Thread(target=move_and_click, args=(angle_rad,)).start()

    # 计算执行时间
    elapsed_time = time.time() - start_time
    print(f"Spend time: {elapsed_time:.4f} seconds")

    # 监听 'q' 退出
    if keyboard.is_pressed("q"):
        print("\n'Q' pressed, exiting...")
        break

    #time.sleep(0.02)
pyautogui.mouseUp(button="left")