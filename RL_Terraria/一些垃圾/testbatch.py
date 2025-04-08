from dxcam_capture import TerrariaAimbot
from memory_read import MemoryReader
from keyboard_Listener import KeyActionListener
from testNN import LiteCNNTransformer
import torch
import pyautogui
import cv2
import threading
import time
import os
import numpy as np

# 定义模型保存路径
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "cnn_bc_batch.pth")

# 设备选择（优先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 初始化各模块
aim_bot = TerrariaAimbot("E:/terraria_project/after_training_weight/10000best.pt")
reader = MemoryReader()
action_listener = KeyActionListener()

# 初始化模型
model = LiteCNNTransformer().to(device)
if os.path.exists(SAVE_PATH):
    print(f"Loading model from {SAVE_PATH} ...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    print("Model loaded successfully!")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 标志位
global is_shooting
is_shooting = False

# 数据存储
player_window_dataset = []
expert_action_dataset = []


def move_and_click(target_x, target_y, boss_exist):
    """ 让鼠标移动到 BOSS 位置并点击 """
    global is_shooting
    pyautogui.moveTo(target_x, target_y, duration=0.002)

    if not is_shooting:
        pyautogui.mouseDown(button='left')
        is_shooting = True

    if not boss_exist:
        pyautogui.mouseUp(button='left')
        is_shooting = False


while True:
    start_time = time.time()
    try:
        env_data = reader.read_memory()
    except Exception as e:
        print(f"Error reading memory: {e}")
        continue

    if not env_data or len(env_data) != 6:
        print("Invalid environment data, skipping frame...")
        continue

    player_hp, boss_hp, player_x, player_y, velocity_x, velocity_y = env_data
    if boss_hp == -1:
        print("BOSS 死亡，停止采集数据，进入训练阶段...")
        break

    frame = aim_bot.grab_screen()
    if frame is None or frame.size == 0:
        print("截图失败，跳过当前帧...")
        continue

    boss_coord = aim_bot.boss_pos(frame)
    if boss_coord:
        boss_x, boss_y = boss_coord
        threading.Thread(target=move_and_click, args=(boss_x, boss_y, bool(boss_coord)), daemon=True).start()

    # 处理帧数据
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    player_window_dataset.append(frame_tensor)

    # 记录专家动作
    expert_actions = action_listener.get_action_labels()
    expert_action_dataset.append(expert_actions.clamp(1e-2, 1 - 1e-2))

    print(f"Data collected: {len(expert_action_dataset)} sequences | Time: {time.time() - start_time:.4f} sec")

# 释放鼠标
pyautogui.mouseUp(button='left')
is_shooting = False
print("释放鼠标")

# 保存数据集
player_window_np = torch.stack(player_window_dataset).cpu().numpy()
expert_action_np = torch.stack(expert_action_dataset).cpu().numpy()

np.save(os.path.join(SAVE_DIR, "player_window_dataset.npy"), player_window_np)
np.save(os.path.join(SAVE_DIR, "expert_action_dataset.npy"), expert_action_np)

print("数据已保存！")
