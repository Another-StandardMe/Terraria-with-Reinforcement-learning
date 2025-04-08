# This code tests a trained reinforcement learning model for Terraria.
# It captures game frames, processes them with a CNN-Transformer model, and evaluates its performance.
# The bot tracks and shoots at the boss while executing movement actions based on model predictions.

from Categorical_policy_model import LiteCNNTransformer
from dxcam_capture import TerrariaAimbot
from memory_read import MemoryReader
from Categorical_action_executor import ActionExecutor

import torch
import time
import cv2
import pyautogui
import threading
import collections


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:\\RL_Terraria\\Project_TAI\\Construct\\checkpoints\\model_alpha_1_best.pth"


model = LiteCNNTransformer().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


aim_bot = TerrariaAimbot("E:/terraria_project/after_training_weight/10000best.pt")
reader = MemoryReader()
executor = ActionExecutor()


SEQ_LEN = 8
frame_buffer = collections.deque(maxlen=SEQ_LEN)

is_shooting = False

def move_and_click(target_x, target_y, boss_exist):
    """ 让鼠标在子线程中移动到 BOSS 位置并点击 """
    global is_shooting
    pyautogui.moveTo(target_x, target_y, duration=0.002)

    if boss_exist and not is_shooting:
        pyautogui.mouseDown(button='left')
        is_shooting = True
    elif not boss_exist:
        pyautogui.mouseUp(button='left')
        is_shooting = False

while True:
    start_time = time.time()

    env_data = reader.read_memory()
    if env_data is None or len(env_data) != 6:
        print("环境数据读取失败，跳过当前帧...")
        continue

    player_hp, boss_hp, player_x, player_y, boss_x, boss_y = env_data

    if boss_hp == -1 or player_hp == 0:
        print("BOSS 死亡，停止射击")
        break

    frame = aim_bot.grab_screen()
    boss_coord = aim_bot.boss_pos(frame)

    if frame is None or frame.size == 0:
        print("截图失败，跳过当前帧...")
        continue

    # **如果检测到 BOSS 位置，则持续射击**
    if boss_coord:
        bossX, bossY = boss_coord
        threading.Thread(target=move_and_click, args=(bossX, bossY, bool(boss_coord))).start()

    # 处理截图
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

    frame_buffer.append(frame_tensor)

    if len(frame_buffer) < SEQ_LEN:
        print(f"等待收集足够帧 ({len(frame_buffer)}/{SEQ_LEN}) ...")
        continue

    input_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)  # [B=1, N=8, 3, 224, 224]

    with torch.no_grad():
        action, _ = model.sample_action(input_tensor)  # **使用 `Categorical` 采样离散动作**
        action = action.squeeze(0)  # [3]

    move, jump, down = action.cpu().numpy().astype(int)

    # 执行动作
    executor.execute_action((move, jump, down))

    print(f"Predicted Action: {move, jump, down}, Time Taken: {time.time() - start_time:.4f} sec")


pyautogui.mouseUp(button='left')
is_shooting = False
print("释放鼠标")
