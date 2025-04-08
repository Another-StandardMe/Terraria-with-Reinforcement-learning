from dxcam_capture import TerrariaAimbot
from memory_read import MemoryReader  # memory_read 名字无错误，勿修改
from keyboard_Listener import KeyActionListener
from CNNTransformer import LiteCNNTransformer
import torch
import pyautogui
import cv2
import threading
import time
import os


# 定义模型保存路径
save_dir = "./checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "cnn_bc_train.pth")

# 设置 device（GPU 优先）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 初始化各模块
aim_bot = TerrariaAimbot("E:/terraria_project/after_training_weight/10000best.pt")
reader = MemoryReader()
action_listener = KeyActionListener()

# 创建模型实例，并将其移动到 GPU
model = LiteCNNTransformer().to(device)

if os.path.exists(save_path):
    print(f"Loading model from {save_path} ...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("Model loaded successfully!")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 标志位，记录是否正在射击
is_shooting = False

# **📌 添加一个缓存列表来存储专家数据**
expert_dataset = []  # 用于存储 [player_window, velocity_tensor, expert_action]


def move_and_click(target_x, target_y, boss_exist):
    """ 让鼠标在子线程中移动到 BOSS 位置并点击 """
    global is_shooting
    pyautogui.moveTo(target_x, target_y, duration=0.002)

    if not is_shooting:  # 只有当未射击时，才按下鼠标
        pyautogui.mouseDown(button='left')
        is_shooting = True

    if not boss_exist:
        pyautogui.mouseUp(button='left')
        is_shooting = False


while True:
    last_time = time.time()

    env_data = reader.read_memory()

    # 确保 env_data 不为空并且数据完整
    if env_data and len(env_data) == 6:
        player_hp, boss_hp, player_x, player_y, velocity_x, velocity_y = env_data
    else:
        print("Failed to read environment data, skipping frame...")
        continue  # 跳过当前循环

    # **🔹 终止射击条件**
    if boss_hp == -1:
        print("BOSS 死亡，停止采集数据，进入训练阶段...")
        break  # 退出循环，进行训练

    frame = aim_bot.grab_screen()  # 获取实时截图
    boss_coord = aim_bot.boss_pos(frame)  # 进行目标检测

    if frame is None or frame.size == 0:  # 检查截图是否为空
        print("截图失败，跳过当前帧...")
        continue  # 直接跳过本次循环，防止训练崩溃

    # 如果检测到 BOSS 位置，则持续射击
    if boss_coord:
        boss_x, boss_y = boss_coord  # 解包坐标
        threading.Thread(target=move_and_click, args=(boss_x, boss_y, bool(boss_coord))).start()

    # **📌 处理数据，存入缓存**
    # 1. 处理截图，并转换为 [1, 3, 224, 224]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0  # 归一化
    player_window = frame_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 2. 处理速度数据，转换为 [1, 2]
    velocity_tensor = torch.tensor([[velocity_x, velocity_y]], dtype=torch.float32).to(device)

    # 3. 获取专家动作标签（one-hot 格式），形状 [1, 3]
    expert_actions = action_listener.get_action_labels().to(device)
    current_expert_action = expert_actions.clamp(min=1e-2, max=1 - 1e-2)

    # **📌 存储数据**
    expert_dataset.append((player_window, velocity_tensor, current_expert_action))

    print(f"Data collected: {len(expert_dataset)} frames | Spending time: {time.time() - last_time} ")

# **确保鼠标释放**
pyautogui.mouseUp(button='left')
is_shooting = False
print("释放鼠标")

# **📌 进入训练阶段**
print(f"Collected {len(expert_dataset)} frames of expert data. Starting training for 100 epochs...")

batch_size = 64  # 设定批量大小

for epoch in range(200):
    total_loss = 0.0
    num_batches = (len(expert_dataset) + batch_size - 1) // batch_size  # 计算总批次数（向上取整）

    print(f"\nEpoch {epoch + 1}/200")  # 打印 epoch 开始

    for batch_idx, i in enumerate(range(0, len(expert_dataset), batch_size), start=1):
        batch = expert_dataset[i:i + batch_size]  # **取 batch_size 个样本，但最后可能小于 batch_size**

        if len(batch) == 0:
            continue  # 避免空 batch

        # **合并 batch 数据**
        player_window_batch = torch.cat([x[0] for x in batch])  # shape: [batch_size 或 <batch_size, 3, 224, 224]
        expert_action_batch = torch.cat([x[2] for x in batch])  # shape: [batch_size 或 <batch_size, action_dim]

        optimizer.zero_grad()
        loss = model.imitation_loss(expert_action_batch, player_window_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # **打印进度条**
        progress = batch_idx / num_batches
        bar_length = 20  # 进度条长度
        filled_length = int(bar_length * progress)  # 已填充的长度
        bar = "🟩" * filled_length + "⬜" * (bar_length - filled_length)  # 进度条
        percentage = progress * 100  # 百分比

        print(f"\rBatch {batch_idx}/{num_batches} [{bar}] {percentage:.1f}%", end="")

    avg_loss = total_loss / num_batches
    print(f"\nEpoch {epoch + 1}/100 | Loss: {avg_loss:.6f}")

# **📌 训练完成后保存模型**
torch.save(model.state_dict(), save_path)
print("训练结束，模型已保存")
