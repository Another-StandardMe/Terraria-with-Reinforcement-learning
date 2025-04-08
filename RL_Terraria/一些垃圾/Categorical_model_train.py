# This code trains a reinforcement learning model for Terraria using expert data.
# It captures gameplay frames, collects human-labeled actions, and trains a CNN-Transformer model.
# The trained model learns to imitate expert actions based on visual and environmental inputs.

from dxcam_capture import TerrariaAimbot
from memory_read import MemoryReader
from Categorical_key_Listener import KeyActionListener
from Categorical_policy_model import LiteCNNTransformer
import torch
import pyautogui
import threading
import cv2
import time
import os
from torch.utils.data import Dataset, DataLoader


class Config:
    save_dir = "./checkpoints"
    model_best = os.path.join(save_dir, "model_alpha_1_best.pth")
    model_last = os.path.join(save_dir, "model_alpha_1_last.pth")
    batch_size = 16
    seq_len = 8  # 历史帧数
    epochs = 100
    accum_steps = 4
    lr = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"


def init_modules():
    os.makedirs(Config.save_dir, exist_ok=True)
    aim_bot = TerrariaAimbot("E:/terraria_project/after_training_weight/10000best.pt")
    reader = MemoryReader()
    listener = KeyActionListener()
    model = LiteCNNTransformer().to(Config.device)

    if os.path.exists(Config.model_last):
        print(f"Loading model from {Config.model_last} ...")
        model.load_state_dict(torch.load(Config.model_last, map_location=Config.device))
        print("Model loaded successfully!")
    return aim_bot, reader, listener, model


def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    return torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

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

def collect_expert_data(aim_bot, reader, listener):
    expert_data = []
    try:
        while True:
            start_time = time.time()
            env_data = reader.read_memory()
            if not env_data or len(env_data) != 6:
                continue
            if env_data[1] == -1:  # [player_hp, boss_hp, player_x, player_y, velocity_x, velocity_y]
                break

            frame = aim_bot.grab_screen()
            if frame is None or frame.size == 0:
                continue

            if boss_coord := aim_bot.boss_pos(frame):
                boss_x, boss_y = boss_coord
                threading.Thread(target=move_and_click, args=(boss_x, boss_y, bool(boss_coord))).start()

            frame_tensor = process_frame(frame)

            expert_actions = listener.get_action_labels().to(Config.device)

            expert_data.append((frame_tensor, expert_actions))
            print(f"Data collected: {len(expert_data)} frames | Time: {time.time() - start_time:.4f}s")
    finally:
        pyautogui.mouseUp(button='left')
        return expert_data

# 定义数据集
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        frames = [self.data[i][0] for i in range(idx, idx + self.seq_len)]
        action = self.data[idx + self.seq_len - 1][1]

        return torch.stack(frames), action

# 训练模型
def train_model(model, dataset):
    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    best_loss = float('inf')

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(loader)

        for batch_idx, (frames, targets) in enumerate(loader):
            assert isinstance(frames, torch.Tensor)
            frames = frames.to(Config.device)  # [B, N, 3, 224, 224]
            targets = targets.to(Config.device).long()

            loss = model.imitation_loss(targets, frames) / Config.accum_steps
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                continue

            loss.backward()

            if (batch_idx + 1) % Config.accum_steps == 0 or batch_idx == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * Config.accum_steps

            # 进度条
            progress = (batch_idx + 1) / num_batches
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = "❇️" * filled_length + "⬜" * (bar_length - filled_length)
            percentage = progress * 100
            print(f"\rEpoch {epoch + 1}/{Config.epochs} | Batch {batch_idx+1}/{num_batches} [{bar}] {percentage:.1f}%", end='', flush=True)

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch + 1}/{Config.epochs} | Loss: {avg_loss:.4f}")

        # **保存最优模型**
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.model_best)
            print(f"New best model saved with loss {best_loss:.6f}")

    torch.save(model.state_dict(), Config.model_last)
    print("训练结束，模型已保存")
if __name__ == "__main__":
    is_shooting = False
    aimbot, reader, listener, model = init_modules()
    print("=== 开始数据采集 ===")
    expert_data = collect_expert_data(aimbot, reader, listener)

    if len(expert_data) < Config.seq_len:
        raise ValueError(f"需要至少 {Config.seq_len} 帧数据，当前只有 {len(expert_data)} 帧")

    print("=== 开始训练 ===")
    dataset = SequenceDataset(expert_data, Config.seq_len)
    train_model(model, dataset)
    print("训练完成，模型已保存至:", Config.model_last)
