from dxcam_capture import TerrariaAimbot
from memory_read import MemoryReader
from Categorical_action_executor import ActionExecutor
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController
import math
import torch
import time
import cv2
import pyautogui
import threading
import collections
import numpy as np
from pynput.keyboard import Key

class TerrariaEnv:
    def __init__(self, model_path, seq_len=8, verbose=True):
        """ 初始化环境 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aim_bot = TerrariaAimbot(model_path)
        self.reader = MemoryReader()
        self.executor = ActionExecutor()
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.verbose = verbose  # 是否打印日志信息

        # **环境状态**
        self.env_data = None
        self.Previous_player_hp = 500
        self.SEQ_LEN = seq_len

        #frame_buffer 长度超过 SEQ_LEN=8,旧数据会被丢弃，保证 frame_buffer 始终只存储最新的 8 帧。
        self.frame_buffer = collections.deque(maxlen=self.SEQ_LEN)

        self.is_shooting = False

    def move_and_click(self, target_x, target_y, boss_exist):
        """ 鼠标移动到 BOSS 位置并点击 """
        pyautogui.moveTo(target_x, target_y, duration=0.002)
        if boss_exist and not self.is_shooting:
            pyautogui.mouseDown(button="left")
            self.is_shooting = True
        elif not boss_exist:
            pyautogui.mouseUp(button="left")
            self.is_shooting = False

    def reset(self):
        """ 重置环境，等待玩家复活，清空帧缓存，召唤 BOSS，返回初始状态 """
        # **等待玩家复活**
        while True:
            env_data = self.reader.read_memory()
            if env_data:
                player_hp = env_data[0]  # `player_hp`
                if player_hp > 0:
                    break
            if self.verbose:
                print("🛑 玩家死亡，等待复活...")
            time.sleep(2)

        if self.verbose:
            print("✅ 玩家已复活，开始 `reset()`")

        # **清空环境状态**
        self.frame_buffer.clear()
        self.keyboard.release("a")
        self.keyboard.release("d")
        self.keyboard.release(Key.space)
        self.keyboard.release("s")

        pyautogui.mouseUp(button="left")
        self.is_shooting = False
        self.Previous_player_hp = 500

        # **执行 `reset()`**
        self._clear_environment()
        self._summon_boss()

        # **尝试获取初始观测值**
        for _ in range(10):  # 最多尝试 10 次
            obs = self._get_observation()
            if obs is not None:
                return obs
            if self.verbose:
                print("⚠️ `reset()` 观测为空，重新尝试获取状态...")
            time.sleep(1)

        print("❌ `reset()` 失败，返回 None")
        return None

    def _clear_environment(self):
        """ 清除地面掉落物和小怪 """
        self.keyboard.press("4")
        time.sleep(0.05)
        self.keyboard.release("4")
        time.sleep(1)
        pyautogui.mouseDown(button="left")
        time.sleep(1)
        pyautogui.mouseUp(button="left")
        print("------- 清除地面掉落物和小怪 -------")

    def _summon_boss(self):
        """ 召唤 BOSS """
        self.keyboard.press("6")
        time.sleep(0.05)
        self.keyboard.release("6")
        time.sleep(1)
        pyautogui.mouseDown(button="left")
        time.sleep(1)
        pyautogui.mouseUp(button="left")
        print("------- 召唤 BOSS -------")

        # **切换武器**
        self.keyboard.press("5")
        time.sleep(0.05)
        self.keyboard.release("5")
        print("------- 切换武器 -------")

    def _get_observation(self):
        """ 读取当前环境状态，包括图像和游戏数据 """
        frame = self.aim_bot.grab_screen()
        if frame is None or frame.size == 0:
            return None

        # **检测 BOSS 位置，并控制鼠标**
        boss_coord = self.aim_bot.boss_pos(frame)
        if boss_coord:
            boss_x, boss_y = boss_coord
            threading.Thread(target=self.move_and_click, args=(boss_x, boss_y, bool(boss_coord))).start()

        # **处理图像**
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0

        self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) < self.SEQ_LEN:
            return None  # 还未收集够 SEQ_LEN 帧

        return torch.stack(list(self.frame_buffer)).unsqueeze(0).to(self.device)

    def step(self, action):
        """ 执行动作，并返回新的状态、奖励、是否终止 """
        self.env_data = self.reader.read_memory()
        if not self.env_data:
            return None, 0, None, {}

        move, jump, down = action
        self.executor.execute_action((move, jump, down))

        # **获取新状态**
        new_state = self._get_observation()
        if new_state is None:
            return None, 0, None, {}

        player_hp, boss_hp, player_x, player_y, boss_x, boss_y = self.env_data
        distance = math.sqrt((player_x - boss_x) ** 2 + (player_y - boss_y) ** 2)

        # **计算奖励**
        reward = self._calculate_reward(player_hp, distance)

        # **检查终止条件**
        done = boss_hp == -1 or player_hp == 0

        return new_state, reward, done, {}

    def _calculate_reward(self, player_hp, distance):
        """ 计算奖励值 """
        reward = player_hp * 0.002  # 奖励与玩家血量正相关
        damage = self.Previous_player_hp - player_hp

        if distance > 750:
            reward -= distance * 0.001  # 远离 BOSS 给予负奖励

        if damage > 0:
            reward -= damage * 24  # 玩家受伤扣分

        self.Previous_player_hp = player_hp  # 更新 HP 记录
        return reward

    def close(self):
        """ 释放鼠标，关闭环境 """
        pyautogui.mouseUp(button="left")
        self.is_shooting = False
        print("✅ 环境关闭")


# **测试环境**
if __name__ == "__main__":
    env = TerrariaEnv(model_path="E:/terraria_project/after_training_weight/10000best.pt")

    obs = env.reset()
    if obs is None:
        print("❌ 无法初始化环境")
    else:
        done = False
        total_reward = 0
        time.sleep(1)

        while not done:
            action = np.random.randint(0, 2, size=(3,))  # 随机动作 (move, jump, down)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            print(f"🔹 done: {done}, 当前奖励: {reward:.2f}")

    env.close()
    print(f"🎯 游戏结束，总奖励: {total_reward:.2f}")
