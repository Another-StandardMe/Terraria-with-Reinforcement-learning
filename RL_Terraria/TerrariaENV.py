from sys import breakpointhook

import torch
import collections
import cv2
import time
import numpy as np
import pyautogui
import threading
import math
import matplotlib.pyplot as plt

from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController
from pynput.keyboard import Key
from ImageCapture import TerrariaAimbot
from MemoryRead import MemoryReader
from CategoricalActionExecutor import ActionExecutor


class TerrariaEnv:
    def __init__(self, seq_len=4, verbose=True):
        """ 初始化环境 """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aim_bot = TerrariaAimbot()
        self.reader = MemoryReader()
        self.executor = ActionExecutor()
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.verbose = verbose  # 是否打印日志信息

        # **环境状态**
        self.env_data = None
        self.Previous_player_hp = 500
        self.Previous_distance = 0
        self.num_step = 0
        self.SEQ_LEN = seq_len

        # **帧缓存：仅存储最新的 8 帧**
        self.frame_buffer = collections.deque(maxlen=self.SEQ_LEN)
        # self.is_shooting = False

    # def move_and_click(self, target_x, target_y, boss_exist):
    #     """ 鼠标移动到 BOSS 位置并点击 """
    #     pyautogui.moveTo(target_x, target_y, duration=0.002)
    #     if boss_exist and not self.is_shooting:
    #         pyautogui.mouseDown(button="left")
    #         self.is_shooting = True
    #     elif not boss_exist:
    #         pyautogui.mouseUp(button="left")
    #         self.is_shooting = False

    def calculate_angle(self, player_x, player_y, boss_x, boss_y):
        """ 计算从玩家到Boss的角度（弧度） """
        dx = boss_x - player_x
        dy = boss_y - player_y
        angle_rad = math.atan2(dy, dx)  # 计算弧度
        return angle_rad

    def move_and_click(self, angle_rad):
        """ 根据计算出的角度移动鼠标到目标位置 """
        # 以 (685, 720) 为中心
        target_x = 685 + math.cos(angle_rad) * 100  # 100 是移动距离
        target_y = 720 + math.sin(angle_rad) * 100

        # 移动鼠标
        pyautogui.moveTo(target_x, target_y, duration=0.05)

    def reset(self):
        """ 重置环境，等待玩家复活，清空帧缓存，召唤 BOSS，返回初始状态 """
        while True:
            env_data = self.reader.read_memory()
            print(f"player HP:{env_data[0]}")
            self._clear_environment()
            env_data = self.reader.read_memory()
            if env_data and env_data[0] >= 250:  # `player_hp`
                self._clear_environment()
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
        pyautogui.moveTo(685, 720, duration=0.02)
        self.is_shooting = False
        self.Previous_player_hp = 500
        self.num_step = 0

        # **执行 `reset()`**
        # self._clear_environment()
        self._summon_boss()

        while True:
            self.env_data = self.reader.read_memory()
            player_hp, boss_hp, player_x, player_y, boss_x, boss_y = self.env_data
            dx, dy, distance = self._calculate_distance(player_x, player_y, boss_x, boss_y)
            if distance < 720:
                break
            else:
                time.sleep(0.5)
                continue

        obs = self._get_observation()
        pyautogui.mouseDown(button="left")
        return obs


        # for _ in range(10):  # 最多尝试 10 次
        #     obs = self._get_observation()
        #     if obs is not None:
        #         pyautogui.mouseDown(button="left")
        #         return obs
        #     if self.verbose:
        #         print("⚠️ `reset()` 观测为空，重新尝试获取状态...")
        #     time.sleep(1)
        #
        # print("❌ `reset()` 失败，返回 None")
        # return None

    def check(self):
        env_data = self.reader.read_memory()

        if env_data is None or len(env_data) < 2:
            return None

        return env_data[0] < 500 or env_data[1] == -1

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

        self.keyboard.press("5")
        time.sleep(0.05)
        self.keyboard.release("5")
        print("------- 切换武器 -------")

    def _get_observation(self):
        """ 读取当前环境状态，包括图像和游戏数据 """
        total_retry = 0

        # 如果4次未能填满缓存，就说明这次状态采集失败，需要重新采集
        while len(self.frame_buffer) < self.SEQ_LEN and total_retry < 4:
            frame = self.aim_bot.grab_screen()

            if frame is not None and frame.size > 0:
                # 成功读取图像后处理
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_resized = cv2.resize(frame_gray, None, fx=0.3, fy=0.3)
                frame_tensor = torch.from_numpy(frame_resized).unsqueeze(0).float() / 255.0
                self.frame_buffer.append(frame_tensor)
            else:
                print(f"⚠️ 第 {total_retry + 1} 次截图失败")
                time.sleep(0.01)

            total_retry += 1

        if len(self.frame_buffer) < self.SEQ_LEN:
            print("❌ 未能填满帧缓存，返回 None")
            return None

        # 构造堆叠帧 [SEQ_LEN, 1, H, W]
        frames_stack = torch.stack(list(self.frame_buffer), dim=0).to(self.device)
        return frames_stack

        # 检测 BOSS 位置，并控制鼠标
        # boss_coord = self.aim_bot.boss_pos(frame)
        #
        # if boss_coord:
        #     boss_x, boss_y = boss_coord
        #     threading.Thread(target=self.move_and_click, args=(boss_x, boss_y, bool(boss_coord))).start()

        # # 处理图像
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # 将灰度图调整尺寸
        # frame_resized = cv2.resize(frame_gray, None, fx=0.3, fy=0.3)
        #
        # # 转换为 tensor，并增加 channel 维度（结果为 [1, 354, 396]）
        # frame_tensor = torch.from_numpy(frame_resized).unsqueeze(0).float() / 255.0
        #
        # self.frame_buffer.append(frame_tensor)
        # if len(self.frame_buffer) < self.SEQ_LEN:
        #     return None
        #
        # # 返回的 tensor 形状为 [SEQ_LEN, 1, 354, 396]
        # frames_stack = torch.stack(list(self.frame_buffer), dim=0).to(self.device)
        # return frames_stack

    def _get_discountFactor(self, step, beta_const=200):
        return step / (step + 2000 * (beta_const / step))

    def _calculate_distance(self, player_x, player_y, boss_x, boss_y):
        dx = player_x - boss_x
        dy = player_y - boss_y
        distance = math.sqrt((player_x - boss_x) ** 2 + (player_y - boss_y) ** 2)
        return dx, dy, distance


    def step(self, action):
        """ 执行动作，并返回新的状态、奖励、是否终止 """

        move, jump = action

        if self.num_step % 50 == 0:
            print(f"move:{move}, jump:{jump}")

        self.executor.execute_action((move, jump))

        # 获取新状态（由 _get_observation 收集 SEQ_LEN 帧组成）
        new_state = self._get_observation()
        if new_state is None:
            return None, 0, False, {}

        # 清空帧缓存，确保下一次状态从空开始重新采集
        self.frame_buffer.clear()

        self.env_data = self.reader.read_memory()
        if not self.env_data:
            return None, 0, False, {}

        player_hp, boss_hp, player_x, player_y, boss_x, boss_y = self.env_data
        # print(f"Player HP: {player_hp}, Boss HP: {boss_hp}, Pos: ({player_x:.2f}, {player_y:.2f}), "
        #       f"boss pos: ({boss_x:.2f}, {boss_y:.2f})")
        dx, dy, distance = self._calculate_distance(player_x, player_y, boss_x, boss_y)

        angle_rad = self.calculate_angle(player_x, player_y, boss_x, boss_y)
        threading.Thread(target=self.move_and_click, args=(angle_rad,)).start()

        # **计算奖励**
        reward, kill = self._calculate_reward(player_hp, boss_hp, dx, dy, distance)

        # **检查终止条件**
        done = boss_hp == -1 or player_hp == 0

        return new_state, reward, done, kill

    def _calculate_reward(self, player_hp, boss_hp, dx, dy, distance):
        """ 改进后的奖励设计，鼓励agent维持安全距离，避免受伤 """
        self.num_step += 1
        reward = 0

        # 受伤扣分，扣除更多惩罚，明确避免受伤
        damage = self.Previous_player_hp - player_hp
        if damage > 0:
            if damage < 10:
                reward -= 0.5
            reward -= damage * 0.1 # 受伤扣更多分

        # 与BOSS保持安全距离奖励设计
        # safe_min, safe_max = 300, 650  # 理想安全距离范围

        # ✅ 判断玩家是否在矩形范围内（以 boss 为中心: 跟玩家之间的安全距离是 宽 580，高 380）
        if abs(dx) <= 520 and abs(dy) <= 400:
            reward += 2  # 矩形内奖励
        else:
            reward -= 1  # 矩形外惩罚

        # elif distance > more_penalty_distance:
        #     reward -= 100
        # if safe_min <= distance <= safe_max:
        #     reward += 0.5  # 安全距离奖励明确给予较大的正强化
        # elif close_penalty_distance <= distance < safe_min:
        #     reward -= (safe_min - distance) * 0.005  # 距离稍微近，轻微惩罚
        # elif safe_max < distance <= far_penalty_distance:
        #     reward -= (distance - safe_max) * 0.005  # 距离稍微远，轻微惩罚
        # elif distance > far_penalty_distance:
        #     reward -= 5
        # else:
        #     reward -= (safe_min - distance) * 0.41

        # if boss_hp > 0:
        #     reward += (3570 - boss_hp) * 0.002

        a = self._get_discountFactor(self.num_step)

        # 玩家死亡给予巨大负面惩罚
        if player_hp <= 0:
            reward -= (1-a) * 500

        kill = False
        # print(f"Previous_distance:{self.Previous_distance}")
        # print(f"distance:{distance}")
        # BOSS死亡给予巨大正奖励4
        if player_hp <= 0:
            reward -= (1-a) * 500
        elif boss_hp == -1 and self.Previous_distance> 2500:
            reward -= (1-a) * 500
            kill = False
            print(" ------ penalty ------ ")
            print(f"Previous_distance:{self.Previous_distance}")
            print(f"distance:{distance}")
        elif boss_hp == -1 and self.Previous_distance< 2500:
            reward += 500
            print(" ------ reward ------ ")
            print(f"Previous_distance:{self.Previous_distance}")
            print(f"distance:{distance}")
            kill = True

        # 更新血量记录
        self.Previous_player_hp = player_hp
        self.Previous_distance = distance

        #print(f"step:{self.num_step}")

        return reward, kill

    def close(self):
        """ 释放鼠标，关闭环境 """
        self.frame_buffer.clear()
        self.keyboard.release("a")
        self.keyboard.release("d")
        self.keyboard.release(Key.space)
        self.keyboard.release("s")
        pyautogui.mouseUp(button="left")
        #self.is_shooting = False

        self._clear_environment()
        self.frame_buffer.clear()

        print("✅ 环境关闭")


if __name__ == "__main__":
    #env = TerrariaEnv(model_path="E:/terraria_project/after_training_weight/10000best.pt")
    env = TerrariaEnv()

    obs = env.reset()
    if obs is None:
        print("❌ 无法初始化环境")
    else:
        done = False
        total_reward = 0
        time.sleep(1)

        while not done:
            action = (0, 1)  # 随机动作 (move, jump)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if obs is not None:
                # 取最后一帧
                obs_img = obs[-1].cpu().numpy().transpose(1, 2, 0)  # [3, 224, 224] -> [224, 224, 3]
                obs_img = (obs_img * 255).astype(np.uint8)  # 转换回 OpenCV 格式

                # **显示画面**
                cv2.imshow("Terraria Observation", obs_img)

                # **按 `q` 退出**
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                continue

            print(f"🔹 done: {done}, 当前奖励: {reward:.2f}")

    env.close()
    cv2.destroyAllWindows()  # 关闭 OpenCV 窗口
    print(f"游戏结束，总奖励: {total_reward:.2f}")
