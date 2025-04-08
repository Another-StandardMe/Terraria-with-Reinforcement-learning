# This code defines an action execution module for controlling a Terraria agent.
# It translates predicted discrete actions into keyboard inputs for movement, jumping, and crouching.
# The module ensures proper key press handling to avoid conflicting inputs.

from pynput.keyboard import Controller, Key
import torch
import time


class ActionExecutor:
    def __init__(self):
        self.keyboard = Controller()
        self.key_status = {"a": False, "d": False, "space": False, "s": False}

    def execute_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy().tolist()[0]

        move, jump = action

        if move == 0:
            if not self.key_status["a"]:
                self.keyboard.press("a")
                self.key_status["a"] = True
            self.keyboard.release("d")
            self.key_status["d"] = False
        elif move == 1:
            if not self.key_status["d"]:
                self.keyboard.press("d")
                self.key_status["d"] = True
            self.keyboard.release("a")
            self.key_status["a"] = False
        else:
            self.keyboard.release("a")
            self.keyboard.release("d")
            self.key_status["a"] = False
            self.key_status["d"] = False

        if jump == 1:
            self.keyboard.press(Key.space)
            time.sleep(0.05)  # 持续
            self.keyboard.release(Key.space)





        # if jump == 1:
        #     if not self.key_status["space"]:  # 如果空格没有被按过
        #         self.keyboard.press(Key.space)  # 按下空格
        #         self.key_status["space"] = True
        #     self.keyboard.release("s")  # 松开 S
        #     self.key_status["s"] = False
        # else:
        #     if not self.key_status["s"]:  # 如果 S 没被按
        #         self.keyboard.press("s")  # 按下 S（下蹲）
        #         self.key_status["s"] = True
        #     self.keyboard.release(Key.space)  # 松开空格
        #     self.key_status["space"] = False

        #
        # if jump == 1:
        #     if not self.key_status["space"]:
        #         self.keyboard.press(Key.space)
        #         self.key_status["space"] = True
        # else:
        #     self.keyboard.release(Key.space)
        #     self.key_status["space"] = False

        # if down == 1:
        #     if not self.key_status["s"]:
        #         self.keyboard.press("s")
        #         self.key_status["s"] = True
        # else:
        #     self.keyboard.release("s")
        #     self.key_status["s"] = False
