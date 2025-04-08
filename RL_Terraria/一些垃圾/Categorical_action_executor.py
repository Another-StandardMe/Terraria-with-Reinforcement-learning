# This code defines an action execution module for controlling a Terraria agent.
# It translates predicted discrete actions into keyboard inputs for movement, jumping, and crouching.
# The module ensures proper key press handling to avoid conflicting inputs.

from pynput.keyboard import Controller, Key
import torch

class ActionExecutor:
    def __init__(self):
        self.keyboard = Controller()
        self.key_status = {"a": False, "d": False, "space": False, "s": False}

    def execute_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().detach().numpy().tolist()[0]

        move, jump, down = action

        if move == 0:
            if not self.key_status["a"]:
                self.keyboard.press("a")
                self.key_status["a"] = True
            self.keyboard.release("d")
            self.key_status["d"] = False
        elif move == 2:
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
            if not self.key_status["space"]:
                self.keyboard.press(Key.space)
                self.key_status["space"] = True
        else:
            self.keyboard.release(Key.space)
            self.key_status["space"] = False

        if down == 1:
            if not self.key_status["s"]:
                self.keyboard.press("s")
                self.key_status["s"] = True
        else:
            self.keyboard.release("s")
            self.key_status["s"] = False
