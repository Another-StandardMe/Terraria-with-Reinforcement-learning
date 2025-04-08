# This script defines a keyboard action listener for collecting expert demonstration data.
# It captures discrete movement actions (left, right, idle), jumping, and crouching in real time.
# The listener converts keypress states into a formatted tensor for reinforcement learning.

from pynput import keyboard
import torch

class KeyActionListener:
    def __init__(self):
        self.key_status = {
            "left": 0,
            "right": 0,
            "jump": 0,
            "down": 0
        }
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.key_status["jump"] = 1
        else:
            try:
                if key.char == "a":
                    self.key_status["left"] = 1
                    self.key_status["right"] = 0
                elif key.char == "d":
                    self.key_status["right"] = 1
                    self.key_status["left"] = 0
                elif key.char == "s":
                    self.key_status["down"] = 1
            except AttributeError:
                pass

    def on_release(self, key):
        if key == keyboard.Key.space:
            self.key_status["jump"] = 0
        else:
            try:
                if key.char == "a":
                    self.key_status["left"] = 0
                elif key.char == "d":
                    self.key_status["right"] = 0
                elif key.char == "s":
                    self.key_status["down"] = 0
            except AttributeError:
                pass

    def get_action_labels(self):
        if self.key_status["right"] and not self.key_status["left"]:
            move = 0
        elif self.key_status["left"] and not self.key_status["right"]:
            move = 2
        else:
            move = 1

        jump = self.key_status["jump"]
        down = self.key_status["down"]

        return torch.tensor([[move, jump, down]], dtype=torch.long)

if __name__ == "__main__":
    n = 0
    listener = KeyActionListener()
    while True:
        expert_actions = listener.get_action_labels()
        print(f"{n} times Current action: {expert_actions.numpy().tolist()}")
        n += 1
