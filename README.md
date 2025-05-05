# Terraria-with-Reinforcement-learning
Applying PPO-based Reinforcement Learning with Transformer Memory (GTrXL) to automate combat strategies and navigation in the game Terraria.

demonstration videos are https://youtu.be/mXUjfhCDtbA

## 📁 File Descriptions

This repository contains a complete implementation of reinforcement learning training and object detection in the game **Terraria**. Below is a description of the key files and folders:

- `YOLO/`  
  Contains the code for YOLO model training and various preprocessing scripts for object detection in game frames.

- `RL_Terraria/TerrariaENV.py`  
  Defines the custom reinforcement learning environment for interacting with the Terraria game.

- `RL_Terraria/TerrariaPPO.py`  
  Implements the Proximal Policy Optimization (PPO) algorithm for training the agent.

- `RL_Terraria/TerrariaTrain.py`  
  The main training script that launches the reinforcement learning training process.

- `RL_Terraria/G_ValueNet.py`  
  Contains the CNN - Gated Transformer-XL-based value network (Critic) used to estimate the value of game states.

- `RL_Terraria/G_PolicyNet.py`  
  Contains the CNN - Gated Transformer-XL-based policy network (Actor) that learns the optimal action distribution.

- `RL_Terraria/TerrariaTesting.py`  
  Testing script used to evaluate the trained agent's performance in the game environment.

- `environment.yml`
  All installed packages (including versions) in the current project environment.
