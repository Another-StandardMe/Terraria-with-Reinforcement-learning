import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from G_PolicyNet import GTrXLPolicyNet, orthogonal_init
from G_ValueNet import GTrXLValueNet, orthogonal_init

class PPO:
    def __init__(self, args):
        self.device = args.device
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.entropy_coef_max = args.entropy_coef
        self.clip_value = args.clip_value
        self.use_grad_clip = getattr(args, "use_grad_clip", False)
        self.use_lr_decay = getattr(args, "use_lr_decay", False)
        self.lamda = args.lamda
        self.max_episodes = args.max_episodes
        self.num_critic_updates = 5
        self.count = 0
        self.trajectory_count = 0
        # 新增记录各 episode 指标的属性
        self.ep_avg_losses = []  # 用于记录每个 episode 的平均策略 loss（也可包含 critic loss）
        self.ep_avg_qs = []      # 用于记录每个 episode 的平均 Q 值（Critic 输出均值）
        self.ep_lengths = []     # 记录每个 episode 的步数（如果在 Worker 中统计，可以在此更新）
        self.ep_rewards = []     # 记录每个 episode 的总奖励（如果在 Worker 中统计，可以在此更新）

        # **创建 Actor 和 Critic**
        self.actor = GTrXLPolicyNet(
            img_feature_dim=args.img_feature_dim,
            transformer_dim=args.transformer_dim,
            hidden_dim=args.hidden_width,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            dropout_rate=args.dropout_rate,
        ).to(self.device)

        self.critic = GTrXLValueNet(
            img_feature_dim=args.img_feature_dim,
            transformer_dim=args.transformer_dim,
            hidden_dim=args.hidden_width,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            dropout_rate=args.dropout_rate,
        ).to(self.device)

        # **🚀 执行正交初始化**
        self._initialize_weights()
        self.set_seed(42)

        if getattr(args, "set_adam_eps", False):
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)

        self.checkpoint_path = "D:/RL_Terraria/Project_TAI/newest/checkpoints/Terraria_final_model_005.pth"
        if os.path.exists(self.checkpoint_path):
            print(f"🔄 加载已保存的模型: {self.checkpoint_path}")
            self.load(self.checkpoint_path)
            # self.load_actor(self.checkpoint_path)
            print("✅ 模型加载成功！")
        else:
            print("⚠️ 未找到已保存的模型，使用随机初始化参数")

        # ✅ 注意：把 scheduler 初始化放到 load() 之后
        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_actor, T_max=self.max_episodes, eta_min=1e-5)
        self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_critic, T_max=self.max_episodes, eta_min=3e-5)

        # self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer_actor, T_max=args.max_train_steps, eta_min=1e-4)
        # self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer_critic, T_max=args.max_train_steps, eta_min=5e-5)
        #
        # self.checkpoint_path = "checkpoints/Terraria_final_model_003.pth"
        # if os.path.exists(self.checkpoint_path):
        #     print(f"🔄 加载已保存的模型: {self.checkpoint_path}")
        #     self.load(self.checkpoint_path)
        #     print("✅ 模型加载成功！")
        # else:
        #     print("⚠️ 未找到已保存的模型，使用随机初始化参数")

    def _initialize_weights(self):
        """
        🚀 **使用正交初始化对 Actor 和 Critic 网络进行权重初始化**
        """
        self.actor.apply(orthogonal_init)
        self.critic.apply(orthogonal_init)
        print("✅ 正交初始化完成: Actor & Critic")

    @staticmethod  # ✅ 添加这个装饰器，使其成为静态方法
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_action(self, state):
        """ 采样动作并返回处理后的 Python 原生类型 """
        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(state_tensor)

        return actions.cpu().numpy().squeeze(0).tolist(), log_probs.cpu().numpy().tolist()

    def update(self, replay_buffer):
        """
        更新 PPO 策略和 Critic
        replay_buffer 中包含一个单独样本： (s, a, a_logprob, r, s_, done)
        """
        s, a, a_logprob, r, s_, done = replay_buffer
        s = s.to(self.device)
        a = a.to(self.device)
        a_logprob = a_logprob.to(self.device)
        r = r.to(self.device)
        s_ = s_.to(self.device)
        done = done.to(self.device)

        # 计算 TD(0) 目标
        with torch.no_grad():
            v_s_ = self.critic(s_, update_memory=False).squeeze(-1)
            v_target = r + self.gamma * (1 - done) * v_s_

        # k_epochs 实际上是决定你要在同一个状态上尝试多少次动作，才能让熵坍塌
        for epoch in range(self.K_epochs):
            update_memory = (epoch == 0)  # 只有第一个 epoch 更新 memory
            # 调用 actor 时传递 update_memory 参数
            move_dist, jump_dist = self.actor.get_dist(s, update_memory=update_memory)

            # 调用 critic 时传递 update_memory 参数
            v_s = self.critic(s, update_memory=update_memory).squeeze(-1)

            advantage = (v_target - v_s).detach()
            # 计算 PPO 损失
            ratio = torch.exp((move_dist.log_prob(a[:, 0]) + jump_dist.log_prob(a[:, 1])).view(-1, 1) - a_logprob)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * (
                        move_dist.entropy().mean() + jump_dist.entropy().mean())

            values_clipped = v_s + (v_s - v_s.detach()).clamp(-self.clip_value, self.clip_value)
            value_loss_unclipped = (v_target - v_s).pow(2)
            value_loss_clipped = (v_target - values_clipped).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            # 更新 Actor
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            # 更新 Critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

        # if self.use_lr_decay:
        #     self.scheduler_actor.step()
        #     self.scheduler_critic.step()

        print(f"💠 第[{self.count}]次 更新......")
        self.count += 1

    def redistribute_final_reward(self, rewards, final_reward):
        """
        将 final_reward 合理地分配到 rewards[:-1] 上，并给最后一个状态设置 ±50 的奖励
        满足：
        1. 分配总和为 final_reward
        2. 最后状态再额外补充 ±50 用作 terminal signal
        """
        eps = 3
        r_tensor = torch.stack(rewards).squeeze()
        r_main = r_tensor[:-1]  # 前 T 个状态
        T = len(r_main)

        # 构建分配权重
        if final_reward >= 0:
            min_r = r_main.min()
            weights = torch.where(r_main > 0, r_main - min_r + eps, torch.full_like(r_main, eps))
        else:
            max_r = r_main.max()
            weights = torch.where(r_main < 0, -r_main + max_r + eps, torch.full_like(r_main, eps))

        # 分配 final_reward 到前 T 个状态
        weights = weights / (weights.sum() + eps)
        redistribution = weights * final_reward
        adjusted = r_main + redistribution

        # ✅ 最后一个状态设定为 ±10（额外奖励，不属于分配）
        terminal_bonus = torch.tensor(10.0 if final_reward >= 0 else -10.0, device=r_tensor.device)
        adjusted_rewards = list(adjusted) + [terminal_bonus]

        return [r.unsqueeze(0) for r in adjusted_rewards]

    def update_trajectory(self, trajectory):
        states = torch.stack(trajectory["states"]).squeeze(1).to(self.device)
        actions = torch.stack(trajectory["actions"]).squeeze(1).to(self.device)
        log_probs = torch.stack(trajectory["log_probs"]).squeeze(1).to(self.device)
        rewards = trajectory["rewards"]
        next_states = torch.stack(trajectory["next_states"]).squeeze(1).to(self.device)
        dones = torch.stack(trajectory["dones"]).squeeze(1).to(self.device)

        final_reward = rewards[-1].item()
        redistributed_rewards = self.redistribute_final_reward(rewards, final_reward)
        rewards_tensor = torch.stack(redistributed_rewards).squeeze(1).to(self.device)

        print(" --------------1-----------------")
        print(f"[显存监控] 已使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"        保留缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        with torch.no_grad():
            self.critic.reset_memory()
            values = []
            for s in states:
                v = self.critic(s.unsqueeze(0), update_memory=True).squeeze()
                values.append(v)
            values = torch.stack(values)

            self.critic.reset_memory()
            next_values = []
            for s_ in next_states:
                v_ = self.critic(s_.unsqueeze(0), update_memory=True).squeeze()
                next_values.append(v_)
            next_values = torch.stack(next_values)

            advantages = torch.zeros_like(rewards_tensor).to(self.device)
            gae = 0
            for t in reversed(range(len(rewards_tensor))):
                next_v = (1 - dones[t]) * next_values[t]
                delta = rewards_tensor[t] + self.gamma * next_v - values[t]
                gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
                advantages[t] = gae

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            with open("gae_stats_log.txt", "a") as log_file:
                log_file.write(f"Advantages mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}\n")
                log_file.write(f"Returns mean: {returns.mean().item():.4f}, value mean: {values.mean().item():.4f}\n")

        print(" --------------2-----------------")
        print(f"[显存监控] 已使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"        保留缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        for epoch in range(self.K_epochs):
            print(f"🟢 轨迹式序列更新: 第 {epoch + 1}/{self.K_epochs} 轮")

            self.actor.reset_memory()
            self.critic.reset_memory()

            # 累积时序性处理：逐帧 forward，批量记录 logits 和 values
            move_logits_list, jump_logits_list, value_preds = [], [], []
            for s in states:
                move_logits, jump_logits = self.actor.forward(s.unsqueeze(0), update_memory=True)
                v_s = self.critic(s.unsqueeze(0), update_memory=True).squeeze()
                move_logits_list.append(move_logits.squeeze(0))
                jump_logits_list.append(jump_logits.squeeze(0))
                value_preds.append(v_s)

            move_logits = torch.stack(move_logits_list)
            jump_logits = torch.stack(jump_logits_list)
            value_preds = torch.stack(value_preds)

            move_dist = Categorical(logits=move_logits)
            jump_dist = Categorical(logits=jump_logits)
            logp_new = move_dist.log_prob(actions[:, 0]) + jump_dist.log_prob(actions[:, 1])
            ratio = torch.exp(logp_new - log_probs.view(-1))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy = move_dist.entropy().mean() + jump_dist.entropy().mean()
            policy_loss -= self.entropy_coef * entropy

            values_clipped = value_preds + (value_preds - value_preds.detach()).clamp(-self.clip_value, self.clip_value)
            value_loss_unclipped = (returns - value_preds).pow(2)
            value_loss_clipped = (returns - values_clipped).pow(2)
            critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            print(" --------------3-----------------")
            print(f"[显存监控] 已使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"        保留缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            # ✅ 一次性反向传播
            self.optimizer_actor.zero_grad()
            policy_loss.backward()
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            for _ in range(self.num_critic_updates - 1):  # 注意 -1，因为上面已经更新过一次了
                self.critic.reset_memory()
                value_preds_extra = []
                for s in states:
                    v_s = self.critic(s.unsqueeze(0), update_memory=True).squeeze()
                    value_preds_extra.append(v_s)
                value_preds_extra = torch.stack(value_preds_extra)

                values_clipped = value_preds_extra + (value_preds_extra - value_preds_extra.detach()).clamp(
                    -self.clip_value, self.clip_value)
                value_loss_unclipped = (returns - value_preds_extra).pow(2)
                value_loss_clipped = (returns - values_clipped).pow(2)
                critic_loss_extra = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                self.optimizer_critic.zero_grad()
                critic_loss_extra.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                # 显存缓存清理
                torch.cuda.empty_cache()

            if self.use_lr_decay and epoch == 0:
                self.scheduler_actor.step()
                self.scheduler_critic.step()

            # 显存缓存清理
            torch.cuda.empty_cache()

            print(" --------------4-----------------")
            print(f"[显存监控] 已使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"        保留缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            avg_policy_loss = policy_loss.item()
            avg_q_value = value_preds.mean().item()
            self.ep_avg_losses.append(avg_policy_loss)
            self.ep_avg_qs.append(avg_q_value)

            print(f"✅ Epoch {self.trajectory_count}: avg_policy_loss: {avg_policy_loss:.6f}, avg_q_value: {avg_q_value:.6f}")

        print(f"🚀 完成轨迹更新，策略 loss: {avg_policy_loss:.6f}，平均 Q 值: {avg_q_value:.6f}")
        self.entropy_coef = self.entropy_coef_max * 0.5 * (
                1 + math.cos(math.pi * self.trajectory_count / self.max_episodes))
        print(f"当前 entropy_coef: {self.entropy_coef:.6f}")
        self.trajectory_count += 1

    def save(self, filepath):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if "actor" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor"])
            print("✅ 成功加载 actor 网络参数")
        else:
            print("⚠️ checkpoint 中没有 'actor' 键，跳过 actor 加载")

        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
            print("✅ 成功加载 critic 网络参数")
        else:
            print("⚠️ checkpoint 中没有 'critic' 键，跳过 critic 加载")

    def load_actor(self, path):
        actor_state_dict = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(actor_state_dict)

    def resetMemory(self):
        self.actor.reset_memory()
        self.critic.reset_memory()
        print("🧿 记忆初始化: Actor & Critic")

