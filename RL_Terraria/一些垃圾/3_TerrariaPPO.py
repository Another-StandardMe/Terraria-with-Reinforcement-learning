import os
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
        self.clip_value = args.clip_value
        self.count = 0
        self.use_grad_clip = getattr(args, "use_grad_clip", False)
        self.use_lr_decay = getattr(args, "use_lr_decay", False)

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
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a, eps=3e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c, eps=8e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)

        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_actor, T_max=args.max_train_steps, eta_min=1e-5)
        self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_critic, T_max=args.max_train_steps, eta_min=3e-5)

        self.checkpoint_path = "checkpoints/Terraria_final_model_3900.pth"
        if os.path.exists(self.checkpoint_path):
            print(f"🔄 加载已保存的模型: {self.checkpoint_path}")
            self.load(self.checkpoint_path)
            print("✅ 模型加载成功！")
        else:
            print("⚠️ 未找到已保存的模型，使用随机初始化参数")

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

        # 初始化指标累加器
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_q_value = 0.0

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

            # 累加指标
            total_policy_loss += policy_loss.item()
            total_critic_loss += critic_loss.item()
            total_q_value += v_s.mean().item()

        # 若你只关心策略网络的 loss，则记录平均 policy loss
        avg_policy_loss = total_policy_loss / self.K_epochs
        avg_critic_loss = total_critic_loss / self.K_epochs
        avg_q_value = total_q_value / self.K_epochs

        self.ep_avg_losses.append(avg_policy_loss)
        self.ep_avg_qs.append(avg_q_value)
        # 注意：ep_lengths 与 ep_rewards 通常由环境交互统计，并在 Worker 中记录后更新到 PPO 模型中

        if self.use_lr_decay:
            self.scheduler_actor.step()
            self.scheduler_critic.step()

        print(f"💠 第[{self.count}]次 更新......")
        self.count += 1

    def save(self, filepath):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def resetMemory(self):
        self.actor.reset_memory()
        self.critic.reset_memory()
        print("🧿 记忆初始化: Actor & Critic")

