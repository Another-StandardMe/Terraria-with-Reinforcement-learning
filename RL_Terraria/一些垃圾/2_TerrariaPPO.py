import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

        if getattr(args, "set_adam_eps", False):
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)

        self.scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_actor, T_max=args.max_train_steps, eta_min=1e-5)
        self.scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_critic, T_max=args.max_train_steps, eta_min=1e-5)

    def _initialize_weights(self):
        """
        🚀 **使用正交初始化对 Actor 和 Critic 网络进行权重初始化**
        """
        self.actor.apply(orthogonal_init)
        self.critic.apply(orthogonal_init)
        print("✅ 正交初始化完成: Actor & Critic")


    def get_action(self, state):
        """ 采样动作并返回处理后的Python原生类型 """
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
            v_s_ = self.critic(s_).squeeze(-1)
            v_target = r + self.gamma * (1 - done) * v_s_

        for _ in range(self.K_epochs):
            # 计算策略分布
            move_dist, jump_dist = self.actor.get_dist(s)

            # 计算当前动作的 log_prob
            a_logprob_now = move_dist.log_prob(a[:, 0]) + jump_dist.log_prob(a[:, 1])
            a_logprob_now = a_logprob_now.view(-1, 1)

            # 计算 PPO clip ratio
            ratio = torch.exp(a_logprob_now - a_logprob)
            surr1 = ratio * (v_target - self.critic(s).squeeze(-1)).detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * (v_target - self.critic(s).squeeze(-1)).detach()
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * (move_dist.entropy().mean() + jump_dist.entropy().mean())

            # 计算价值函数 loss（clipped loss）
            v_s = self.critic(s).squeeze(-1)
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
