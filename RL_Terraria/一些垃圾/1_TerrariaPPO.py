import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from PolicyNet import SwinPolicyNet
from ValueNet import CriticNetwork
from T_PolicyNet_ import CNNTransformer
from T_ValueNet import CriticNetwork

class PPO:
    def __init__(self, args):
        self.device = args.device
        self.gamma = args.gamma  # 折扣因子
        self.lamda = args.lamda  # GAE参数
        self.epsilon = args.epsilon  # PPO clip范围
        self.K_epochs = args.K_epochs  # 策略更新轮数
        self.entropy_coef = args.entropy_coef  # 熵系数
        self.clip_value = args.clip_value  # 价值函数clip阈值
        self.max_episodes = args.max_episodes
        self.count = 0

        # Swin 网络初始化
        # self.actor = SwinPolicyNet(
        #     embed_dim=args.embed_dim,
        #     hidden_dim=args.hidden_dim,
        #     dropout_rate=args.dropout_rate
        # ).to(self.device)
        #
        # self.critic = CriticNetwork(
        #     embed_dim=args.embed_dim,
        #     hidden_dim=args.hidden_dim,
        #     dropout_rate=args.dropout_rate
        # ).to(self.device)

        # 网络初始化
        self.actor = CNNTransformer(
            img_feature_dim=args.img_feature_dim,
            transformer_dim=args.transformer_dim,
            hidden_dim=args.hidden_width,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            dropout_rate=args.dropout_rate,
        ).to(self.device)

        self.critic = nn.Sequential(
            CriticNetwork(
                img_feature_dim=args.img_feature_dim,
                transformer_dim=args.transformer_dim,
                hidden_dim=args.hidden_width,
                transformer_heads=args.transformer_heads,
                transformer_layers=args.transformer_layers,
                dropout_rate=args.dropout_rate,
            )
        ).to(self.device)


        # 优化器配置
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': args.lr_a},
            {'params': self.critic.parameters(), 'lr': args.lr_c}
        ])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_episodes, eta_min=1e-5)

        self.training_log = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "actor_grad": [],
            "critic_grad": []
        }
        self.log_file = "training_log.pt"  # ✅ 日志存储文件

    def _save_log(self):
        if len(self.training_log["policy_loss"]) % 100 == 0:
            torch.save(self.training_log, self.log_file)

    def get_action(self, state):
        """ 采样动作并返回处理后的Python原生类型 """
        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(state_tensor)
        return actions.cpu().numpy().squeeze(0).tolist(), log_probs.cpu().numpy().tolist()

    def update(self, batch_samples):
        """ 批量更新核心逻辑 """
        # 显式地从 batch_samples 中分别 stack 后，再转换到 GPU
        states = torch.stack([torch.as_tensor(s[0], device=self.device) for s in batch_samples])
        actions = torch.stack([torch.as_tensor(s[1], device=self.device) for s in batch_samples])
        old_log_probs = torch.stack([torch.as_tensor(s[2], device=self.device) for s in batch_samples])
        rewards = torch.stack([torch.as_tensor(s[3], device=self.device) for s in batch_samples])
        next_states = torch.stack([torch.as_tensor(s[4], device=self.device) for s in batch_samples])
        dones = torch.stack([torch.as_tensor(s[5], device=self.device) for s in batch_samples])
        #print(f"✅ 输入到 state 的维度: {states.shape}")

        # 检查数据是否都在正确的设备上
        assert states.device == self.device, f"states设备不一致: {states.device} vs {self.device}"

        # 价值预测
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)

            # GAE计算
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                next_value = (1 - dones[t]) * next_values[t].detach()
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values

        # 优势值标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 训练 K_epochs 轮
        for _ in range(self.K_epochs):
            self._optimize_epoch(states, actions, old_log_probs, returns, advantages)
        self.scheduler.step()

    def _optimize_epoch(self, states, actions, old_log_probs, returns, advantages):
        """ 单轮优化步骤 """
        assert states.device == self.device, f"states设备不一致: {states.device} vs {self.device}"
        assert actions.device == self.device, f"actions设备不一致: {actions.device} vs {self.device}"

        # 动作分布获取
        move_dist, jump_dist = self.actor.get_dist(states)

        # 新策略概率计算
        new_log_probs = sum([
            1.5 * move_dist.log_prob(actions[:, 0]),
            1.0 * jump_dist.log_prob(actions[:, 1])
        ])

        # PPO目标计算
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # 价值函数优化（带clip）
        values = self.critic(states).squeeze(-1)
        values_clipped = values + (values - values.detach()).clamp(-self.clip_value, self.clip_value)
        # "Proximal Policy Optimization Algorithms" 给出的 价值损失 (Value Loss) 公式
        value_loss_unclipped = (values - returns).pow(2)
        value_loss_clipped = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # 熵正则项
        entropy = torch.mean(torch.stack([
            dist.entropy().mean()
            for dist in [move_dist, jump_dist]
        ]))

        # 总损失计算
        total_loss = policy_loss + value_loss + self.entropy_coef * entropy

        # 反向传播与梯度裁剪
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        # 训练监控
        actor_grad = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
        critic_grad = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)

        # ✅ 存储数据
        self.training_log["policy_loss"].append(policy_loss.item())
        self.training_log["value_loss"].append(value_loss.item())
        self.training_log["entropy"].append(entropy.item())
        self.training_log["actor_grad"].append(actor_grad)
        self.training_log["critic_grad"].append(critic_grad)

        # ✅ 训练日志写入文件
        self._save_log()

        self.count += 1
        print(f"💠 第[{self.count}]次 更新......")