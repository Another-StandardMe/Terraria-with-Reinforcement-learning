import torch
import torch.nn as nn
import torch.nn.functional as F
from Categorical_policy_model import LiteCNNTransformer
import numpy as np


class PPO:
    def __init__(self, args):
        # 超参数配置
        self.device = args.device
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size

        # 网络初始化
        self.actor = LiteCNNTransformer(
            img_feature_dim=args.img_feature_dim,
            transformer_dim=args.transformer_dim,
            hidden_dim=args.hidden_width,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            dropout_rate=args.dropout_rate,
            max_seq_len=args.seq_len
        ).to(self.device)

        self.critic = nn.Sequential(
            LiteCNNTransformer(
                img_feature_dim=args.img_feature_dim,
                transformer_dim=args.transformer_dim,
                hidden_dim=args.hidden_width,
                transformer_heads=args.transformer_heads,
                transformer_layers=args.transformer_layers,
                dropout_rate=args.dropout_rate,
                max_seq_len=args.seq_len
            ),
            nn.Linear(args.hidden_width, 1)
        ).to(self.device)

        # 优化器配置
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': args.lr_a},
            {'params': self.critic.parameters(), 'lr': args.lr_c}
        ])

    def get_action(self, state):
        """获取动作"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if state.ndim == 4:
            state = state.unsqueeze(1)  # 添加序列维度

        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(state)  # ✅ 直接调用 sample_action()

        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def update(self, buffer):
        """核心优化逻辑"""
        states, actions, old_log_probs, rewards, next_states, dones = buffer
        #obs, action, log_prob, reward, next_obs, done
        # 数据预处理
        states = self._format_input(states)
        print(f"🚀 处理后的 states 形状: {states.shape}")
        next_states = self._format_input(next_states)
        actions = torch.LongTensor(actions).to(self.device)

        # GAE计算
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            deltas = rewards + self.gamma * (1 - dones) * next_values - values
            advantages = self._compute_gae(deltas, dones)
            returns = advantages + values

        # 策略优化
        for _ in range(self.K_epochs):
            for indices in self._generate_batches():
                self._optimize(
                    states[indices],
                    actions[indices],
                    old_log_probs[indices],
                    returns[indices],
                    advantages[indices]
                )

    def _format_input(self, x):
        """统一输入格式 [B, T, C, H, W]"""
        x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if x.ndim == 4:  # 补充序列维度
            x = x.unsqueeze(1).expand(-1, self.actor.seq_len, -1, -1, -1)
        return x

    def _compute_gae(self, deltas, dones):
        """向量化GAE计算"""
        advantages = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + self.gamma * self.lamda * (1 - dones[t]) * gae
            advantages[t] = gae
        return (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def _generate_batches(self):
        """生成随机mini-batch"""
        perm = torch.randperm(self.batch_size)
        for i in range(0, self.batch_size, self.mini_batch_size):
            yield perm[i:i + self.mini_batch_size]

    def _optimize(self, states, actions, old_log_probs, returns, advantages):
        """策略和价值网络联合优化"""
        # 策略损失
        move_dist, jump_dist, down_dist = self.actor.get_dist(states)
        new_log_probs = (
                move_dist.log_prob(actions[..., 0]) +
                jump_dist.log_prob(actions[..., 1]) +
                down_dist.log_prob(actions[..., 2])
        )

        ratios = (new_log_probs - old_log_probs).exp()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        values = self.critic(states)
        value_loss = F.mse_loss(returns, values)

        # 熵正则化
        entropy = sum(d.entropy().mean() for d in [move_dist, jump_dist, down_dist])

        # 联合优化
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()