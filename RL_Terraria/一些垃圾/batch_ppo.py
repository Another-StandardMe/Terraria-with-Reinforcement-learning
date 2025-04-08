import torch
import torch.nn as nn
import torch.nn.functional as F
from actor_batch import LiteCNNTransformer
from model_critic_batch import CriticNetwork


class PPO:
    def __init__(self, args):
        """ PPO 初始化 """
        self.device = args.device
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef

        # **1️⃣ Actor-策略网络**
        self.actor = LiteCNNTransformer(
            img_feature_dim=args.img_feature_dim,
            transformer_dim=args.transformer_dim,
            hidden_dim=args.hidden_width,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            dropout_rate=args.dropout_rate
        ).to(self.device)

        # **2️⃣ Critic-价值网络**
        self.critic = CriticNetwork(
            img_feature_dim=args.img_feature_dim,
            transformer_dim=args.transformer_dim,
            hidden_dim=args.hidden_width,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            dropout_rate=args.dropout_rate
        ).to(self.device)

        # **3️⃣ 优化器**
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': args.lr_a},
            {'params': self.critic.parameters(), 'lr': args.lr_c}
        ])

    def get_action(self, state):
        """ 🎯 获取单个动作 """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            actions, log_probs = self.actor.sample_action(state)

        actions = actions.cpu().numpy().squeeze(0).tolist()  # ✅ 确保返回 `[move, jump, down]`
        log_probs = log_probs.cpu().numpy().tolist()

        return actions, log_probs

    def update(self, batch_samples):
        """ ✅ **批量更新（16 个样本一起更新，使用 GAE）** """
        print("------ 开始批量更新 --------")

        # **转换 batch 数据**
        states = torch.stack([s[0] for s in batch_samples]).to(self.device)  # [16, 8, 3, 224, 224]
        actions = torch.stack([s[1] for s in batch_samples]).to(self.device)  # [16, 3]
        old_log_probs = torch.stack([s[2] for s in batch_samples]).to(self.device)  # [16]
        rewards = torch.stack([s[3] for s in batch_samples]).to(self.device)  # [16]
        next_states = torch.stack([s[4] for s in batch_samples]).to(self.device)  # [16, 8, 3, 224, 224]
        dones = torch.stack([s[5] for s in batch_samples]).to(self.device)  # [16]

        # **计算 Critic 预测的值**
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)  # [16]
            next_values = self.critic(next_states).squeeze(-1)  # [16]

            # **✅ 使用 GAE 计算 Advantage**
            advantages = torch.zeros_like(rewards).to(self.device)
            gae = 0  # 初始化 GAE 值
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
                gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae  # ✅ GAE 递推计算
                advantages[t] = gae

            returns = advantages + values  # 计算目标 Q 值

        # **策略优化**
        for _ in range(self.K_epochs):
            self._optimize(states, actions, old_log_probs, returns, advantages)

    def _optimize(self, states, actions, old_log_probs, returns, advantages):
        """ ✅ **批量梯度更新** """
        move_dists, jump_dists, down_dists = self.actor.get_dist(states)

        # **计算新 log_prob**
        new_log_probs = (
            move_dists.log_prob(actions[:, 0]) +
            jump_dists.log_prob(actions[:, 1]) +
            down_dists.log_prob(actions[:, 2])
        )  # [16]

        # **PPO 目标**
        ratio = torch.exp(new_log_probs - old_log_probs)  # [16]
        surr1 = ratio * advantages  # [16]
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages  # [16]
        policy_loss = -torch.min(surr1, surr2).mean()  # ✅ 计算均值

        # **价值损失**
        values = self.critic(states).squeeze(-1)  # [16]
        value_loss = F.mse_loss(returns, values)  # ✅ 计算均方误差

        # **熵正则化**
        entropy = sum(d.entropy().mean() for d in [move_dists, jump_dists, down_dists])

        # **优化**
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        # **打印日志**
        print(f"策略损失: {policy_loss.item():.4f}, 价值损失: {value_loss.item():.4f}, 熵: {entropy.item():.4f}")
