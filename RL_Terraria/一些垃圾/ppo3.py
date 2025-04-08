import torch
import torch.nn as nn
import torch.nn.functional as F
from model4 import LiteCNNTransformer
from model_critic import CriticNetwork



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

        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def update(self, sample):
        """ ✅ 单样本更新 """
        print("------ 开始更新 --------")
        state, action, old_log_prob, reward, next_state, done = sample

        # **确保数据是 `[8, 3, 224, 224]`，不额外增加 batch 维度**
        state = state.to(self.device)  # [8, 3, 224, 224]
        next_state = next_state.to(self.device)
        action = action.to(self.device)  # [3]
        old_log_prob = old_log_prob.to(self.device)  # 标量
        reward = reward.to(self.device)  # 标量
        done = done.to(self.device)  # 标量

        # **计算 Critic 预测的值**
        with torch.no_grad():
            value = self.critic(state).squeeze(-1)
            next_value = self.critic(next_state).squeeze(-1)

            # **计算 Advantage**
            advantage = reward + self.gamma * (1 - done) * next_value - value
            return_val = advantage + value  # 目标 Q 值

        # **策略优化**
        for _ in range(self.K_epochs):
            self._optimize(state, action, old_log_prob, return_val, advantage)

    def _optimize(self, state, action, old_log_prob, return_val, advantage):
        """ ✅ 单样本梯度更新 """
        move_dist, jump_dist, down_dist = self.actor.get_dist(state)

        new_log_prob = (
            move_dist.log_prob(action[0]) +
            jump_dist.log_prob(action[1]) +
            down_dist.log_prob(action[2])
        )

        # **PPO 目标**
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2)

        # **价值损失**
        value = self.critic(state).squeeze(-1)  # 计算 V(s)
        value_loss = F.mse_loss(return_val, value)

        # **熵正则化**
        entropy = sum(d.entropy() for d in [move_dist, jump_dist, down_dist])

        # **优化**
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer.step()

        # **打印日志**
        print(f"策略损失: {policy_loss.item():.4f}, 价值损失: {value_loss.item():.4f}, 熵: {entropy.item():.4f}")
