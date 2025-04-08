import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import collections
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # **缓存位置编码**

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class LiteCNNTransformer(nn.Module):
    def __init__(
            self,
            img_feature_dim=128,
            transformer_dim=64,
            hidden_dim=128,
            transformer_heads=2,
            transformer_layers=3,
            dropout_rate=0.1,
            max_seq_len=8  # 设定最大帧序列长度
    ):
        super(LiteCNNTransformer, self).__init__()
        self.seq_len = max_seq_len

        # **1️⃣ CNN 提取图像特征**
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.cnn_fc = nn.Linear(28 * 28 * 64, img_feature_dim)

        # **2️⃣ Transformer 进行时序建模**
        self.state_proj = nn.Linear(img_feature_dim, transformer_dim)
        self.positional_encoding = PositionalEncoding(transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # **3️⃣ 预计算因果掩码**
        self.register_buffer("causal_mask", self.create_causal_mask(max_seq_len))

        # **4️⃣ 独立策略头**
        self.hidden_fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.move_output = nn.Linear(hidden_dim, 3)  # 移动 (左/不动/右)
        self.jump_output = nn.Linear(hidden_dim, 2)  # 跳跃 (不跳/跳)
        self.down_output = nn.Linear(hidden_dim, 2)  # 下蹲 (不蹲/蹲)

        # **5️⃣ CNN 特征缓存**
        self.feature_cache = collections.deque(maxlen=max_seq_len - 1)
        self._init_cache()

    def _init_cache(self):
        """ 初始化缓存为零值 """
        dummy_feature = torch.zeros(1, 128)
        for _ in range(self.seq_len - 1):
            self.feature_cache.append(dummy_feature)

    def reset_cache(self):
        """ 重置缓存 """
        self.feature_cache.clear()
        self._init_cache()


    @staticmethod
    def create_causal_mask(seq_len):
        """ 预计算因果掩码 """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def extract_cnn_feature(self, images):
        """ CNN 提取特征 """
        x = torch.tanh(self.conv1(images))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(x.shape[0], -1)
        return torch.tanh(self.cnn_fc(x))

    def forward(self, images):
        B, N, C, H, W = images.shape
        assert N == self.seq_len, f"Expected {self.seq_len} frames, but got {N}"

        # **计算最新 1 帧 CNN 特征**
        new_feature = self.extract_cnn_feature(images[:, -1, :, :, :])  # ✅ 移除 unsqueeze(0)

        # **确保缓存至少有 7 帧**
        if len(self.feature_cache) == 0:
            # **初始化缓存**
            all_features = self.extract_cnn_feature(images.view(B * N, C, H, W)).view(B, N, -1)
            self.feature_cache.extend([f.squeeze(0) for f in all_features[:, :-1].unbind(1)])  # ✅ 正确拆分为单帧特征
        else:
            # **用零填充缺失帧**
            while len(self.feature_cache) < self.seq_len - 1:
                self.feature_cache.append(torch.zeros_like(new_feature).cpu())

            # **堆叠缓存特征**
            cached_features = torch.stack(list(self.feature_cache), dim=1).to(images.device)  # [B, 7, D]

            # **更新缓存**
            self.feature_cache.append(new_feature.detach().cpu())

            # **合并特征**
            all_features = torch.cat([
                cached_features,
                new_feature.unsqueeze(1)  # ✅ 添加序列维度
            ], dim=1)

        # **Transformer 处理所有历史帧**
        state_emb = self.state_proj(all_features)
        transformer_out = self.transformer(state_emb, mask=self.causal_mask[:N, :N].to(state_emb.device))

        # **只取最后一帧**
        last_frame_feat = transformer_out[:, -1, :]

        # **计算每个动作类别**
        hidden = self.hidden_fc(last_frame_feat)
        move_logits = self.move_output(hidden)  # [B, 3]
        jump_logits = self.jump_output(hidden)  # [B, 2]
        down_logits = self.down_output(hidden)  # [B, 2]

        return move_logits, jump_logits, down_logits

    def get_dist(self, images):
        """ 获取 Categorical 分布，用于 PPO 训练 """
        move_logits, jump_logits, down_logits = self.forward(images)
        return (
            Categorical(logits=move_logits),
            Categorical(logits=jump_logits),
            Categorical(logits=down_logits)
        )

    def sample_action(self, images):
        """ 采样离散动作，用于 PPO 训练 """
        move_dist, jump_dist, down_dist = self.get_dist(images)
        move = move_dist.sample()
        jump = jump_dist.sample()
        down = down_dist.sample()
        log_prob = move_dist.log_prob(move) + jump_dist.log_prob(jump) + down_dist.log_prob(down)

        # **确保返回 `[move, jump, down]` 形状 `(3,)`**
        return torch.stack([move, jump, down], dim=-1).squeeze(0), log_prob
