import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class LiteCNNTransformer(nn.Module):
    def __init__(
            self,
            img_feature_dim=128,  # CNN 输出的特征维度
            transformer_dim=128,  # Transformer 维度
            hidden_dim=128,
            transformer_heads=2,
            transformer_layers=3,
            dropout_rate=0.1,
            max_seq_len=8
    ):
        super(LiteCNNTransformer, self).__init__()
        self.seq_len = max_seq_len
        self.img_feature_dim = img_feature_dim

        # **CNN 特征提取**
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.cnn_fc = nn.Linear(28 * 28 * 64, img_feature_dim)

        # **Transformer 进行时序建模**
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

        # **策略头**
        self.hidden_fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.move_output = nn.Linear(hidden_dim, 3)  # 移动 (左/不动/右)
        self.jump_output = nn.Linear(hidden_dim, 2)  # 跳跃 (不跳/跳)
        self.down_output = nn.Linear(hidden_dim, 2)  # 下蹲 (不蹲/蹲)

    def extract_cnn_feature(self, images):
        """ CNN 提取单帧图像特征 """
        B, C, H, W = images.shape  # [8, 3, 224, 224]
        x = torch.tanh(self.conv1(images))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(B, -1)  # Flatten
        return torch.tanh(self.cnn_fc(x))  # [8, img_feature_dim]

    def forward(self, images):
        """ 处理 `[batch_size, 8, 3, 224, 224]` 并输出策略动作 """
        if images.ndim == 5:  # ✅ 处理批量数据
            B, N, C, H, W = images.shape  # [16, 8, 3, 224, 224]
            images = images.view(B * N, C, H, W)  # **展平批量维度**
        else:  # 处理单帧情况
            images = images.unsqueeze(0)  # 变为 `[1, 8, 3, 224, 224]`
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)

        # **CNN 提取特征**
        cnn_features = self.extract_cnn_feature(images)  # [B * 8, img_feature_dim]
        cnn_features = cnn_features.view(B, N, -1)  # **恢复 `[B, 8, img_feature_dim]`**

        # **Transformer 处理时序信息**
        state_emb = self.state_proj(cnn_features)
        state_emb = self.positional_encoding(state_emb)
        transformer_out = self.transformer(state_emb)[:, -1, :]  # 取最后一帧输出 `[B, transformer_dim]`

        # **策略头**
        hidden = self.hidden_fc(transformer_out)
        move_logits = self.move_output(hidden)  # [B, 3]
        jump_logits = self.jump_output(hidden)  # [B, 2]
        down_logits = self.down_output(hidden)  # [B, 2]

        return move_logits, jump_logits, down_logits

    def get_dist(self, images):
        """ 获取 Categorical 分布 """
        move_logits, jump_logits, down_logits = self.forward(images)
        return (
            Categorical(logits=move_logits),
            Categorical(logits=jump_logits),
            Categorical(logits=down_logits)
        )

    def sample_action(self, images):
        """ 采样离散动作 """
        move_dist, jump_dist, down_dist = self.get_dist(images)
        move = move_dist.sample()
        jump = jump_dist.sample()
        down = down_dist.sample()
        log_prob = move_dist.log_prob(move) + jump_dist.log_prob(jump) + down_dist.log_prob(down)

        return torch.stack([move, jump, down], dim=-1), log_prob
