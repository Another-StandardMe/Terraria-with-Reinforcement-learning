import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CriticNetwork(nn.Module):
    def __init__(
            self,
            img_feature_dim=128,  # CNN 提取的特征维度
            transformer_dim=128,  # Transformer 维度
            hidden_dim=128,  # 隐藏层维度
            transformer_heads=2,  # 多头注意力
            transformer_layers=3,  # Transformer 层数
            dropout_rate=0.1,  # Dropout
            max_seq_len=8
    ):
        super(CriticNetwork, self).__init__()
        self.seq_len = max_seq_len

        # **CNN 提取特征**
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

        # **最终的 Critic 价值头**
        self.hidden_fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )
        self.value_head = nn.Linear(hidden_dim, 1)  # 价值网络最终只输出一个标量 V(s)

    def extract_cnn_feature(self, images):
        """ CNN 提取单帧图像特征 """
        B, C, H, W = images.shape  # [8, 3, 224, 224]
        x = torch.tanh(self.conv1(images))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(B, -1)  # Flatten
        return torch.tanh(self.cnn_fc(x))  # [8, img_feature_dim]

    def forward(self, images):
        """ 计算当前状态的总价值 V(s) """

        if images.ndim == 5:  # ✅ 处理批量数据 `[B, 8, 3, 224, 224]`
            B, N, C, H, W = images.shape  # [16, 8, 3, 224, 224]
            images = images.view(B * N, C, H, W)  # **展开批量维度**
        else:  # 处理单帧情况
            images = images.unsqueeze(0)  # 变为 `[1, 8, 3, 224, 224]`
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)

        # **CNN 提取特征**
        cnn_features = self.extract_cnn_feature(images)  # [B * 8, img_feature_dim]
        cnn_features = cnn_features.view(B, N, -1)  # **重新 reshape 成 `[B, 8, img_feature_dim]`**

        # **Transformer 处理时序信息**
        state_emb = self.state_proj(cnn_features)
        state_emb = self.positional_encoding(state_emb)
        transformer_out = self.transformer(state_emb)[:, -1, :]  # 取最后一帧输出 `[B, transformer_dim]`

        # **计算 V(s)**
        hidden = self.hidden_fc(transformer_out)
        value = self.value_head(hidden)  # [B, 1]

        return value.squeeze(-1)  # **变成 `[B]`**

