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
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :].to(x.device)


class GTrXLTransformerEncoderLayer(nn.Module):
    """
    一个 GTrXL 层：
    - 对输入进行 LayerNorm 后计算自注意力，
      然后通过门控（learnable scalar gate）将新信息加权后加到残差上。
    - 接着对结果进行 LayerNorm 后计算前馈网络，
      同样通过门控加权后加到残差上。
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(GTrXLTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, memory=None):
        if memory is not None:
            src_cat = torch.cat([memory, src], dim=1)
        else:
            src_cat = src
        src_norm = self.attn_layer_norm(src)
        src_cat_norm = self.attn_layer_norm(src_cat)
        attn_output, _ = self.self_attn(src_norm, src_cat_norm, src_cat_norm)
        attn_output = self.dropout(attn_output)
        src = src + torch.sigmoid(self.attn_gate) * attn_output
        src_norm = self.ff_layer_norm(src)
        ff_output = self.linear2(self.ff_dropout(F.relu(self.linear1(src_norm))))
        ff_output = self.dropout(ff_output)
        src = src + torch.sigmoid(self.ff_gate) * ff_output
        return src


class GTrXLPolicyNet(nn.Module):
    def __init__(self,
                 img_feature_dim=128,  # CNN 输出的特征维度
                 transformer_dim=128,  # Transformer 内部维度
                 hidden_dim=128,
                 transformer_heads=4,
                 transformer_layers=2,
                 dropout_rate=0.1,
                 memory_length=8  # 保存最近 memory_length 个状态的隐藏表示
                 ):
        super(GTrXLPolicyNet, self).__init__()
        self.memory_length = memory_length
        self.num_layers = transformer_layers

        # 输入尺寸为 (354, 396, 1)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # 全连接层输入尺寸调整为 44*49*64
        self.cnn_fc = nn.Linear(44 * 49 * 64, img_feature_dim)

        self.state_proj = nn.Linear(img_feature_dim, transformer_dim)
        self.positional_encoding = PositionalEncoding(transformer_dim, max_len=memory_length + 1)

        self.transformer_layers = nn.ModuleList([
            GTrXLTransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout_rate
            ) for _ in range(transformer_layers)
        ])

        self.hidden_fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.move_output = nn.Linear(hidden_dim, 2)  # 移动（左/右）
        self.jump_output = nn.Linear(hidden_dim, 2)  # 跳跃（不跳/跳）

        self.memory_queue = [None] * self.num_layers

    def extract_cnn_feature(self, images):
        """ CNN 提取单帧图像特征 """
        images = images.to(self.conv1.weight.device)
        B, C, H, W = images.shape  # 例如：[B, 1, 384, 384]
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(B, -1)

        # print(f"CNN features shape before fc: {x.shape}")
        return F.relu(self.cnn_fc(x))

    def forward(self, images, update_memory=True):
        """
        Args:
            images: [B, seq_len, 3, H, W] 当前输入序列（例如 seq_len=1 表示当前状态，为4帧叠加后的状态）
            memory_list: list，每个元素对应一层 Transformer 的 memory，形状为 [B, memory_length, transformer_dim]
                         如果为 None，则表示当前是轨迹的起始阶段
        Returns:
            move_logits, jump_logits: 策略头输出（用于采样动作）
            new_memory_list: 更新后的 memory_list，供下次 forward 使用
        """

        # images: [B, seq_len, 1, 384, 384] 或 [seq_len, 1, 384, 384]
        if images.ndim == 5:
            B, N, C, H, W = images.shape
            images = images.reshape(B * N, C, H, W)
        elif images.ndim == 4:
            images = images.unsqueeze(0)
            B, N, C, H, W = images.shape
            images = images.reshape(B * N, C, H, W)
        else:
            raise ValueError(f"Unsupported input shape: {images.shape}")

        cnn_features = self.extract_cnn_feature(images).view(B, N, -1)
        state_emb = self.state_proj(cnn_features)
        x = self.positional_encoding(state_emb)

        new_memory_list = []
        for i, layer in enumerate(self.transformer_layers):
            mem = self.memory_queue[i]
            x = layer(x, memory=mem)
            # 仅在 update_memory=True 时更新 memory
            if update_memory:
                new_memory = x[:, -self.memory_length:, :].detach()
            else:
                new_memory = mem  # 保持 memory 不变
            new_memory_list.append(new_memory)

        if update_memory:
            self.memory_queue = new_memory_list  # 仅在第一个 epoch 更新

        transformer_out = x[:, -1, :]
        hidden = self.hidden_fc(transformer_out)
        move_logits = self.move_output(hidden)
        jump_logits = self.jump_output(hidden)
        return move_logits, jump_logits

    def get_dist(self, images, update_memory=True):
        move_logits, jump_logits = self.forward(images,update_memory)
        return Categorical(logits=move_logits), Categorical(logits=jump_logits)

    def sample_action(self, images, update_memory=True):
        move_dist, jump_dist = self.get_dist(images,update_memory)
        move = move_dist.sample()
        jump = jump_dist.sample()
        log_prob = move_dist.log_prob(move) + jump_dist.log_prob(jump)
        return torch.stack([move, jump], dim=-1), log_prob

    def reset_memory(self):
        self.memory_queue = [None] * self.num_layers


# 定义正交初始化函数（保持不变）
def orthogonal_init(m):
    if isinstance(m, nn.MultiheadAttention):
        if hasattr(m, 'out_proj') and m.out_proj is not None:
            nn.init.orthogonal_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
        if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
            d_model = m.embed_dim
            for i in range(3):
                start = i * d_model
                end = (i + 1) * d_model
                nn.init.orthogonal_(m.in_proj_weight[start:end, :])
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
    elif isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
