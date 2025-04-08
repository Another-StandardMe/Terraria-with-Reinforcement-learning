import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from jinja2.nodes import Tuple


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

    def forward(self, src, memory=None, update_memory=True):
        # 当 update_memory 为 False 或 memory 为 None 时，不进行 memory 拼接
        if memory is not None and update_memory:
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


class GTrXLValueNet(nn.Module):
    def __init__(self,
                 img_feature_dim=128,
                 transformer_dim=128,
                 hidden_dim=128,
                 transformer_heads=2,
                 transformer_layers=2,
                 dropout_rate=0.1,
                 memory_length=8):
        super(GTrXLValueNet, self).__init__()
        self.memory_length = memory_length
        self.num_layers = transformer_layers

        # 输入尺寸为 (354, 396, 1)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
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

        self.value_hidden = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.value_head = nn.Linear(hidden_dim, 1)

        self.memory_queue = [None] * self.num_layers

    def extract_cnn_feature(self, images):
        """
        CNN 提取单状态图像特征
        images: [B, 4, H, W]，其中4个通道为4帧堆叠而成
        """
        images = images.to(self.conv1.weight.device)
        B, C, H, W = images.shape  # 例如：[B, 4, 384, 384]
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(B, -1)
        return F.relu(self.cnn_fc(x))

    def forward(self, images, update_memory=True):
        """
        Args:
            images: [B, 4, 1, H, W] 或 [B, 4, H, W]
                    表示4帧灰度图像堆叠成一个状态。
            update_memory: 是否更新 Transformer 的 memory
        Returns:
            move_logits, jump_logits: 策略头输出（用于采样动作）
        """
        # 如果输入为 [B, 4, 1, H, W]，则将通道维 squeeze 掉
        if images.ndim == 5:
            # 假设 shape 为 [B, 4, 1, H, W]
            images = images.squeeze(2)  # 变成 [B, 4, H, W]
        elif images.ndim != 4:
            raise ValueError("Input must be [B, 4, H, W] or [B, 4, 1, H, W] representing 4 stacked frames.")

        B, C, H, W = images.shape  # 此时 C 应为 4

        # 提取CNN特征，得到每个状态的表示 [B, img_feature_dim]
        cnn_features = self.extract_cnn_feature(images)

        # 为符合Transformer输入格式，增加一个时间步维度（状态序列长度为1）
        cnn_features = cnn_features.unsqueeze(1)  # [B, 1, img_feature_dim]

        state_emb = self.state_proj(cnn_features)  # [B, 1, transformer_dim]
        x = self.positional_encoding(state_emb)  # [B, 1, transformer_dim]

        new_memory_list = []
        for i, layer in enumerate(self.transformer_layers):
            mem = self.memory_queue[i]
            x = layer(x, memory=mem, update_memory=update_memory)
            # 更新 memory: 如果序列长度不足 memory_length，则保留现有 x
            if update_memory:
                new_memory = x[:, -self.memory_length:, :].detach()
            else:
                new_memory = mem  # 保持 memory 不变
            new_memory_list.append(new_memory)

        if update_memory:
            self.memory_queue = new_memory_list

        transformer_out = x[:, -1, :]
        hidden = self.value_hidden(transformer_out)
        value = self.value_head(hidden)
        return value

    def reset_memory(self):
        self.memory_queue = [None] * self.num_layers


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
