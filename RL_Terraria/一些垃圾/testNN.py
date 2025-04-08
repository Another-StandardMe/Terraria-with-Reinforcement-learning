import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class LiteCNNTransformer(nn.Module):
    def __init__(
            self,
            img_feature_dim=128,
            transformer_dim=64,
            hidden_dim=128,
            action_dim=3,
            transformer_heads=2,
            transformer_layers=3,
            dropout_rate=0.1
    ):
        super(LiteCNNTransformer, self).__init__()

        self.epsilon = 1e-6

        # --------------------- 1️⃣ 轻量级 CNN 处理 **RGB 图像** (224x224) ---------------------
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # stride=2 代替池化
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # 计算 CNN 输出尺寸
        self.cnn_output_size = (28 * 28 * 64)
        self.cnn_fc = nn.Linear(self.cnn_output_size, img_feature_dim)

        # --------------------- 2️⃣ Transformer 进行特征融合 ---------------------
        self.state_proj = nn.Linear(img_feature_dim, transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # --------------------- 3️⃣ 输出 Beta 分布参数 ---------------------
        self.hidden_fc = nn.Sequential(
            nn.Linear(transformer_dim, hidden_dim),
            nn.Tanh(),  # 用 Tanh 让特征更加平滑
            nn.Dropout(dropout_rate)
        )
        self.output_layer = nn.Linear(hidden_dim, action_dim * 2)

    def forward(self, images):
        """
        参数：
          - images: [B, N, 3, 224, 224] (批量大小 B=16, N=64 张 RGB 图像)
        返回：
          - alpha, beta: [B, N, action_dim]
        """
        B, N, C, H, W = images.shape  # 确保 N 维度正确

        # CNN 处理每张图片
        x = images.view(B * N, C, H, W)  # 让 CNN 处理 B*N 张图片
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.view(B * N, -1)
        img_feature = torch.tanh(self.cnn_fc(x))

        # 变换为 Transformer 输入
        img_feature = img_feature.view(B, N, -1)  # 重新变回 [B, N, img_feature_dim]
        state_emb = self.state_proj(img_feature)  # [B, N, transformer_dim]

        # Transformer 处理所有图片
        transformer_out = self.transformer(state_emb)  # [B, N, transformer_dim]

        # 输出每个 token 的特征
        hidden = self.hidden_fc(transformer_out)  # [B, N, hidden_dim]

        # 计算 Beta 分布的参数 α 和 β
        alpha_beta = self.output_layer(hidden)  # [B, N, action_dim * 2]
        alpha, beta = torch.chunk(F.softplus(alpha_beta) + self.epsilon, 2, dim=-1)  # [B, N, action_dim]

        return alpha, beta

    def get_dist(self, images):
        alpha, beta = self.forward(images)
        return Beta(alpha, beta)

    def sample_action(self, images):
        dist = self.get_dist(images)
        action = dist.rsample()  # 采样动作
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算 log_prob
        return action, log_prob

    def imitation_loss(self, expert_actions, images):
        """
        计算模仿学习损失 - 专家动作在当前策略分布下的负对数似然。
        """
        dist = self.get_dist(images)
        log_prob = dist.log_prob(expert_actions)  # 确保维度匹配 [B, N, action_dim]
        return -log_prob.sum(dim=-1).mean()
