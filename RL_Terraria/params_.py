import torch
from PolicyNet import SwinPolicyNet
# 加载模型
model = SwinPolicyNet(embed_dim=64, hidden_dim=128, dropout_rate=0.1)

# 加载训练好的权重
checkpoint = torch.load("D:\\RL_Terraria\\Project_TAI\\newest\\checkpoints\\Terraria_checkpoint_64.pth", map_location=torch.device("cpu"))

# 加载到模型中
model.load_state_dict(checkpoint)

# 计算总参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"🚀 训练好的模型总参数量: {count_parameters(model) / 1e6:.2f} M")  # 单位：百万(M)

# 打印每层参数形状
print("\n📌 模型参数细节：")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")