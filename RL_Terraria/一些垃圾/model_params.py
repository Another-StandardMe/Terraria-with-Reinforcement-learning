# import torch
# from Categorical_policy_model import LiteCNNTransformer
#
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model_path = "D:/RL_Terraria/Project_TAI/newest/checkpoints/Terraria_checkpoint_352.pth"
# model = LiteCNNTransformer().to(device)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.eval()
#
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# print(f"总参数量: {total_params:,}")
# print(f"可训练参数量: {trainable_params:,}")



