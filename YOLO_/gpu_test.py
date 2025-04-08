import torch

print("GPU 是否可用:", torch.cuda.is_available())
print("使用的 GPU 数量:", torch.cuda.device_count())
print("当前 GPU 设备:", torch.cuda.current_device())
print("GPU 名称:", torch.cuda.get_device_name(0))
