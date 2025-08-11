import torch

print(torch.version.cuda)  # 输出PyTorch编译时使用的CUDA版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
