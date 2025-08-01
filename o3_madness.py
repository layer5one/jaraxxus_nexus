import torch, platform
print("Torch CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0))
