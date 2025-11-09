import torch

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cudnn.version:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ PyTorch 未检测到可用 GPU。")
