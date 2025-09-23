import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)

# Simple tensor test
x = torch.rand(3, 3).cuda()
print("Tensor on GPU:\n", x)

print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))