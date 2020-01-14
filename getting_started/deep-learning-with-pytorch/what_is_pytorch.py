# What is PyTorch?

# Replacement for NumPy to use the power of GPUs
# Deep learning research platform

from __future__ import print_function
import torch

# 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)

# randomly initialized matrix
x = torch.rand(5, 3)
print(x)

# matrix filled with zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# tensor based on an existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.rand_like(x, dtype=torch.float)
print(x)

# get its size
print(x.size())

# addition: syntax 1
y = torch.rand(5, 3)
print(x + y)

# addition: syntax 2
print(torch.add(x, y))

# addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# addition: in-place
y.add_(x)
print(y)

# standard NumPy-like indexing
print(x[:, 1])

# resizing
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# get the value of a one element tensor
x = torch.randn(1)
print(x)
print(x.item())

# Torch tensor and NumPy array will share the underlying memory location
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# converting Numpy Array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA Tensors
if torch.cuda.is_available():
  device = torch.device("cuda")
  y = torch.ones_like(x, device=device)
  x = x.to(device)
  z = x + y
  print(z)
  print(z.to("cpu", torch.double))
