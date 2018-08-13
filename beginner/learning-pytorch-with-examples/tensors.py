# In this tutorial, the main features of PyTorch are introduced using a fully-connected ReLU 
# network. This network will be implemented with numpy and with PyTorch.

import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
