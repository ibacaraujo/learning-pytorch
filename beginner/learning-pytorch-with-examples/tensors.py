# In this tutorial, the main features of PyTorch are introduced using a fully-connected ReLU 
# network. This network will be implemented with numpy and with PyTorch.

import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# Training the network
learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y
  h = x.dot(w1)
  h_relu = np.max(0, h)
  y_pred = h_relu.dot(w2)

  # Compute and print loss
  loss = np.square(y_pred - y).sum()
  print(t, loss)
