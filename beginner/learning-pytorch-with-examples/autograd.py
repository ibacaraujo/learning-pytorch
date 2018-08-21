import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # forward pass
  y_pred = x.mm(w1).clamp(min=0).mm(w2)
  loss = (y_pred - y).pow(2).sum()
  print(t, loss.item())

  # backward pass
  loss.backward()
  with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad

    w1.grad.zero_()
    w2.grad.zero_()

# For large neural networks raw autograd can be a bit too low-level
# Use of the nn package that defines a set of Modules

model = torch.nn.Sequential(
  torch.nn.Linear(D_in, H),
  torch.nn.ReLU(),
  torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
  y_pred = model(x)
  loss = loss_fn(y_pred, y)
  print(t, loss.item())
  model.zero_grad()
  loss.backward()
  
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad

# To use more sophisticated optimizers, we can use the optim package

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
  y_pred = model(x)

  loss = loss_fn(y_pred, y)
  print(t, loss.item())

  optimizer.zero_grad()
  
  loss.backward()

  optimizer.step()
  
