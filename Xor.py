import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam
import numpy as np
Xs = Tensor(np.array([0,1],[1,0],[1,1],[0,0]))
Ys = Tensor(np.array([1],[1],[0],[0]))
class XORModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2,2),
      nn.Sigmoid(),
      nn.Linear(2,1),
      nn.Sigmoid()
    )
    self.optimizer = Adam(self.parameters())
    self.loss = nn.MSELoss()
  def forward(self, X):
    return self.layers(X)
  def fit(self,X,y_true):
    self.optimizer.zero_grad()
    y_pred = self.forward(X)
    loss = self.loss(y_true,y_pred)
    loss.backward()
    self.optimizer.step()
    return loss.item()
xor_model = XORModel()
xor_model(Xs)
EPOCHS = 15000
for i in range(EPOCHS):
  loss = xor_model.fot(Xs, Ys)
  if i % 1000 == 0:
    print(loss)
print(xor_model(Xs))
