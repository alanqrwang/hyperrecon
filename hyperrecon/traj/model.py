import torch.nn as nn

class TrajNet(nn.Module):
  def __init__(self, in_dim=1, h_dim=8, out_dim=2):
    super(TrajNet, self).__init__()
    self.lin1 = nn.Linear(in_dim, h_dim)
    self.lin2 = nn.Linear(h_dim, h_dim)
    self.lin3 = nn.Linear(h_dim, h_dim)
    self.lin4 = nn.Linear(h_dim, h_dim)
    self.lin5 = nn.Linear(h_dim, out_dim)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.lin1(x)
    x = self.relu(x)
    x = self.lin2(x)
    x = self.relu(x)
    x = self.lin3(x)
    x = self.relu(x)
    x = self.lin2(x)
    x = self.relu(x)
    x = self.lin5(x)
    out = self.sigmoid(x)
    return out

