import torch
import torch.nn as nn

class SimpleImage(nn.Module):
  def __init__(self, dims, init_img=None):
    super(SimpleImage, self).__init__()
    self.pmask = nn.Parameter(torch.FloatTensor(1, 1, *dims))         
    self.pmask.requires_grad = True
    if init_img is None:
      self.pmask.data.uniform_(0, 1)
    else:
      self.pmask.data = init_img
    self.pmask.data = self.pmask.data.cuda()

  def forward(self, zf):
    return self.pmask.clone()