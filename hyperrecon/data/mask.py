import numpy as np
import torch
import torch.nn as nn

class BaseMask(nn.Module):
  def __init__(self, mask_path):
    super(BaseMask, self).__init__()
    print('Loading mask:', mask_path)
    mask = np.load(mask_path)
    mask = np.fft.fftshift(mask)
    self.mask = torch.tensor(mask, requires_grad=False).float()

  def forward(self, num_samples):
    mask_stack = self.mask[None, None].repeat(num_samples, 1, 1, 1)
    return mask_stack