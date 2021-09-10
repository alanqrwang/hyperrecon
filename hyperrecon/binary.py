import torch
import numpy as np
from hyperrecon.util.train import BaseTrain

class Binary(BaseTrain):
  """Constant sampling for hypernetwork."""

  def __init__(self, args):
    super(Binary, self).__init__(args=args)

  def train_epoch_begin(self):
    super().train_epoch_begin()
    print('Binary Sampling')
  
  def sample_hparams(self, num_samples):
    return torch.bernoulli(torch.empty(num_samples, self.num_hparams).fill_(0.5))

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)