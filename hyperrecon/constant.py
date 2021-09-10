import torch
import numpy as np
from hyperrecon.util.train import BaseTrain

class Constant(BaseTrain):
  """Constant sampling for hypernetwork."""

  def __init__(self, args):
    super(Constant, self).__init__(args=args)

  def train_epoch_begin(self):
    super().train_epoch_begin()
    print('Constant Sampling')
  
  def sample_hparams(self, num_samples):
    hyperparams = torch.ones((num_samples, self.num_hparams)) * self.hyperparameters
    return hyperparams
  
  def set_eval_hparams(self):
    self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
    self.test_hparams = torch.tensor(self.hyperparameters).view(-1, 1)