import torch
import random
from hyperrecon.util.train import BaseTrain

class BinaryConstantBatch(BaseTrain):
  """BinaryConstantBatch."""

  def __init__(self, args):
    super(BinaryConstantBatch, self).__init__(args=args)

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('Binary Constant Batches')
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    if random.random() < 0.5:
      return torch.zeros(num_samples, self.num_hparams)
    else:
      return torch.ones(num_samples, self.num_hparams)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)