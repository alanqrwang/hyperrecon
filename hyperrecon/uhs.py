import torch
import numpy as np
from hyperrecon.util.train import BaseTrain

class UHS(BaseTrain):
  """UHS."""

  def __init__(self, args):
    super(UHS, self).__init__(args=args)

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('UHS Sampling')
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    # self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    # self.test_hparams = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).view(-1, 1)
    self.val_hparams = torch.tensor([[0.,0.], [1.,1.]])
    hparams = []
    for i in np.linspace(0, 1, 50):
      for j in np.linspace(0, 1, 50):
        hparams.append([i, j])
    self.test_hparams = torch.tensor(hparams).float()