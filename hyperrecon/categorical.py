import torch
import random
from hyperrecon.util.train import BaseTrain

class CategoricalConstant(BaseTrain):
  """CategoricalConstant."""

  def __init__(self, args):
    super(CategoricalConstant, self).__init__(args=args)
    self.categories = [0, 0.5, 1]

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('Categorical Constant')
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    cat = random.choice(self.categories)
    return torch.ones(num_samples, self.num_hparams) * cat

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)