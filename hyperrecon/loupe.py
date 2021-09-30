import torch
from hyperrecon.util.train import BaseTrain

class RateAgnostic(BaseTrain):
  """RateAgnostic."""

  def __init__(self, args):
    super(RateAgnostic, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
  
  def mask_inference(self, x):
    return self.mask_module(x)