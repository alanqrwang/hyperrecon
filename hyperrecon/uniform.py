import torch
import random
from hyperrecon.util.train import BaseTrain

class Uniform(BaseTrain):
  """Uniform."""

  def __init__(self, args):
    super(Uniform, self).__init__(args=args)

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('UHS Sampling')
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
    # self.val_hparams = torch.tensor([[0.,0.], [1.,1.]])
    # hparams = []
    # for i in np.linspace(0, 1, 50):
    #   for j in np.linspace(0, 1, 50):
    #     hparams.append([i, j])
    # self.test_hparams = torch.tensor(hparams).float()

class UniformConstant(BaseTrain):
  """UniformConstant."""

  def __init__(self, args):
    super(UniformConstant, self).__init__(args=args)

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('Uniform Constant')
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    cat = random.random()
    return torch.ones(num_samples, self.num_hparams) * cat

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)