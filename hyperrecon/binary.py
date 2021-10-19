import torch
import random
import numpy as np
from hyperrecon.util.train import BaseTrain


class BinaryAnneal(BaseTrain):
  """BinaryAnneal.
  
  TODO: For now, expects that model weights are pretrained on binary constant batches.
  """

  def __init__(self, args):
    super(BinaryAnneal, self).__init__(args=args)
    self.p = self.p_min

  def set_monitor(self):
    self.list_of_monitor = [
      'learning_rate', 
      'time:train',
      'p_value',
    ]

  def set_metrics(self):
    self.list_of_metrics = [
      'loss:train',
      'psnr:train',
    ]
    self.list_of_val_metrics = [
      'loss:val:' + self.stringify_list(l.tolist()) for l in self.val_hparams
    ] + [
      'psnr:val:' + self.stringify_list(l.tolist()) for l in self.val_hparams
    ]
    self.list_of_test_metrics = [
    ]

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('Binary Annealing')
      print('p-value:', self.p)
      self.monitor['p_value'].append(self.p)
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.
    
    Linearly increase p from p_min to p_max.
    '''
    self.p = min((self.p_max - self.p_min) / self.epoch_of_p_max * self.epoch + self.p_min, self.p_max)
    if random.random() < 0.5:
      samples = torch.bernoulli(torch.empty(num_samples, self.num_hparams).fill_(self.p))
    else:
      samples = torch.bernoulli(torch.empty(num_samples, self.num_hparams).fill_(1-self.p))
    print(self.p)
    print(samples)
    return samples

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)
  
