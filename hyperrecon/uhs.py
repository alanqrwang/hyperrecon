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
  
  def set_eval_hparams(self):
    # self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    # self.test_hparams = torch.tensor([0., 0.25, 0.5, 0.75, 1.]).view(-1, 1)
    self.val_hparams = torch.tensor([[0.,0.], [1.,1.]])
    hparams = []
    for i in np.linspace(0, 1, 50):
      for j in np.linspace(0, 1, 50):
        hparams.append([i, j])
    self.test_hparams = torch.tensor(hparams).float()

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
      'loss:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'psnr:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'ssim:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'hfen:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'watson:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ] + [
      'mae:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    # ] + [
    #   'dice:test:' + self.stringify_list(l.tolist()) + ':sub{}'.format(s) for l in self.test_hparams for s in np.arange(self.num_val_subjects)
    ]
