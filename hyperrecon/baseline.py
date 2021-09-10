import torch
import numpy as np
from tqdm import tqdm

from hyperrecon.util.train import BaseTrain
from hyperrecon.util import utils
from hyperrecon.model.unet import Unet

class Baseline(BaseTrain):
  """Baseline."""

  def __init__(self, args):
    super(Baseline, self).__init__(args=args)

  def train_epoch_begin(self):
    print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
    print('Learning rate:', self.scheduler.get_last_lr())
    print('Baseline')
  

  def get_model(self):
    self.network = Unet(
                      in_ch=self.n_ch_in,
                      out_ch=self.n_ch_out,
                      h_ch=self.unet_hdim,
                      use_batchnorm=self.use_batchnorm
                   ).to(self.device)
    utils.summary(self.network)
    return self.network

  def inference(self, zf, hyperparams):
    return self.network(zf)

  def sample_hparams(self, num_samples):
    hyperparams = torch.ones((num_samples, self.num_hyperparams)) * self.hyperparameters
    return hyperparams
  
  def set_eval_hparams(self):
    self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
    self.test_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
  
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