import torch
import time
from tqdm import tqdm

from hyperrecon.util.train import BaseTrain
from hyperrecon.util import utils
from hyperrecon.model.unet import Unet

class Baseline(BaseTrain):
  """Baseline."""

  def __init__(self, args):
    super(Baseline, self).__init__(args=args)

  def train_epoch_begin(self):
    self.r1 = self.hyperparameters
    self.r2 = self.hyperparameters

    print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
    print('Learning rate:',
        self.lr if self.force_lr is None else self.force_lr)
    print('Sampling bounds [%.2f, %.2f]' % (self.r1, self.r2))
    print('Baseline')
  
  def inference(self, zf, hyperparams):
    return self.network(zf)

  def get_model(self):
    self.network = Unet(in_ch=self.n_ch_in,
                    out_ch=self.n_ch_out,
                    h_ch=self.unet_hdim).to(self.device)
    utils.summary(self.network)
    return self.network

  def sample_hparams(self, num_samples):
    hyperparams = torch.ones((num_samples, self.num_hyperparams)) * self.hyperparameters
    return hyperparams
  
  def set_hparams_for_eval(self):
    self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
  
  def set_metrics(self):
    self.list_of_metrics = [
      'loss.train',
      'psnr.train',
      'time.train',
    ] 
    self.list_of_eval_metrics = [
      'loss.val{:02f}'.format(self.hyperparameters)
    ] + [
      'psnr.val{:02f}'.format(self.hyperparameters) 
    ] + [
      'time.val{:02f}'.format(self.hyperparameters) 
    ]
      # 'ssim.val',
      # 'hfen.val',
    # ] 