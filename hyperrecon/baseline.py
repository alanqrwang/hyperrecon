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
    print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
    print('Learning rate:', self.scheduler.get_last_lr())
    print('Baseline')
  

  def get_model(self):
    self.network = Unet(in_ch=self.n_ch_in,
                    out_ch=self.n_ch_out,
                    h_ch=self.unet_hdim).to(self.device)
    utils.summary(self.network)
    return self.network

  def inference(self, zf, hyperparams):
    return self.network(zf)

  def sample_hparams(self, num_samples):
    hyperparams = torch.ones((num_samples, self.num_hyperparams)) * self.hyperparameters
    return hyperparams
  
  def set_val_hparams(self):
    self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
    self.test_hparams = torch.tensor(self.hyperparameters).view(-1, 1)