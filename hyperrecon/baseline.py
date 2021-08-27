import torch

from hyperrecon.util.train import BaseTrain
from hyperrecon.util import utils
from hyperrecon.model.unet import Unet

class Baseline(BaseTrain):
  """Baseline."""

  def __init__(self, args):
    super(Baseline, self).__init__(args=args)

  def process_loss(self, loss):
    return loss.mean()

  def train_epoch_begin(self):
    assert self.hyperparameters is not None
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

  def sample_hparams(self, num_samples, is_training=False):
    hyperparams = torch.ones((num_samples, self.num_hyperparams)).to(
          self.device) * self.hyperparameters
    return hyperparams