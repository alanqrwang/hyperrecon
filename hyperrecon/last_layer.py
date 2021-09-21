import torch
from hyperrecon.util.train import BaseTrain
from hyperrecon.util import utils
from hyperrecon.model.unet_v2 import LastLayerHyperUnet

class LastLayer(BaseTrain):
  """LastLayer."""

  def __init__(self, args):
    super(LastLayer, self).__init__(args=args)

  def get_model(self):
    self.network = LastLayerHyperUnet(
                      self.num_coeffs,
                      self.hnet_hdim,
                      in_ch_main=self.n_ch_in,
                      out_ch_main=self.n_ch_out,
                      h_ch_main=self.unet_hdim,
                      use_batchnorm=self.use_batchnorm
                    ).to(self.device)
    utils.summary(self.network)
    return self.network
  
  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

  def set_eval_hparams(self):
    self.val_hparams = torch.tensor([0., 1.]).view(-1, 1)
    self.test_hparams = torch.tensor([0., 1.]).view(-1, 1)