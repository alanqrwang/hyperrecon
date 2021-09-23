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

  def inference(self, zf, coeffs):
    return self.network(zf)

  def sample_hparams(self, num_samples):
    hyperparams = torch.ones((num_samples, self.num_hparams)) * self.hyperparameters
    return hyperparams
  
  def set_eval_hparams(self):
    self.val_hparams = torch.tensor(self.hyperparameters).view(-1, 1)
    self.test_hparams = torch.tensor(self.hyperparameters).view(-1, 1)