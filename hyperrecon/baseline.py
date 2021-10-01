import torch
from hyperrecon.util.train import BaseTrain

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