import torch

from hyperrecon.util.train import BaseTrain

class DHS(BaseTrain):
  """DHS."""

  def __init__(self, args):
    super(DHS, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    hyperparams = self.sampler.sample(
      num_samples, 0, 1)
    return hyperparams

  def process_loss(self, loss):
    dc_losses = loss_dict['dc']
    _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
    sort_losses = self.losses[perm] # Reorder total losses by lowest to highest DC loss
    hyperparams = hyperparams[perm]
    loss = torch.mean(sort_losses[:self.topK]) # Take only the losses with lowest DC

    return loss

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('DHS Sampling')