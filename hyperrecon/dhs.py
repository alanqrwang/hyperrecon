import torch

from hyperrecon.util.train import BaseTrain

class DataDriven(BaseTrain):
  """DataDriven."""

  def __init__(self, args):
    super(DataDriven, self).__init__(args=args)

  def sample_hparams(self, num_samples):
    '''Samples hyperparameters from distribution.'''
    return torch.FloatTensor(num_samples, self.num_hparams).uniform_(0, 1)

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