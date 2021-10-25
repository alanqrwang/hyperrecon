import torch

from hyperrecon.util.train import BaseTrain

class DataDriven(BaseTrain):
  """DataDriven."""

  def __init__(self, args):
    super(DataDriven, self).__init__(args=args)

  def process_loss(self, loss, loss_dict):
    dc_losses = loss_dict['dc']
    _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
    sort_losses = loss[perm] # Reorder total losses by lowest to highest DC loss
    loss = torch.mean(sort_losses[:self.topK]) # Take only the losses with lowest DC

    return loss