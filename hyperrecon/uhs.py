import torch

from hyperrecon.util.train import BaseTrain

class UHS(BaseTrain):
  """UHS."""

  def __init__(self, args):
    super(UHS, self).__init__(args=args)

  def process_loss(self, loss):
    return loss.mean()

  def train_epoch_begin(self, epoch):
      super().train_epoch_begin(epoch)
      print('UHS Sampling')