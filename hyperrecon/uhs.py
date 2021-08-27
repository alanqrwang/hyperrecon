import torch

from hyperrecon.util.train import BaseTrain

class UHS(BaseTrain):
  """UHS."""

  def __init__(self, args):
    super(UHS, self).__init__(args=args)

  def train_epoch_begin(self):
      super().train_epoch_begin()
      print('UHS Sampling')