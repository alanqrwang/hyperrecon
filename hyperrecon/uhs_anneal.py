import torch

from hyperrecon.util.train import BaseTrain

class UHSAnneal(BaseTrain):
  """UHSAnneal."""

  def __init__(self, args):
    super(UHSAnneal, self).__init__(args=args)

  def process_loss(self, loss):
    return loss.mean()

  def train_epoch_begin(self):
    if self.epoch < 500:
      self.r1 = 0.5
      self.r2 = 0.5
    elif self.epoch < 1000:
      self.r1 = 0.4
      self.r2 = 0.6
    elif self.epoch < 1500:
      self.r1 = 0.2
      self.r2 = 0.8
    elif self.epoch < 2000:
      self.r1 = 0
      self.r2 = 1

    print('\nEpoch %d/%d' % (self.epoch, self.num_epochs))
    print('Learning rate:',
        self.lr if self.force_lr is None else self.force_lr)
    print('Sampling bounds [%.2f, %.2f]' % (self.r1, self.r2))
    print('UHS Anneal Sampling')