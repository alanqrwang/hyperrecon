import numpy as np
import torch
from .data_util import ArrDataset

class KneeBase():
  def __init__(self, batch_size):
    self.batch_size = batch_size

  def load(self):
    train_loader = torch.utils.data.DataLoader(self.trainset, 
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=0,
          pin_memory=True)
    val_loader = torch.utils.data.DataLoader(self.valset,
          batch_size=self.batch_size*2,
          shuffle=False,
          num_workers=0,
          pin_memory=True)
    test_loader = torch.utils.data.DataLoader(self.testset,
          batch_size=self.batch_size*2,
          shuffle=False,
          num_workers=0,
          pin_memory=True)
    return train_loader, val_loader, test_loader

class KneeArr(KneeBase):
  def __init__(self, batch_size):
    super(KneeArr, self).__init__(batch_size)

    train_gt = get_train_gt()
    test_gt = get_test_gt()
    self.trainset = ArrDataset(
      train_gt[:int(len(train_gt)*0.8)], None)
    self.valset = ArrDataset(
      train_gt[int(len(train_gt)*0.8):], None)
    self.testset = ArrDataset(
      test_gt, None)

def get_train_gt():
  gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_train_normalized.npy'
  gt = np.load(gt_path)
  return gt

def get_test_gt():
  gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_test_normalized.npy'
  gt = np.load(gt_path)
  return gt