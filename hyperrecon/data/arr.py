import numpy as np
import torch

class Arr:
  def __init__(self, batch_size, train_path, test_path):
    self.batch_size = batch_size
    train_gt = np.load(train_path)
    test_gt = np.load(test_path)
    assert len(train_gt.shape) == 4 and len(test_gt.shape) == 4, \
      'Invalid dataset shape'
    assert train_gt.shape[1] == 1 and test_gt.shape[1] == 1, \
      'Invalid channel dimensions'
    self.trainset = ArrDataset(train_gt)
    self.valset = ArrDataset(test_gt)

  def load(self):
    train_loader = torch.utils.data.DataLoader(self.trainset, 
          batch_size=self.batch_size,
          shuffle=True,
          num_workers=0,
          pin_memory=True,
          drop_last=True)
    val_loader = torch.utils.data.DataLoader(self.valset,
          batch_size=self.batch_size*2,
          shuffle=False,
          num_workers=0,
          pin_memory=True)
    return train_loader, val_loader

class ArrDataset(torch.utils.data.Dataset):
  def __init__(self, x):
    self.x = x

  def __len__(self):
    return len(self.x)

  def __getitem__(self, index):
    # Load data and get label
    return self.x[index]