import numpy as np
import torch
from .data_util import ArrDataset
import os
from torchvision import transforms
from glob import glob
from hyperrecon.model.layers import ClipByPercentile

class KneeBase():
  def __init__(self, batch_size):
    self.batch_size = batch_size

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
    train_gt = np.moveaxis(train_gt, [0,1,2,3], [0,2,3,1])
    test_gt = np.moveaxis(test_gt, [0,1,2,3], [0,2,3,1])
    self.trainset = ArrDataset(
      train_gt[:int(len(train_gt)*0.8)])
    self.valset = ArrDataset(
      train_gt[int(len(train_gt)*0.8):])
    self.testset = ArrDataset(test_gt)

def get_train_gt():
  gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_train_normalized.npy'
  print('loading', gt_path)
  gt = np.load(gt_path)
  return gt

def get_test_gt():
  gt_path = '/share/sablab/nfs02/users/aw847/data/knee/knee_test_normalized.npy'
  print('loading', gt_path)
  gt = np.load(gt_path)
  return gt

class FastMRI(KneeBase):
  def __init__(self, 
               batch_size,
               eval_subsample=50,
               img_dims=(256,256)
              ):
    super(FastMRI, self).__init__(batch_size)
    self.batch_size = batch_size
    self.data_path = '/share/sablab/nfs02/data/fastmri/'

    transform = transforms.Compose([transforms.ToPILImage('F'), transforms.Resize(img_dims), transforms.ToTensor(), ClipByPercentile()])
    self.trainset = SliceDataset(os.path.join(self.data_path, 'singlecoil_train/slices'), transform=transform)
    self.valset = SliceDataset(os.path.join(self.data_path, 'singlecoil_val/slices'), subsample=eval_subsample, transform=transform)
    self.testset = SliceDataset(os.path.join(self.data_path, 'singlecoil_val/slices'), subsample=eval_subsample, transform=transform)

class SliceDataset(torch.utils.data.Dataset):
  def __init__(self, data_path, subsample=None, transform=None):
    super(SliceDataset, self).__init__()
    self.data_path = data_path
    self.transform = transform
    self.subsample = subsample

    print('Gathering subjects for dataloader')
    self.slices = self._gather_slices()
    print('done')

  def _gather_slices(self):
    glob_path = os.path.join(self.data_path, '*/*.npy')
    print(glob_path)
    paths = glob(glob_path)
    if self.subsample is not None:
      indices = np.random.choice(len(paths), size=self.subsample, replace=False)
      paths = [paths[i] for i in indices]
    return paths

  def __len__(self):
    return len(self.slices)

  def __getitem__(self, index):
    # Load data and get label
    path = self.slices[index]
    x = np.load(path).astype('float32')
    if self.transform is not None:
      x = self.transform(x)

    return x