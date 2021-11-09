"""
Dataloaders for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
from torch.utils import data
import numpy as np
import os
from glob import glob
from torchvision import transforms
from hyperrecon.model.layers import ClipByPercentile, ZeroPad
from hyperrecon.util.noise import RicianNoise
from .data_util import ArrDataset

class BrainBase():
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
          batch_size=1,
          shuffle=False,
          num_workers=0,
          pin_memory=True)
    return train_loader, val_loader, test_loader

class BrainArr(BrainBase):
  def __init__(self, batch_size, split_ratio=0.8):
    super(BrainArr, self).__init__(batch_size)
    self.split_ratio = split_ratio
    train_gt = get_train_gt()
    test_gt = get_test_gt()[:10]
    self.trainset = ArrDataset(
      train_gt[:int(len(train_gt)*self.split_ratio)])
    self.valset = ArrDataset(
      train_gt[int(len(train_gt)*self.split_ratio):])
    self.testset = ArrDataset(test_gt)

class Abide(BrainBase):
  def __init__(self, 
               batch_size, 
               num_train_subjects=50,
               num_val_subjects=5,
               subsample_test=False,
               img_dims=(256,256),
               rician_snr=5,
               clip_percentile=100,
               fixed_noise=False):
    super(Abide, self).__init__(batch_size)
    self.data_path = '/share/sablab/nfs02/users/aw847/data/brain/abide/'

    transform = transforms.Compose([ClipByPercentile(clip_percentile), ZeroPad(img_dims), RicianNoise(img_dims, snr=rician_snr, fixed=fixed_noise)])
    target_transform = transforms.Compose([ZeroPad(img_dims)])

    self.trainset = SliceDataset(
      self.data_path, 'train', total_subjects=num_train_subjects, transform=transform, target_transform=target_transform)
    self.valset = SliceDataset(
      self.data_path, 'validate', total_subjects=num_val_subjects, transform=transform, target_transform=target_transform)
    self.testset = SliceVolDataset(
      self.data_path, 'validate', total_subjects=num_val_subjects, transform=transform, target_transform=target_transform,
      subsample=subsample_test)

class SliceDataset(data.Dataset):
  def __init__(self, data_path, split, total_subjects=None, transform=None, target_transform=None):
    super(SliceDataset, self).__init__()
    self.data_path = data_path
    self.split = split
    self.transform = transform
    self.target_transform = target_transform
    self.total_subjects = total_subjects

    print('Gathering subjects for ABIDE dataloader')
    self.slices = self._gather_slices()
    print('done')

  def _gather_slices(self):
    paths = sorted(glob(os.path.join(self.data_path, self.split, '*.npy')))
    aseg_paths = sorted(glob(os.path.join(self.data_path, self.split, 'asegs', '*.npy')))

    subject_names = set() # Save subject names for purposes of limiting total subjects
    slices = [] # Save slice paths

    for path in paths:
      path_name = path.split('/')[-1][:-4] # e.g. <>.npy
      subject_name, subject_slice = path_name[:-9], path_name[-5:] # e.g. '<>_orig_0000'
      dir_name = os.path.dirname(path)
      subject_names.add(subject_name)
      # Check if number of subjects exceeds total subjects
      if len(subject_names) > self.total_subjects:
        break
      elif os.path.isfile(path):
        aseg_path = os.path.join(dir_name, 'asegs', subject_name + 'aseg' + subject_slice + '.npy')
        assert aseg_path in aseg_paths, 'Invalid aseg path name'
        slices.append((path, aseg_path))

    return slices

  def __len__(self):
    return len(self.slices)

  def __getitem__(self, index):
    # Load data and get label
    path, aseg_path = self.slices[index]
    x = np.load(path)
    y = np.load(aseg_path)
    if self.transform is not None:
      x = self.transform(x)
    if self.target_transform is not None:
      y = self.target_transform(y)

    return x[None], y[None]

class SliceVolDataset(data.Dataset):
  def __init__(self, data_path, split, total_subjects=None, transform=None, target_transform=None, subsample=False):
    super(SliceVolDataset, self).__init__()
    self.data_path = data_path
    self.split = split
    self.transform = transform
    self.target_transform = target_transform
    self.total_subjects = total_subjects
    self.subsample = subsample

    print('Gathering volumes for ABIDE dataloader')
    self.vols = self._gather_vols()
    print('done')

  def _gather_vols(self):
    paths = sorted(glob(os.path.join(self.data_path, self.split, '*.npy')))
    aseg_paths = sorted(glob(os.path.join(self.data_path, self.split, 'asegs', '*.npy')))

    subject_names = set() # Save subject names for purposes of limiting total subjects
    vols = {} # Save slice paths

    for path in paths:
      # Check if number of subjects exceeds total subjects
      path_name = path.split('/')[-1][:-4] # e.g. <>.npy
      subject_name, subject_slice = path_name[:-9], path_name[-5:] # e.g. '<>_orig_0000'
      dir_name = os.path.dirname(path)
      subject_names.add(subject_name)
      if len(subject_names) > self.total_subjects:
        break
      elif os.path.isfile(path):
        aseg_path = os.path.join(dir_name, 'asegs', subject_name + 'aseg' + subject_slice + '.npy')
        assert aseg_path in aseg_paths, 'Invalid aseg path name'
        if subject_name in vols:
          vols[subject_name].append((path, aseg_path))
        else:
          vols[subject_name] = [(path, aseg_path)]

    return list(vols.values())

  def __len__(self):
    return len(self.vols)

  def __getitem__(self, index):
    # Load data and get label
    xs, ys = None, None
    vol = np.array(self.vols[index])
    if self.subsample:
      indices = np.arange(0, 192, 20)
      vol = vol[indices]
    for path, aseg_path in vol:
      x = np.load(path)
      y = np.load(aseg_path)
      if self.transform is not None:
        x = self.transform(x)
      if self.target_transform is not None:
        y = self.target_transform(y)
      
      xs = x[None] if xs is None else np.concatenate((xs, x[None]), axis=0)
      ys = y[None] if ys is None else np.concatenate((ys, y[None]), axis=0)

    return xs, ys

def get_train_gt():
  # gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_train_normalized_rician_snr5_mu0_sigma1.npy'
  gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_train_normalized.npy'
  # gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/train_mri_normalized_{img_dims}.npy'.format(img_dims=img_dims)
  print('loading', gt_path)
  gt = np.load(gt_path)
  if gt.shape[1] != 1:
    gt = np.moveaxis(gt, [0,1,2,3], [0,2,3,1])
  return gt

def get_test_gt():
  # gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized_rician_snr5_mu0_sigma1.npy'
  gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/brain_test_normalized.npy'
  # gt_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/test_mri_normalized_{img_dims}.npy'.format(img_dims=img_dims)
  print('loading', gt_path)
  gt = np.load(gt_path)
  if gt.shape[1] != 1:
    gt = np.moveaxis(gt, [0,1,2,3], [0,2,3,1])
  return gt

def get_seg_data(split, img_dims):
  seg_path = '/share/sablab/nfs02/users/aw847/data/brain/adrian/segs/{split}_seg_{img_dims}.npy'.format(split=split, img_dims=img_dims)
  seg = np.load(seg_path)
  return seg
