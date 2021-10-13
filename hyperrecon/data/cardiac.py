import numpy as np
import torch
import os
from glob import glob
from torchvision import transforms
from hyperrecon.model.layers import ClipByPercentile

class ACDC():
  def __init__(self, 
               batch_size,
               eval_subsample=50,
               img_dims=(256,256)
              ):
    super(ACDC, self).__init__()
    self.batch_size = batch_size
    self.data_path = '/share/sablab/nfs02/data/acdc/'

    transform = transforms.Compose([transforms.ToPILImage('F'), transforms.Resize(img_dims), transforms.ToTensor(), ClipByPercentile()])
    self.trainset = SliceDataset(os.path.join(self.data_path, 'training'), transform=transform)
    self.valset = SliceDataset(os.path.join(self.data_path, 'testing'), subsample=eval_subsample, transform=transform)
    self.testset = SliceDataset(os.path.join(self.data_path, 'testing'), subsample=eval_subsample, transform=transform)
    
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
    glob_path = os.path.join(self.data_path, '*/4d_slices/*.npy')
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