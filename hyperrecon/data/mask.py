import numpy as np
import torch

def get_mask(mask_dims, undersampling_rate, as_tensor=True, centered=False):
  # mask = np.load('data/mask.npy')
  path = '/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_{maskname}_{mask_dims}.npy'.format(maskname=undersampling_rate, mask_dims=mask_dims)
  print('Loading mask:', path)
  mask = np.load(path)

  if not centered:
    mask = np.fft.fftshift(mask)
  if as_tensor:
    return torch.tensor(mask, requires_grad=False).float()
  else:
    return mask