import numpy as np
import torch

def get_mask(mask_type, mask_dims, undersampling_rate, centered=False):
  '''Load mask.

  Args:
    mask_type: Type of mask
    mask_dims: Dimension of mask
    undersampling_rate: Undersampling rate of mask
    as_tensor: Whether or not to return as torch tensor
    centered: Center low-frequency in image. Should be un-centered for everything,
      use centered for visualization only.
  '''
  if mask_type == 'poisson':
    mask = vds_poisson(mask_dims, undersampling_rate)
  elif mask_type == 'epi_horizontal':
    mask = epi_horizontal(mask_dims, undersampling_rate)
  elif mask_type == 'epi_vertical':
    mask = epi_vertical(mask_dims, undersampling_rate)
  elif mask_type == 'first_half':
    mask = first_half(mask_dims)
  elif mask_type == 'second_half':
    mask = second_half(mask_dims)
  elif mask_type == 'center_patch':
    mask = center_patch(mask_dims)

  if not centered:
    mask = np.fft.fftshift(mask)
  return torch.tensor(mask, requires_grad=False).float()

def vds_poisson(mask_dims, undersampling_rate):
  path = '/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_{maskname}_{dim1}_{dim2}.npy'.format(maskname=undersampling_rate, dim1=mask_dims[0], dim2=mask_dims[1])
  print('Loading mask:', path)
  mask = np.load(path)

  return mask
  
def epi_horizontal(mask_dims, undersampling_rate):
  path = '/share/sablab/nfs02/users/aw847/data/masks/EPI_{rate}_{dim1}_{dim2}.npy'.format(rate=undersampling_rate, dim1=mask_dims[0], dim2=mask_dims[1])
  print('Loading mask:', path)
  mask = np.load(path)
  return mask

def epi_vertical(mask_dims, undersampling_rate, percent_acs=0.2):
  '''See https://arxiv.org/pdf/1811.08839.pdf section 4.9.'''
  # total_lines = mask_dims[1]
  # mask = np.zeros(mask_dims)
  # num_sampled_lines = int(np.prod(mask_dims) / int(undersampling_rate) / mask_dims[0])
  # middle_lines = int(num_sampled_lines * percent_acs)

  # center_line_idx = np.arange((total_lines - middle_lines) // 2,
  #                     (total_lines + middle_lines) // 2).astype(np.int)

  # # Find remaining candidates
  # outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx).astype(np.int)

  # # Sample remaining lines from outside the ACS at random
  # random_line_idx = np.random.choice(outer_line_idx,
  #           size=int(num_sampled_lines - middle_lines), replace=False)

  # # Create a mask and place ones at the right locations
  # mask[:, center_line_idx] = 1.
  # mask[:, random_line_idx] = 1.
  # print('Generated EPI Vertical with mean', mask.mean())

  path = '/share/sablab/nfs02/users/aw847/data/masks/EPI_vertical_{rate}_{dim1}_{dim2}.npy'.format(rate=undersampling_rate, dim1=mask_dims[0], dim2=mask_dims[1])
  print('Loading mask:', path)
  mask = np.load(path)
  return mask

def first_half(mask_dims):
  mask = np.zeros(mask_dims)
  mask[mask_dims[0]//2::, :] = 1
  return mask

def second_half(mask_dims):
  mask = np.zeros(mask_dims)
  mask[:mask_dims[0]//2, :] = 1
  return mask

def center_patch(mask_dims):
  p_dim = (50, 50)
  mask = np.ones(mask_dims)
  mask[mask_dims[0]//2 - p_dim[0]//2 : mask_dims[0]//2 + p_dim[0]//2, \
       mask_dims[1]//2 - p_dim[1]//2 : mask_dims[1]//2 + p_dim[1]//2] = 0
  return mask