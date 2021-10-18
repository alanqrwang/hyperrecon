import numpy as np
import torch
import torch.nn as nn
import sys

class BaseMask(nn.Module):
  def __init__(self, dims, rate):
    super(BaseMask, self).__init__()

    self.dims = dims
    self.rate = rate
    self.mask = self._config()

  def _config(self):
    pass

  def forward(self, num_samples):
    mask_stack = self.mask[None, None].repeat(num_samples, 1, 1, 1)
    return mask_stack

class VDSPoisson(BaseMask):
  def _config(self):
    path = '/share/sablab/nfs02/users/aw847/data/masks/poisson_disk_{maskname}_{dim1}_{dim2}.npy'.format(maskname=self.rate, dim1=self.dims[0], dim2=self.dims[1])
    print('Loading mask:', path)
    mask = np.load(path)
    mask = np.fft.fftshift(mask)
    return torch.tensor(mask, requires_grad=False).float()
  
class EPIHorizontal(BaseMask):
  def _config(self):
    path = '/share/sablab/nfs02/users/aw847/data/masks/EPI_{rate}_{dim1}_{dim2}.npy'.format(rate=self.rate, dim1=self.dims[0], dim2=self.dims[1])
    print('Loading mask:', path)
    mask = np.load(path)
    mask = np.fft.fftshift(mask)
    return torch.tensor(mask, requires_grad=False).float()

class EPIVertical(BaseMask):
  def _config(self):
    '''See https://arxiv.org/pdf/1811.08839.pdf section 4.9.'''
    # total_lines = mask_dims[1]
    # mask = np.zeros(mask_dims)
    # num_sampled_lines = int(np.prod(mask_dims) / int(rate) / mask_dims[0])
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

    if self.rate == '4':
      path = '/share/sablab/nfs02/users/aw847/data/masks/EPI_vertical_{rate}_percentacs50_{dim1}_{dim2}.npy'.format(rate=self.rate, dim1=self.dims[0], dim2=self.dims[1])
    elif self.rate == '8':
      path = '/share/sablab/nfs02/users/aw847/data/masks/EPI_vertical_{rate}_percentacs50_{dim1}_{dim2}.npy'.format(rate=self.rate, dim1=self.dims[0], dim2=self.dims[1])
    else:
      raise ValueError('no mask path found')
    print('Loading mask:', path)
    mask = np.load(path)
    mask = np.fft.fftshift(mask)
    return torch.tensor(mask, requires_grad=False).float()

class FirstHalf(BaseMask):
  def _config(self):
    mask = np.zeros(self.dims)
    mask[self.dims[0]//2:, :] = 1
    return torch.tensor(mask, requires_grad=False).float()

class SecondHalf(BaseMask):
  def _config(self):
    mask = np.zeros(self.dims)
    mask[:self.dims[0]//2, :] = 1
    return torch.tensor(mask, requires_grad=False).float()

class CenterPatch(BaseMask):
  def _config(self):
    p_dim = (50, 50)
    mask = np.ones(self.dims)
    mask[self.dims[0]//2 - p_dim[0]//2 : self.dims[0]//2 + p_dim[0]//2, \
        self.dims[1]//2 - p_dim[1]//2 : self.dims[1]//2 + p_dim[1]//2] = 0
    return torch.tensor(mask, requires_grad=False).float()
    
class RandomBox(nn.Module):
  def __init__(self, 
               dims, 
               scale=(0.02, 0.3),
               ratio=3.3):
    super(RandomBox, self).__init__()

    self.dims = dims
    self.scale = scale
    self.ratio = ratio

  def forward(self, num_samples):
    mask_stack = None
    for _ in range(num_samples):
        ones = torch.ones(*self.dims)
        mask = self._cutpatch(ones, self.scale, self.ratio)[None, None]
        mask_stack = mask if mask_stack is None else torch.cat((mask_stack, mask), dim=0)
    return mask_stack
  
  def _cutpatch(self, img, scale, ratio):
    image_height, image_width = self.dims
    area = image_height * image_width
    erase_area = torch.FloatTensor(1).uniform_(scale[0], scale[1]) * area
    aspect_ratio = torch.FloatTensor(1).uniform_(1, ratio)
    aspect_ratio = aspect_ratio if torch.rand(1) > 0.5 else 1.0 / aspect_ratio
    pad_h = int(torch.round(torch.sqrt(erase_area * aspect_ratio)).long())
    pad_h = min(pad_h, image_height - 1)
    pad_w = int(torch.round(torch.sqrt(erase_area / aspect_ratio)).long())
    pad_w = min(pad_w, image_width - 1)

    cutout_center_height = torch.FloatTensor(1).uniform_(0, (image_height - pad_h)).long()
    cutout_center_width = torch.FloatTensor(1).uniform_(0, (image_width - pad_w)).long()

    lower_pad = cutout_center_height
    upper_pad = max(0, image_height - cutout_center_height - pad_h)
    left_pad = cutout_center_width
    right_pad = max(0, image_width - cutout_center_width - pad_w)

    # randomly cut patch
    img[lower_pad:image_height - upper_pad, left_pad:image_width - right_pad] = 0
    return img

class Loupe(nn.Module):
  def __init__(self, dims, pmask_slope=5, temp=0.8):
    super(Loupe, self).__init__()

    self.dims = dims
    self.pmask_slope = pmask_slope
    self.temp = temp
    self.sigmoid = nn.Sigmoid()
    self.init_parameters()

  def init_parameters(self, mask_eps=0.01):
    self.pmask = nn.Parameter(torch.FloatTensor(*self.dims))         
    self.pmask.requires_grad = True
    self.pmask.data.uniform_(mask_eps, 1-mask_eps)
    self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
    self.pmask.data = self.pmask.data.cuda()
    
  def sample_gumbel(self, shape, eps=sys.float_info.epsilon):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, p, eps=sys.float_info.epsilon):
    g1 = self.sample_gumbel(p.size())
    g2 = self.sample_gumbel(p.size())
    return 1-self.sigmoid((torch.log(1-p+eps) - torch.log(p+eps) + g1 - g2)/self.temp)

  def binary_gumbel_softmax(self, pmask):
    """Shape-agnostic binary Gumbel-Softmax sampler

    input: (*) probabilistic mask
    return: (*) pixel-wise Bernoulli realization
    """
    y = self.gumbel_softmax_sample(pmask)
    y_hard = y.round()
    return (y_hard - y).detach() + y

  def squash_mask(self, mask):
    return self.sigmoid(self.pmask_slope*mask)

  def sparsify(self, masks, rate, eps=sys.float_info.epsilon):
    xbar = masks.mean(dim=(1, 2, 3))
    r = rate / (xbar+eps)
    beta = (1-rate) / (1-xbar+eps)
    le = (r <= 1).float()
    r = r[..., None, None, None]
    beta = beta[..., None, None, None]
    le = le[..., None, None, None]
    return le * masks * r + (1-le) * (1 - (1 - masks) * beta)

  def forward(self, num_samples, rate):
    '''
    Args:
      num_samples: Number of masks to generate
      rate: (num_samples, 1) Rates for each mask
    '''
    mask = self.pmask.clone()
    mask = self.squash_mask(mask)
    mask = mask[None, None].repeat(num_samples, 1, 1, 1)
    mask = self.sparsify(mask, rate.squeeze())
    mask = self.binary_gumbel_softmax(mask)
    return mask

class ConditionalLoupe(Loupe):
  def init_parameters(self, mask_eps=0.01):
    self.fc1 = nn.Linear(1, 32) 
    self.fc2 = nn.Linear(32, 128)
    self.fc_last = nn.Linear(128, np.prod(self.dims))
    self.relu = nn.ReLU()

  def forward(self, rate):
    '''
    Args:
      num_samples: Number of masks to generate
      rate: (num_samples, 1) Rates for each mask
    '''
    x = self.relu(self.fc1(rate))
    x = self.relu(self.fc2(x))
    x = self.fc_last(x).view(-1, 1, *self.dims)
    mask = self.squash_mask(x)
    mask = self.sparsify(mask, rate.squeeze())
    mask = self.binary_gumbel_softmax(mask)
    return mask