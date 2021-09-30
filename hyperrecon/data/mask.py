import numpy as np
import torch
import torch.nn as nn

class BaseMask(nn.Module):
  def __init__(self, dims, rate):
    super(BaseMask, self).__init__()

    self.dims = dims
    self.rate = rate
    self.mask = self._config()

  def _config(self):
    pass

  def forward(self):
    return torch.tensor(self.mask, requires_grad=False).float()


class Loupe(BaseMask):
  def __init__(self, dims, rate, pmask_slope=5, mask_eps=0.01, temp=0.8):
    super(Loupe, self).__init__(dims, rate)

    self.pmask_slope = pmask_slope
    self.temp = temp
    self.sigmoid = nn.Sigmoid()
    self.mask_eps = mask_eps

    self.pmask = nn.Parameter(torch.FloatTensor(*self.dims))         
    self.pmask.requires_grad = True
    self.pmask.data.uniform_(self.mask_eps, 1-self.mask_eps)
    self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope
    self.pmask.data = self.pmask.data.to(self.device)
    
  def sample_gumbel(self, shape, eps=1e-20):
    U = torch.rand(shape).to(self.device)
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, p, temperature):
    g1 = self.sample_gumbel(p.size())
    g2 = self.sample_gumbel(p.size())
    return 1-self.sigmoid((torch.log(1-p) - torch.log(p) + g1 - g2)/temperature)

  def binary_gumbel_softmax(self, pmask, temperature):
    """Shape-agnostic binary Gumbel-Softmax sampler

    input: (*) probabilistic mask
    return: (*) pixel-wise Bernoulli realization
    """
    y = self.gumbel_softmax_sample(pmask, temperature)
    y_hard = y.round()
    return (y_hard - y).detach() + y

  def squash_mask(self, mask):
    return self.sigmoid(self.pmask_slope*mask)

  def sparsify(self, mask, rate):
    xbar = mask.mean()
    r = rate / xbar
    beta = (1-rate) / (1-xbar)
    le = (r <= 1).float()
    return le * mask * r + (1-le) * (1 - (1 - mask) * beta)

  def forward(self, rate):
    mask = self.squash_mask(self.pmask)
    mask = self.sparsify(mask, rate)
    mask = self.binary_gumbel_softmax(mask, self.temp)
    return mask

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

    path = '/share/sablab/nfs02/users/aw847/data/masks/EPI_vertical_{rate}_{dim1}_{dim2}.npy'.format(rate=self.rate, dim1=self.dims[0], dim2=self.dims[1])
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