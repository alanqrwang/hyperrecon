from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from .utils import fft, ifft, scale

class BaseForward(ABC):
  def __init__(self):
    '''Base forward abstract class.

    Args:
      mask: Binary mask of where to under-sample (l, w)
    '''
    super().__init__()

  @abstractmethod
  def __call__(self, fullysampled, mask):
    pass

class CSMRIForward(BaseForward):
  '''Forward model for CS-MRI.'''
  def __init__(self):
    super(CSMRIForward, self).__init__()

  def __call__(self, fullysampled, mask):
    '''Generate under-sampled k-space data with given binary mask.
    
    Args:
      fullysampled: Clean image in image space (N, n_ch, l, w)
      mask: Stack of masks (N, 1, l, w)
    '''
    ksp = fft(fullysampled)
    under_ksp = ksp * mask
    zf = ifft(under_ksp)
    under_ksp, zf = scale(under_ksp, zf)
    return under_ksp

class InpaintingForward(BaseForward):
  '''Forward model for inpainting.'''
  def __init__(self):
    super(InpaintingForward, self).__init__()

  def __call__(self, fullysampled, mask):
    '''Generate masked version of input.
    
    Args:
      fullysampled: Clean image in image space (N, n_ch, l, w)
      mask: Stack of masks (N, 1, l, w)
    '''
    masked = fullysampled * mask
    masked = torch.cat((masked, torch.zeros_like(masked)), dim=1)
    return masked
  
class SuperresolutionForward(BaseForward):
  '''Forward model for super-resolution.'''
  def __init__(self, factor):
    super(SuperresolutionForward, self).__init__()
    self.factor = float(1/int(factor))

  def __call__(self, x, *args):
    '''Downsample input.
    
    Args:
      x: Clean image in image space (N, n_ch, l, w)
    '''
    del args
    x_down = F.interpolate(x, scale_factor=self.factor, mode='bicubic', align_corners=False) 
    x_down = F.upsample(x_down, scale_factor=int(1/self.factor), mode='nearest')
    x_down = torch.cat((x_down, torch.zeros_like(x_down)), dim=1)
    return x_down

class DenoisingForward(BaseForward):
  '''Forward model for de-noising.'''
  def __init__(self, sigma):
    super(DenoisingForward, self).__init__()
    self.sigma = sigma

  def __call__(self, x, *args):
    '''Downsample input.
    
    Args:
      x: Clean image in image space (N, n_ch, l, w)
    '''
    del args
    x_noise = x + torch.normal(0, self.sigma, size=x.shape).cuda()
    x_noise = torch.cat((x_noise, torch.zeros_like(x_noise)), dim=1)
    return x_noise