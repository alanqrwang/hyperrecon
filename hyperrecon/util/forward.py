from abc import ABC, abstractmethod
import torch
from .utils import fft, ifft, scale

class BaseForward(ABC):
  def __init__(self, mask):
    '''Base forward abstract class.

    Args:
      mask: Binary mask of where to under-sample (l, w)
    '''
    self.mask = mask.unsqueeze(0)
    super().__init__()

  @abstractmethod
  def generate_measurement(self, fullysampled):
    pass

class CSMRIForward(BaseForward):
  '''Forward model for CS-MRI.'''
  def __init__(self, mask):
    super(CSMRIForward, self).__init__(mask)

  def generate_measurement(self, fullysampled):
    '''Generate under-sampled k-space data with given binary mask.
    
    Args:
      fullysampled: Clean image in image space (N, n_ch, l, w)
    '''
    ksp = fft(fullysampled)
    under_ksp = ksp * self.mask
    zf = ifft(under_ksp)
    under_ksp, zf = scale(under_ksp, zf)
    return zf, under_ksp

class InpaintingForward(BaseForward):
  '''Forward model for inpainting.'''
  def __init__(self, mask):
    super(InpaintingForward, self).__init__(mask)

  def generate_measurement(self, fullysampled):
    '''Generate masked version of input.
    
    Args:
      fullysampled: Clean image in image space (N, n_ch, l, w)
    '''
    print(fullysampled.shape, self.mask.shape)
    masked = fullysampled * self.mask
    masked = torch.cat((masked, torch.zeros_like(masked)), dim=1)
    return masked, fft(masked)