from abc import ABC, abstractmethod
import torch
from .utils import fft, ifft, scale

class BaseForward(ABC):
  def __init__(self):
    '''Base forward abstract class.

    Args:
      mask: Binary mask of where to under-sample (l, w)
    '''
    super().__init__()

  @abstractmethod
  def generate_measurement(self, fullysampled, mask):
    pass

class CSMRIForward(BaseForward):
  '''Forward model for CS-MRI.'''
  def __init__(self):
    super(CSMRIForward, self).__init__()

  def generate_measurement(self, fullysampled, mask):
    '''Generate under-sampled k-space data with given binary mask.
    
    Args:
      fullysampled: Clean image in image space (N, n_ch, l, w)
      mask: Stack of masks (N, 1, l, w)
    '''
    ksp = fft(fullysampled)
    under_ksp = ksp * mask
    zf = ifft(under_ksp)
    under_ksp, zf = scale(under_ksp, zf)
    return zf, under_ksp

class InpaintingForward(BaseForward):
  '''Forward model for inpainting.'''
  def __init__(self):
    super(InpaintingForward, self).__init__()

  def generate_measurement(self, fullysampled, mask):
    '''Generate masked version of input.
    
    Args:
      fullysampled: Clean image in image space (N, n_ch, l, w)
      mask: Stack of masks (N, 1, l, w)
    '''
    mask = mask.unsqueeze(0)
    masked = fullysampled * mask
    masked = torch.cat((masked, torch.zeros_like(masked)), dim=1)
    return masked, fft(masked)