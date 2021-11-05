import numpy as np
import torch

class AdditiveGaussianNoise(object):
  def __init__(self, image_dims, mean=0., std=1., fixed=False):
    self.std = std
    self.mean = mean
    self.fixed = fixed
    if fixed:
      self.noise = torch.normal(mean, std, size=image_dims).cuda()

  def __call__(self, img):
    if self.fixed:
      noise = self.noise
    else:
      noise = torch.normal(self.mean, self.std, size=img.shape).cuda()
    return img + noise
    
class RicianNoise(object):
  def __init__(self, snr=5, mean=0., std=1.):
    self.snr = snr
    self.mean = mean
    self.std = std
  
  def __call__(self, img):
    level = self.snr * np.max(img) / 100
    x = level * np.random.normal(self.mean, self.std, size=img.shape) + img
    y = level * np.random.normal(self.mean, self.std, size=img.shape)
    return np.sqrt(x**2 + y**2)