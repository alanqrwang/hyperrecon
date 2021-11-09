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
  def __init__(self, img_dims, snr=5, mean=0., std=1., fixed=False):
    self.snr = snr
    self.mean = mean
    self.std = std
    self.fixed = fixed
    if fixed:
      self.x_noise = np.random.normal(self.mean, self.std, size=img_dims)
      self.y_noise = np.random.normal(self.mean, self.std, size=img_dims)
  
  def __call__(self, img):
    level = self.snr * np.max(img) / 100
    if self.fixed:
      x_noise = self.x_noise
      y_noise = self.y_noise
    else:
      x_noise = np.random.normal(self.mean, self.std, size=img.shape)
      y_noise = np.random.normal(self.mean, self.std, size=img.shape)

    x = level * x_noise + img
    y = level * y_noise
    return np.sqrt(x**2 + y**2)