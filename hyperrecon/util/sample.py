import torch
import random
import numpy as np

class Uniform():
  def __init__(self, r1=0, r2=1):
    self.r1 = r1
    self.r2 = r2
  def __call__(self, size):
    return torch.FloatTensor(*size).uniform_(self.r1, self.r2)

class UniformOversample():
  '''Sample a random hyperparameter. Over-samples 0 and 1.
  Only supports 1d hyperparameter sampling.
  '''
  def __init__(self, r1=0, r2=1, p_end=0):
    self.r1 = r1
    self.r2 = r2
    self.p_end = p_end
    
  def random_hyperparam(self):
    rand = random.uniform(self.r1, self.r2)
    return np.random.choice([0.0, 1.0, rand], p=[self.p_end, self.p_end, 1-2*self.p_end])
  def __call__(self, size):
    batch_samp = [self.random_hyperparam() for _ in range(size[0])]
    return torch.tensor(batch_samp).reshape(*size).float()

class Constant():
  def __init__(self, value):
    self.value = value
  def __call__(self, size):
    return self.value.repeat(size[0], 1)