import torch
import random

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
  def __init__(self, r1=0, r2=1):
    self.r1 = r1
    self.r2 = r2
  def random_hyperparam(self):
    rand = random.uniform(self.r1, self.r2)
    return random.choice([0.0, 1.0, rand])
  def __call__(self, size):
    batch_samp = [self.random_hyperparam() for _ in range(size[0])]
    return torch.tensor(batch_samp).reshape(*size).float()

class UniformConstant():
  def __init__(self, r1=0, r2=1):
    self.r1 = r1
    self.r2 = r2
  def __call__(self, size):
    cat = random.uniform(self.r1, self.r2)
    return torch.ones(*size) * cat

class Constant():
  def __init__(self, value):
    self.value = value
  def __call__(self, size):
    return self.value.repeat(size[0], 1)

class CategoricalConstant():
  def __init__(self, categories):
    self.categories = categories
  def __call__(self, size):
    cat = random.choice(self.categories)
    return torch.ones(*size) * cat

class Binary():
  def __call__(self, size):
    return torch.bernoulli(torch.empty(*size).fill_(0.5))

class BinaryConstant():
  def __call__(self, size):
    if random.random() < 0.5:
      return torch.zeros(*size)
    else:
      return torch.ones(*size)