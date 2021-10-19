import torch
import random

class Uniform():
  def __init__(self, r1=0, r2=1):
    self.r1 = r1
    self.r2 = r2
    print(self.r1, self.r2)
  def __call__(self, size):
    return torch.FloatTensor(*size).uniform_(self.r1, self.r2)

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
    return torch.ones(*size) * self.value

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