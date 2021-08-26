"""
Sampler for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torch.nn as nn

class Sampler(nn.Module):
  """Hyperparameter sampler class"""
  def __init__(self, num_hyperparams, binary=False):
    """
    Parameters
    ----------
    num_hyperparams : int
      Number of hyperparameters (i.e. number of regularization functions)
    """
    super(Sampler, self).__init__()

    self.num_hyperparams = num_hyperparams
    self.binary = binary

  def sample(self, batch_size, r1=0,r2=1):
    """Uniform sampling

    Parameters
    ----------
    batch_size: int
      Size of batch
    """
    if self.binary:
      hyperparams = torch.bernoulli(torch.empty(batch_size, self.num_hyperparams).fill_(0.5))
    else:
      hyperparams = torch.FloatTensor(batch_size, self.num_hyperparams).uniform_(r1, r2)
    return hyperparams
