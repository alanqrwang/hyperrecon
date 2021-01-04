"""
Sampler for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import numpy as np
import torch
import torch.nn as nn

class HpSampler(nn.Module):
    """Hyperparameter sampler class"""
    def __init__(self, num_hyperparams, bounds):
        """
        Parameters
        ----------
        num_hyperparams : int
            Number of hyperparameters (i.e. number of regularization functions)
        bounds : numpy.array (num_hyperparams, 2)
            Bounds of uniform sampling
        """
        super(HpSampler, self).__init__()

        assert num_hyperparams == len(bounds), 'Bounds and num_hyperparams must match'
        self.num_hyperparams = num_hyperparams
        self.bounds = bounds

    def _uniform_bound(self, batch_size, bound):
        r1 = float(bound[0])
        r2 = float(bound[1])

        sample = ((r1 - r2) * torch.rand((batch_size, 1)) + r2)
        return sample

    def sample(self, batch_size):
        """Uniform sampling, can be extended for >2 hyperparameters

        Parameters
        ----------
        batch_size: int
            Size of batch
        """
        if self.num_hyperparams == 2:
            alpha = self._uniform_bound(batch_size, self.bounds[0])
            beta = self._uniform_bound(batch_size, self.bounds[1])

            hyperparams = torch.cat([alpha, beta], dim=1)
        else:
            hyperparams = self._uniform_bound(batch_size, self.bounds[0])
        return hyperparams
