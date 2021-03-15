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
    def __init__(self, num_hyperparams):
        """
        Parameters
        ----------
        num_hyperparams : int
            Number of hyperparameters (i.e. number of regularization functions)
        """
        super(HpSampler, self).__init__()

        self.num_hyperparams = num_hyperparams

    def sample(self, batch_size):
        """Uniform sampling

        Parameters
        ----------
        batch_size: int
            Size of batch
        """
        hyperparams = torch.rand((batch_size, self.num_hyperparams))
        return hyperparams
