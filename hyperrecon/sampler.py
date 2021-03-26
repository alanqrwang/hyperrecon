"""
Sampler for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import numpy as np
import torch
import torch.nn as nn
from . import utils

class HpSampler(nn.Module):
    """Hyperparameter sampler class"""
    def __init__(self, num_hyperparams, device, range_restrict):
        """
        Parameters
        ----------
        num_hyperparams : int
            Number of hyperparameters (i.e. number of regularization functions)
        """
        super(HpSampler, self).__init__()

        self.num_hyperparams = num_hyperparams
        self.device = device
        self.range_restrict = range_restrict

    def sample(self, batch_size, val=False):
        """Uniform sampling

        Parameters
        ----------
        batch_size: int
            Size of batch
        """
        if val:
            ref_hps = utils.get_reference_hps(self.num_hyperparams, self.range_restrict)
            hyperparams = ref_hps.repeat(int(np.ceil(batch_size / len(ref_hps))), 1)[:batch_size]
        else:
            hyperparams = torch.rand((batch_size, self.num_hyperparams))
        return hyperparams.to(self.device)
