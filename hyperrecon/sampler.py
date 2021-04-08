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
    def __init__(self, num_hyperparams, range_restrict, weights):
        """
        Parameters
        ----------
        num_hyperparams : int
            Number of hyperparameters (i.e. number of regularization functions)
        """
        super(HpSampler, self).__init__()

        self.num_hyperparams = num_hyperparams
        self.range_restrict = range_restrict
        if weights is not None:
            self.weights = torch.tensor(weights)
        else:
            self.weights = None

    def sample(self, batch_size, phase):
        """Uniform sampling

        Parameters
        ----------
        batch_size: int
            Size of batch
        """
        if self.weights is not None:
            # Fixed hyperparameters
            hyperparams = self.weights.unsqueeze(0)
            hyperparams = hyperparams.repeat(batch_size, 1)

        elif phase == 'val':
            ref_hps = utils.get_reference_hps(self.num_hyperparams, self.range_restrict)
            print('ref', ref_hps)
            hyperparams = ref_hps.repeat(int(np.ceil(batch_size / len(ref_hps))), 1)[:batch_size]
        else:
            hyperparams = torch.rand((batch_size, self.num_hyperparams))
        return hyperparams
