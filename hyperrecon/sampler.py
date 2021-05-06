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
    def __init__(self, num_hyperparams, range_restrict, debug=False, hps=None):
        """
        Parameters
        ----------
        num_hyperparams : int
            Number of hyperparameters (i.e. number of regularization functions)
        debug : bool
            Samples from 2d uniform and converts to 3d hypernet, debugging only
        hps : list of floats
            Fixed hyperparameters for baselines
        """
        super(HpSampler, self).__init__()

        self.num_hyperparams = num_hyperparams
        self.range_restrict = range_restrict
        self.debug = debug
        self.hps = None if hps is None else torch.tensor(hps)

    def sample(self, batch_size, val=None):
        """Uniform sampling

        Parameters
        ----------
        batch_size: int
            Size of batch
        """
        if self.hps is not None:
            hyperparams = torch.ones((batch_size, self.num_hyperparams)) * self.hps
        elif val == 'one':
            # ref_hps = utils.get_reference_hps(self.num_hyperparams, self.range_restrict)
            # hyperparams = ref_hps.repeat(int(np.ceil(batch_size / len(ref_hps))), 1)[:batch_size]
            hyperparams = torch.ones((batch_size, self.num_hyperparams))
        elif val == 'zero':
            # ref_hps = utils.get_reference_hps(self.num_hyperparams, self.range_restrict)
            # hyperparams = ref_hps.repeat(int(np.ceil(batch_size / len(ref_hps))), 1)[:batch_size]
            hyperparams = torch.zeros((batch_size, self.num_hyperparams))
        elif self.debug:
            hyperparams = torch.rand((batch_size, 2))
            hyperparams = utils.oldloss2newloss(hyperparams)

        else:
            hyperparams = torch.rand((batch_size, self.num_hyperparams))
            # hyperparams = torch.bernoulli(torch.empty(batch_size, self.num_hyperparams).fill_(0.5))
        return hyperparams
