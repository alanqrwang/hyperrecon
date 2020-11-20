import numpy as np
import torch
import torch.nn as nn

class HpSampler(nn.Module):
    def __init__(self, sampling_method, num_hyperparams, alpha_bound, beta_bound):
        super(HpSampler, self).__init__()

        self.sampling_method = sampling_method
        self.num_hyperparams = num_hyperparams
        self.alpha_bound = alpha_bound
        self.beta_bound = beta_bound

    def uniform(self, batch_size):
        if self.num_hyperparams == 2:
            alpha = self._uniform_bound(batch_size, self.alpha_bound)
            beta = self._uniform_bound(batch_size, self.beta_bound)

            hyperparams = torch.cat([alpha, beta], dim=1)
        else:
            hyperparams = self._uniform_bound(batch_size, self.alpha_bound)
        return hyperparams

    def _uniform_bound(self, batch_size, bound):
        r1 = float(bound[0])
        r2 = float(bound[1])

        sample = ((r1 - r2) * torch.rand((batch_size, 1)) + r2)
        return sample

    def sup_map_sample(self, batch_size, sup_map, top_percentage=25, hard_cutoff=None):
        assert sup_map is not None

        samples = np.sqrt(len(sup_map))

        if hard_cutoff is not None:
            sup_map = [1 if i > hard_cutoff else 0 for i in sup_map] # Threshold
        else:
            percentile = np.percentile(sup_map, 100-top_percentage)
            sup_map = [1 if i > percentile else 0 for i in sup_map]

        sup_map_prob = sup_map / np.sum(sup_map) # Make into uniform distribution

        x_jitter = np.random.uniform(-1/2, 1/2, batch_size)
        y_jitter = np.random.uniform(-1/2, 1/2, batch_size)
        rand_point = np.random.choice(len(sup_map_prob), batch_size, p=sup_map_prob)
        rand_x = rand_point % samples + x_jitter
        rand_y = rand_point // samples + y_jitter

        alpha_scaled = torch.tensor(np.interp(rand_x, [-1/2,(samples-1)+1/2], [0, 1]).reshape(-1,1))
        beta_scaled = torch.tensor(np.interp(rand_y, [-1/2,(samples-1)+1/2], [0, 1]).reshape(-1,1))
        hyperparams = torch.cat([alpha_scaled, beta_scaled], dim=1)
        return hyperparams

    def sample(self, batch_size, sup_map=None):
        if self.sampling_method == 'bestsup' and sup_map is not None:
            # print('sampling sup map')
            s = self.sup_map_sample(batch_size, sup_map)
        else:
            # print('sampling uniform')
            s = self.uniform(batch_size)
        return s
