import torch
import torch.nn as nn
from hyperhqsnet import test as hypertest
from hyperhqsnet import utils
from hyperhqsnet import model as reconnet
import parse
import myutils
from myutils.array import make_imshowable as mims
import os
import matplotlib.pyplot as plt

class TrajectoryNet(nn.Module):
    '''
    dim_bounds is out_dim by 2 array, each row contains bounds for that dimension
    '''
    def __init__(self, dim_bounds, in_dim=1, h_dim=8, out_dim=2):
        super(HyperReduction, self).__init__()

        self.low = dim_bounds[:,0]
        self.high = dim_bounds[:,1]
        self.lin1 = nn.Linear(in_dim, h_dim)
#         self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
#         x = self.sigmoid(x)
        x = self.low + (self.high - self.low) / (1 + torch.exp(-x))
        return x 
