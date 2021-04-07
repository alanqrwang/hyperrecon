"""
Layers for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Upsample(nn.Module):
    """Upsample a multi-channel input image"""
    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class HyperConv2d(nn.Module):
    """
    Conv2D for a batch of images and weights
    For batch size B of images and weights, convolutions are computed between
    images[0] and weights[0], images[1] and weights[1], ..., images[B-1] and weights[B-1]

    Takes hypernet output and transforms it to weights and biases
    """
    def __init__(self, in_channels, out_channels, hyp_out_units, stride=1,
                 padding=0, dilation=1, ks=3, use_tanh=True):
        super(HyperConv2d, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ks = ks
        self.use_tanh = use_tanh

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = None
        self.bias = None

        kernel_units = np.prod(self.get_weight_shape())
        bias_units = np.prod(self.get_bias_shape())
        self.hyperkernel = nn.Linear(hyp_out_units, kernel_units)
        self.hyperbias = nn.Linear(hyp_out_units, bias_units)
        self.tanh = nn.Tanh()

    def forward(self, x, hyp_out, include_bias=True):
        assert x.shape[0] == hyp_out.shape[0], "dim=0 of x must be equal in size to dim=0 of hypernet output"

        x = x.unsqueeze(1)
        b_i, b_j, c, h, w = x.shape

        # Reshape input and get weights from hyperkernel
        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        self.kernel = self.hyperkernel(hyp_out)
        if self.use_tanh:
            self.kernel = self.tanh(self.kernel)
        weight = self.kernel.view(b_i * self.out_channels, self.in_channels, self.ks, self.ks)

        out = F.conv2d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)

        out = out.view(b_j, b_i, self.out_channels, out.shape[-2], out.shape[-1])
        out = out.permute([1, 0, 2, 3, 4])

        if include_bias:
            # Get weights from hyperbias
            self.bias = self.tanh(self.hyperbias(hyp_out))
            out = out + self.bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

        out = out[:,0,...]
        return out

    def get_weight_shape(self):
        return [self.out_channels, self.in_channels, self.ks, self.ks]

    def get_bias_shape(self):
        return [self.out_channels]


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    Setting hyp_out_units activates batched convolutional weight training for hypernetworks
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 padding=0, dilation=1, ks=3, hyp_out_units=None, use_tanh=True):
        super().__init__()

        if hyp_out_units is not None:
            self.conv = HyperConv2d(in_channels, out_channels, hyp_out_units, stride=stride,
                 padding=padding, dilation=dilation, ks=ks, use_tanh=use_tanh)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, ks, stride=stride, padding=padding, dilation=dilation)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, hyp_out):
        if hyp_out is None:
            out = self.conv(x)
        else:
            out = self.conv(x, hyp_out)
        out = self.activation(out)
        return out
    
    def get_l1_weight_penalty(self):
        if isinstance(self.conv, nn.Conv2d):
            l1 = torch.norm(self.conv.weight.data, p=1) + torch.norm(self.conv.bias.data, p=1)
        elif isinstance(self.conv, HyperConv2d):
            assert self.conv.kernel is not None and self.conv.bias is not None
            l1 = torch.norm(self.conv.kernel, dim=1, p=1) + torch.norm(self.conv.bias, dim=1, p=1)
        else:
            raise Exception()

        return l1
