"""
Layers for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    """Upsample a multi-channel input image"""
    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class BatchConv2DLayer(nn.Module):
    """
    Conv2D for a batch of images and weights
    For batch size B of images and weights, convolutions are computed between
    images[0] and weights[0], images[1] and weights[1], ..., images[B-1] and weights[B-1]
    """
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=0, dilation=1, ks=3):
        super(BatchConv2DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ks = ks

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        x = x.unsqueeze(1)
        b_i, b_j, c, h, w = x.shape
        b_i, out_channels, in_channels, kernel_height_size, kernel_width_size = weight.shape
        assert out_channels == self.out_channels, 'Invalid out_channels in weights'
        assert in_channels == self.in_channels, 'Invalid in_channels in weights'
        assert kernel_height_size == self.ks and kernel_width_size == self.ks, 'Invalid kernel size in weights'

        out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
        weight = weight.contiguous().view(b_i * out_channels, in_channels, kernel_height_size, kernel_width_size)

        out = F.conv2d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)


        out = out.view(b_j, b_i, out_channels, out.shape[-2], out.shape[-1])

        out = out.permute([1, 0, 2, 3, 4])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

        out = out[:,0,...]
        return out

    def get_weight_shape(self):
        return [self.out_channels, self.in_channels, self.ks, self.ks]

    def get_bias_shape(self):
        return [self.out_channels]
