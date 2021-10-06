import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MultiSequential(nn.Sequential):
  def forward(self, *inputs):
    for module in self._modules.values():
      if isinstance(module, (Conv2d, BatchConv2d)):
        x = module(*inputs)
      else:
        x = module(inputs[0])
    return x

class Conv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, 
               stride=1, padding=0, dilation=1, **kwargs):
    super(Conv2d, self).__init__()
    self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)

  def forward(self, x, *args):
    return self.layer(x)

class BatchConv2d(nn.Module):
  """
  Conv2D for a batch of images and weights
  For batch size B of images and weights, convolutions are computed between
  images[0] and weights[0], images[1] and weights[1], ..., images[B-1] and weights[B-1]

  Takes hypernet output and transforms it to weights and biases
  """
  def __init__(self, in_channels, out_channels, kernel_size=3, 
               stride=1, padding=0, dilation=1, **kwargs):
    super(BatchConv2d, self).__init__()

    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.out_channels = out_channels
    hyp_out_units = kwargs['hyp_out_units']

    kernel_units = np.prod(self.get_kernel_shape())
    bias_units = np.prod(self.get_bias_shape())
    self.hyperkernel = nn.Linear(hyp_out_units, kernel_units)
    self.hyperbias = nn.Linear(hyp_out_units, bias_units)

  def forward(self, x, hyp_out):
    assert x.shape[0] == hyp_out.shape[0], "dim=0 of x must be equal in size to dim=0 of hypernet output"

    x = x.unsqueeze(1)
    b_i, b_j, c, h, w = x.shape

    # Reshape input and get weights from hyperkernel
    out = x.permute([1, 0, 2, 3, 4]).contiguous().view(b_j, b_i * c, h, w)
    kernel = self.get_kernel(hyp_out)
    kernel = kernel.view(b_i * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
    out = F.conv2d(out, weight=kernel, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
             padding=self.padding)

    out = out.view(b_j, b_i, self.out_channels, out.shape[-2], out.shape[-1])
    out = out.permute([1, 0, 2, 3, 4])

    bias = self.get_bias(hyp_out)
    out = out + bias.unsqueeze(1).unsqueeze(3).unsqueeze(3)

    out = out[:,0,...]
    return out

  def get_kernel(self, hyp_out):
    return self.hyperkernel(hyp_out)
  def get_bias(self, hyp_out):
    return self.hyperbias(hyp_out)
  def get_kernel_shape(self):
    return [self.out_channels, self.in_channels, self.kernel_size, self.kernel_size]
  def get_bias_shape(self):
    return [self.out_channels]