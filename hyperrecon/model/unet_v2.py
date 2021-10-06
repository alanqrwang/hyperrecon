import torch
import torch.nn as nn
from hyperrecon.model import layers
from hyperrecon.model import layers_v2
from hyperrecon.model.hypernetwork import HyperNetwork
from hyperrecon.data.mask import Loupe
from hyperrecon.util.forward import CSMRIForward

class BaseUnet(nn.Module):
  def __init__(self, in_ch, out_ch, h_ch, residual=True, use_batchnorm=False, **kwargs):
    '''Main Unet architecture.
    
    Args:
      in_ch: Number of input channels
      out_ch: Number of output channels
      h_ch: Number of hidden channels
      residual: Include residual mapping from input to output
      use_batchnorm: Include batchnorm layers
      hnet_hdim: Required for hypernetwork, specifies hypernetwork hidden dimension
    '''
    super(BaseUnet, self).__init__()
        
    self.residual = residual
    self.use_batchnorm = use_batchnorm
    self.conv2d_module = self.set_conv2d_module()
    self.kwargs = kwargs

    self.dconv_down1 = self.double_conv(in_ch, h_ch)
    self.dconv_down2 = self.double_conv(h_ch, h_ch)
    self.dconv_down3 = self.double_conv(h_ch, h_ch)
    self.dconv_down4 = self.double_conv(h_ch, h_ch)        

    self.maxpool = nn.MaxPool2d(2)
    self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
    
    self.dconv_up3 = self.double_conv(h_ch+h_ch, h_ch)
    self.dconv_up2 = self.double_conv(h_ch+h_ch, h_ch)
    self.dconv_up1 = self.double_conv(h_ch+h_ch, h_ch)
    
    self.conv_last = self.single_conv(h_ch, out_ch)
    
  def set_conv2d_module(self):
    pass

  def double_conv(self, in_channels, out_channels):
    if self.use_batchnorm:
      return layers_v2.MultiArgSequential(
        self.conv2d_module(in_channels, out_channels, kernel_size=3, padding=1, **self.kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        self.conv2d_module(out_channels, out_channels, kernel_size=3, padding=1, **self.kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
      )   
    else:
      return layers_v2.MultiArgSequential(
        self.conv2d_module(in_channels, out_channels, kernel_size=3, padding=1, **self.kwargs),
        nn.ReLU(inplace=True),
        self.conv2d_module(out_channels, out_channels, kernel_size=3, padding=1, **self.kwargs),
        nn.ReLU(inplace=True)
      )   
  
  def single_conv(self, in_channels, out_channels):
    return self.conv2d_module(in_channels, out_channels, kernel_size=1, padding=0, **self.kwargs)

  def forward(self, zf, *args):
    x = zf

    conv1 = self.dconv_down1(x, *args)
    x = self.maxpool(conv1)

    conv2 = self.dconv_down2(x, *args)
    x = self.maxpool(conv2)
    
    conv3 = self.dconv_down3(x, *args)
    x = self.maxpool(conv3)   
    
    x = self.dconv_down4(x, *args)
    
    x = self.upsample(x)        
    x = torch.cat([x, conv3], dim=1)
    x = self.dconv_up3(x, *args)

    x = self.upsample(x)        
    x = torch.cat([x, conv2], dim=1)       
    x = self.dconv_up2(x, *args)

    x = self.upsample(x)        
    x = torch.cat([x, conv1], dim=1)   
    x = self.dconv_up1(x, *args)

    out = self.conv_last(x, *args)

    if self.residual:
      zf = zf.norm(p=2, dim=1, keepdim=True)
      out = zf + out 
    
    return out

class Unet(BaseUnet):
  def set_conv2d_module(self):
    return layers_v2.Conv2d

class HyperUnet(BaseUnet):
  def __init__(self, 
               in_units_hnet, 
               h_units_hnet, 
               in_ch_main, 
               out_ch_main, 
               h_ch_main, 
               residual=True, 
               use_batchnorm=False):
    super(HyperUnet, self).__init__(in_ch_main, out_ch_main, h_ch_main, residual, use_batchnorm, hnet_hdim=h_units_hnet)
    self.hnet = HyperNetwork(
                    in_dim=in_units_hnet, 
                    h_dim=h_units_hnet
                ) 
  
  def set_conv2d_module(self):
    return layers_v2.BatchConv2d
  
  def forward(self, x, hparams):
    hyp_out = self.hnet(hparams)
    return super().forward(x, hyp_out)
  
class LastLayerHyperUnet(Unet):
  def __init__(self, in_units_hnet, h_units_hnet, in_ch_main, out_ch_main, h_ch_main, residual=True, use_batchnorm=False):
    self.hnet_hdim = h_units_hnet
    super(LastLayerHyperUnet, self).__init__(in_ch_main, out_ch_main, h_ch_main, residual, use_batchnorm)
    self.hnet = HyperNetwork(
                    in_dim=in_units_hnet, 
                    h_dim=h_units_hnet
                ) 
  
  def single_conv(self, in_channels, out_channels):
    return layers.BatchConv2d(in_channels, out_channels, self.hnet_hdim, kernel_size=1)
  
  def forward(self, x, hparams):
    hyp_out = self.hnet(hparams)
    return super().forward(x, hyp_out)
  
class LoupeUnet(Unet):
  """LoupeUnet."""
  def __init__(self, 
               in_ch, 
               out_ch, 
               h_ch, 
               image_dims, 
               residual=True, 
               use_batchnorm=False):
    super(LoupeUnet, self).__init__(in_ch, out_ch, h_ch, residual, use_batchnorm)
    self.loupe = Loupe(image_dims).cuda()
    self.forward_model = CSMRIForward()

  def forward(self, x, hyperparams):
    """
    Args:
      x : Input (batch_size, 2, img_height, img_width)
      hyperparams : Hyperparameter values (batch_size, num_hyperparams)
    """
    batch_size = len(x)
    undersample_mask = self.loupe(batch_size, hyperparams)
    measurement, measurement_ft = self.forward_model.generate_measurement(x, undersample_mask)
    out = super().forward(measurement)
    return out, measurement, measurement_ft

class LoupeHyperUnet(HyperUnet):
  """HyperUnet for hyperparameter-agnostic image reconstruction"""
  def __init__(self, 
               in_units_hnet, 
               h_units_hnet, 
               in_ch_main, 
               out_ch_main, 
               h_ch_main, 
               image_dims, 
               residual=True, 
               use_batchnorm=False):
    """
    Args:
      in_units_hnet : Input dimension for hypernetwork
      h_units_hnet : Hidden dimension for hypernetwork
      in_ch_main : Input channels for Unet
      out_ch_main : Output channels for Unet
      h_ch_main : Hidden channels for Unet
      residual : Whether or not to use residual U-Net architecture
    """
    super(LoupeHyperUnet, self).__init__(in_units_hnet, h_units_hnet, in_ch_main, out_ch_main, h_ch_main, residual, use_batchnorm)
    self.loupe = Loupe(image_dims).cuda()
    self.forward_model = CSMRIForward()

  def forward(self, x, hyperparams):
    """
    Args:
      x : Input (batch_size, 2, img_height, img_width)
      hyperparams : Hyperparameter values (batch_size, num_hyperparams)
    """
    batch_size = len(x)
    undersample_mask = self.loupe(batch_size, hyperparams)
    measurement, measurement_ft = self.forward_model.generate_measurement(x, undersample_mask)
    out = super().forward(measurement, hyperparams)
    return out, measurement, measurement_ft