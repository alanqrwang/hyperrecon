import torch
import torch.nn as nn
from hyperrecon.model import layers
from hyperrecon.model.hypernetwork import HyperNetwork

class Unet(nn.Module):
  def __init__(self, in_ch, out_ch, h_ch, residual=True, use_batchnorm=False):
    '''Main Unet architecture.
    
    Args:
      in_ch: Number of input channels
      out_ch: Number of output channels
      h_ch: Number of hidden channels
    '''
    super(Unet, self).__init__()
        
    self.residual = residual
    self.use_batchnorm = use_batchnorm

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
    
  def double_conv(self, in_channels, out_channels):
    if self.use_batchnorm:
      return layers.MultiSequential(
        layers.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        layers.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
      )   
    else:
      return layers.MultiSequential(
        layers.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        layers.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
      )   
  
  def single_conv(self, in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

  def forward(self, zf, hyp_out=None):
    x = zf

    conv1 = self.dconv_down1(x, hyp_out)
    x = self.maxpool(conv1)

    conv2 = self.dconv_down2(x, hyp_out)
    x = self.maxpool(conv2)
    
    conv3 = self.dconv_down3(x, hyp_out)
    x = self.maxpool(conv3)   
    
    x = self.dconv_down4(x, hyp_out)
    
    x = self.upsample(x)        
    x = torch.cat([x, conv3], dim=1)
    x = self.dconv_up3(x, hyp_out)

    x = self.upsample(x)        
    x = torch.cat([x, conv2], dim=1)       
    x = self.dconv_up2(x, hyp_out)

    x = self.upsample(x)        
    x = torch.cat([x, conv1], dim=1)   
    x = self.dconv_up1(x, hyp_out)

    out = self.conv_last(x, hyp_out)

    if self.residual:
      zf = zf.norm(p=2, dim=1, keepdim=True)
      out = zf + out 
    
    return out

class HyperUnet(Unet):
  def __init__(self, in_units_hnet, h_units_hnet, in_ch_main, out_ch_main, h_ch_main, residual=True, use_batchnorm=False):
    self.hnet_hdim = h_units_hnet
    super(HyperUnet, self).__init__(in_ch_main, out_ch_main, h_ch_main, residual, use_batchnorm)
    self.hnet = HyperNetwork(
                    in_dim=in_units_hnet, 
                    h_dim=h_units_hnet
                ) 
  
  def double_conv(self, in_channels, out_channels):
    if self.use_batchnorm:
      return layers.MultiSequential(
        layers.BatchConv2d(in_channels, out_channels, self.hnet_hdim, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        layers.BatchConv2d(out_channels, out_channels, self.hnet_hdim, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
      )   
    else:
      return layers.MultiSequential(
        layers.BatchConv2d(in_channels, out_channels, self.hnet_hdim, padding=1),
        nn.ReLU(inplace=True),
        layers.BatchConv2d(out_channels, out_channels, self.hnet_hdim, padding=1),
        nn.ReLU(inplace=True)
      )   
  
  def single_conv(self, in_channels, out_channels):
    return layers.BatchConv2d(in_channels, out_channels, self.hnet_hdim, kernel_size=1)
  
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