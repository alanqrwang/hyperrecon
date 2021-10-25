"""
Model architecture for HyperRecon
For more details, please read:
  Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
  "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torch.nn as nn
from . import layers
from .hypernetwork import HyperNetwork

class Unet(nn.Module):
  def __init__(self, in_ch, out_ch, h_ch, hnet_hdim=None, residual=True, use_batchnorm=False):
    '''Main Unet architecture.
    
    hnet_hdim activates hypernetwork for Unet.
    '''
    super(Unet, self).__init__()
        
    self.residual = residual
    self.hnet_hdim = hnet_hdim
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
    
    if hnet_hdim is not None:
      self.conv_last = layers.BatchConv2d(h_ch, out_ch, hnet_hdim, kernel_size=1)
    else:
      self.conv_last = nn.Conv2d(h_ch, out_ch, 1)
    

  def double_conv(self, in_channels, out_channels):
    if self.hnet_hdim is not None:
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
    else:
      if self.use_batchnorm:
        return layers.MultiSequential(
          nn.Conv2d(in_channels, out_channels, 3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, 3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
        )   
      else:
        return layers.MultiSequential(
          nn.Conv2d(in_channels, out_channels, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, 3, padding=1),
          nn.ReLU(inplace=True)
        )   
    
  def forward(self, zf, hyp_out=None):
    x = zf
    feature_mean = 0

    conv1 = self.dconv_down1(x, hyp_out)
    feature_mean = feature_mean + conv1.mean(dim=(1,2,3))
    x = self.maxpool(conv1)

    conv2 = self.dconv_down2(x, hyp_out)
    feature_mean = feature_mean + conv2.mean(dim=(1,2,3))
    x = self.maxpool(conv2)
    
    conv3 = self.dconv_down3(x, hyp_out)
    feature_mean = feature_mean + conv3.mean(dim=(1,2,3))
    x = self.maxpool(conv3)   
    
    x = self.dconv_down4(x, hyp_out)
    feature_mean = feature_mean + x.mean(dim=(1,2,3))

    self.feature_mean = feature_mean
    
    x = self.upsample(x)        
    x = torch.cat([x, conv3], dim=1)
    x = self.dconv_up3(x, hyp_out)

    x = self.upsample(x)        
    x = torch.cat([x, conv2], dim=1)       
    x = self.dconv_up2(x, hyp_out)

    x = self.upsample(x)        
    x = torch.cat([x, conv1], dim=1)   
    x = self.dconv_up1(x, hyp_out)

    if self.hnet_hdim is not None:
      out = self.conv_last(x, hyp_out)
    else:
      out = self.conv_last(x)

    if self.residual:
      zf = zf.norm(p=2, dim=1, keepdim=True)
      out = zf + out 
    
    return out

  def get_feature_mean(self):
    return self.feature_mean

  def get_conv_weights(self):
    weights = []
    modules = [module for module in self.modules() if (not isinstance(module, layers.MultiSequential) and isinstance(module, nn.Conv2d))]
    for l in modules:
      weights.append(l.weight.data.flatten().view(1, -1))
      weights.append(l.bias.data.flatten().view(1, -1))
    return weights

class HyperUnet(nn.Module):
  """HyperUnet for hyperparameter-agnostic image reconstruction"""
  def __init__(self, in_units_hnet, h_units_hnet, in_ch_main, out_ch_main, h_ch_main, residual=True, use_batchnorm=False):
    """
    Args:
      in_units_hnet : Input dimension for hypernetwork
      h_units_hnet : Hidden dimension for hypernetwork
      in_ch_main : Input channels for Unet
      out_ch_main : Output channels for Unet
      h_ch_main : Hidden channels for Unet
      residual : Whether or not to use residual U-Net architecture
    """
    super(HyperUnet, self).__init__()

    # HyperNetwork
    self.hnet = HyperNetwork(
                    in_dim=in_units_hnet, 
                    h_dim=h_units_hnet
                )
    self.unet = Unet(
                    in_ch=in_ch_main, 
                    out_ch=out_ch_main, 
                    h_ch=h_ch_main, 
                    hnet_hdim=h_units_hnet,
                    residual=residual,
                    use_batchnorm=use_batchnorm
                )

  def forward(self, x, hyperparams):
    """
    Args:
      x : Input (batch_size, 2, img_height, img_width)
      hyperparams : Hyperparameter values (batch_size, num_hyperparams)
    """
    hyp_out = self.hnet(hyperparams)
    out = self.unet(x, hyp_out)
    return out
  
  def get_hyp_out(self, hyperparams):
    return self.hnet(hyperparams)

  def get_conv_weights(self):
    #TODO: hacky implementation, assumes that forward pass on hyperkernel and hyperbias has been called.
    #      Also, is dependent on batch size of the forward pass.
    weights = []
    modules = [module for module in self.unet.modules() if (not isinstance(module, layers.MultiSequential) and isinstance(module, layers.BatchConv2d))]
    for l in modules:
      weights.append(l.get_kernel())
      weights.append(l.get_bias())
    return weights