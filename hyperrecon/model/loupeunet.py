import torch
import torch.nn as nn

from hyperrecon.data.mask import Loupe, ConditionalLoupe
from hyperrecon.util.forward import CSMRIForward
from hyperrecon.model.unet import Unet
from hyperrecon.model.hypernetwork import HyperNetwork
from hyperrecon.util import utils

class LoupeUnet(nn.Module):
  """LoupeUnet"""
  def __init__(self, in_ch, out_ch, h_ch, image_dims, undersampling_rate, residual=True, use_batchnorm=False):
    """
    Args:
      in_units_hnet : Input dimension for hypernetwork
      h_units_hnet : Hidden dimension for hypernetwork
      in_ch_main : Input channels for Unet
      out_ch_main : Output channels for Unet
      h_ch_main : Hidden channels for Unet
      residual : Whether or not to use residual U-Net architecture
    """
    super(LoupeUnet, self).__init__()

    self.undersampling_rate = 1 / torch.tensor(float(undersampling_rate)).cuda()
    self.unet = Unet(
                    in_ch=in_ch, 
                    out_ch=out_ch, 
                    h_ch=h_ch, 
                    residual=residual,
                    use_batchnorm=use_batchnorm
                )
    self.loupe = Loupe(image_dims).cuda()
    self.forward_model = CSMRIForward()

  def forward(self, x):
    """
    Args:
      x : Input (batch_size, 2, img_height, img_width)
    """
    batch_size = len(x)
    rate = self.undersampling_rate.repeat(batch_size)
    undersample_mask = self.loupe(batch_size, rate)
    measurement = self.forward_model(x, undersample_mask)
    inputs = utils.ifft(measurement)
    out = self.unet(inputs)
    return out, inputs
  
class LoupeHyperUnet(nn.Module):
  """HyperUnet for hyperparameter-agnostic image reconstruction"""
  def __init__(self, in_units_hnet, h_units_hnet, in_ch_main, out_ch_main, h_ch_main, image_dims, residual=True, use_batchnorm=False):
    """
    Args:
      in_units_hnet : Input dimension for hypernetwork
      h_units_hnet : Hidden dimension for hypernetwork
      in_ch_main : Input channels for Unet
      out_ch_main : Output channels for Unet
      h_ch_main : Hidden channels for Unet
      residual : Whether or not to use residual U-Net architecture
    """
    super(LoupeHyperUnet, self).__init__()

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
    measurement = self.forward_model(x, undersample_mask)
    inputs = utils.ifft(measurement)
    hyp_out = self.get_hyp_out(hyperparams)
    out = self.unet(inputs, hyp_out)
    return out, inputs
  
  def get_hyp_out(self, hyperparams):
    return self.hnet(hyperparams)

class ConditionalLoupeHyperUnet(nn.Module):
  """HyperUnet for hyperparameter-agnostic image reconstruction"""
  def __init__(self, in_units_hnet, h_units_hnet, in_ch_main, out_ch_main, h_ch_main, image_dims, residual=True, use_batchnorm=False):
    """
    Args:
      in_units_hnet : Input dimension for hypernetwork
      h_units_hnet : Hidden dimension for hypernetwork
      in_ch_main : Input channels for Unet
      out_ch_main : Output channels for Unet
      h_ch_main : Hidden channels for Unet
      residual : Whether or not to use residual U-Net architecture
    """
    super(ConditionalLoupeHyperUnet, self).__init__()

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
    self.loupe = ConditionalLoupe(image_dims).cuda()
    self.forward_model = CSMRIForward()

  def forward(self, x, hyperparams):
    """
    Args:
      x : Input (batch_size, 2, img_height, img_width)
      hyperparams : Hyperparameter values (batch_size, num_hyperparams)
    """
    undersample_mask = self.loupe(hyperparams)
    measurement = self.forward_model(x, undersample_mask)
    inputs = utils.ifft(measurement)
    hyp_out = self.get_hyp_out(hyperparams)
    out = self.unet(inputs, hyp_out)
    return out, inputs
  
  def get_hyp_out(self, hyperparams):
    return self.hnet(hyperparams)