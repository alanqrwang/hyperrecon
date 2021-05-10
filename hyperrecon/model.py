"""
Model architecture for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import utils
from . import loss as losslayer
from . import layers
import torch
import torch.nn as nn
import sys
import torch.autograd.profiler as profiler

class HyperNetwork(nn.Module):
    """Hypernetwork architecture and forward pass

    Takes hyperparameters and outputs weights of main U-Net
    """
    def __init__(self, normalize, in_dim=1, h_dim=32):
        """
        Parameters
        ----------
        normalize : bool
            Whether or not to normalize the input; removes a degree of freedom
        in_dim : int
            Input dimension
        h_dim : int
            Hidden dimension
        """
        super(HyperNetwork, self).__init__()
        
        self.normalize = normalize
        # Network layers
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, h_dim)
        self.lin4 = nn.Linear(h_dim, h_dim)

        # Activations
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values
        """
        if self.normalize:
            x = x / torch.sum(x, dim=1, keepdim=True)

        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        return x

class HyperUnet(nn.Module):
    """Main U-Net for image reconstruction"""
    def __init__(self, device, num_hyperparams, hnet_hdim, unet_hdim, hnet_norm, n_ch_in, n_ch_out, residual=True, use_tanh=True):
        """
        Parameters
        ----------
        device : PyTorch device, 'cpu' or 'cuda:<gpu_id>'
        num_hyperparams : Number of hyperparameters (i.e. number of regularization functions)
        hnet_hdim : Hidden channel dimension of HyperNetwork
        unet_hdim : Hidden channel dimension of U-Net
        hnet_norm : Whether or not to normalize hypernet inputs
        n_ch_out : Number of output channels
        residual : Whether or not to use residual U-Net architecture
        """
        super(HyperUnet, self).__init__()

        self.residual = residual
        self.unet_hdim = unet_hdim
        self.device = device
        self.n_ch_out = n_ch_out

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.relu = nn.ReLU(inplace=True)

        # UNet
        self.conv_down0 = layers.BatchConv2DLayer(n_ch_in, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down1 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down2 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down3 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down4 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down5 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down6 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_down7 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)

        self.conv_up8 = layers.BatchConv2DLayer(unet_hdim+unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_up9 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_up10 = layers.BatchConv2DLayer(unet_hdim+unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_up11 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_up12 = layers.BatchConv2DLayer(unet_hdim+unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)
        self.conv_up13 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1, use_tanh=use_tanh)

        self.conv_last14 = layers.BatchConv2DLayer(unet_hdim, n_ch_out, hnet_hdim, ks=1, use_tanh=use_tanh)

        # HyperNetwork
        self.hnet = HyperNetwork(hnet_norm, in_dim=num_hyperparams, h_dim=hnet_hdim)

    def forward(self, zf, hyperparams):
        """
        Parameters
        ----------
        zf : torch.Tensor (batch_size, 2, img_height, img_width)
            Zero-filled reconstruction of under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values

        """
        x = zf
        # x = x.permute(0, 3, 1, 2)

        hyp_out = self.hnet(hyperparams)
        penalty = torch.zeros(len(zf), requires_grad=True).to(self.device)

        # conv_down1
        x, l1 = self.conv_down0(x, hyp_out)
        x = self.relu(x)
        # x.register_hook(self.printnorm)
        penalty = penalty + l1
        x, l1 = self.conv_down1(x, hyp_out)
        conv1 = self.relu(x)
        penalty = penalty + l1
        x = self.maxpool(conv1)

        # conv_down2
        x, l1 = self.conv_down2(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x, l1 = self.conv_down3(x, hyp_out)
        conv2 = self.relu(x)
        penalty = penalty + l1
        x = self.maxpool(conv2)

        # conv_down3
        x, l1 = self.conv_down4(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x, l1 = self.conv_down5(x, hyp_out)
        conv3 = self.relu(x)
        penalty = penalty + l1
        x = self.maxpool(conv3)   

        # conv_down4
        x, l1 = self.conv_down6(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x, l1 = self.conv_down7(x, hyp_out)
        conv4 = self.relu(x)
        penalty = penalty + l1
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)

        # conv_up3
        x, l1 = self.conv_up8(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x, l1 = self.conv_up9(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        # conv_up2
        x, l1 = self.conv_up10(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x, l1 = self.conv_up11(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        # conv_up1
        x, l1 = self.conv_up12(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1
        x, l1 = self.conv_up13(x, hyp_out)
        x = self.relu(x)
        penalty = penalty + l1

        # last
        out, l1 = self.conv_last14(x, hyp_out)
        penalty = penalty + l1
        # out.register_hook(self.printnorm)

        # out = out.permute(0, 2, 3, 1)
        if self.residual:
            if self.n_ch_out == 1:
                zf = zf.norm(-1, keepdim=True)
            out = zf + out 

        assert out.shape[-1] == self.n_ch_out, 'Incorrect output channels'
        return out, penalty

    def printnorm(self, x):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('norm0:', x[0].data.norm())
        print('norm1:', x[1].data.norm())
        print('norm2:', x[2].data.norm())
        print('norm3:', x[3].data.norm())
        print('norm4:', x[4].data.norm())
        print('norm5:', x[5].data.norm())
        print('norm6:', x[6].data.norm())
        print('norm7:', x[7].data.norm())

class TrajNet(nn.Module):
    def __init__(self, in_dim=1, h_dim=8, out_dim=2):
        super(TrajNet, self).__init__()
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, h_dim)
        self.lin4 = nn.Linear(h_dim, h_dim)
        self.lin5 = nn.Linear(h_dim, out_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin5(x)
        out = self.sigmoid(x)
        return out

class Unet(nn.Module):
    def __init__(self, n_ch_in=2, n_ch_out=1, unet_hdim=32, residual=True):
        super(Unet, self).__init__()
                
        self.residual = residual
        self.dconv_down1 = self.double_conv(n_ch_in, unet_hdim)
        self.dconv_down2 = self.double_conv(unet_hdim, unet_hdim)
        self.dconv_down3 = self.double_conv(unet_hdim, unet_hdim)
        self.dconv_down4 = self.double_conv(unet_hdim, unet_hdim)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = self.double_conv(unet_hdim+unet_hdim, unet_hdim)
        self.dconv_up2 = self.double_conv(unet_hdim+unet_hdim, unet_hdim)
        self.dconv_up1 = self.double_conv(unet_hdim+unet_hdim, unet_hdim)
        
        self.conv_last = nn.Conv2d(unet_hdim, n_ch_out, 1)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )   
        
    def forward(self, zf, hyperparams=None):
        x = zf
        # x = x.permute(0, 3, 1, 2)

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        # out = out.permute(0, 2, 3, 1)
        if self.residual:
            zf = zf.norm(-1, keepdim=True)
            out = zf + out 
        
        return out, None

