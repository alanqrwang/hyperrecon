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
    def __init__(self, normalize, f_size=3, in_dim=1, h_dim=32, unet_nh=64):
        """
        Parameters
        ----------
        normalize : bool
            Whether or not to normalize the input; removes a degree of freedom
        f_size : int 
            Kernel size
        in_dim : int
            Input dimension
        h_dim : int
            Hidden dimension
        unet_nh : int
            Hidden dimension of main U-Net
        """
        super(HyperNetwork, self).__init__()

        # Initialize weights
        constant_scale = f_size*f_size*unet_nh
        init_std = lambda d_i : (2 / (d_i * constant_scale))**0.5
        self.normalize = normalize

        # Network layers
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin3 = nn.Linear(h_dim, h_dim)
        self.lin4 = nn.Linear(h_dim, h_dim)

        self.lin1.weight.data.normal_(std=init_std(in_dim))
        self.lin1.bias.data.fill_(0)
        self.lin2.weight.data.normal_(std=init_std(h_dim))
        self.lin2.bias.data.fill_(0)
        self.lin3.weight.data.normal_(std=init_std(h_dim))
        self.lin3.bias.data.fill_(0)
        self.lin4.weight.data.normal_(std=init_std(h_dim))
        self.lin4.bias.data.fill_(0)

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
            x = x / np.sum(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        return x

class Unet(nn.Module):
    """Main U-Net for image reconstruction"""
    def __init__(self, device, num_hyperparams, hnet_hdim, unet_hdim, residual=True):
        """
        Parameters
        ----------
        device : PyTorch device, 'cpu' or 'cuda:<gpu_id>'
        num_hyperparams : Number of hyperparameters (i.e. number of regularization functions)
        hnet_hdim : Hidden channel dimension of HyperNetwork
        unet_hdim : Hidden channel dimension of U-Net
        residual : Whether or not to use residual U-Net architecture
        """
        super(Unet, self).__init__()

        self.residual = residual
        self.unet_hdim = unet_hdim
        self.device = device

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.relu = nn.ReLU(inplace=True)

        # UNet
        self.conv_down0 = layers.BatchConv2DLayer(2, unet_hdim, hnet_hdim, padding=1)
        self.conv_down1 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_down2 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_down3 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_down4 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_down5 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_down6 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_down7 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)

        self.conv_up8 = layers.BatchConv2DLayer(unet_hdim+unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_up9 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_up10 = layers.BatchConv2DLayer(unet_hdim+unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_up11 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_up12 = layers.BatchConv2DLayer(unet_hdim+unet_hdim, unet_hdim, hnet_hdim, padding=1)
        self.conv_up13 = layers.BatchConv2DLayer(unet_hdim, unet_hdim, hnet_hdim, padding=1)

        self.conv_last14 = layers.BatchConv2DLayer(unet_hdim, 2, hnet_hdim, ks=1)

        # HyperNetwork
        self.hnet = HyperNetwork(in_dim=num_hyperparams, h_dim=hnet_hdim, unet_nh=unet_hdim)

    def forward(self, zf, hyperparams):
        """
        Parameters
        ----------
        zf : torch.Tensor (batch_size, img_height, img_width, 2)
            Zero-filled reconstruction of under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values

        """
        x = zf
        x = x.permute(0, 3, 1, 2)

        hyp_out = self.hnet(hyperparams)
        penalty = torch.zeros(len(zf), requires_grad=True).to(self.device)

        # conv_down1
        x, l1 = self.conv_down0(x, hyp_out)
        x = self.relu(x)
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
        x, l1 = self.conv_last14(x, hyp_out)
        penalty = penalty + l1

        x = x.permute(0, 2, 3, 1)
        if self.residual:
            x = zf + x

        return x, penalty

    def get_l1_weight_penalty(self, num_examples):
        '''Compute layer-wise L1 weight penalty'''
        penalty = torch.zeros(num_examples, requires_grad=True).to(self.device)
        penalty = penalty + self.conv_down0.get_l1_weight_penalty()
        penalty = penalty + self.conv_down1.get_l1_weight_penalty()
        penalty = penalty + self.conv_down2.get_l1_weight_penalty()
        penalty = penalty + self.conv_down3.get_l1_weight_penalty()
        penalty = penalty + self.conv_down4.get_l1_weight_penalty()
        penalty = penalty + self.conv_down5.get_l1_weight_penalty()
        penalty = penalty + self.conv_down6.get_l1_weight_penalty()
        penalty = penalty + self.conv_down7.get_l1_weight_penalty()

        penalty = penalty + self.conv_up8.get_l1_weight_penalty()
        penalty = penalty + self.conv_up9.get_l1_weight_penalty()
        penalty = penalty + self.conv_up10.get_l1_weight_penalty()
        penalty = penalty + self.conv_up11.get_l1_weight_penalty()
        penalty = penalty + self.conv_up12.get_l1_weight_penalty()
        penalty = penalty + self.conv_up13.get_l1_weight_penalty()
        penalty = penalty + self.conv_last14.get_l1_weight_penalty()

        # Old code that's probably wrong
        # cap_reg = torch.zeros(num_examples, requires_grad=True).to(self.device)
        # for k, b in zip(kl, bl):
        #     k_flat = k.view(num_examples, -1)
        #     b_flat = b.view(num_examples, -1)
        #     cap_reg += torch.norm(k_flat, dim=1)
        #     cap_reg += torch.norm(b_flat, dim=1)

        return penalty

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

class BaseUnet(nn.Module):
    def __init__(self, residual=True):
        super(BaseUnet, self).__init__()
                
        self.residual = residual
        self.dconv_down1 = self.double_conv(2, 64)
        self.dconv_down2 = self.double_conv(64, 64)
        self.dconv_down3 = self.double_conv(64, 64)
        self.dconv_down4 = self.double_conv(64, 64)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = self.double_conv(128, 64)
        self.dconv_up2 = self.double_conv(128, 64)
        self.dconv_up1 = self.double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, 2, 1)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )   
        
    def forward(self, zf):
        x = zf
        x = x.permute(0, 3, 1, 2)

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
        out = out.permute(0, 2, 3, 1)
        if self.residual:
            out = zf + out 
        
        return out

