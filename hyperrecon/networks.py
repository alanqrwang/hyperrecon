"""
Model architecture for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import utils, layers
import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    """Hypernetwork architecture and forward pass

    Takes hyperparameters and outputs weights of main U-Net
    """
    def __init__(self, normalize, in_dim=1, h_dim=32):
        """
        Parameters
        ----------
        conv_layers : dict
            Dictionary of convolutional layers in main U-Net
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
            print('normalizing')
            x = x / torch.sum(x, dim=1, keepdim=True)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        return x


class Unet(nn.Module):
    """Main U-Net for image reconstruction"""
    def __init__(self, uh, hh=None, use_tanh=True):
        """
        Parameters
        ----------
        uh : Hidden channel dimension of U-Net
        hh : Hidden channel dimension of HyperNetwork
        residual : Whether or not to use residual U-Net architecture
        """
        super(Unet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.relu = nn.ReLU(inplace=True)

        self.conv_down0 = layers.ConvBlock(2,     uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down1 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down2 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down3 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down4 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down5 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down6 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_down7 = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)

        self.conv_up0   = layers.ConvBlock(uh+uh, uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_up1   = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_up2   = layers.ConvBlock(uh+uh, uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_up3   = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_up4   = layers.ConvBlock(uh+uh, uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)
        self.conv_up5   = layers.ConvBlock(uh,    uh, padding=1, hyp_out_units=hh, use_tanh=use_tanh)

        self.conv_last  =  layers.ConvBlock(uh,   1,  ks=1, hyp_out_units=hh, use_tanh=use_tanh)


    def forward(self, x, hyp_out=None):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, 2, img_height, img_width)
            Input image
        hyp_out : torch.Tensor (batch_size, hyp_out_units)
            Output of hypernetwork, if provided

        """
        # Encoding
        x = self.conv_down0(x, hyp_out)
        conv1 = self.conv_down1(x, hyp_out)
        x = self.maxpool(conv1)

        x = self.conv_down2(x, hyp_out)
        conv2 = self.conv_down3(x, hyp_out)
        x = self.maxpool(conv2)

        x = self.conv_down4(x, hyp_out)
        conv3 = self.conv_down5(x, hyp_out)
        x = self.maxpool(conv3)   

        x = self.conv_down6(x, hyp_out)
        conv4 = self.conv_down7(x, hyp_out)

        # Decoding
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up0(x, hyp_out)
        x = self.conv_up1(x, hyp_out)

        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        x = self.conv_up2(x, hyp_out)
        x = self.conv_up3(x, hyp_out)

        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        x = self.conv_up4(x, hyp_out)
        x = self.conv_up5(x, hyp_out)

        # last
        x = self.conv_last(x, hyp_out)
        return x

    def get_l1_weight_penalty(self):
        '''Compute layer-wise L1 weight penalty'''
        penalty = torch.tensor(0., requires_grad=True).cuda()
        penalty = penalty + self.conv_down0.get_l1_weight_penalty()
        penalty = penalty + self.conv_down1.get_l1_weight_penalty()
        penalty = penalty + self.conv_down2.get_l1_weight_penalty()
        penalty = penalty + self.conv_down3.get_l1_weight_penalty()
        penalty = penalty + self.conv_down4.get_l1_weight_penalty()
        penalty = penalty + self.conv_down5.get_l1_weight_penalty()
        penalty = penalty + self.conv_down6.get_l1_weight_penalty()
        penalty = penalty + self.conv_down7.get_l1_weight_penalty()

        penalty = penalty + self.conv_up0.get_l1_weight_penalty()
        penalty = penalty + self.conv_up1.get_l1_weight_penalty()
        penalty = penalty + self.conv_up2.get_l1_weight_penalty()
        penalty = penalty + self.conv_up3.get_l1_weight_penalty()
        penalty = penalty + self.conv_up4.get_l1_weight_penalty()
        penalty = penalty + self.conv_up5.get_l1_weight_penalty()
        penalty = penalty + self.conv_last.get_l1_weight_penalty()

        return penalty

class HyperUnet(nn.Module):
    """Main U-Net for image reconstruction"""
    def __init__(self, num_hyperparams, normalize, uh, residual=True, hh=None, use_tanh=True):
        """
        Parameters
        ----------
        num_hyperparams : Number of hyperparameters
        uh : Hidden channel dimension of U-Net
        residual : Whether or not to use residual U-Net architecture
        hh : Hidden channel dimension of HyperNetwork, activates hypernetwork if provided
        """
        super(HyperUnet, self).__init__()

        self.residual = residual
        self.hh = hh

        if hh is not None:
            self.hypernet = HyperNetwork(normalize, in_dim=num_hyperparams, h_dim=hh)
        self.unet = Unet(uh, hh=hh, use_tanh=use_tanh)

    def forward(self, y, hyperparams):
        """
        Parameters
        ----------
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values

        """
        if self.hh is not None:
            hyp_out = self.hypernet(hyperparams)
        else:
            hyp_out = None

        zf = utils.ifft(y)
        unet_out = self.unet(zf.permute(0, 3, 1, 2), hyp_out).permute(0, 2, 3, 1)
        if self.residual:
            out = zf.norm(-1) + unet_out 

        return out 

    def get_l1_weight_penalty(self):
        '''Compute layer-wise L1 weight penalty'''
        return self.unet.get_l1_weight_penalty()

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

