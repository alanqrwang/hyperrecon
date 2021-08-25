"""
Model architecture for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
from . import layers
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
    def __init__(self, in_units_hnet, h_units_hnet, hnet_norm, in_ch_main, out_ch_main, h_ch_main, residual=True):
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

        # HyperNetwork
        self.hnet = HyperNetwork(hnet_norm, 
                                 in_dim=in_units_hnet, 
                                 h_dim=h_units_hnet)
        self.unet = Unet(in_ch=in_ch_main, 
                         out_ch=out_ch_main, 
                         h_ch=h_ch_main, 
                         hnet_hdim=h_units_hnet)

    def forward(self, x, hyperparams):
        """
        Parameters
        ----------
        zf : torch.Tensor (batch_size, 2, img_height, img_width)
            Zero-filled reconstruction of under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values

        """
        hyp_out = self.hnet(hyperparams)
        out = self.unet(x, hyp_out)

        return out

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch, hnet_hdim=None, residual=True):
        '''
        hnet_hdim activates hypernetwork for Unet
        '''
        super(Unet, self).__init__()
                
        self.residual = residual
        self.hnet_hdim = hnet_hdim

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
            self.conv_last = layers.BatchConv2d(h_ch, out_ch, hnet_hdim, ks=1)
        else:
            self.conv_last = nn.Conv2d(h_ch, out_ch, 1)
        

    def double_conv(self, in_channels, out_channels):
        if self.hnet_hdim is not None:
            return layers.MultiSequential(
                layers.BatchConv2d(in_channels, out_channels, self.hnet_hdim, padding=1),
                nn.ReLU(inplace=True),
                layers.BatchConv2d(out_channels, out_channels, self.hnet_hdim, padding=1),
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
        x = x.permute(0, 3, 1, 2)

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

        if self.hnet_hdim is not None:
            out = self.conv_last(x, hyp_out)
        else:
            out = self.conv_last(x)

        out = out.permute(0, 2, 3, 1)
        if self.residual:
            zf = zf.norm(p=2, dim=-1, keepdim=True)
            out = zf + out 
        
        return out

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

