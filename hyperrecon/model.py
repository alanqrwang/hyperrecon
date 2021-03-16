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

class HyperNetwork(nn.Module):
    """Hypernetwork architecture and forward pass

    Takes hyperparameters and outputs weights of main U-Net
    """
    def __init__(self, hyparch, f_size=3, in_dim=1, h_dim=32, unet_nh=64):
        """
        Parameters
        ----------
        conv_layers : dict
            Dictionary of convolutional layers in main U-Net
        hyparch : str
            Hypernetwork architecture [small, medium, large]
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
        self.hyparch = hyparch

        # weight_shapes = []
        # bias_shapes = []
        # for layer in conv_layers.values():
        #     weight_shapes.append(layer.get_weight_shape())
        #     bias_shapes.append(layer.get_bias_shape())
        # self.weight_shapes = torch.tensor(weight_shapes)
        # self.bias_shapes = torch.tensor(bias_shapes)

        # # Compute output dimension from conv layer dimensions
        # out_dim = (torch.sum(torch.prod(self.weight_shapes, dim=1))+torch.sum(torch.prod(self.bias_shapes, dim=1))).item()

        # # Compute weight and bias indices for flattened array
        # self.wind = torch.cumsum(torch.prod(self.weight_shapes, dim=1), dim=0)
        # self.bind = torch.cumsum(torch.prod(self.bias_shapes, dim=1), dim=0) + torch.sum(torch.prod(self.weight_shapes, dim=1))

        # Initialize weights
        constant_scale = f_size*f_size*unet_nh
        init_std = lambda d_i : (2 / (d_i * constant_scale))**0.5

        # Network layers
        if hyparch == 'small':
            print('Small Hypernetwork')
            dim1, dim2 = 2, 4
        elif hyparch == 'medium':
            print('Medium Hypernetwork')
            dim1, dim2 = 8, 32
        elif hyparch == 'large':
            print('Large Hypernetwork')
            dim1, dim2 = 8, 32
        elif hyparch =='huge':
            print('Huge Hypernetwork')
            dim1, dim2 = 16, 64
        elif hyparch == 'massive':
            print('Massive Hypernetwork')
            dim1, dim2 = 32, 128

        self.lin1 = nn.Linear(in_dim, dim1)
        self.lin2 = nn.Linear(dim1, dim2)
        # self.lin_out = nn.Linear(dim2, out_dim)
        self.lin_out = nn.Linear(dim2, dim2)

        # self.batchnorm1 = nn.BatchNorm1d(dim1)
        # self.batchnorm2 = nn.BatchNorm1d(dim2)

        self.lin1.weight.data.normal_(std=init_std(in_dim))
        self.lin1.bias.data.fill_(0)
        self.lin2.weight.data.normal_(std=init_std(dim1))
        self.lin2.bias.data.fill_(0)
        self.lin_out.weight.data.normal_(std=init_std(dim2))
        self.lin_out.bias.data.fill_(0)

        if hyparch is not 'small':
            self.lin3 = nn.Linear(dim2, dim2)
            self.lin4 = nn.Linear(dim2, dim2)
            # self.batchnorm3 = nn.BatchNorm1d(dim2)
            # self.batchnorm4 = nn.BatchNorm1d(dim2)

            self.lin3.weight.data.normal_(std=init_std(dim2))
            self.lin3.bias.data.fill_(0)
            self.lin4.weight.data.normal_(std=init_std(dim2))
            self.lin4.bias.data.fill_(0)

        # Activations
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values

        Returns
        ----------
        wl : list
            Weights indexed by layer
        bl : list 
            Biases indexed by layer
        """
        if self.hyparch == 'small' or self.hyparch == 'medium':
            # x = self.batchnorm1(self.relu(self.lin1(x)))
            # x = self.batchnorm2(self.relu(self.lin2(x)))
            x = self.relu(self.lin1(x))
            x = self.relu(self.lin2(x))

        elif self.hyparch in ['large', 'huge', 'massive', 'gigantic']:
            # x = self.batchnorm1(self.relu(self.lin1(x)))
            # x = self.batchnorm2(self.relu(self.lin2(x)))
            # x = self.batchnorm3(self.relu(self.lin3(x)))
            # x = self.batchnorm4(self.relu(self.lin4(x)))
            x = self.relu(self.lin1(x))
            x = self.relu(self.lin2(x))
            x = self.relu(self.lin3(x))
            x = self.relu(self.lin4(x))
        else:
            sys.exit('Error with hypernet forward pass')

        hyp_out = self.lin_out(x)
        return hyp_out

class Unet(nn.Module):
    """Main U-Net for image reconstruction"""
    def __init__(self, device, num_hyperparams, hyparch, nh=64, residual=True):
        """
        Parameters
        ----------
        device : str 
            PyTorch device, 'cpu' or 'cuda:<gpu_id>'
        num_hyperparams : int
            Number of hyperparameters (i.e. number of regularization functions)
        nh : int
            Hidden channel dimension of U-Net
        residual : bool
            Whether or not to use residual U-Net architecture
        """
        super(Unet, self).__init__()

        self.residual = residual
        self.nh = nh
        self.device = device

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.relu = nn.ReLU(inplace=True)

        if hyparch == 'small':
            hyp_out_units = 4
        elif hyparch == 'medium':
            hyp_out_units = 32 
        elif hyparch == 'large':
            hyp_out_units = 32
        else:
            raise Exception()

        # UNet
        self.conv_down0 = layers.BatchConv2DLayer(2, nh, hyp_out_units, device, padding=1)
        self.conv_down1 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_down2 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_down3 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_down4 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_down5 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_down6 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_down7 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)

        self.conv_up8 = layers.BatchConv2DLayer(nh+nh, nh, hyp_out_units, device, padding=1)
        self.conv_up9 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_up10 = layers.BatchConv2DLayer(nh+nh, nh, hyp_out_units, device, padding=1)
        self.conv_up11 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)
        self.conv_up12 = layers.BatchConv2DLayer(nh+nh, nh, hyp_out_units, device, padding=1)
        self.conv_up13 = layers.BatchConv2DLayer(nh, nh, hyp_out_units, device, padding=1)

        self.conv_last14 = layers.BatchConv2DLayer(nh, 2, hyp_out_units, device, ks=1)

        # HyperNetwork
        self.hnet = HyperNetwork(hyparch, in_dim=num_hyperparams, unet_nh=nh)

    def forward(self, zf, y, hyperparams):
        """
        Parameters
        ----------
        zf : torch.Tensor (batch_size, img_height, img_width, 2)
            Zero-filled reconstruction of under-sampled measurement
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Undersampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Hyperparameter values

        Returns
        ----------
        x : torch.Tensor (batch_size, img_height, img_width, 2)
            Reconstructed image
        """
        x = zf
        x = x.permute(0, 3, 1, 2)

        hyp_out = self.hnet(hyperparams)

        # conv_down1
        x = self.relu(self.conv_down0(x, hyp_out))
        conv1 = self.relu(self.conv_down1(x, hyp_out))
        x = self.maxpool(conv1)
        # conv_down2
        x = self.relu(self.conv_down2(x, hyp_out))
        conv2 = self.relu(self.conv_down3(x, hyp_out))
        x = self.maxpool(conv2)
        # conv_down3
        x = self.relu(self.conv_down4(x, hyp_out))
        conv3 = self.relu(self.conv_down5(x, hyp_out))
        x = self.maxpool(conv3)   
        # conv_down4
        x = self.relu(self.conv_down6(x, hyp_out))
        conv4 = self.relu(self.conv_down7(x, hyp_out))
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        # conv_up3
        x = self.relu(self.conv_up8(x, hyp_out))
        x = self.relu(self.conv_up9(x, hyp_out))
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        # conv_up2
        x = self.relu(self.conv_up10(x, hyp_out))
        x = self.relu(self.conv_up11(x, hyp_out))
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        # conv_up1
        x = self.relu(self.conv_up12(x, hyp_out))
        x = self.relu(self.conv_up13(x, hyp_out))
        # last
        x = self.conv_last14(x, hyp_out)

        x = x.permute(0, 2, 3, 1)
        if self.residual:
            x = zf + x

        return x

    def get_l1_weight_penalty(self, num_examples):
        # Compute layer-wise L1 capacity regularization
        kl = []
        bl = []
        kl.append(self.conv_down0.kernel)
        bl.append(self.conv_down0.bias)
        kl.append(self.conv_down1.kernel)
        bl.append(self.conv_down1.bias)
        kl.append(self.conv_down2.kernel)
        bl.append(self.conv_down2.bias)
        kl.append(self.conv_down3.kernel)
        bl.append(self.conv_down3.bias)
        kl.append(self.conv_down4.kernel)
        bl.append(self.conv_down4.bias)
        kl.append(self.conv_down5.kernel)
        bl.append(self.conv_down5.bias)
        kl.append(self.conv_down6.kernel)
        bl.append(self.conv_down6.bias)
        kl.append(self.conv_down7.kernel)
        bl.append(self.conv_down7.bias)

        kl.append(self.conv_up8.kernel)
        bl.append(self.conv_up8.bias)
        kl.append(self.conv_up9.kernel)
        bl.append(self.conv_up9.bias)
        kl.append(self.conv_up10.kernel)
        bl.append(self.conv_up10.bias)
        kl.append(self.conv_up11.kernel)
        bl.append(self.conv_up11.bias)
        kl.append(self.conv_up12.kernel)
        bl.append(self.conv_up12.bias)
        kl.append(self.conv_up13.kernel)
        bl.append(self.conv_up13.bias)

        kl.append(self.conv_last14.kernel)
        bl.append(self.conv_last14.bias)

        cap_reg = torch.zeros(num_examples, requires_grad=True).to(self.device)
        for k, b in zip(kl, bl):
            k_flat = k.view(num_examples, -1)
            b_flat = b.view(num_examples, -1)
            cap_reg += torch.norm(k_flat, dim=1)
            cap_reg += torch.norm(b_flat, dim=1)

        return cap_reg

class TrajNet(nn.Module):
    '''
    dim_bounds is out_dim by 2 array, each row contains bounds for that dimension
    '''
    def __init__(self, in_dim=1, h_dim=8, out_dim=2):
        super(TrajNet, self).__init__()

        self.out_dim = out_dim
        # self.lin1 = nn.Linear(in_dim, h_dim)
#         self.lin1 = nn.Linear(in_dim, out_dim)
        # self.lin2 = nn.Linear(h_dim, h_dim)
        # self.lin3 = nn.Linear(h_dim, out_dim)
        self.lin = nn.Linear(in_dim, out_dim)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.lin1(x)
        # x = self.relu(x)
        # x = self.lin2(x)
        # x = self.relu(x)
        # x = self.lin3(x)
        # dc, regs = torch.split(x, [1, self.out_dim-1], dim=1)
        # dc = self.dc_lower_bound + (1 - self.dc_lower_bound) / (1 + torch.exp(-dc))
        # regs = self.sigmoid(regs)
        # out = torch.cat((dc, regs), dim=1)
        x = self.lin(x)
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

