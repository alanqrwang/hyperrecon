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

class HyperNetwork(nn.Module):
    """Hypernetwork architecture and forward pass

    Takes hyperparameters and outputs weights of main U-Net
    """
    def __init__(self, conv_layers, hyparch, f_size=3, in_dim=1, h_dim=32, unet_nh=64):
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

        weight_shapes = []
        bias_shapes = []
        for layer in conv_layers.values():
            weight_shapes.append(layer.get_weight_shape())
            bias_shapes.append(layer.get_bias_shape())
        self.weight_shapes = torch.tensor(weight_shapes)
        self.bias_shapes = torch.tensor(bias_shapes)

        # Compute output dimension from conv layer dimensions
        out_dim = (torch.sum(torch.prod(self.weight_shapes, dim=1))+torch.sum(torch.prod(self.bias_shapes, dim=1))).item()

        # Compute weight and bias indices for flattened array
        self.wind = torch.cumsum(torch.prod(self.weight_shapes, dim=1), dim=0)
        self.bind = torch.cumsum(torch.prod(self.bias_shapes, dim=1), dim=0) + torch.sum(torch.prod(self.weight_shapes, dim=1))

        # Initialize weights
        constant_scale = f_size*f_size*unet_nh
        init_std = lambda d_i : (2 / (d_i * constant_scale))**0.5

        # Network layers
        if hyparch == 'small':
            print('Small Hypernetwork')
            self.lin1 = nn.Linear(in_dim, 2)
            self.lin2 = nn.Linear(2, 4)
            self.lin_out = nn.Linear(4, out_dim)

            self.batchnorm1 = nn.BatchNorm1d(2)
            self.batchnorm2 = nn.BatchNorm1d(4)

            self.lin1.weight.data.normal_(std=init_std(in_dim))
            self.lin1.bias.data.fill_(0)
            self.lin2.weight.data.normal_(std=init_std(2))
            self.lin2.bias.data.fill_(0)
            self.lin_out.weight.data.normal_(std=init_std(4))
            self.lin_out.bias.data.fill_(0)

        elif hyparch == 'medium':
            print('Medium Hypernetwork')
            self.lin1 = nn.Linear(in_dim, 8)
            self.lin2 = nn.Linear(8, 32)
            self.lin_out = nn.Linear(32, out_dim)

            self.batchnorm1 = nn.BatchNorm1d(8)
            self.batchnorm2 = nn.BatchNorm1d(32)

            self.lin1.weight.data.normal_(std=init_std(in_dim))
            self.lin1.bias.data.fill_(0)
            self.lin2.weight.data.normal_(std=init_std(8))
            self.lin2.bias.data.fill_(0)
            self.lin_out.weight.data.normal_(std=init_std(32))
            self.lin_out.bias.data.fill_(0)
            
        elif hyparch =='large':
            print('Large Hypernetwork')
            self.lin1 = nn.Linear(in_dim, 8)
            self.lin2 = nn.Linear(8, 32)
            self.lin3 = nn.Linear(32, 32)
            self.lin4 = nn.Linear(32, 32)
            self.lin_out = nn.Linear(32, out_dim)

            self.batchnorm1 = nn.BatchNorm1d(8)
            self.batchnorm2 = nn.BatchNorm1d(32)
            self.batchnorm3 = nn.BatchNorm1d(32)
            self.batchnorm4 = nn.BatchNorm1d(32)

            self.lin1.weight.data.normal_(std=init_std(in_dim))
            self.lin1.bias.data.fill_(0)
            self.lin2.weight.data.normal_(std=init_std(8))
            self.lin2.bias.data.fill_(0)
            self.lin3.weight.data.normal_(std=init_std(32))
            self.lin3.bias.data.fill_(0)
            self.lin4.weight.data.normal_(std=init_std(32))
            self.lin4.bias.data.fill_(0)
            self.lin_out.weight.data.normal_(std=init_std(32))
            self.lin_out.bias.data.fill_(0)

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
            x = self.batchnorm1(self.relu(self.lin1(x)))
            x = self.batchnorm2(self.relu(self.lin2(x)))

        elif self.hyparch == 'large':
            x = self.batchnorm1(self.relu(self.lin1(x)))
            x = self.batchnorm2(self.relu(self.lin2(x)))
            x = self.batchnorm3(self.relu(self.lin3(x)))
            x = self.batchnorm4(self.relu(self.lin4(x)))

        weights = self.lin_out(x)

        # Reorganize weight vector into weight and bias shapes
        wl = []
        bl = []
        wl.append(weights[:, :self.wind[0]].view(-1, *self.weight_shapes[0]))
        bl.append(weights[:, self.wind[14]:self.bind[0]].view(-1, *self.bias_shapes[0]))
        for i in range(len(self.weight_shapes)-1):
            wl.append(weights[:, self.wind[i]:self.wind[i+1]].view(-1, *self.weight_shapes[i+1]))
            bl.append(weights[:, self.bind[i]:self.bind[i+1]].view(-1, *self.bias_shapes[i+1]))

        return wl, bl 

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

        # UNet
        self.convs = {}
        self.convs['down0'] = layers.BatchConv2DLayer(2, nh, padding=1)
        self.convs['down1'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['down2'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['down3'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['down4'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['down5'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['down6'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['down7'] = layers.BatchConv2DLayer(nh, nh, padding=1)

        self.convs['up8'] = layers.BatchConv2DLayer(nh+nh, nh, padding=1)
        self.convs['up9'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['up10'] = layers.BatchConv2DLayer(nh+nh, nh, padding=1)
        self.convs['up11'] = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.convs['up12'] = layers.BatchConv2DLayer(nh+nh, nh, padding=1)
        self.convs['up13'] = layers.BatchConv2DLayer(nh, nh, padding=1)

        self.convs['last14'] = layers.BatchConv2DLayer(nh, 2, ks=1)

        # HyperNetwork
        self.hnet = HyperNetwork(self.convs, hyparch, in_dim=num_hyperparams, unet_nh=nh)

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

        wl, bl = self.hnet(hyperparams)

        # Compute layer-wise L1 capacity regularization
        cap_reg = torch.zeros(len(y), requires_grad=True).to(self.device)
        for w, b in zip(wl, bl):
            w_flat = w.view(len(y), -1)
            b_flat = b.view(len(y), -1)
            cap_reg += torch.sum(torch.abs(w_flat), dim=1)
            cap_reg += torch.sum(torch.abs(b_flat), dim=1)

        # conv_down1
        x = self.relu(self.convs['down0'](x, wl[0], bias=bl[0]))
        conv1 = self.relu(self.convs['down1'](x, wl[1], bias=bl[1]))
        x = self.maxpool(conv1)
        # conv_down2
        x = self.relu(self.convs['down2'](x, wl[2], bias=bl[2]))
        conv2 = self.relu(self.convs['down3'](x, wl[3], bias=bl[3]))
        x = self.maxpool(conv2)
        # conv_down3
        x = self.relu(self.convs['down4'](x, wl[4], bias=bl[4]))
        conv3 = self.relu(self.convs['down5'](x, wl[5], bias=bl[5]))
        x = self.maxpool(conv3)   
        # conv_down4
        x = self.relu(self.convs['down6'](x, wl[6], bias=bl[6]))
        conv4 = self.relu(self.convs['down7'](x, wl[7], bias=bl[7]))
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        # conv_up3
        x = self.relu(self.convs['up8'](x, wl[8], bias=bl[8]))
        x = self.relu(self.convs['up9'](x, wl[9], bias=bl[9]))
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       
        # conv_up2
        x = self.relu(self.convs['up10'](x, wl[10], bias=bl[10]))
        x = self.relu(self.convs['up11'](x, wl[11], bias=bl[11]))
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        # conv_up1
        x = self.relu(self.convs['up12'](x, wl[12], bias=bl[12]))
        x = self.relu(self.convs['up13'](x, wl[13], bias=bl[13]))
        # last
        x = self.convs['last14'](x, wl[14], bias=bl[14])

        x = x.permute(0, 2, 3, 1)
        if self.residual:
            x = zf + x
        return x, cap_reg
