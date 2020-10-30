from . import utils
from . import loss as losslayer
from . import layers
import torch
import torch.nn as nn
import sys

class HyperNetwork(nn.Module):
    def __init__(self, conv_layers, f_size=3, in_dim=1, h_dim=32, unet_nh=64):
        super(HyperNetwork, self).__init__()

        weight_shapes = []
        bias_shapes = []
        for layer in conv_layers.values():
            weight_shapes.append(layer.get_weight_shape())
            bias_shapes.append(layer.get_bias_shape())
        self.weight_shapes = torch.tensor(weight_shapes)
        self.bias_shapes = torch.tensor(bias_shapes)

        # Compute output dimension from conv layer dimensions
        out_dim = (torch.sum(torch.prod(self.weight_shapes, dim=1))+torch.sum(torch.prod(self.bias_shapes, dim=1))).item()
        print('out dimension', out_dim)

        # Compute weight and bias indices for flattened array
        self.wind = torch.cumsum(torch.prod(self.weight_shapes, dim=1), dim=0)
        self.bind = torch.cumsum(torch.prod(self.bias_shapes, dim=1), dim=0) + torch.sum(torch.prod(self.weight_shapes, dim=1))

        # Network layers
        self.lin1 = nn.Linear(in_dim, 100)
        self.lin2 = nn.Linear(100, 1000)
        self.lin3 = nn.Linear(1000, 10000)
        self.lin4 = nn.Linear(10000,10000)
        self.lin5 = nn.Linear(10000,10000)
        # self.lin6 = nn.Linear(20000,30000)
        self.lin_out = nn.Linear(10000, out_dim)


        # Network layers
        # self.lin1 = nn.Linear(in_dim, h_dim)
        # self.lin2 = nn.Linear(h_dim, h_dim)
        # self.lin_out = nn.Linear(h_dim, out_dim)

        # Activations
        self.relu = nn.LeakyReLU(inplace=True)
        self.tanh = nn.Tanh()

        # Initalize weights
        self.lin1.weight.data.normal_(std=1/(in_dim*unet_nh)**(1/2))
        self.lin1.bias.data.normal_(std=1/(in_dim*unet_nh)**(1/2))
        self.lin2.weight.data.normal_(std=1/(h_dim*unet_nh)**(1/2))
        self.lin2.bias.data.normal_(std=1/(h_dim*unet_nh)**(1/2))
        self.lin_out.weight.data.normal_(std=1/(h_dim*unet_nh)**(1/2))
        self.lin_out.bias.data.normal_(std=1/(h_dim*unet_nh)**(1/2))

    def forward(self, x):
        # x = self.lin1(x)
        # x = self.lin2(x)

        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.relu(self.lin3(x))
        x = self.relu(self.lin4(x))
        x = self.relu(self.lin5(x))
        # x = self.relu(self.lin6(x))
        # x = self.relu(self.lin4(x))
        # weights = self.tanh(self.lin_out(x))
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
    def __init__(self, device, num_hyperparams, nh=64, residual=True):
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
        self.hnet = HyperNetwork(self.convs, in_dim=num_hyperparams, unet_nh=nh)
        
    def forward(self, zf, y, hyperparams):
        x = zf
        x = x.permute(0, 3, 1, 2)

        wl, bl = self.hnet(hyperparams)
        l1_weights = 0
        for i, (w, b) in enumerate(zip(wl, bl)):
            l1_weights = l1_weights + torch.sum(torch.abs(w), dim=(1, 2, 3, 4)) + torch.sum(torch.abs(b), dim=(1))
            print('l1 of %d layer:' % i, wl[i].abs().sum().item())

        # conv_down1
        x = x.unsqueeze(1)
        x = self.relu(self.convs['down0'](x, wl[0], bias=bl[0]))
        conv1 = self.relu(self.convs['down1'](x, wl[1], bias=bl[1]))
        conv1 = conv1[:,0,...]
        x = self.maxpool(conv1)

        # conv_down2
        x = x.unsqueeze(1)
        x = self.relu(self.convs['down2'](x, wl[2], bias=bl[2]))
        conv2 = self.relu(self.convs['down3'](x, wl[3], bias=bl[3]))
        conv2 = conv2[:,0,...]
        x = self.maxpool(conv2)
        
        # conv_down3
        x = x.unsqueeze(1)
        x = self.relu(self.convs['down4'](x, wl[4], bias=bl[4]))
        conv3 = self.relu(self.convs['down5'](x, wl[5], bias=bl[5]))
        conv3 = conv3[:,0,...]
        x = self.maxpool(conv3)   
        
        # conv_down4
        x = x.unsqueeze(1)
        x = self.relu(self.convs['down6'](x, wl[6], bias=bl[6]))
        conv4 = self.relu(self.convs['down7'](x, wl[7], bias=bl[7]))
        conv4 = conv4[:,0,...]
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        
        # conv_up3
        x = x.unsqueeze(1)
        x = self.relu(self.convs['up8'](x, wl[8], bias=bl[8]))
        x = self.relu(self.convs['up9'](x, wl[9], bias=bl[9]))
        x = x[:,0,...]
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        # conv_up2
        x = x.unsqueeze(1)
        x = self.relu(self.convs['up10'](x, wl[10], bias=bl[10]))
        x = self.relu(self.convs['up11'](x, wl[11], bias=bl[11]))
        x = x[:,0,...]
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        # conv_up1
        x = x.unsqueeze(1)
        x = self.relu(self.convs['up12'](x, wl[12], bias=bl[12]))
        x = self.relu(self.convs['up13'](x, wl[13], bias=bl[13]))
        
        # last
        x = self.convs['last14'](x, wl[14], bias=bl[14])
        x = x[:,0,...]

        x = x.permute(0, 2, 3, 1)
        if self.residual:
            x = zf + x
        return x, l1_weights
