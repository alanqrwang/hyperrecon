from . import utils
from . import loss as losslayer
from . import layers
import torch
import torch.nn as nn
import sys

def get_indices(unet_nh, ks=3):
    unet_kern_shapes = torch.tensor([
        [unet_nh, 2, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh+unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh+unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [unet_nh, unet_nh+unet_nh, ks, ks], \
        [unet_nh, unet_nh, ks, ks], \
        [2, unet_nh, 1, 1] \
    ])
    unet_bias_shapes = torch.tensor([
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [unet_nh], \
        [2]
        ])

    out_dim = (torch.sum(torch.prod(unet_kern_shapes, dim=1))+torch.sum(torch.prod(unet_bias_shapes, dim=1))).item()
    kind = torch.cumsum(torch.prod(unet_kern_shapes, dim=1), dim=0)
    bind = torch.cumsum(torch.prod(unet_bias_shapes, dim=1), dim=0) + torch.sum(torch.prod(unet_kern_shapes, dim=1))
    return unet_kern_shapes.numpy(), unet_bias_shapes.numpy(), kind, bind, out_dim


class HyperNetwork(nn.Module):
    def __init__(self, f_size=3, in_dim=1, h_dim=32, unet_nh=64):
        super(HyperNetwork, self).__init__()

        self.kern_shapes, self.bias_shapes, self.kind, self.bind, out_dim = get_indices(unet_nh)

        self.lin1 = nn.Linear(in_dim, h_dim)
        self.lin2 = nn.Linear(h_dim, h_dim)
        self.lin_out = nn.Linear(h_dim, out_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.lin1.weight.data.normal_(std=1/(in_dim*unet_nh)**(1/2))
        self.lin1.bias.data.normal_(std=1/(in_dim*unet_nh)**(1/2))
        self.lin2.weight.data.normal_(std=1/(h_dim*unet_nh)**(1/2))
        self.lin2.bias.data.normal_(std=1/(h_dim*unet_nh)**(1/2))
        self.lin_out.weight.data.normal_(std=1/(h_dim*unet_nh)**(1/2))
        self.lin_out.bias.data.normal_(std=1/(h_dim*unet_nh)**(1/2))

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        weights = self.tanh(self.lin_out(x))

        # Reorganize weight vector into kernel and bias shapes
        weight_list = []
        bias_list = []

        weight_list.append(weights[:, :self.kind[0]].view(-1, *self.kern_shapes[0]))
        bias_list.append(weights[:, self.kind[14]:self.bind[0]].view(-1, *self.bias_shapes[0]))
        for i in range(len(self.kern_shapes)-1):
            weight_list.append(weights[:, self.kind[i]:self.kind[i+1]].view(-1, *self.kern_shapes[i+1]))
            bias_list.append(weights[:, self.bind[i]:self.bind[i+1]].view(-1, *self.bias_shapes[i+1]))

        return weight_list, bias_list 

class Unet(nn.Module):
    def __init__(self, device, num_hyperparams, nh=64, residual=True):
        super(Unet, self).__init__()

        self.residual = residual
        self.nh = nh
        self.device = device

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = layers.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.relu = nn.ReLU(inplace=True)


        # HyperNetwork
        self.hnet = HyperNetwork(in_dim=num_hyperparams, unet_nh=nh)

        # UNet
        self.conv_down_1_0 = layers.BatchConv2DLayer(2, nh, padding=1)
        self.conv_down_1_1 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_down_2_0 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_down_2_1 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_down_3_0 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_down_3_1 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_down_4_0 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_down_4_1 = layers.BatchConv2DLayer(nh, nh, padding=1)

        self.conv_up_3_0 = layers.BatchConv2DLayer(nh+nh, nh, padding=1)
        self.conv_up_3_1 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_up_2_0 = layers.BatchConv2DLayer(nh+nh, nh, padding=1)
        self.conv_up_2_1 = layers.BatchConv2DLayer(nh, nh, padding=1)
        self.conv_up_1_0 = layers.BatchConv2DLayer(nh+nh, nh, padding=1)
        self.conv_up_1_1 = layers.BatchConv2DLayer(nh, nh, padding=1)

        self.conv_last = layers.BatchConv2DLayer(nh, 2)
        
    def forward(self, zf, y, hyperparams):
        x = zf
        x = x.permute(0, 3, 1, 2)

        wl, bl = self.hnet(hyperparams)
        l1_weights = 0
        for w, b in zip(wl, bl):
            l1_weights = l1_weights + torch.sum(torch.abs(w), dim=(1, 2, 3, 4)) + torch.sum(torch.abs(b), dim=(1))

        # conv_down1
        x = x.unsqueeze(1)
        x = self.relu(self.conv_down_1_0(x, wl[0], bias=bl[0]))
        conv1 = self.relu(self.conv_down_1_1(x, wl[1], bias=bl[1]))
        conv1 = conv1[:,0,...]
        x = self.maxpool(conv1)

        # conv_down2
        x = x.unsqueeze(1)
        x = self.relu(self.conv_down_2_0(x, wl[2], bias=bl[2]))
        conv2 = self.relu(self.conv_down_2_1(x, wl[3], bias=bl[3]))
        conv2 = conv2[:,0,...]
        x = self.maxpool(conv2)
        
        # conv_down3
        x = x.unsqueeze(1)
        x = self.relu(self.conv_down_3_0(x, wl[4], bias=bl[4]))
        conv3 = self.relu(self.conv_down_3_1(x, wl[5], bias=bl[5]))
        conv3 = conv3[:,0,...]
        x = self.maxpool(conv3)   
        
        # conv_down4
        x = x.unsqueeze(1)
        x = self.relu(self.conv_down_4_0(x, wl[6], bias=bl[6]))
        conv4 = self.relu(self.conv_down_4_1(x, wl[7], bias=bl[7]))
        conv4 = conv4[:,0,...]
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        
        # conv_up3
        x = x.unsqueeze(1)
        x = self.relu(self.conv_up_3_0(x, wl[8], bias=bl[8]))
        x = self.relu(self.conv_up_3_1(x, wl[9], bias=bl[9]))
        x = x[:,0,...]
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        # conv_up2
        x = x.unsqueeze(1)
        x = self.relu(self.conv_up_2_0(x, wl[10], bias=bl[10]))
        x = self.relu(self.conv_up_2_1(x, wl[11], bias=bl[11]))
        x = x[:,0,...]
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        # conv_up1
        x = x.unsqueeze(1)
        x = self.relu(self.conv_up_1_0(x, wl[12], bias=bl[12]))
        x = self.relu(self.conv_up_1_1(x, wl[13], bias=bl[13]))
        
        # last
        x = self.conv_last(x, wl[14], bias=bl[14])
        x = x[:,0,...]

        x = x.permute(0, 2, 3, 1)
        if self.residual:
            x = zf + x
        return x, l1_weights
