from . import utils
from . import loss as losslayer
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

def get_indices(unet_nh):
    unet_kern_shapes = torch.tensor([
        [unet_nh, 2, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh+unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh+unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
        [unet_nh, unet_nh+unet_nh, 3, 3], \
        [unet_nh, unet_nh, 3, 3], \
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

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class HyperNetwork(nn.Module):
    def __init__(self, f_size=3, in_dim=1, h_dim=32, unet_nh=64):
        super(HyperNetwork, self).__init__()

        _,_,kind, bind, out_dim = get_indices(unet_nh)

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
        x = self.tanh(self.lin_out(x))

        return x 

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

    def forward(self, x, w, b):
        return F.conv2d(x, w, bias=b, padding=1)

class Unet(nn.Module):
    def __init__(self, device, num_hyperparams, nh=64, residual=True):
        super(Unet, self).__init__()
        # UNet
        self.residual = residual
        self.device = device

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.relu = nn.ReLU(inplace=True)

        self.kern_shapes, self.bias_shapes,self.kind, self.bind, out_dim = get_indices(nh)
        # HyperNetwork
        self.hnet = HyperNetwork(in_dim=num_hyperparams, unet_nh=nh)

        
    def forward(self, zf, y, hyperparams):
        x = zf
        x = x.permute(0, 3, 1, 2)

        hyperparams = hyperparams.view(-1) # vectorize any length input
        weights = self.hnet(hyperparams)
        conv_down1_0_w = weights[:self.kind[0]].view(tuple(self.kern_shapes[0]))
        conv_down1_1_w = weights[self.kind[0]:self.kind[1]].view(tuple(self.kern_shapes[1]))
        conv_down2_0_w = weights[self.kind[1]:self.kind[2]].view(tuple(self.kern_shapes[2]))
        conv_down2_1_w = weights[self.kind[2]:self.kind[3]].view(tuple(self.kern_shapes[3]))
        conv_down3_0_w = weights[self.kind[3]:self.kind[4]].view(tuple(self.kern_shapes[4]))
        conv_down3_1_w = weights[self.kind[4]:self.kind[5]].view(tuple(self.kern_shapes[5]))
        conv_down4_0_w = weights[self.kind[5]:self.kind[6]].view(tuple(self.kern_shapes[6]))
        conv_down4_1_w = weights[self.kind[6]:self.kind[7]].view(tuple(self.kern_shapes[7]))
        conv_up3_0_w = weights[self.kind[7]:self.kind[8]].view(tuple(self.kern_shapes[8]))
        conv_up3_1_w = weights[self.kind[8]:self.kind[9]].view(tuple(self.kern_shapes[9]))
        conv_up2_0_w = weights[self.kind[9]:self.kind[10]].view(tuple(self.kern_shapes[10]))
        conv_up2_1_w = weights[self.kind[10]:self.kind[11]].view(tuple(self.kern_shapes[11]))
        conv_up1_0_w = weights[self.kind[11]:self.kind[12]].view(tuple(self.kern_shapes[12]))
        conv_up1_1_w = weights[self.kind[12]:self.kind[13]].view(tuple(self.kern_shapes[13]))
        conv_last_w = weights[self.kind[13]:self.kind[14]].view(tuple(self.kern_shapes[14]))
        
        conv_down1_0_b = weights[self.kind[14]:self.bind[0]].view(tuple(self.bias_shapes[0]))
        conv_down1_1_b = weights[self.bind[0]:self.bind[1]].view(tuple(self.bias_shapes[1]))
        conv_down2_0_b = weights[self.bind[1]:self.bind[2]].view(tuple(self.bias_shapes[2]))
        conv_down2_1_b = weights[self.bind[2]:self.bind[3]].view(tuple(self.bias_shapes[3]))
        conv_down3_0_b = weights[self.bind[3]:self.bind[4]].view(tuple(self.bias_shapes[4]))
        conv_down3_1_b = weights[self.bind[4]:self.bind[5]].view(tuple(self.bias_shapes[5]))
        conv_down4_0_b = weights[self.bind[5]:self.bind[6]].view(tuple(self.bias_shapes[6]))
        conv_down4_1_b = weights[self.bind[6]:self.bind[7]].view(tuple(self.bias_shapes[7]))
        conv_up3_0_b = weights[self.bind[7]:self.bind[8]].view(tuple(self.bias_shapes[8]))
        conv_up3_1_b = weights[self.bind[8]:self.bind[9]].view(tuple(self.bias_shapes[9]))
        conv_up2_0_b = weights[self.bind[9]:self.bind[10]].view(tuple(self.bias_shapes[10]))
        conv_up2_1_b = weights[self.bind[10]:self.bind[11]].view(tuple(self.bias_shapes[11]))
        conv_up1_0_b = weights[self.bind[11]:self.bind[12]].view(tuple(self.bias_shapes[12]))
        conv_up1_1_b = weights[self.bind[12]:self.bind[13]].view(tuple(self.bias_shapes[13]))
        conv_last_b = weights[self.bind[13]:self.bind[14]].view(tuple(self.bias_shapes[14]))

        # conv_down1
        x = self.relu(F.conv2d(x, conv_down1_0_w, bias=conv_down1_0_b, padding=1))
        conv1 = self.relu(F.conv2d(x, conv_down1_1_w, bias=conv_down1_1_b, padding=1))
        x = self.maxpool(conv1)

        # conv_down2
        x = self.relu(F.conv2d(x, conv_down2_0_w, bias=conv_down2_0_b, padding=1))
        conv2 = self.relu(F.conv2d(x, conv_down2_1_w, bias=conv_down2_1_b, padding=1))
        x = self.maxpool(conv2)
        
        # conv_down3
        x = self.relu(F.conv2d(x, conv_down3_0_w, bias=conv_down3_0_b, padding=1))
        conv3 = self.relu(F.conv2d(x, conv_down3_1_w, bias=conv_down3_1_b, padding=1))
        x = self.maxpool(conv3)   
        
        # conv_down4
        x = self.relu(F.conv2d(x, conv_down4_0_w, bias=conv_down4_0_b, padding=1))
        conv4 = self.relu(F.conv2d(x, conv_down4_1_w, bias=conv_down4_1_b, padding=1))
        x = self.upsample(conv4)        
        x = torch.cat([x, conv3], dim=1)
        
        # conv_up3
        x = self.relu(F.conv2d(x, conv_up3_0_w, bias=conv_up3_0_b, padding=1))
        x = self.relu(F.conv2d(x, conv_up3_1_w, bias=conv_up3_1_b, padding=1))
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        # conv_up2
        x = self.relu(F.conv2d(x, conv_up2_0_w, bias=conv_up2_0_b, padding=1))
        x = self.relu(F.conv2d(x, conv_up2_1_w, bias=conv_up2_1_b, padding=1))
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        # conv_up1
        x = self.relu(F.conv2d(x, conv_up1_0_w, bias=conv_up1_0_b, padding=1))
        x = self.relu(F.conv2d(x, conv_up1_1_w, bias=conv_up1_1_b, padding=1))
        
        # last
        x = F.conv2d(x, conv_last_w, bias=conv_last_b)

        x = x.permute(0, 2, 3, 1)
        if self.residual:
            x = zf + x
        return x
