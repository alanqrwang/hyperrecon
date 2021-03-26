"""
Loss for RegAgnosticCSMRI
For more details, please read:
    Alan Q. Wang, Adrian V. Dalca, and Mert R. Sabuncu. 
    "Regularization-Agnostic Compressed Sensing MRI with Hypernetworks" 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import sys
sys.path.append('/home/aw847/PerceptualSimilarity/src/')
sys.path.append('/home/aw847/torch-radon/')
from torch_radon.shearlet import ShearletTransform
from . import utils, dataset
import matplotlib.pyplot as plt
# from perceptualloss.loss_provider import LossProvider
from . import utils

class AmortizedLoss(nn.Module):
    """Loss function for model"""
    def __init__(self, reg_types, range_restrict, sampling_method, topK, device, mask, take_avg=True):
        """
        Parameters
        ----------
        reg_types : List of strings which describes which regularization
                functions to use and in what order
        range_restrict : Bool, whether or not to range restrict hyperparameter values
            if True, then for 1 and 2 hyperparameters, respectively:
                Loss = (1-alpha)*DC + alpha*Reg1
                Loss = alpha*DC + (1-alpha)*beta*Reg1 + (1-alpha)*(1-beta)*Reg2
            else:
                Loss = DC + alpha*Reg1
                Loss = DC + alpha*Reg1 + beta*Reg2
        mask : Undersampling mask, must be Torch.tensor
        sampling_method : Either uhs or dhs
        """
        super(AmortizedLoss, self).__init__()
        self.reg_types = reg_types
        self.range_restrict = range_restrict
        self.sampling_method = sampling_method
        self.topK = topK
        self.device = device
        self.mask = mask
        self.take_avg = take_avg

        self.l1 = torch.nn.L1Loss(reduction='none')

        if 'w' in reg_types:
            self.xfm = DWTForward(J=3, mode='zero', wave='db4').to(self.device)
        if 'sh' in reg_types:
            scales = [0.5] * 2
            self.shearlet = ShearletTransform(*mask.shape, scales)#, cache='/home/aw847/shear_cache/')

    def get_tv(self, x):
        """Total variation loss

        Parameters
        ----------
        x : torch.Tensor (batch_size, img_height, img_width, 2)
            Input image

        Returns
        ----------
        tv_loss : TV loss
        """
        x = x.permute(0, 3, 1, 2)
        tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs() + (x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs(), dim=(1, 2))
        tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs() + (x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs(), dim=(1, 2))
        return tv_x + tv_y
        
    def get_wavelets(self, x):
        """L1-penalty on wavelets

        Parameters
        ----------
        x : torch.Tensor (batch_size, img_height, img_width, 2)
            Input image

        Returns
        ----------
        l1_wave : wavelets loss
        """
        Yl, Yh = self.xfm(x)

        batch_size = x.shape[0]
        channels = x.shape[1]
        rows = utils.nextPowerOf2(Yh[0].shape[-2]*2)
        cols = utils.nextPowerOf2(Yh[0].shape[-1]*2)
        wavelets = torch.zeros(batch_size, channels, rows, cols).to(self.device)
        # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
        for i, band in enumerate(Yh):
            irow = rows // 2**(i+1)
            icol = cols // 2**(i+1)
            wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
            wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
            wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

        wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients

        l1_wave = torch.sum(self.l1(wavelets, torch.zeros_like(wavelets)), dim=(1, 2, 3))
        return l1_wave

    def get_shearlets(self, x):
        x = utils.absval(x)
        shears = self.shearlet.forward(x)
        l1_shear = torch.sum(self.l1(shears, torch.zeros_like(shears)), dim=(1, 2, 3))
        return l1_shear

    def forward(self, x_hat, y, hyperparams, cap_reg, schedule):
        '''
        Parameters
        ----------
        x_hat : torch.Tensor (batch_size, img_height, img_width, 2)
            Reconstructed image
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Matrix of hyperparameter values
        cap_reg : float
            Layer-wise L1 capacity penalty
        topK : int
            K for DHS sampling
        schedule : bool
            Loss scheduler

        Returns
        ----------
        loss : torch.Tensor (batch_size,)
            Total amortized loss per batch image
        loss_dict : dict
            Losses separated by different terms
        hyperparams : torch.Tensor
            Hyperparmeters used
        '''
        
        loss_dict = {}
        loss_dict['dc'] = get_dc_loss(x_hat, y, self.mask)

        # Regularization
        loss_dict['cap'] = cap_reg
        if 'tv' in self.reg_types:
            loss_dict['tv'] = self.get_tv(x_hat)
        if 'w' in self.reg_types:
            loss_dict['w'] = self.get_wavelets(x_hat)
        if 'sh' in self.reg_types:
            loss_dict['sh'] = self.get_shearlets(x_hat)

        if self.range_restrict and len(self.reg_types) == 1:
            assert len(self.reg_types) == hyperparams.shape[1], 'num_hyperparams and reg mismatch'
            alpha = hyperparams[:, 0]
            loss = (1-alpha)*loss_dict['dc'] + alpha*loss_dict[self.reg_types[0]]

        elif self.range_restrict and len(self.reg_types) == 2:
            assert len(self.reg_types) == hyperparams.shape[1], 'num_hyperparams and reg mismatch'
            alpha = hyperparams[:, 0]
            beta = hyperparams[:, 1]
            loss = alpha*loss_dict['dc'] + (1-alpha)*beta * loss_dict[self.reg_types[0]] + (1-alpha)*(1-beta) * loss_dict[self.reg_types[1]]

        else:
            assert len(self.reg_types) == hyperparams.shape[1] - 1, 'num_hyperparams and reg mismatch'
            loss = hyperparams[:,0]*loss_dict['dc']
            for i in range(schedule):
                loss = loss + hyperparams[:,i+1] * loss_dict[self.reg_types[i]]
            loss = loss / torch.sum(hyperparams, dim=1)

        loss, sort_hyperparams = self._process_losses(loss, loss_dict['dc'], hyperparams)
        return loss, loss_dict, sort_hyperparams

    def _process_losses(self, unsup_losses, dc_losses, hyperparams):
        '''Performs DHS sampling if topK is not None, otherwise UHS sampling

        Parameters
        ----------
        unsup_losses : torch.Tensor (batch_size,)
            Per-batch losses
        dc_losses : torch.Tensor (batch_size,)
            Per-batch DC losses
        topK : int
            K for DHS sampling
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Matrix of hyperparameter values

        Returns
        ----------
        loss : float
            Total amortized loss
        sort_hyperparams : torch.Tensor
            Hyperparmeters sorted by best DC
        '''
        if self.topK is not None and self.take_avg:
            assert self.sampling_method == 'dhs'
            _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
            sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss
            sort_hyperparams = hyperparams[perm]

            loss = torch.sum(sort_losses[:self.topK]) # Take only the losses with lowest DC
        elif self.topK is None and self.take_avg:
            loss = torch.sum(unsup_losses)
            sort_hyperparams = hyperparams
        else:
            loss = unsup_losses
            sort_hyperparams = hyperparams

        return loss, sort_hyperparams

def get_dc_loss(x_hat, y, mask):
    l2 = torch.nn.MSELoss(reduction='none')
    mask_expand = mask.unsqueeze(2)
    loss_dict = {}

    # Data consistency term
    Fx_hat = utils.fft(x_hat)
    UFx_hat = Fx_hat * mask_expand
    dc = torch.sum(l2(UFx_hat, y), dim=(1, 2, 3))
    return dc

def trajloss(recons, dc_losses, lmbda, device, loss_type, mse=None):
    '''Trajectory Net Loss

    recons: (batch_size, num_points, n1, n2, 2)
    recons: (batch_size, num_points, n1, n2, 2)
    Loss = (1-lmbda)*dist_loss + lmbda*dc_loss
    '''
    # provider = LossProvider()
    # loss_function = provider.get_loss_function('Watson-DFT', colorspace='grey', pretrained=True, reduction='sum').to(device)

    batch_size, num_points = recons.shape[0], recons.shape[1]
    dist_loss = 0
    reg_loss = 0
    for b in range(batch_size):
        if loss_type == 'perceptual':
            for i in range(num_points-1):
                for j in range(i+1, num_points):
                    img1 = utils.absval(recons[b, i:i+1, ...]).unsqueeze(1)
                    img2 = utils.absval(recons[b, j:j+1, ...]).unsqueeze(1)
                    dist_loss = dist_loss + loss_function(img1, img2)

        elif loss_type == 'l2':
            dist_loss = dist_loss - torch.sum(F.pdist(recons[b].reshape(num_points, -1)))
        else:
            raise Exception()

        reg_loss = reg_loss + torch.sum(dc_losses[b])

    if mse is None:
        print(lmbda)
        loss = (1-lmbda)*dist_loss + lmbda*reg_loss
    else:
        print(lmbda)
        loss = (1-lmbda)*dist_loss + lmbda*mse
    
    return loss
