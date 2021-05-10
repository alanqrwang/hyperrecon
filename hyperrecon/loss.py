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
from perceptualloss.loss_provider import LossProvider
from . import utils
import pytorch_ssim

class AmortizedLoss(nn.Module):
    """Loss function for model"""
    def __init__(self, losses, range_restrict, sampling_method, topK, device, mask, take_avg=True):
        """
        Parameters
        ----------
        losses : List of strings which describes which loss
                functions to use and in what order
        range_restrict : bool, whether or not to range restrict hyperparameter values
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
        assert len(losses) > 1, 'Need more hyperparameters'
        self.losses = losses
        self.range_restrict = range_restrict
        self.sampling_method = sampling_method
        self.topK = topK
        self.device = device
        self.mask = mask
        self.take_avg = take_avg

        self.l1 = torch.nn.L1Loss(reduction='none')

        if 'w' in losses:
            self.xfm = DWTForward(J=3, mode='zero', wave='db4').to(self.device)
        if 'sh' in losses:
            scales = [0.5] * 2
            self.shearlet = ShearletTransform(*mask.shape, scales)#, cache='/home/aw847/shear_cache/')
        if 'perc' in losses:
            provider = LossProvider()
            self.watson_dft = provider.get_loss_function('Watson-DFT', colorspace='grey', pretrained=True, reduction='none').to(device)
        if 'ssim' in losses:
            self.ssim_loss = pytorch_ssim.SSIM(size_average=False)

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
        # x = x.permute(0, 3, 1, 2)
        tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs(), dim=(1, 2))
        tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs(), dim=(1, 2))
        if x.shape[1] == 2:
            tv_x += torch.sum((x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs(), dim=(1, 2))
            tv_y += torch.sum((x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs(), dim=(1, 2))
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

    def get_ssim(self, input, target):
        # input = input.permute(0, 3, 1, 2)
        # target = target.permute(0, 3, 1, 2)
        assert input.shape[1] == 1 and target.shape[1] == 1, 'Channel dimension incorrect'
        ssim_out = 1-self.ssim_loss(input, target)
        ssim_out = ssim_out / 0.31786856
        return ssim_out

    def get_watson_dft(self, input, target):
        # input = input.permute(0, 3, 1, 2)
        # target = target.permute(0, 3, 1, 2)
        loss = self.watson_dft(input, target)
        return loss

    def get_dc_loss(self, x_hat, y, mask):
        l2 = torch.nn.MSELoss(reduction='none')

        UFx_hat = utils.undersample(x_hat, mask)
        dc = torch.sum(l2(UFx_hat, y), dim=(1, 2, 3))
        return dc

    def get_l1(self, input, target):
        l1 = torch.mean(self.l1(input, target), dim=(1, 2, 3))
        l1 = l1 / 0.045254722
        return l1

    def get_mse(self, input, target):
        mse_loss = torch.nn.MSELoss(reduction='none')
        return torch.mean(mse_loss(input, target), dim=(1, 2, 3))

    def forward(self, input, y, hyperparams, cap_reg, target=None):
        '''
        Parameters
        ----------
        input : torch.Tensor (batch_size, img_height, img_width, 2)
            Reconstructed image
        y : torch.Tensor (batch_size, img_height, img_width, 2)
            Under-sampled measurement
        hyperparams : torch.Tensor (batch_size, num_hyperparams)
            Matrix of hyperparameter values
        cap_reg : float
            Layer-wise L1 capacity penalty

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

        # Regularization
        loss_dict['cap'] = cap_reg
        if 'dc' in self.losses:
            loss_dict['dc'] = self.get_dc_loss(input, y, self.mask)
        if 'tv' in self.losses:
            loss_dict['tv'] = self.get_tv(input)
        if 'w' in self.losses:
            loss_dict['w'] = self.get_wavelets(input)
        if 'sh' in self.losses:
            loss_dict['sh'] = self.get_shearlets(input)
        if 'dice' in self.losses:
            dice_loss = get_dice(input, target)
            loss_dict['dice'] = dice_loss
        if 'l1' in self.losses:
            assert target is not None, 'supervised loss requires ground truth'
            loss_dict['l1'] = self.get_l1(input, target)
        if 'mse' in self.losses:
            assert target is not None, 'supervised loss requires ground truth'
            loss_dict['mse'] = self.get_mse(input, target)
        if 'ssim' in self.losses:
            assert target is not None, 'supervised loss requires ground truth'
            loss_dict['ssim'] = self.get_ssim(input, target)
        if 'perc' in self.losses:
            assert target is not None, 'supervised loss requires ground truth'
            loss_dict['perc'] = self.get_watson_dft(input, target)

        if self.range_restrict and len(self.losses) == 2:
            assert len(self.losses) == hyperparams.shape[1] + 1, 'num_hyperparams and loss mismatch'
            alpha = hyperparams[:, 0]
            print(loss_dict[self.losses[0]].shape, loss_dict[self.losses[1]].shape) 
            loss = (1-alpha)*loss_dict[self.losses[0]] + alpha*loss_dict[self.losses[1]]

        elif self.range_restrict and len(self.losses) == 3:
            assert len(self.losses) == hyperparams.shape[1] + 1, 'num_hyperparams and loss mismatch'
            alpha = hyperparams[:, 0]
            beta = hyperparams[:, 1]
            loss = alpha * loss_dict[self.losses[0]] \
                    + (1-alpha)*beta * loss_dict[self.losses[1]] \
                    + (1-alpha)*(1-beta) * loss_dict[self.losses[2]]

        else:
            assert len(self.losses) == hyperparams.shape[1], 'num_hyperparams and loss mismatch'
            loss = torch.tensor(0., requires_grad=True)
            for i in range(len(self.losses)):
                loss = loss + hyperparams[:,i] * loss_dict[self.losses[i]]
            loss = loss / torch.sum(hyperparams, dim=1)

        loss, hyperparams = self._process_losses(loss, loss_dict, hyperparams)
        return loss, loss_dict, hyperparams

    def _process_losses(self, losses, loss_dict, hyperparams):
        '''Performs DHS sampling if topK is not None, otherwise UHS sampling

        Parameters
        ----------
        unsup_losses : torch.Tensor (batch_size,)
            Per-batch losses
        loss_dict : torch.Tensor (batch_size,)
            Dictionary of losses
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
            dc_losses = loss_dict['dc']
            _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
            sort_losses = losses[perm] # Reorder total losses by lowest to highest DC loss
            sort_hyperparams = hyperparams[perm]
            loss = torch.mean(sort_losses[:self.topK]) # Take only the losses with lowest DC
        elif self.topK is None and self.take_avg:
            loss = torch.mean(losses)
            sort_hyperparams = hyperparams
        else:
            loss = losses
            sort_hyperparams = hyperparams

        return loss, sort_hyperparams

def get_dice(pred, target):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'

    if len(target.size())==4:
        n,c,x,y = target.size()
    if len(target.size())==5:
        n,c,x,y,z = target.size()

    target = target.view(n,c,-1)
    pred = pred.view(n,c,-1)

    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss.mean(1) # Average across channels

    return ind_avg

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
