import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from . import utils

class AmortizedLoss(nn.Module):
    def __init__(self, reg_types, range_restrict, mask, sampling_method, evaluate=False):
        super(AmortizedLoss, self).__init__()
        self.reg_types = reg_types
        self.range_restrict = range_restrict
        self.mask = mask
        self.sampling_method = sampling_method
        self.evaluate = evaluate

    def get_tv(self, x):
        x = x.permute(0, 3, 1, 2)
        tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs() + (x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs(), dim=(1, 2))
        tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs() + (x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs(), dim=(1, 2))
        return tv_x + tv_y
        
    def forward(self, x_hat, y, hyperparams, cap_reg, topK, schedule):
        '''
        Loss = (1-alpha) * DC + alpha * Reg1
        Loss = alpha * DC + (1-alpha)*beta * Reg1 + (1-alpha)*(1-beta) * Reg2
        hyperparams: matrix of hyperparams (batch_size, num_hyperparams)
        '''
        assert len(self.reg_types) == hyperparams.shape[1], 'num_hyperparams and reg mismatch'
        
        l1 = torch.nn.L1Loss(reduction='none')
        l2 = torch.nn.MSELoss(reduction='none')
     
        mask_expand = self.mask.unsqueeze(2)
     
        # Data consistency term
        Fx_hat = utils.fft(x_hat)
        UFx_hat = Fx_hat * mask_expand
        dc = torch.sum(l2(UFx_hat, y), dim=(1, 2, 3))

        # Regularization
        regs = {}

        regs['dc'] = dc
        regs['cap'] = cap_reg

        tv = self.get_tv(x_hat)
        regs['tv'] = tv

        # wavelets = get_wavelets(x_hat, device)
        # l1_wavelet = torch.sum(l1(wavelets, torch.zeros_like(wavelets)), dim=(1, 2, 3)) 
        # regs['w'] = l1_wavelet
     
        if len(self.reg_types) == 1:
            alpha = hyperparams[:, 0]
            if self.range_restrict:
                # print('range-restricted loss on %s' % self.reg_types[0])
                loss = (1-alpha)*dc + alpha*regs[self.reg_types[0]]
            else:
                # print('non-range-restricted loss on %s' % self.reg_types[0])
                loss = dc + alpha*regs[self.reg_types[0]]

        elif len(self.reg_types) == 2:
            alpha = hyperparams[:, 0]
            beta = hyperparams[:, 1]
            if self.range_restrict:
                # print('range-restricted loss on %s and %s' % (self.reg_types[0], self.reg_types[1]))
                if schedule:
                    # print('both regs for rest')
                    loss = alpha * dc + (1-alpha)*beta * regs[self.reg_types[0]] + (1-alpha)*(1-beta) * regs[self.reg_types[1]]
                else:
                    # print('just tv in beginning')
                    loss = alpha * dc + (1-alpha) * regs[self.reg_types[1]]
            else:
                # print('non-range-restricted loss on %s and %s' % (self.reg_types[0], self.reg_types[1]))
                loss = dc + alpha * regs[self.reg_types[0]] + beta * regs[self.reg_types[1]]
        else:
            raise NameError('Bad loss')

        if self.evaluate:
            return loss, regs, hyperparams
        else:
            loss, sort_hyperparams = self.process_losses(loss, dc, topK, hyperparams)
            return loss, regs, sort_hyperparams

    def process_losses(self, unsup_losses, dc_losses, topK, hyperparams):
        if topK is not None:
            assert self.sampling_method == 'bestdc'
            _, perm = torch.sort(dc_losses) # Sort by DC loss, low to high
            sort_losses = unsup_losses[perm] # Reorder total losses by lowest to highest DC loss
            sort_hyperparams = hyperparams[perm]

            loss = torch.sum(sort_losses[:topK]) # Take only the losses with lowest DC
        else:
            loss = torch.sum(unsup_losses)
            sort_hyperparams = hyperparams

        return loss, sort_hyperparams

# def get_wavelets(x, device):
#     xfm = DWTForward(J=3, mode='zero', wave='db4').to(device) # Accepts all wave types available to PyWavelets
#     Yl, Yh = xfm(x)

#     batch_size = x.shape[0]
#     channels = x.shape[1]
#     rows = nextPowerOf2(Yh[0].shape[-2]*2)
#     cols = nextPowerOf2(Yh[0].shape[-1]*2)
#     wavelets = torch.zeros(batch_size, channels, rows, cols).to(device)
#     # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
#     for i, band in enumerate(Yh):
#         irow = rows // 2**(i+1)
#         icol = cols // 2**(i+1)
#         wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
#         wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
#         wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

#     wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients
#     return wavelets

# def nextPowerOf2(n):
#     count = 0;

#     # First n in the below  
#     # condition is for the  
#     # case where n is 0 
#     if (n and not(n & (n - 1))):
#         return n

#     while( n != 0):
#         n >>= 1
#         count += 1

#     return 1 << count;

