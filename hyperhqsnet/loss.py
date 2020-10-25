import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from . import utils

def data_consistency(k, k0, mask, lmbda=0):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    mask = mask.unsqueeze(-1)
    mask = mask.expand_as(k)

    return (1 - mask) * k + mask * (lmbda*k + k0) / (1 + lmbda)

def get_tv_iso(x):
    tv_x = torch.sum(((x[:, 0, :, :-1] - x[:, 0, :, 1:])**2 + (x[:, 1, :, :-1] - x[:, 1, :, 1:])**2)**0.5)
    tv_y = torch.sum(((x[:, 0, :-1, :] - x[:, 0, 1:, :])**2 + (x[:, 1, :-1, :] - x[:, 1, 1:, :])**2)**0.5)
    return tv_x + tv_y

def get_tv(x):
    tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs() + (x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs(), dim=(1, 2))
    tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs() + (x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs(), dim=(1, 2))
    return tv_x + tv_y

def get_tv_all(x):
    tv_x = torch.sum((x[:, 0, :, :-1] - x[:, 0, :, 1:]).abs() + (x[:, 1, :, :-1] - x[:, 1, :, 1:]).abs())
    tv_y = torch.sum((x[:, 0, :-1, :] - x[:, 0, 1:, :]).abs() + (x[:, 1, :-1, :] - x[:, 1, 1:, :]).abs())
    return tv_x + tv_y

def get_wavelets(x, device):
    xfm = DWTForward(J=3, mode='zero', wave='db4').to(device) # Accepts all wave types available to PyWavelets
    Yl, Yh = xfm(x)

    batch_size = x.shape[0]
    channels = x.shape[1]
    rows = nextPowerOf2(Yh[0].shape[-2]*2)
    cols = nextPowerOf2(Yh[0].shape[-1]*2)
    wavelets = torch.zeros(batch_size, channels, rows, cols).to(device)
    # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
    for i, band in enumerate(Yh):
        irow = rows // 2**(i+1)
        icol = cols // 2**(i+1)
        wavelets[:, :, 0:(band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,0,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), 0:(band[:,:,0,:,:].shape[-1])] = band[:,:,1,:,:]
        wavelets[:, :, irow:(irow+band[:,:,0,:,:].shape[-2]), icol:(icol+band[:,:,0,:,:].shape[-1])] = band[:,:,2,:,:]

    wavelets[:,:,:Yl.shape[-2],:Yl.shape[-1]] = Yl # Put in LL coefficients
    return wavelets

def nextPowerOf2(n):
    count = 0;

    # First n in the below  
    # condition is for the  
    # case where n is 0 
    if (n and not(n & (n - 1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;

def unsup_loss(x_hat, y, mask, hyperparams, device, reg_types, cap_reg, range_restrict=True):
    '''
    Loss = (1-alpha) * DC + alpha * Reg1
    Loss = (alpha*beta) * DC + (1-alpha)*beta * Reg1 + (1-alpha)*(1-beta) * Reg2
    hyperparams: matrix of hyperparams (batch_size, num_hyperparams)
    '''
    assert len(reg_types) == hyperparams.shape[1], 'num_hyperparams and reg mismatch'
    
    l1 = torch.nn.L1Loss(reduction='none')
    l2 = torch.nn.MSELoss(reduction='none')
 
    mask_expand = mask.unsqueeze(2)
 
    # Data consistency term
    Fx_hat = utils.fft(x_hat)
    UFx_hat = Fx_hat * mask_expand
    dc = torch.sum(l2(UFx_hat, y), dim=(1, 2, 3))

    # Regularization
    regs = {}
    regs['cap'] = cap_reg

    x_hat = x_hat.permute(0, 3, 1, 2)
    tv = get_tv(x_hat)
    regs['tv'] = tv

    wavelets = get_wavelets(x_hat, device)
    l1_wavelet = torch.sum(l1(wavelets, torch.zeros_like(wavelets)), dim=(1, 2, 3)) 
    regs['w'] = l1_wavelet
 
    if len(reg_types) == 1:
        alpha = hyperparams[:, 0]
        if range_restrict:
            print('range-restricted loss on %s' % reg_types[0])
            loss = (1-alpha)*dc + alpha*regs[reg_types[0]]
        else:
            print('non-range-restricted loss on %s' % reg_types[0])
            loss = dc + alpha*regs[reg_types[0]]

    elif len(reg_types) == 2:
        alpha = hyperparams[:, 0]
        beta = hyperparams[:, 1]
        if range_restrict:
            print('range-restricted loss on %s and %s' % (reg_types[0], reg_types[1]))
            loss = (alpha*beta) * dc + (1-alpha)*beta * regs[reg_types[0]] + (1-alpha)*(1-beta) * regs[reg_types[1]]
        else:
            print('non-range-restricted loss on %s and %s' % (reg_types[0], reg_types[1]))
            loss = dc + alpha * regs[reg_types[0]] + beta * regs[reg_types[1]]
    else:
        raise NameError('Bad loss')

    return loss, dc

def unsup_loss_single_batch(x_hat, y, mask, hyperparams, device):
    '''
    Loss = (1-alpha) * DC + alpha * Reg
    Loss = (alpha*beta) * DC + (1-alpha)*beta * Reg1 + (1-alpha)*(1-beta) * Reg2
    '''
    l1 = torch.nn.L1Loss(reduction='sum')
    l2 = torch.nn.MSELoss(reduction='sum')
 
    mask_expand = mask.unsqueeze(2)
 
    # Data consistency term
    Fx_hat = utils.fft(x_hat)
    UFx_hat = Fx_hat * mask_expand
    dc = l2(UFx_hat, y)

    # Regularization
    x_hat = x_hat.permute(0, 3, 1, 2)
    tv = get_tv_all(x_hat)
    wavelets = get_wavelets(x_hat, device)
    l1_wavelet = l1(wavelets, torch.zeros_like(wavelets)) # we want L1 value by itself, not the error
 
    if len(hyperparams) == 2:
        w_coeff = hyperparams[0]
        tv_coeff = hyperparams[1]
        loss = (w_coeff*tv_coeff) * dc + (1-w_coeff)*tv_coeff * l1_wavelet + (1-w_coeff)*(1-tv_coeff) * tv
    else:
        tv_coeff = hyperparams[0]
        loss = (1-tv_coeff)*dc + tv_coeff*tv

    return loss, dc
