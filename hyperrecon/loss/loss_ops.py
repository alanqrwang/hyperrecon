import sys
sys.path.append('/home/aw847/PerceptualSimilarity/src/')
sys.path.append('/home/aw847/torch-radon/')
import torch
from pytorch_wavelets import DWTForward
import pytorch_ssim
from hyperrecon.util import utils

class DataConsistency(object):
  def __init__(self, forward_model, mask_module, reduction='sum'):
    self.forward_model = forward_model
    self.mask_module = mask_module
    self.l2 = torch.nn.MSELoss(reduction='none')
    self.reduction = reduction
    assert self.reduction in ['sum', 'mean']

  def __call__(self, gt, pred, **kwargs):
    del kwargs
    batch_size = len(pred)
    mask = self.mask_module(batch_size).cuda()
    measurement = self.forward_model(pred, mask)
    measurement_gt = self.forward_model(gt, mask)
    if self.reduction == 'sum':
      dc = torch.sum(self.l2(measurement, measurement_gt), dim=(1, 2, 3)) 
    else:
      dc = torch.mean(self.l2(measurement, measurement_gt), dim=(1, 2, 3)) * 2
    return dc

class TotalVariation(object):
  def __init__(self, reduction='sum'):
    self.reduction = reduction
    assert self.reduction in ['sum', 'mean']

  def __call__(self, gt, pred, **kwargs):
    """Total variation loss.

    x : torch.Tensor (batch_size, n_ch, img_height, img_width)
      Input image
    """
    del gt, kwargs
    tv = 0
    for c in range(pred.shape[1]):
      if self.reduction == 'sum':
        tv_x = torch.sum((pred[:, c, :, :-1] - pred[:, c, :, 1:]).abs(), dim=(1, 2))
        tv_y = torch.sum((pred[:, c, :-1, :] - pred[:, c, 1:, :]).abs(), dim=(1, 2))
      else:
        tv_x = torch.mean((pred[:, c, :, :-1] - pred[:, c, :, 1:]).abs(), dim=(1, 2))
        tv_y = torch.mean((pred[:, c, :-1, :] - pred[:, c, 1:, :]).abs(), dim=(1, 2))
      tv += tv_x + tv_y
    return tv


class L1Wavelets(object):
  def __init__(self, device):
    self.xfm = DWTForward(J=3, mode='zero', wave='db4').to(device)
    self.l1 = torch.nn.L1Loss(reduction='none')

  def __call__(self, gt, pred, **kwargs):

    def nextPowerOf2(n):
      """Get next power of 2"""
      count = 0
      if (n and not(n & (n - 1))):
        return n
      while( n != 0):
        n >>= 1
        count += 1
      return 1 << count

    """L1-penalty on wavelets.

    x : torch.Tensor (batch_size, 2, img_height, img_width)
      Input image

    """
    del gt, kwargs
    Yl, Yh = self.xfm(pred)

    batch_size = pred.shape[0]
    channels = pred.shape[1]
    rows = nextPowerOf2(Yh[0].shape[-2]*2)
    cols = nextPowerOf2(Yh[0].shape[-1]*2)
    wavelets = torch.zeros(batch_size, channels,
                 rows, cols).to(self.device)
    # Yl is LL coefficients, Yh is list of higher bands with finest frequency in the beginning.
    for i, band in enumerate(Yh):
      irow = rows // 2**(i+1)
      icol = cols // 2**(i+1)
      wavelets[:, :, 0:(band[:, :, 0, :, :].shape[-2]), icol:(icol +
                                  band[:, :, 0, :, :].shape[-1])] = band[:, :, 0, :, :]
      wavelets[:, :, irow:(irow+band[:, :, 0, :, :].shape[-2]),
           0:(band[:, :, 0, :, :].shape[-1])] = band[:, :, 1, :, :]
      wavelets[:, :, irow:(irow+band[:, :, 0, :, :].shape[-2]),
           icol:(icol+band[:, :, 0, :, :].shape[-1])] = band[:, :, 2, :, :]

    # Put in LL coefficients
    wavelets[:, :, :Yl.shape[-2], :Yl.shape[-1]] = Yl

    l1_wave = torch.mean(
      self.l1(wavelets, torch.zeros_like(wavelets)), dim=(1, 2, 3))
    return l1_wave


class SSIM(object):
  def __init__(self):
    self.ssim_loss = pytorch_ssim.SSIM(size_average=False)

  def __call__(self, gt, pred, **kwargs):
    del kwargs
    return 1-self.ssim_loss(gt, pred)

class L1(object):
  def __init__(self):
    self.l1 = torch.nn.L1Loss(reduction='none')
  def __call__(self, gt, pred, **kwargs):
    del kwargs
    l1 = torch.mean(self.l1(gt, pred), dim=(1, 2, 3))
    return l1


class MSE(object):
  def __init__(self):
    self.mse_loss = torch.nn.MSELoss(reduction='none')
  def __call__(self, gt, pred, **kwargs):
    del kwargs
    return torch.mean(self.mse_loss(gt, pred), dim=(1, 2, 3))

class PSNR(object):
  def __init__(self, max_pixel=1.0):
    self.max_pixel = max_pixel
    self.mse = MSE()
  def __call__(self, gt, pred, **kwargs):
    del kwargs
    m = self.mse(gt, pred)
    return 20 * torch.log10(self.max_pixel / torch.sqrt(m))

class rPSNR(object):
  def __init__(self):
    self.psnr = PSNR()
  def __call__(self, gt, pred, **kwargs):
    gt = gt.norm(dim=1, keepdim=True)
    pred = pred.norm(dim=1, keepdim=True)
    zf = kwargs['zf'].norm(dim=1, keepdim=True)
    gt = utils.linear_normalization(gt, (0, 1))
    pred = utils.linear_normalization(pred, (0, 1))
    zf = utils.linear_normalization(zf, (0, 1))
    return self.psnr(gt, pred) - self.psnr(gt, zf)

class L1PenaltyWeights(object):
  def __call__(self, gt, pred, **kwargs):
    del gt
    network = kwargs['network']
    weights = network.get_conv_weights()

    cap_reg = torch.zeros(len(pred), requires_grad=True).cuda()
    for w in weights:
      w_flat = w.view(len(w), -1)
      if len(w_flat) != len(pred):
        w_flat = w_flat.repeat(len(pred), 1)
      cap_reg += torch.sum(torch.abs(w_flat), dim=1)
    return cap_reg