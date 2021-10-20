import sys
sys.path.append('/home/aw847/PerceptualSimilarity/src/')
sys.path.append('/home/aw847/torch-radon/')
import torch
from pytorch_wavelets import DWTForward
from torch_radon.shearlet import ShearletTransform
from perceptualloss.loss_provider import LossProvider
import pytorch_ssim
# from unetsegmentation.predict import Segmenter
from hyperrecon.model.layers import GaussianSmoothing
from torch.nn import functional as F

class Data_Consistency(object):
  def __init__(self, forward_model, mask_module):
    self.forward_model = forward_model
    self.mask_module = mask_module
    self.l2 = torch.nn.MSELoss(reduction='none')

  def __call__(self, pred, gt):
    batch_size = len(pred)
    mask = self.mask_module(batch_size).cuda()
    measurement = self.forward_model(pred, mask)
    measurement_gt = self.forward_model(gt, mask)
    dc = torch.sum(self.l2(measurement, measurement_gt), dim=(1, 2, 3))
    return dc

class Total_Variation(object):
  def __call__(self, pred, gt):
    """Total variation loss.

    x : torch.Tensor (batch_size, n_ch, img_height, img_width)
      Input image
    """
    del gt
    tv_x = torch.sum((pred[:, 0, :, :-1] - pred[:, 0, :, 1:]).abs(), dim=(1, 2))
    tv_y = torch.sum((pred[:, 0, :-1, :] - pred[:, 0, 1:, :]).abs(), dim=(1, 2))
    if pred.shape[1] == 2:
      tv_x += torch.sum((pred[:, 1, :, :-1] -
                 pred[:, 1, :, 1:]).abs(), dim=(1, 2))
      tv_y += torch.sum((pred[:, 1, :-1, :] -
                 pred[:, 1, 1:, :]).abs(), dim=(1, 2))
    return tv_x + tv_y


class L1_Wavelets(object):
  def __init__(self, device):
    self.xfm = DWTForward(J=3, mode='zero', wave='db4').to(device)
    self.l1 = torch.nn.L1Loss(reduction='none')

  def __call__(self, pred, gt):

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
    del gt
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


class L1_Shearlets(object):
  def __init__(self, dims):
    scales = [0.5] * 2
    self.shearlet = ShearletTransform(*dims, scales)

  def __call__(self, pred, gt):
    del gt
    pred = pred.norm(dim=-1) # Absolute value of complex image
    shears = self.shearlet.forward(pred)
    l1_shear = torch.sum(
      self.l1(shears, torch.zeros_like(shears)), dim=(1, 2, 3))
    return l1_shear


class SSIM(object):
  def __init__(self):
    self.ssim_loss = pytorch_ssim.SSIM(size_average=False)

  def __call__(self, pred, gt):
    assert pred.shape[1] == 1 and gt.shape[1] == 1, 'Channel dimension incorrect'
    ssim_out = 1-self.ssim_loss(pred, gt)
    return ssim_out


class Watson_DFT(object):
  def __init__(self, device):
    provider = LossProvider()
    self.watson_dft = provider.get_loss_function(
      'Watson-DFT', colorspace='grey', pretrained=True, reduction='none').to(device)

  def __call__(self, pred, gt):
    loss = self.watson_dft(pred, gt) 
    return loss


class L1(object):
  def __init__(self):
    self.l1 = torch.nn.L1Loss(reduction='none')
  def __call__(self, pred, gt):
    l1 = torch.mean(self.l1(pred, gt), dim=(1, 2, 3))
    return l1


class MSE(object):
  def __init__(self):
    self.mse_loss = torch.nn.MSELoss(reduction='none')
  def __call__(self, pred, gt):
    return torch.mean(self.mse_loss(pred, gt), dim=(1, 2, 3))

class L2Loss(object):
  def __call__(self, pred, gt):
    pred_vec = pred.view(len(pred), -1)
    gt_vec = gt.view(len(gt), -1)
    return (pred_vec - gt_vec).norm(p=2, dim=1)

# class DICE():
#   '''Compute Dice score against segmentation labels of clean images.

#   TODO: segtest.tester currently only supports performing testing on
#     full volumes, not slices.
#   '''
#   def __init__(self):
#     pretrained_seg_path = '/share/sablab/nfs02/users/aw847/models/UnetSegmentation/abide-dataloader-evan-dice/May_26/0.001_64_32_2/'
#     self.segmenter = Segmenter(pretrained_seg_path)

#   def __call__(self, pred, **kwargs):
#     seg = kwargs['seg']
#     loss = self.segmenter.predict(
#                   recon=pred,
#                   seg_data=seg)
#     return loss

class UnetEncFeat(object):
  def __call__(self, unet_network):
    feat_mean = unet_network.get_feature_mean()
    N = len(feat_mean)
    batch1 = feat_mean[:N//2]
    batch2 = feat_mean[N//2:]
    return (batch1 - batch2).norm(p=2)

class LPF_L2():
  def __init__(self):
    kernel_size = 5
    sigma = 10
    self.smoothing = GaussianSmoothing(1, kernel_size, sigma)
  
  def __call__(self, pred, gt):
    pred = F.pad(pred, (2, 2, 2, 2), mode='reflect')
    gt = F.pad(gt, (2, 2, 2, 2), mode='reflect')
    pred_smooth = self.smoothing(pred)
    gt_smooth = self.smoothing(gt)
    return (pred_smooth - gt_smooth).norm(p=2)