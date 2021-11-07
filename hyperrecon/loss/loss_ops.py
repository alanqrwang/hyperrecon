import sys
sys.path.append('/home/aw847/PerceptualSimilarity/src/')
sys.path.append('/home/aw847/torch-radon/')
import torch
from pytorch_wavelets import DWTForward
from torch_radon.shearlet import ShearletTransform
from perceptualloss.loss_provider import LossProvider
import pytorch_ssim
from unetsegmentation.predict import Segmenter
from hyperrecon.model.layers import GaussianSmoothing
from torch.nn import functional as F
import lpips
from hyperrecon.model.unet import Unet
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

class MinNormDataConsistency(object):
  def __init__(self, forward_model, mask_module):
    self.forward_model = forward_model
    self.mask_module = mask_module
    self.l2 = torch.nn.MSELoss(reduction='none')

  def __call__(self, gt, pred, lmbda=10, **kwargs):
    del kwargs
    batch_size = len(pred)
    mask = self.mask_module(batch_size).cuda()
    measurement = self.forward_model(pred, mask)
    measurement_gt = self.forward_model(gt, mask)
    dc = torch.sum(self.l2(measurement, measurement_gt), dim=(1, 2, 3))
    norm = pred.view(len(pred), -1).norm(p=2, dim=1)
    return dc + lmbda*norm

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


class L1Shearlets(object):
  def __init__(self, dims):
    scales = [0.5] * 2
    self.shearlet = ShearletTransform(*dims, scales)

  def __call__(self, gt, pred, **kwargs):
    del gt, kwargs
    pred = pred.norm(dim=-1) # Absolute value of complex image
    shears = self.shearlet.forward(pred)
    l1_shear = torch.sum(
      self.l1(shears, torch.zeros_like(shears)), dim=(1, 2, 3))
    return l1_shear


class SSIM(object):
  def __init__(self):
    self.ssim_loss = pytorch_ssim.SSIM(size_average=False)

  def __call__(self, gt, pred, **kwargs):
    del kwargs
    return 1-self.ssim_loss(gt, pred)

class WatsonDFT(object):
  def __init__(self):
    provider = LossProvider()
    self.watson_dft = provider.get_loss_function(
      'Watson-DFT', colorspace='grey', pretrained=True, reduction='none').cuda()

  def __call__(self, gt, pred, **kwargs):
    del kwargs
    gt = utils.linear_normalization(gt, (0, 1))
    pred = utils.linear_normalization(pred, (0, 1))
    loss = self.watson_dft(gt, pred) 
    return loss

class LPIPS(object):
  def __init__(self):
    self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() 

  def __call__(self, gt, pred, **kwargs):
    del kwargs
    gt = utils.linear_normalization(gt, (-1, 1))
    pred = utils.linear_normalization(pred, (-1, 1))
    gt = utils.gray2rgb(gt)
    pred = utils.gray2rgb(pred)
    return self.loss_fn_vgg(gt, pred)[..., 0, 0, 0]

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

class L2Loss(object):
  def __call__(self, gt, pred, **kwargs):
    del kwargs
    gt_vec = gt.view(len(gt), -1)
    pred_vec = pred.view(len(pred), -1)
    return (gt_vec - pred_vec).norm(p=2, dim=1)

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

class PretrainedDICE():
  '''Compute Dice score against segmentation labels of clean images.

  TODO: segtest.tester currently only supports performing testing on
    full volumes, not slices.
  '''
  def __init__(self):
    # pretrained_seg_path = '/share/sablab/nfs02/users/aw847/models/UnetSegmentation/abide-dataloader-evan-dice/May_26/0.001_64_32_2/'
    ckpt_path = '/share/sablab/nfs02/users/aw847/models/HyperRecon/segment-snr5/Nov_05/datasetabide_archunet_methodsegment_distconstant_forwardcsmri_maskpoisson_rate8p3_lr0.001_bs32_l1+ssim_hnet128_unet32_topKNone_restrictTrue_hp0.0_gauss0.0_resFalse_dcscale0.5/checkpoints/model.0200.h5'

    n_classes = 5 
    network = Unet(
                in_ch=1,
                out_ch=n_classes,
                h_ch=32,
                residual=False,
                use_batchnorm=True
              ).cuda()
    self.trained_model = utils.load_checkpoint(network, ckpt_path)
    self.trained_model.eval()
    for param in self.trained_model.parameters():
      param.requires_grad = False
    self.criterion = DiceLoss(hard=True)

  def predict(self, recon, seg_data):
    target_onehot = utils.get_onehot(seg_data).cuda()
    preds = self.trained_model(recon)
    with torch.set_grad_enabled(True):
      loss, _, _ = self.criterion(preds, target_onehot, ign_first_ch=True)
    return loss

  def __call__(self, gt, pred, **kwargs):
    del gt
    seg = kwargs['seg']
    loss = self.predict(
                  recon=pred,
                  seg_data=seg)
    return loss

class UnetEncFeat(object):
  def __call__(self, gt, pred, **kwargs):
    del gt, pred
    feat_mean = kwargs['network'].get_feature_mean()
    N = len(feat_mean)
    batch1 = feat_mean[:N//2]
    batch2 = feat_mean[N//2:]
    return (batch1 - batch2).norm(p=2)

class LPF_L2():
  def __init__(self):
    kernel_size = 5
    sigma = 10
    self.smoothing = GaussianSmoothing(1, kernel_size, sigma)
  
  def __call__(self, gt, pred, **kwargs):
    del kwargs
    gt = F.pad(gt, (2, 2, 2, 2), mode='reflect')
    pred = F.pad(pred, (2, 2, 2, 2), mode='reflect')
    gt_smooth = self.smoothing(gt)
    pred_smooth = self.smoothing(pred)
    return (gt_smooth - pred_smooth).norm(p=2)

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

class DiceLoss(torch.nn.Module):
  '''Dice loss.

  Supports 2d or 3d inputs.
  Supports hard dice or soft dice.
  If soft dice, returns scalar dice loss for entire slice/volume.
  If hard dice, returns:
      total_avg: scalar dice loss for entire slice/volume ()
      regions_avg: dice loss per region (n_ch)
      ind_avg: dice loss per region per pixel (bs, n_ch)

  '''
  def __init__(self, hard=False):
    super(DiceLoss, self).__init__()
    self.hard = hard

  def forward(self, pred, target, ign_first_ch=False):
    eps = 1
    assert pred.size() == target.size(), 'Input and target are different dim'
    
    if len(target.size())==4:
      n,c,_,_ = target.size()
    if len(target.size())==5:
      n,c,_,_,_ = target.size()

    target = target.contiguous().view(n,c,-1)
    pred = pred.contiguous().view(n,c,-1)
    
    if self.hard: # hard Dice
      pred_onehot = torch.zeros_like(pred)
      pred = torch.argmax(pred, dim=1, keepdim=True)
      pred = torch.scatter(pred_onehot, 1, pred, 1.)
    if ign_first_ch:
      target = target[:,1:,:]
      pred = pred[:,1:,:]
  
    num = torch.sum(2*(target*pred),2) + eps
    den = (pred*pred).sum(2) + (target*target).sum(2) + eps
    dice_loss = 1-num/den
    ind_avg = dice_loss
    total_avg = torch.mean(dice_loss, 1)
    regions_avg = torch.mean(dice_loss, 0)
    
    if not self.hard:
      return total_avg
    else:
      return total_avg, regions_avg, ind_avg