import numpy as np
from skimage.metrics import structural_similarity
import math
import scipy.ndimage as nd
from unetsegmentation import test as segtest
from perceptualloss.loss_provider import LossProvider
from hyperrecon.util import utils

def psnr(gt, img):
  '''PSNR of two images.  
  
  Args:
    img1: (n1, n2)
    img2: (n1, n2)
  '''
  mse = np.mean((gt - img) ** 2)
  if mse == 0:  
    return 100
  max_pixel = 1.0
  return 20 * math.log10(max_pixel / math.sqrt(mse))

def rpsnr(gt, img, zfimg):
  return psnr(gt, img) - psnr(gt, zfimg)

def bpsnr(bimg1, bimg2, normalized=False):
  '''Compute average PSNR of a batch of images (possibly complex).

  Args:
    bimg1: (batch_size, c, n1, n2)
    bimg2: (batch_size, c, n1, n2)
  '''
  bimg1 = bimg1.norm(dim=1, keepdim=True)
  bimg2 = bimg2.norm(dim=1, keepdim=True)
  if normalized:
      bimg1 = utils.unit_rescale(bimg1)
      bimg2 = utils.unit_rescale(bimg2)

  metrics = []
  for img1, img2 in zip(bimg1, bimg2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    metrics.append(psnr(img1, img2))
  return float(np.array(metrics).mean())

def brpsnr(bgt, bimg, bzf, normalized=False):
  '''Compute average PSNR of a batch of images (possibly complex).

  Args:
    bimg1: (batch_size, c, n1, n2)
    bimg2: (batch_size, c, n1, n2)
  '''
  bgt = bgt.norm(dim=1, keepdim=True)
  bimg = bimg.norm(dim=1, keepdim=True)
  bzf = bzf.norm(dim=1, keepdim=True)
  if normalized:
      bgt = utils.unit_rescale(bgt)
      bimg = utils.unit_rescale(bimg)
      bzf = utils.unit_rescale(bzf)

  metrics = []
  for gt, img, zf in zip(bgt, bimg, bzf):
    gt = gt.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    zf = zf.cpu().detach().numpy()
    metrics.append(rpsnr(gt, img, zf))
  return float(np.array(metrics).mean())

def ssim(img1, img2):
  '''PSNR of two images.  
  
  Args:
    img1: (n1, n2)
    img2: (n1, n2)
  '''

  return structural_similarity(img1, img2)

def bssim(bimg1, bimg2):
  '''Compute average SSIM of a batch of images (possibly complex).

  Args:
    bimg1: (batch_size, c, n1, n2)
    bimg2: (batch_size, c, n1, n2)
  '''
  bimg1 = bimg1.norm(dim=1)
  bimg2 = bimg2.norm(dim=1)
  # if normalized:
  #     recons = rescale(recons)
  #     gt = rescale(gt)
  #     zf = rescale(zf)

  metrics = []
  for img1, img2 in zip(bimg1, bimg2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    metrics.append(ssim(img1, img2))
  return float(np.array(metrics).mean())

def hfen(img, gt, window_size=15, sigma=1.5):
  '''High-frequency error norm.'''
  t = (((window_size - 1)/2)-0.5)/sigma
  LoG_img = nd.gaussian_laplace(img, sigma=sigma, truncate=t)
  LoG_gt = nd.gaussian_laplace(gt, sigma=sigma, truncate=t)
  return np.linalg.norm(LoG_img - LoG_gt) / np.linalg.norm(LoG_gt)

def bhfen(bimg1, bimg2):
  '''Compute average HFEN of a batch of images (possibly complex).

  Args:
    bimg1: (batch_size, c, n1, n2)
    bimg2: (batch_size, c, n1, n2)
  '''
  bimg1 = bimg1.norm(dim=1)
  bimg2 = bimg2.norm(dim=1)
  # if normalized:
  #     recons = rescale(recons)
  #     gt = rescale(gt)
  #     zf = rescale(zf)

  metrics = []
  for img1, img2 in zip(bimg1, bimg2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    metrics.append(hfen(img1, img2))
  return float(np.array(metrics).mean())

def mae(img1, img2):
  '''Mean-absolute-error.'''
  return np.mean(np.abs(img1 - img2))

def bmae(bimg1, bimg2):
  '''Compute average mean-absolute-error of a batch of images (possibly complex).

  Args:
    bimg1: (batch_size, c, n1, n2)
    bimg2: (batch_size, c, n1, n2)
  '''
  bimg1 = bimg1.norm(dim=1)
  bimg2 = bimg2.norm(dim=1)
  # if normalized:
  #     recons = rescale(recons)
  #     gt = rescale(gt)
  #     zf = rescale(zf)

  metrics = []
  for img1, img2 in zip(bimg1, bimg2):
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    metrics.append(mae(img1, img2))
  return float(np.array(metrics).mean())

def bwatson(bimg1, bimg2):
  '''Compute average mean-absolute-error of a batch of images (possibly complex).

  Args:
    bimg1: (batch_size, c, n1, n2)
    bimg2: (batch_size, c, n1, n2)
  '''
  bimg1 = bimg1.norm(dim=1, keepdim=True).to('cuda:0')
  bimg2 = bimg2.norm(dim=1, keepdim=True).to('cuda:0')
  watson_dft = LossProvider().get_loss_function(
    'Watson-DFT', colorspace='grey', pretrained=True).to('cuda:0')
  return float(watson_dft(bimg1, bimg2))

def dice(recon, gt, seg):
  '''Compute Dice score against segmentation labels of clean images.

  TODO: segtest.tester currently only supports performing testing on
    full volumes, not slices.
  '''
  pretrained_segmentation_path = '/share/sablab/nfs02/users/aw847/models/UnetSegmentation/abide-dataloader-evan-dice/May_26/0.001_64_32_2/'

  roi_loss, roi_loss_gt, preds, preds_gt, targets = segtest.tester(pretrained_segmentation_path,
                xdata=recon.cpu().detach().numpy(),
                gt_data=gt.cpu().detach().numpy(),
                seg_data=seg.cpu().detach().numpy())
  
  return roi_loss, roi_loss_gt, preds, preds_gt, targets