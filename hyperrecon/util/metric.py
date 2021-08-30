import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import scipy.ndimage as nd

def psnr(img1, img2):
  '''PSNR of two images.  
  
  Args:
    img1: (n1, n2)
    img2: (n1, n2)
  '''

  return peak_signal_noise_ratio(img1, img2)

def bpsnr(bimg1, bimg2):
  '''Compute average PSNR of a batch of images (possibly complex).

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
    metrics.append(psnr(img1, img2))
  return np.array(metrics).mean()

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
  return np.array(metrics).mean()

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
  return np.array(metrics).mean()