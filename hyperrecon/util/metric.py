import numpy as np
import math
import scipy.ndimage as nd

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