import numpy as np
from skimage.metrics import peak_signal_noise_ratio

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