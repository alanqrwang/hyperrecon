import myutils
import numpy as np
from . import utils
import torch
import math

def psnr(img1, img2):
    if torch.is_tensor(img1):
        mse = torch.mean((img1 - img2)**2)
    else:
        mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def hfen(img, gt, window_size=15, sigma=1.5):
    t = (((window_size - 1)/2)-0.5)/sigma
    LoG_img = nd.gaussian_laplace(img, sigma=sigma, truncate=t)
    LoG_gt = nd.gaussian_laplace(gt, sigma=sigma, truncate=t)
    return np.linalg.norm(LoG_img - LoG_gt) / np.linalg.norm(LoG_gt)

def ssim(img1, img2):
    if len(img1.shape) > 2:
        img1 = img1.squeeze()
    if len(img2.shape) > 2:
        img2 = img2.squeeze()
    return structural_similarity(img1, img2, data_range=img2.max() - img2.min())

def relative_psnr(zf, recon, gt):
    zf_metric = psnr(zf, gt)
    r_psnr = psnr(recon, gt) - zf_metric
    return r_psnr

def relative_ssim(zf, recon, gt):
    zf_metric = ssim(zf, gt)
    r_ssim = ssim(recon, gt) - zf_metric
    return r_ssim

def relative_hfen(zf, recon, gt):
    zf_metric = hfen(zf, gt)
    r_hfen = hfen(recon, gt) - zf_metric
    return r_hfen

def get_metric(recon, gt, metric_type, zero_filled=None):
    if 'relative' in metric_type and zero_filled is None:
        raise Exception('No zf provided')
    if metric_type == 'mse':
        metric = mse_img(recon, gt)
    elif metric_type == 'psnr':
        metric = psnr(recon, gt)
    elif metric_type == 'relative psnr':
        metric = relative_psnr(zero_filled, recon, gt)
    elif metric_type == 'ssim':
        metric = ssim(recon, gt)
    elif metric_type == 'relative ssim':
        metric = relative_ssim(zero_filled, recon, gt)
    elif metric_type == 'hfen':
        metric = hfen(recon, gt)
    elif metric_type == 'relative hfen':
        metric = relative_hfen(zero_filled, recon, gt)
    else:
        raise Exception('no metric')

    return metric

def get_metrics(gt, recons, zf, metric_type, normalized, reduction='sum'):
    metrics = []
    if normalized:
        recons = utils.rescale(recons)
        gt = utils.rescale(gt)
        zf = utils.rescale(zf)


    if len(recons.shape) > 2:
        for i in range(len(recons)):
            metric = get_metric(recons[i], gt[i], metric_type, zero_filled=zf[i])
            metrics.append(metric)
    else:
        metric = myutils.metrics.get_metric(recons, gt, metric_type)
        metrics.append(metric)

    if reduction == 'mean':
        res = np.array(metrics).mean()
    elif reduction == 'sum':
        res = np.array(metrics).sum()
    else:
        res = np.array(metrics)
    return res
